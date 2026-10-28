"""Background noise removal for one caller's microphone, on NVIDIA's Maxine BNR NIM (gRPC at
BNR_GRPC_URL; off when unset).

The NIM's streaming mode takes 10 ms chunks of 32-bit float audio at the rate its model profile
was built for (48 kHz: profile v2-48k) and returns each chunk cleaned, in order. Measured
(2026-10-28, an L40S NIM on the LAN): ~13 ms per round trip, and the cleaned audio lags the input
by ~40 ms (the model's lookahead); the noise floor drops by ~37 dB while speech keeps its level.
The mic (16 kHz, after echo cancellation) goes up to 48 kHz and back down with LiveKit's QUICK
resampler, which holds back no audio. In calls the final transcript came ~70 ms later with it
(median 718 vs 649 ms after speech end; clients/asr_timing_test.py) and the translation was heard
~60-90 ms later; in noise (5 dB SNR) the Hindi meaning came through in 4 of 4 runs, against
wrong words in 4 of 6 runs without it (README, "Background noise removal").

If the NIM can't be reached, errors or falls behind, the stream turns itself off and the agent
feeds the raw mic to speech recognition again: a call never stops because of noise removal.

The two messages are encoded by hand (bnr.proto in github.com/NVIDIA-Maxine/nim-clients):
  EnhanceAudioRequest  oneof { bytes audio_stream_data = 1; EnhanceAudioConfig config = 2; }
  EnhanceAudioConfig   { optional float intensity_ratio = 1; }
  EnhanceAudioResponse oneof { bytes audio_stream_data = 1; EnhanceAudioConfig config = 2; }
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import struct

import grpc
import numpy as np
from livekit import rtc

log = logging.getLogger("s2s.bnr")

BNR_GRPC_URL = os.environ.get("BNR_GRPC_URL", "")
BNR_RATE = int(os.environ.get("BNR_SAMPLE_RATE", "48000"))    # the NIM's model profile
INTENSITY = float(os.environ.get("BNR_INTENSITY", "1.0"))      # 0..1, how much noise to take out
LAG_S = float(os.environ.get("BNR_LAG_MS", "40")) / 1000       # the cleaned audio's delay (measured)
METHOD = "/nvidia.ai4m.bnr.v1.BNR/EnhanceAudio"
CHUNK = BNR_RATE // 100                                       # 10 ms, as the NIM's streaming mode expects
MAX_IN_FLIGHT = 50                                            # 0.5 s unanswered: the NIM fell behind


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def request_audio(f32: bytes) -> bytes:
    return b"\x0a" + _varint(len(f32)) + f32


def request_config(intensity: float) -> bytes:
    inner = b"\x0d" + struct.pack("<f", intensity)
    return b"\x12" + _varint(len(inner)) + inner


def response_audio(msg: bytes) -> bytes | None:
    """The audio in a response, or None for the config echo."""
    if not msg or msg[0] != 0x0A:
        return None
    n = shift = 0
    i = 1
    while True:
        b = msg[i]
        i += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return msg[i: i + n]


class BnrStream:
    """One mic through the NIM: push() 16-bit PCM at `rate` with the wall time it arrived; read
    the cleaned audio, at the same rate, from .out as (pcm, wall time its content was spoken).
    None in .out means the stream turned itself off."""

    def __init__(self, rate: int):
        self.rate = rate
        q = rtc.AudioResamplerQuality.QUICK
        self.up = rtc.AudioResampler(rate, BNR_RATE, num_channels=1, quality=q)
        self.down = rtc.AudioResampler(BNR_RATE, rate, num_channels=1, quality=q)
        self.pending = np.zeros(0, np.float32)        # 48 kHz samples short of a whole chunk
        self.pushed = 0                               # 48 kHz samples pushed so far
        self.chunked = 0                              # ... of which sent in chunks
        self.in_flight: collections.deque[float] = collections.deque()   # when each sent chunk was spoken
        self.out: asyncio.Queue[tuple[bytes, float] | None] = asyncio.Queue()
        self.writes: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.alive = False
        self.channel: grpc.aio.Channel | None = None
        self.tasks: list[asyncio.Task] = []

    async def start(self) -> bool:
        try:
            self.channel = grpc.aio.insecure_channel(BNR_GRPC_URL)
            await asyncio.wait_for(self.channel.channel_ready(), 3)
            call = self.channel.stream_stream(METHOD, request_serializer=bytes, response_deserializer=bytes)()
            await call.write(request_config(INTENSITY))
        except Exception as e:  # noqa: BLE001 - no noise removal is better than no call
            log.warning("background noise removal unavailable (%s): using the raw mic", e or type(e).__name__)
            await self.close()
            return False
        self.alive = True
        self.tasks = [asyncio.create_task(self._write(call)), asyncio.create_task(self._read(call))]
        log.info("background noise removal on (%s, intensity %.2f)", BNR_GRPC_URL, INTENSITY)
        return True

    def push(self, pcm16: bytes, at: float) -> bool:
        """Queue mic audio that ended at wall time `at`. False if the stream is off (feed it raw)."""
        if not self.alive:
            return False
        for f in self.up.push(bytearray(pcm16)):
            x = np.frombuffer(f.data, np.int16).astype(np.float32) / 32768
            self.pending = np.concatenate([self.pending, x])
            self.pushed += len(x)
        while len(self.pending) >= CHUNK:
            chunk, self.pending = self.pending[:CHUNK], self.pending[CHUNK:]
            self.chunked += CHUNK
            self.in_flight.append(at - (self.pushed - self.chunked) / BNR_RATE)
            self.writes.put_nowait(request_audio(chunk.tobytes()))
        if len(self.in_flight) > MAX_IN_FLIGHT:
            self._off(f"no answer for {len(self.in_flight) * 10} ms")
        return True

    async def _write(self, call):
        try:
            while (msg := await self.writes.get()) is not None:
                await call.write(msg)
            await call.done_writing()
        except Exception as e:  # noqa: BLE001
            self._off(f"send failed: {e}")

    async def _read(self, call):
        try:
            async for msg in call:
                audio = response_audio(msg)
                if audio is None or not self.in_flight:
                    continue
                spoken = self.in_flight.popleft() - LAG_S
                y = np.clip(np.frombuffer(audio, np.float32), -1, 1)
                for f in self.down.push(bytearray((y * 32767).astype(np.int16).tobytes())):
                    self.out.put_nowait((f.data.tobytes(), spoken))
            self._off("the NIM ended the stream")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self._off(f"receive failed: {e}")

    def _off(self, reason: str):
        if self.alive:
            self.alive = False
            log.warning("background noise removal off for this mic: %s; using the raw mic", reason)
            self.out.put_nowait(None)

    async def close(self):
        self.alive = False
        self.writes.put_nowait(None)
        for t in self.tasks:
            t.cancel()
        if self.channel is not None:
            await self.channel.close()
