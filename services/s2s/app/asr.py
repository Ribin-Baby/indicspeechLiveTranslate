"""Speech recognition for one caller, on the ASR NIM (AI4Bharat IndicConformer models
served by NVIDIA Riva, HTTP API).

The models are deployed in Riva's offline mode: each request transcribes a finished piece
of audio, in ~25 ms. The agent cuts each caller's microphone into utterances itself
(Silero VAD, vad.py) and sends every utterance to the NIM. While someone is still
talking, the utterance so far is re-transcribed every half second for live captions.

A stream emits the same events a streaming ASR would:
  {"type": "speech_start", "segment_id", "start_ms"}
  {"type": "partial", "segment_id", "text"}
  {"type": "final", "segment_id", "text", "start_ms", "end_ms", "forced", "decode_ms"}
  {"type": "error", "segment_id", "message"}
Timestamps are milliseconds of this stream's audio.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave

import aiohttp
import numpy as np

from .languages import ASR_CODE
from .vad import SR, Segmenter, SegmenterConfig, SileroVAD

log = logging.getLogger("s2s.asr")

ASR_URL = os.environ.get("ASR_URL", "http://127.0.0.1:8001")         # the NIM's HTTP API
PARTIAL_INTERVAL_S = float(os.environ.get("S2S_ASR_PARTIAL_INTERVAL_S", "0.5"))
MAX_SEGMENT_S = 15.0            # force a cut in a monologue so finals keep coming
# after this much speech, a breath this long also ends the utterance (vad.SegmenterConfig.phrase_s)
PHRASE_S = float(os.environ.get("S2S_PHRASE_S", "3.0"))
BREATH_MS = int(os.environ.get("S2S_BREATH_MS", "200"))
_vad: SileroVAD | None = None


def shared_vad() -> SileroVAD:
    """One ONNX session for every caller; each stream keeps its own recurrent state."""
    global _vad
    if _vad is None:
        _vad = SileroVAD()
    return _vad


def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


async def transcribe(http: aiohttp.ClientSession, audio: np.ndarray, lang: str) -> tuple[str, float]:
    """One offline request: 16 kHz float32 audio -> (text, milliseconds taken)."""
    form = aiohttp.FormData()
    form.add_field("file", wav_bytes(audio), filename="utterance.wav", content_type="audio/wav")
    form.add_field("language", ASR_CODE[lang])      # Riva's language code for that model
    t0 = time.perf_counter()
    async with http.post(f"{ASR_URL}/v1/audio/transcriptions", data=form) as r:
        if r.status != 200:
            raise RuntimeError(f"ASR: HTTP {r.status} {(await r.text())[:200]}")
        res = await r.json(content_type=None)
    return (res.get("text") or "").strip(), (time.perf_counter() - t0) * 1000


class AsrStream:
    """Feed 16 kHz PCM16 mic audio with feed(); read events from .events."""

    def __init__(self, http: aiohttp.ClientSession, lang: str, partials: bool, min_silence_ms: int):
        self.http, self.lang, self.partials = http, lang, partials
        self.seg = Segmenter(shared_vad(), SegmenterConfig(min_silence_ms=min_silence_ms,
                                                           max_segment_s=MAX_SEGMENT_S,
                                                           phrase_s=PHRASE_S, breath_ms=BREATH_MS))
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.segment_id = 0
        self.last_partial_at = 0          # samples into the utterance when the last partial was taken
        self.partial_task: asyncio.Task | None = None
        self.final_chain: asyncio.Task | None = None

    def feed(self, pcm16: bytes):
        x = np.frombuffer(pcm16, np.int16).astype(np.float32) / 32768.0
        for ev in self.seg.feed(x):
            if ev.kind == "start":
                self.segment_id += 1
                self.last_partial_at = 0
                self.events.put_nowait({"type": "speech_start", "segment_id": self.segment_id,
                                        "start_ms": ev.start_ms})
            else:
                self._final(self.segment_id, ev)
        if self.partials:
            self._maybe_partial()

    def close(self):
        for t in (self.partial_task, self.final_chain):
            if t is not None:
                t.cancel()

    def _maybe_partial(self):
        audio = self.seg.current_audio()
        if audio is None or len(audio) - self.last_partial_at < PARTIAL_INTERVAL_S * SR:
            return
        if self.partial_task is not None and not self.partial_task.done():
            return                        # one partial in flight per caller; skip, don't queue
        self.last_partial_at = len(audio)
        self.partial_task = asyncio.create_task(self._partial(self.segment_id, audio))

    async def _partial(self, seg_id: int, audio: np.ndarray):
        try:
            text, _ = await transcribe(self.http, audio, self.lang)
        except Exception as e:  # noqa: BLE001 - a lost partial is harmless; the final follows
            log.warning("partial transcription failed: %s", e)
            return
        # drop it if the utterance already ended (its final is coming)
        if text and seg_id == self.segment_id and self.seg.current_audio() is not None:
            self.events.put_nowait({"type": "partial", "segment_id": seg_id, "text": text})

    def _final(self, seg_id: int, ev):
        prev = self.final_chain

        async def run():
            try:
                text, ms = await transcribe(self.http, ev.audio, self.lang)
                msg = {"type": "final", "segment_id": seg_id, "text": text, "start_ms": ev.start_ms,
                       "end_ms": ev.end_ms, "forced": ev.forced, "decode_ms": round(ms, 1)}
            except Exception as e:  # noqa: BLE001 - one failed utterance doesn't end the stream
                log.exception("transcription of segment %d failed", seg_id)
                msg = {"type": "error", "segment_id": seg_id, "message": f"transcription failed: {e}"}
            if prev is not None:
                await prev                # finals go out in utterance order
            self.events.put_nowait(msg)

        self.final_chain = asyncio.create_task(run())
