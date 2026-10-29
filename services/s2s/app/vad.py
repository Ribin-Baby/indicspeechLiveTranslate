"""Silero VAD (ONNX, CPU) and a per-stream utterance segmenter (cuts the mic stream into utterances)."""
from __future__ import annotations

import collections
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

SR = 16_000
FRAME = 512                 # Silero v5+ takes exactly 512 samples at 16 kHz (32 ms)
CONTEXT = 64                # ...prefixed with the last 64 samples of the previous frame
FRAME_MS = FRAME * 1000 // SR


def silero_model_path() -> str:
    # find the package's bundled ONNX file without importing it (its __init__ needs torch)
    import importlib.util
    pkg = importlib.util.find_spec("silero_vad").submodule_search_locations[0]
    return os.path.join(pkg, "data", "silero_vad.onnx")


class SileroVAD:
    """One shared ONNX session; every stream keeps its own recurrent state."""

    def __init__(self, path: str | None = None):
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.log_severity_level = 3
        self.session = ort.InferenceSession(path or silero_model_path(), so,
                                            providers=["CPUExecutionProvider"])
        self.sr = np.array(SR, dtype=np.int64)

    def new_state(self) -> dict:
        return {"state": np.zeros((2, 1, 128), np.float32), "context": np.zeros((1, CONTEXT), np.float32)}

    def prob(self, frame: np.ndarray, st: dict) -> float:
        x = np.concatenate([st["context"], frame.reshape(1, -1)], axis=1)
        out, st["state"] = self.session.run(None, {"input": x, "state": st["state"], "sr": self.sr})
        st["context"] = x[:, -CONTEXT:]
        return float(out.reshape(-1)[0])


@dataclass
class SegmenterConfig:
    enabled: bool = True
    threshold: float = 0.5
    min_silence_ms: int = 500        # pause that ends an utterance
    speech_pad_ms: int = 200         # audio kept before speech start / after speech end
    min_speech_ms: int = 200         # shorter blips are dropped ("haan" is ~250 ms)
    max_segment_s: float = 15.0      # force a cut so a monologue still produces finals
    # In long speech a shorter pause ends the utterance too: once it has run phrase_s, a breath
    # of breath_ms between sentences cuts it, so a monologue is translated sentence by sentence
    # instead of in 15 s blocks (whose translations then queue up). 0 = off.
    phrase_s: float = 0.0
    breath_ms: int = 200
    # End-of-turn model (turn.py): once a pause reaches turn_check_ms, a "turn" event asks it
    # whether the speaker has finished; the answer comes back through turn_decided(). Finished:
    # the utterance ends then. Unfinished: a short utterance waits up to turn_wait_ms for the
    # rest; a long one skips the breath cut and waits the normal min_silence_ms. Until the
    # answer arrives, the rules above apply (without the breath cut). 0 = off.
    turn_check_ms: int = 0
    turn_wait_ms: int = 700


@dataclass
class Event:
    kind: str                        # "start" | "end" | "turn" (ask the end-of-turn model)
    start_ms: int
    end_ms: int = 0
    audio: np.ndarray | None = None  # set on "end"; on "turn", the utterance so far
    forced: bool = False
    token: int = 0                   # on "turn": pass back to turn_decided()


class Segmenter:
    """Feed 16 kHz float32 audio; get speech start/end events back.

    Timestamps are milliseconds of stream audio since the session started.
    """

    def __init__(self, vad: SileroVAD, cfg: SegmenterConfig):
        self.vad, self.cfg = vad, cfg
        self.st = vad.new_state()
        self.pending = np.zeros(0, np.float32)
        self.pos = 0                                    # samples consumed (frame-aligned)
        pad_frames = max(1, cfg.speech_pad_ms // FRAME_MS)
        self.pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=pad_frames)
        self.frames: list[np.ndarray] = []              # current segment
        self.seg_start = 0
        self.speech_ms = 0
        self.silence_ms = 0
        self.confirmed = False                          # has a "start" been emitted
        self.in_speech = False
        self.turn: str | None = None                    # this pause: None | "asked" | "finished" | "unfinished"
        self.turn_token = 0

    @property
    def neg_threshold(self) -> float:
        return max(self.cfg.threshold - 0.15, 0.01)

    def current_audio(self) -> np.ndarray | None:
        if not (self.in_speech and self.confirmed and self.frames):
            return None
        return np.concatenate(self.frames)

    def feed(self, audio: np.ndarray) -> list[Event]:
        self.pending = np.concatenate([self.pending, audio]) if len(self.pending) else audio
        n = len(self.pending) // FRAME
        events: list[Event] = []
        for i in range(n):
            events += self._frame(self.pending[i * FRAME: (i + 1) * FRAME])
        self.pending = self.pending[n * FRAME:]
        return events

    def flush(self) -> list[Event]:
        """Close the current utterance now (end of stream, or client-side endpointing)."""
        events: list[Event] = []
        if len(self.pending):
            if not self.cfg.enabled and not self.frames:
                # no VAD: everything since the last cut is one utterance
                self.seg_start = self.pos
                self.in_speech = self.confirmed = True
                events.append(Event("start", self._ms(self.seg_start)))
            if self.in_speech:
                self.frames.append(self.pending)
            self.pos += len(self.pending)
            self.pending = np.zeros(0, np.float32)
        if self.in_speech and self.confirmed:
            return events + self._end(forced=False, trim_silence=False)
        self._reset()
        return events

    # ------------------------------------------------------------------
    def _ms(self, samples: int) -> int:
        return samples * 1000 // SR

    def _frame(self, frame: np.ndarray) -> list[Event]:
        cfg = self.cfg
        events: list[Event] = []
        start_of_frame = self.pos
        self.pos += FRAME

        if not cfg.enabled:
            if not self.frames:
                self.seg_start = start_of_frame
                events.append(Event("start", self._ms(start_of_frame)))
            self.frames.append(frame)
            self.in_speech = self.confirmed = True
            if self._ms(self.pos - self.seg_start) >= cfg.max_segment_s * 1000:
                events += self._end(forced=True, trim_silence=False)
            return events

        p = self.vad.prob(frame, self.st)
        if not self.in_speech:
            if p >= cfg.threshold:
                self.in_speech, self.confirmed = True, False
                self.frames = list(self.pre_roll) + [frame]
                self.seg_start = start_of_frame - FRAME * len(self.pre_roll)
                self.speech_ms, self.silence_ms = FRAME_MS, 0
            else:
                self.pre_roll.append(frame)
            return events

        self.frames.append(frame)
        if p >= self.neg_threshold:
            self.speech_ms += FRAME_MS
            self.silence_ms = 0
            self.turn = None                            # talking again: a later pause is asked anew
        else:
            self.silence_ms += FRAME_MS

        if not self.confirmed:
            if self.speech_ms >= cfg.min_speech_ms:
                self.confirmed = True
                events.append(Event("start", self._ms(self.seg_start)))
            elif self.silence_ms >= cfg.min_silence_ms:
                self._reset()                       # a blip, not speech
                return events

        long = bool(cfg.phrase_s) and self._ms(self.pos - self.seg_start) >= cfg.phrase_s * 1000
        if (cfg.turn_check_ms and self.confirmed and self.turn is None
                and self.silence_ms >= cfg.turn_check_ms):
            self.turn, self.turn_token = "asked", self.turn_token + 1
            events.append(Event("turn", self._ms(self.seg_start), audio=np.concatenate(self.frames),
                                 token=self.turn_token))
        pause_ms = cfg.min_silence_ms
        if long and not cfg.turn_check_ms:
            pause_ms = min(pause_ms, cfg.breath_ms)
        if self.turn == "unfinished" and not long:
            pause_ms = max(pause_ms, cfg.turn_wait_ms)
        if self.confirmed and self.silence_ms >= pause_ms:
            events += self._end(forced=False, trim_silence=True)
        elif self.confirmed and self._ms(self.pos - self.seg_start) >= cfg.max_segment_s * 1000:
            events += self._end(forced=True, trim_silence=False)
            # the speaker is still talking: next frames open a new segment directly
            self.in_speech, self.confirmed = True, True
            self.seg_start = self.pos
            self.speech_ms = self.silence_ms = 0
            events.append(Event("start", self._ms(self.seg_start)))
        return events

    def _end(self, forced: bool, trim_silence: bool) -> list[Event]:
        frames = self.frames
        if trim_silence:
            # keep speech_pad_ms of the trailing silence, drop the rest
            drop = max(0, (self.silence_ms - self.cfg.speech_pad_ms) // FRAME_MS)
            frames = frames[: len(frames) - drop] if drop else frames
        audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
        ev = Event("end", self._ms(self.seg_start), self._ms(self.seg_start + len(audio)), audio, forced)
        self._reset()
        return [ev] if len(audio) else []

    def turn_decided(self, token: int, finished: bool) -> list[Event]:
        """The end-of-turn model's answer to the "turn" event `token`. If the speaker finished
        and is still silent, the utterance ends now (its "end" event is returned)."""
        if token != self.turn_token or self.turn != "asked":
            return []                                   # stale: they spoke again, or it already ended
        self.turn = "finished" if finished else "unfinished"
        if finished and self.in_speech and self.confirmed and self.silence_ms > 0:
            return self._end(forced=False, trim_silence=True)
        return []

    def _reset(self):
        self.frames = []
        self.in_speech = self.confirmed = False
        self.speech_ms = self.silence_ms = 0
        self.turn = None
        self.pre_roll.clear()
