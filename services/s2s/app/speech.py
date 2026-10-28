"""Shaping the translated speech before it plays (16-bit mono PCM), as meetmind's interpreter does:

- SilenceTrimmer: Magpie pads each clip with ~0.1-0.15 s of silence before the speech (up to
  0.45 s before a long sentence) and ~0.15-0.25 s after it. Dropping the leading silence starts
  the speech sooner; dropping the trailing silence keeps clips played back to back from adding
  0.3-0.4 s of dead air between them. Pauses inside a clip are kept.
- time_stretch / catch_up_rate: a translation that had to wait for the previous one to finish
  plays faster, pitch unchanged (WSOLA), the way a human interpreter speeds up when behind.
  Magpie ignores SSML speaking rates, so it's done on the audio. Hindi runs ~1.6x as long as the
  English it translates, so in a long English turn the voice otherwise falls further behind
  with every sentence.
"""
from __future__ import annotations

import numpy as np

QUIET = 250          # below this a sample is silence (about -42 dBFS): the voice's padding
LEAD_MS = 50         # kept before the first sound, so soft starts aren't clipped
TAIL_MS = 120        # and after the last: with the next clip's lead, a natural ~170 ms pause
MAX_CATCH_UP = 1.4


class SilenceTrimmer:
    """Feed a voice stream's bytes as they arrive; get them back without the silence around the
    speech. Before the first sound, silence is dropped as it arrives; after it, a quiet stretch
    is held back until more sound follows (a pause: passed on whole) or the stream ends (the
    padding: cut to TAIL_MS)."""

    def __init__(self, rate: int):
        self.lead = rate * LEAD_MS // 1000
        self.tail = rate * TAIL_MS // 1000
        self.started = False
        self.held = np.zeros(0, np.int16)
        self.odd = b""

    def feed(self, data: bytes) -> bytes:
        data = self.odd + data
        self.odd, data = (data[-1:], data[:-1]) if len(data) % 2 else (b"", data)
        x = np.concatenate([self.held, np.frombuffer(data, np.int16)])
        loud = np.flatnonzero(np.abs(x.astype(np.int32)) >= QUIET)
        if not self.started:
            if not len(loud):
                self.held = x[-self.lead:] if self.lead else x[:0]    # still silence: keep only a lead's worth
                return b""
            self.started = True
            start = max(0, loud[0] - self.lead)
            x, loud = x[start:], loud - start
        end = loud[-1] + 1 if len(loud) else 0
        self.held = x[end:]
        return x[:end].tobytes()

    def end(self) -> bytes:
        return self.held[: self.tail].tobytes() if self.started else b""


def catch_up_rate(waited_ms: float) -> float:
    """How much faster to speak, given how long this translation waited for the previous one to
    finish: from 300 ms on (1.06x), 1.2x at 1 s, up to 1.4x from 2 s (meetmind's rule)."""
    if waited_ms < 300:
        return 1.0
    return min(MAX_CATCH_UP, round(1 + waited_ms / 5000, 2))


def time_stretch(x: np.ndarray, rate: float, sr: int) -> np.ndarray:
    """`rate` times faster, same pitch (WSOLA): overlapping 30 ms windows, each taken `rate`
    times further along the input and nudged (within +-10 ms) to where its waveform best
    continues what's already written, so the joins don't click. The search runs on every 4th
    sample, then refines around the best match."""
    if rate <= 1.001 or len(x) < sr // 10:
        return x
    win = round(sr * 0.03)
    hop = win // 2
    seek = round(sr * 0.01)
    corr = win // 4
    step = 4
    xf = x.astype(np.float32)
    out_len = int(len(x) / rate)
    out = np.zeros(out_len + win, np.float32)
    norm = np.zeros(out_len + win, np.float32)
    hann = np.hanning(win).astype(np.float32)
    prev_end = 0
    for pos in range(0, out_len, hop):
        nominal = round(pos * rate)
        best = nominal
        ref = xf[prev_end: prev_end + corr]
        lo, hi = max(0, nominal - seek), min(len(x) - win, nominal + seek)
        if pos > 0 and hi > lo and len(ref) == corr:
            c = np.correlate(xf[lo: hi + corr][::step], ref[::step], "valid")
            k = lo + step * int(np.argmax(c))
            a, b = max(lo, k - step), min(hi, k + step)
            best = a + int(np.argmax(np.correlate(xf[a: b + corr], ref, "valid")))
        best = max(0, min(best, len(x) - win))
        n = min(win, len(out) - pos)
        out[pos: pos + n] += xf[best: best + n] * hann[:n]
        norm[pos: pos + n] += hann[:n]
        prev_end = best + hop
    y = np.divide(out[:out_len], norm[:out_len], out=np.zeros(out_len, np.float32), where=norm[:out_len] > 1e-3)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)
