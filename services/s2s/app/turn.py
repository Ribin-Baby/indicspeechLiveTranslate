"""End-of-turn detection: Pipecat's Smart Turn v3.2 (BSD-2, github.com/pipecat-ai/smart-turn;
model pipecat-ai/smart-turn-v3 on Hugging Face), an 8 MB int8 ONNX model on the CPU, 23
languages including Hindi and English.

Given the last (up to) 8 s of an utterance, it scores from 0 to 1 how likely the speaker has
finished what they were saying, from the words and the intonation together. The agent asks it
when a pause starts (vad.py): a finished thought ends the utterance at once, and an
unfinished one ("so I was thinking that...") waits a little longer for the rest, so it isn't
translated as a fragment.

Its input is Whisper's log-mel spectrogram of 8 s of 16 kHz audio (80 mel bands, 25 ms
windows every 10 ms -> 80 x 800), with shorter audio padded with silence at the start. It is
computed here in numpy exactly as transformers' WhisperFeatureExtractor(chunk_length=8) with
do_normalize=True does in Smart Turn's own inference.py, so the image needs neither
transformers nor torch.
"""
from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort

MODEL_PATH = os.environ.get("S2S_TURN_MODEL", "/srv/models/smart-turn-v3.2-cpu.onnx")
SR = 16_000
SECONDS = 8
N_FFT, HOP, N_MELS = 400, 160, 80


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, np.float64)
    mel = 3.0 * f / 200.0
    log = f >= 1000.0
    return np.where(log, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) * (27.0 / np.log(6.4)), mel)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64)
    hz = 200.0 * m / 3.0
    log = m >= 15.0
    return np.where(log, 1000.0 * np.exp(np.log(6.4) / 27.0 * (m - 15.0)), hz)


def _mel_filters() -> np.ndarray:
    """(201, 80) triangular filters, slaney-normalised, 0-8 kHz (transformers' mel_filter_bank)."""
    fft_freqs = np.linspace(0, SR // 2, 1 + N_FFT // 2)
    mel = np.linspace(_hz_to_mel(0.0), _hz_to_mel(8000.0), N_MELS + 2)
    filter_freqs = _mel_to_hz(mel)
    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    return filters * (2.0 / (filter_freqs[2:N_MELS + 2] - filter_freqs[:N_MELS]))[None, :]


_FILTERS = _mel_filters()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))      # periodic Hann


def features(audio: np.ndarray) -> np.ndarray:
    """16 kHz float audio -> (1, 80, 800) float32 input for the model."""
    n = SR * SECONDS
    x = np.asarray(audio, np.float64)[-n:]
    x = np.pad(x, (n - len(x), 0))                                         # silence first
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)                           # do_normalize
    x = np.pad(x, N_FFT // 2, mode="reflect")                              # center=True
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]     # (801, 400)
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=N_FFT, axis=1)) ** 2    # (801, 201)
    log = np.log10(np.maximum(1e-10, _FILTERS.T @ power.T))[:, :-1]        # (80, 800)
    log = np.maximum(log, log.max() - 8.0)
    return ((log + 4.0) / 4.0).astype(np.float32)[None]


class SmartTurn:
    """One CPU session shared by every caller (ONNX Runtime sessions are thread-safe)."""

    def __init__(self, path: str = MODEL_PATH):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])

    def finished(self, audio: np.ndarray) -> float:
        """0..1: how likely the speaker has finished, given the utterance so far (16 kHz float)."""
        return float(self.session.run(None, {"input_features": features(audio)})[0].reshape(-1)[0])
