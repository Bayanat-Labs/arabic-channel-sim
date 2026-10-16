# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Signal-processing building blocks used by the conditions. All functions take and return float64 arrays."""
from __future__ import annotations

import math
import shutil
import subprocess
from fractions import Fraction
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import signal

FULL_SCALE = 32768.0


def to_int16(x: np.ndarray) -> np.ndarray:
    """Round to nearest 16-bit integer and clip; no dither."""
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * FULL_SCALE), -32768, 32767).astype(np.int16)


def quantise16(x: np.ndarray) -> np.ndarray:
    """Float signal -> the float value of its 16-bit quantisation."""
    return to_int16(x).astype(np.float64) / FULL_SCALE


def resample(x: np.ndarray, from_hz: int, to_hz: int, window=("kaiser", 5.0)) -> np.ndarray:
    if from_hz == to_hz:
        return np.asarray(x, dtype=np.float64)
    f = Fraction(int(to_hz), int(from_hz))
    return signal.resample_poly(np.asarray(x, dtype=np.float64), f.numerator, f.denominator, window=tuple(window))


def bandpass(x: np.ndarray, fs: int, low_hz: float, high_hz: float, numtaps: int, window=("kaiser", 8.0)) -> np.ndarray:
    """Linear-phase FIR band-pass, applied centred so that the output has zero delay."""
    if numtaps % 2 != 1:
        raise ValueError("numtaps must be odd for a zero-delay linear-phase filter")
    h = signal.firwin(numtaps, [low_hz, high_hz], pass_zero=False, fs=fs, window=tuple(window))
    return signal.oaconvolve(np.asarray(x, dtype=np.float64), h, mode="same")


def ffmpeg_path() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError("ffmpeg was not found on PATH; it is needed for codec conditions")
    return p


def ffmpeg_version() -> str:
    out = subprocess.run([ffmpeg_path(), "-hide_banner", "-version"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()[0].strip() if out else "unknown"


def ffmpeg_codec(x: np.ndarray, fs: int, encode: Sequence[str], decode_input: Sequence[str]) -> np.ndarray:
    """Encode the 16-bit version of x with ffmpeg, decode it again, return float samples at fs."""
    ff = ffmpeg_path()
    pcm = to_int16(x).tobytes()
    enc_cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "s16le", "-ar", str(fs), "-ac", "1",
               "-i", "pipe:0", *encode, "pipe:1"]
    coded = subprocess.run(enc_cmd, input=pcm, capture_output=True, check=True).stdout
    dec_cmd = [ff, "-hide_banner", "-loglevel", "error", "-nostdin", *decode_input, "-i", "pipe:0",
               "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(fs), "-ac", "1", "pipe:1"]
    raw = subprocess.run(dec_cmd, input=coded, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / FULL_SCALE


def estimate_delay(ref: np.ndarray, deg: np.ndarray, max_lag: int) -> int:
    """Lag d (samples) maximising the cross-correlation, so that deg[n + d] ~ ref[n]. |d| <= max_lag."""
    n = len(ref) + len(deg) - 1
    nfft = 1 << (n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(deg, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    lags = np.concatenate([xc[: max_lag + 1], xc[-max_lag:]]) if max_lag > 0 else xc[:1]
    idx = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)]) if max_lag > 0 else np.array([0])
    return int(idx[int(np.argmax(lags))])


def shift_to_length(y: np.ndarray, delay: int, length: int) -> np.ndarray:
    """Return y advanced by ``delay`` samples (y[n + delay]) and cut or zero-padded to ``length``."""
    out = np.zeros(length, dtype=np.float64)
    if delay >= 0:
        seg = y[delay: delay + length]
        out[: len(seg)] = seg
    else:
        seg = y[: max(0, length + delay)]
        out[-delay: -delay + len(seg)] = seg
    return out


def active_mask(x: np.ndarray, fs: int, frame_ms: float = 20.0, within_db: float = 30.0) -> np.ndarray:
    """Boolean per-sample mask of frames whose energy is within ``within_db`` of the loudest frame."""
    flen = int(round(fs * frame_ms / 1000.0))
    nfr = int(math.ceil(len(x) / flen))
    pad = np.zeros(nfr * flen)
    pad[: len(x)] = x
    e = np.mean(pad.reshape(nfr, flen) ** 2, axis=1)
    thr = e.max() * 10.0 ** (-within_db / 10.0)
    keep = e >= thr if e.max() > 0 else np.ones(nfr, dtype=bool)
    return np.repeat(keep, flen)[: len(x)]


def active_rms(x: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x)[mask] ** 2)))


def level_match(ref: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    r, v = active_rms(ref, mask), active_rms(y, mask)
    if v <= 0:
        return y, 0.0
    g = r / v
    return y * g, 20.0 * math.log10(g)


def clip_guard(y: np.ndarray, max_fraction: float) -> Tuple[np.ndarray, float]:
    """Lower the level just enough that at most ``max_fraction`` of the samples reach 16-bit full scale.

    Returns (signal, applied change in dB; 0.0 when nothing was needed).
    """
    a = np.sort(np.abs(np.asarray(y, dtype=np.float64)))[::-1]
    k = int(math.floor(max_fraction * len(a)))
    if k >= len(a):
        return y, 0.0
    limit = 32766.0 / FULL_SCALE
    v = float(a[k])
    if v < limit:
        return y, 0.0
    g = limit / v
    return y * g, 20.0 * math.log10(g)


def convolve_rir(x: np.ndarray, h: np.ndarray, direct: int, length: int) -> Tuple[np.ndarray, float]:
    """Convolve, align on the direct path and cut to ``length``. Returns (signal, trimmed tail energy in dB)."""
    full = signal.fftconvolve(np.asarray(x, dtype=np.float64), np.asarray(h, dtype=np.float64), mode="full")
    y = full[direct: direct + length]
    kept = float(np.sum(y ** 2))
    tail = float(np.sum(full[direct + length:] ** 2))
    tail_db = 10.0 * math.log10(tail / kept) if tail > 0 and kept > 0 else float("-inf")
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
    return y, tail_db


def spectral_rolloff(x: np.ndarray, fs: int, fraction: float = 0.95, nperseg: int = 1024,
                     mask: Optional[np.ndarray] = None) -> float:
    """Frequency (Hz) below which ``fraction`` of the signal power lies (Welch spectrum over active samples)."""
    x = np.asarray(x, dtype=np.float64)
    if mask is not None and mask.any():
        x = x[mask]
    f, p = signal.welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    c = np.cumsum(p)
    if c[-1] <= 0:
        return 0.0
    return float(f[int(np.searchsorted(c, fraction * c[-1]))])
