# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Degradation measurements of a variant against its clean parent."""
from __future__ import annotations

import math

import numpy as np


def si_sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (both signals mean-removed)."""
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    n = min(len(ref), len(est))
    ref, est = ref[:n] - ref[:n].mean(), est[:n] - est[:n].mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = alpha * ref
    noise = est - target
    return 10.0 * math.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps))


def estoi(ref: np.ndarray, est: np.ndarray, fs: int = 16000) -> float:
    """Extended short-time objective intelligibility (pystoi, extended=True)."""
    from pystoi import stoi

    n = min(len(ref), len(est))
    return float(stoi(np.asarray(ref[:n], dtype=np.float64), np.asarray(est[:n], dtype=np.float64), fs, extended=True))


def clipped_fraction(x16: np.ndarray) -> float:
    """Share of 16-bit samples at full scale (-32768 or 32767)."""
    x16 = np.asarray(x16)
    return float(np.mean((x16 >= 32767) | (x16 <= -32768))) if len(x16) else 0.0
