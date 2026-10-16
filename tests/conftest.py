# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess

import numpy as np
import pytest


def _has_encoder(name: str) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    out = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


HAS_MULAW = _has_encoder("pcm_mulaw")
HAS_OPUS = _has_encoder("libopus")


@pytest.fixture(scope="session")
def speechlike():
    """Two seconds of a deterministic speech-like test signal on the 16-bit grid (harmonics + noise bursts)."""
    fs = 16000
    rng = np.random.default_rng(7)
    t = np.arange(2 * fs) / fs
    f0 = 140 + 30 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    x = sum((0.5 / k) * np.sin(k * phase) for k in range(1, 40))
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t)) ** 2
    x = x * env + 0.02 * rng.standard_normal(len(t)) * (env > 0.5)
    x = 0.3 * x / np.max(np.abs(x))
    return np.round(x * 32768) / 32768
