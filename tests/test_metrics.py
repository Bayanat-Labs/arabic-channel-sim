# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arabic_channel_sim import clipped_fraction, si_sdr


def test_si_sdr_is_scale_invariant():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(16000)
    n = rng.standard_normal(16000)
    a = si_sdr(x, x + 0.1 * n)
    b = si_sdr(x, 3.0 * (x + 0.1 * n))
    assert a == pytest.approx(b, abs=1e-9)
    assert a == pytest.approx(20.0, abs=0.5)
    assert si_sdr(x, x) > 100


def test_estoi_identity_and_degradation(speechlike):
    pytest.importorskip("pystoi")
    from arabic_channel_sim import estoi

    rng = np.random.default_rng(5)
    assert estoi(speechlike, speechlike) == pytest.approx(1.0, abs=1e-6)
    noisy = speechlike + 0.2 * rng.standard_normal(len(speechlike))
    assert estoi(speechlike, noisy) < 0.9


def test_clipped_fraction():
    x = np.array([0, 32767, -32768, 100, -100], dtype=np.int16)
    assert clipped_fraction(x) == pytest.approx(0.4)
