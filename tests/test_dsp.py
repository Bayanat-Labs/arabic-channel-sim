# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from arabic_channel_sim import dsp

FS = 16000


def tone(freq, seconds=1.0, fs=FS, amp=0.5):
    t = np.arange(int(seconds * fs)) / fs
    return amp * np.sin(2 * np.pi * freq * t)


def gain_db(x, y):
    sl = slice(len(x) // 4, 3 * len(x) // 4)
    return 20 * np.log10(np.sqrt(np.mean(y[sl] ** 2)) / np.sqrt(np.mean(x[sl] ** 2)))


def test_to_int16_rounds_and_clips():
    x = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5 / 32768, 1.5 / 32768])
    assert dsp.to_int16(x).tolist() == [0, 32767, -32768, 32767, -32768, 0, 2]


def test_resample_round_trip_keeps_length_and_low_band():
    x = tone(500)
    y = dsp.resample(dsp.resample(x, FS, 8000), 8000, FS)
    assert len(y) == len(x)
    assert abs(gain_db(x, y)) < 0.1
    assert dsp.estimate_delay(x, y, 160) == 0


def test_resample_4k_removes_content_above_2k():
    x = tone(3000)
    y = dsp.resample(dsp.resample(x, FS, 4000), 4000, FS)
    assert gain_db(x, y) < -40


@pytest.mark.parametrize("freq,inside", [(1000, True), (3000, True), (100, False), (3900, False)])
def test_telephone_bandpass(freq, inside):
    x = tone(freq, fs=8000)
    y = dsp.bandpass(x, 8000, 300, 3400, 511, ("kaiser", 8.0))
    g = gain_db(x, y)
    if inside:
        assert abs(g) < 0.5
    else:
        assert g < -30


def test_bandpass_is_zero_delay():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8000)
    y = dsp.bandpass(x, 8000, 300, 3400, 511, ("kaiser", 8.0))
    assert dsp.estimate_delay(x, y, 400) == 0


@pytest.mark.parametrize("shift", [-37, 0, 5, 120])
def test_estimate_delay_and_shift(shift):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000)
    y = np.roll(x, shift)
    d = dsp.estimate_delay(x, y, 200)
    assert d == shift
    z = dsp.shift_to_length(y, d, len(x))
    inner = slice(200, 3800)
    assert np.allclose(z[inner], x[inner])


def test_level_match_sets_active_rms():
    x = tone(440)
    y = 0.25 * x
    mask = dsp.active_mask(x, FS)
    z, g = dsp.level_match(x, y, mask)
    assert abs(g - 20 * np.log10(4)) < 1e-9
    assert abs(dsp.active_rms(z, mask) - dsp.active_rms(x, mask)) < 1e-12


def test_clip_guard_limits_full_scale_share():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(100000) * 0.5
    z, db = dsp.clip_guard(y, 0.001)
    assert db < 0
    q = dsp.to_int16(z)
    assert np.mean((q >= 32767) | (q <= -32768)) <= 0.001
    quiet = 0.1 * y
    same, db0 = dsp.clip_guard(quiet, 0.001)
    assert db0 == 0.0 and same is quiet


def test_spectral_rolloff():
    x = tone(1000) + tone(5000, amp=0.05)
    r = dsp.spectral_rolloff(x, FS)
    assert 950 <= r <= 1050


def test_convolve_rir_aligns_direct_path():
    x = np.zeros(1000)
    x[100] = 1.0
    h = np.zeros(300)
    h[40] = 1.0
    h[200] = 0.5
    y, tail_db = dsp.convolve_rir(x, h, 40, len(x))
    assert len(y) == len(x)
    assert y[100] == pytest.approx(1.0)
    assert y[260] == pytest.approx(0.5)
    assert tail_db < -200  # only FFT round-off beyond the cut
