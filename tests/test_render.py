# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

import arabic_channel_sim as acs
from arabic_channel_sim import dsp

from conftest import HAS_MULAW, HAS_OPUS

NEEDS = {"CD1": HAS_MULAW, "CD6": HAS_OPUS}


def conditions():
    return [c["id"] for c in acs.load_conditions()["conditions"]]


@pytest.mark.parametrize("cid", conditions())
def test_render_contract(cid, speechlike):
    if not NEEDS.get(cid, True):
        pytest.skip("ffmpeg encoder not available")
    doc = acs.load_conditions()
    y16, rec = acs.render(speechlike, cid, "example.wav", doc)
    assert y16.dtype == np.int16
    assert len(y16) == len(speechlike) == rec["num_samples"]
    assert rec["seed"] == acs.variant_seed("example.wav", cid)
    assert rec["clipped_fraction"] <= 0.001
    y = y16.astype(np.float64) / 32768
    mask = dsp.active_mask(speechlike, 16000, 20, 30)
    level = 20 * np.log10(dsp.active_rms(y, mask) / dsp.active_rms(speechlike, mask))
    assert abs(level) <= 0.5
    ck = doc["by_id"][cid].get("checks", {})
    if "rolloff95_max_hz" in ck:
        assert dsp.spectral_rolloff(y, 16000, mask=mask) < ck["rolloff95_max_hz"]
    if cid == "RV2":
        from arabic_channel_sim import rirs

        assert rec["rir_id"] == rirs.rir_ids(0.6)[rec["seed"] % 20]
    # deterministic
    y16b, rec_b = acs.render(speechlike, cid, "example.wav", doc)
    assert np.array_equal(y16, y16b) and rec == rec_b


def test_bandlimit_removes_high_band(speechlike):
    from scipy import signal

    y16, _ = acs.render(speechlike, "BL1", "x.wav")
    f, p = signal.welch(y16 / 32768, fs=16000, nperseg=1024)
    f0, p0 = signal.welch(speechlike, fs=16000, nperseg=1024)
    hi = f > 4000
    assert p[hi].sum() / p.sum() < 1e-4 < p0[hi].sum() / p0.sum()


def test_flac_round_trip_is_lossless(speechlike):
    x16 = dsp.to_int16(speechlike)
    data = acs.encode_flac(x16)
    y, fs = acs.decode_audio(data)
    assert fs == 16000
    assert np.array_equal(dsp.to_int16(y), x16)


def test_unknown_condition(speechlike):
    with pytest.raises(KeyError):
        acs.render(speechlike, "XX9", "x.wav")
