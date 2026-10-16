# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os

import numpy as np
import pytest
import soundfile as sf

from arabic_channel_sim import load_conditions, rirs


def test_rir_set_matches_spec():
    spec = load_conditions()["rirs"]
    params = rirs.load_params()
    assert len(params) == len(spec["rt60_targets_s"]) * spec["rooms_per_target"]
    for t in spec["rt60_targets_s"]:
        assert len(rirs.rir_ids(t)) == spec["rooms_per_target"]
    for p in params:
        assert p["synthetic"] is True
        assert abs(p["measured_rt60_s"] - p["rt60_target_s"]) / p["rt60_target_s"] <= spec["rt60_tolerance"]
        lo, hi = spec["source_mic_distance_m"]
        assert lo <= p["source_mic_distance_m"] <= hi
        path = os.path.join(rirs.default_dir(), p["file"])
        with open(path, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == p["sha256"]
        info = sf.info(path)
        assert info.samplerate == 16000 and info.channels == 1 and info.frames == p["length_samples"]


def test_recorded_rt60_and_drr_recompute_from_files():
    for p in rirs.load_params()[::7]:
        h, _ = rirs.load_rir(p["rir_id"])
        assert rirs.schroeder_rt60(h, 16000) == pytest.approx(p["measured_rt60_s"], abs=2e-3)
        assert rirs.drr_db(h, 16000) == pytest.approx(p["drr_db"], abs=1e-2)
        assert rirs.direct_index(h) == p["direct_index"]


def test_schroeder_rt60_on_exponential_decay():
    fs = 16000
    rng = np.random.default_rng(6)
    t = np.arange(int(1.5 * fs)) / fs
    for rt60 in (0.3, 0.6, 1.0):
        h = rng.standard_normal(len(t)) * 10 ** (-3 * t / rt60)
        assert rirs.schroeder_rt60(h, fs) == pytest.approx(rt60, rel=0.05)


def test_unknown_rir_id():
    with pytest.raises(KeyError):
        rirs.load_rir("rirs_v1/0.6/room99")
