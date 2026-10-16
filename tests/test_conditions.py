# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os

import arabic_channel_sim as acs

EXPECTED = ["BL1", "RS2", "CD1", "CD6", "RV2"]


def test_conditions_file_is_packaged_and_parses():
    doc = acs.load_conditions()
    assert os.path.exists(acs.DEFAULT_CONDITIONS)
    assert doc["version"] == "conditions_v1"
    assert [c["id"] for c in doc["conditions"]] == EXPECTED
    for c in doc["conditions"]:
        assert c["family"] in {"bandlimit", "resample", "codec", "reverb"}
        assert c["steps"], c["id"]
        assert c["description"]


def test_conditions_hash_is_stable_function_of_file():
    with open(acs.DEFAULT_CONDITIONS, "rb") as f:
        assert acs.conditions_sha256() == hashlib.sha256(f.read()).hexdigest()


def test_variant_seed_formula():
    pid, cid = "10019390257075855641.wav", "RV2"
    expected = int(hashlib.sha256((pid + cid).encode()).hexdigest()[:8], 16)
    assert acs.variant_seed(pid, cid) == expected
    assert acs.variant_seed(pid, "CD1") != expected


def test_every_step_resamples_back_to_16k():
    doc = acs.load_conditions()
    for c in doc["conditions"]:
        rate = 16000
        for st in c["steps"]:
            if st["op"] == "resample":
                rate = st["to_hz"]
            if st["op"] == "ffmpeg_codec":
                assert st["sample_rate_hz"] == rate
        assert rate == 16000, c["id"]
