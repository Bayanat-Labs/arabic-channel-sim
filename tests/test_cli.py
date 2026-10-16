# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import soundfile as sf

from arabic_channel_sim.cli import main


def test_list(capsys):
    assert main(["list", "--json"]) == 0
    ids = [c["id"] for c in json.loads(capsys.readouterr().out)]
    assert ids == ["BL1", "RS2", "CD1", "CD6", "RV2"]


def test_render_and_measure(tmp_path, capsys, speechlike):
    src = tmp_path / "clean.wav"
    sf.write(src, speechlike, 16000, subtype="PCM_16")
    out = tmp_path / "rs2.flac"
    assert main(["render", "--condition", "RS2", "--input", str(src), "--output", str(out)]) == 0
    rec = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rec["condition_id"] == "RS2" and rec["num_samples"] == len(speechlike)
    y, fs = sf.read(out)
    assert fs == 16000 and len(y) == len(speechlike)
    assert main(["measure", str(src), str(out)]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["rolloff95_hz"] < 2100
    assert np.isfinite(m["si_sdr_db"])
