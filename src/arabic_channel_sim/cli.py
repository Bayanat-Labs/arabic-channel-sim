# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Command line: ``arabic-channel-sim list | render | measure | rirs``."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _read_parent(path: str) -> np.ndarray:
    import soundfile as sf

    from .dsp import quantise16

    x, fs = sf.read(path, dtype="float64", always_2d=True)
    if fs != 16000:
        raise SystemExit(f"{path}: expected 16 kHz audio, got {fs} Hz (resample it first)")
    return quantise16(x.mean(axis=1))


def cmd_list(a) -> int:
    from .pipeline import load_conditions

    doc = load_conditions(a.conditions)
    if a.json:
        print(json.dumps([{k: c[k] for k in ("id", "name", "family", "description")} for c in doc["conditions"]],
                         ensure_ascii=False, indent=1))
    else:
        for c in doc["conditions"]:
            print(f"{c['id']:5} {c['family']:9} {c['name']:20} {c['description']}")
    return 0


def cmd_render(a) -> int:
    import os

    import soundfile as sf

    from .pipeline import load_conditions, render

    doc = load_conditions(a.conditions)
    x = _read_parent(a.input)
    pid = a.parent_id or os.path.basename(a.input)
    ids = list(doc["by_id"]) if a.condition == "all" else [a.condition]
    for cid in ids:
        y16, rec = render(x, cid, pid, doc, a.rir_dir)
        out = a.output if len(ids) == 1 else os.path.join(a.output, f"{os.path.splitext(pid)[0]}__{cid}.flac")
        if len(ids) > 1:
            os.makedirs(a.output, exist_ok=True)
        sf.write(out, y16, 16000, format="FLAC", subtype="PCM_16")
        rec["output"] = out
        print(json.dumps(rec, ensure_ascii=False))
    return 0


def cmd_measure(a) -> int:
    import soundfile as sf

    from .dsp import spectral_rolloff
    from .metrics import estoi, si_sdr

    x, fx = sf.read(a.clean, dtype="float64")
    y, fy = sf.read(a.variant, dtype="float64")
    if fx != fy:
        raise SystemExit("sample rates differ")
    print(json.dumps({"estoi": round(estoi(x, y, fx), 6), "si_sdr_db": round(si_sdr(x, y), 4),
                      "rolloff95_hz": spectral_rolloff(y, fy)}))
    return 0


def cmd_rirs(a) -> int:
    from .pipeline import load_conditions
    from .rirs import generate_set

    doc = load_conditions(a.conditions)
    rows = generate_set(doc["rirs"], a.out, log=(lambda m: print(m, file=sys.stderr)))
    print(json.dumps({"rooms": len(rows), "out": a.out}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="arabic-channel-sim", description=__doc__)
    ap.add_argument("--conditions", default=None, help="conditions YAML (default: the packaged conditions_v1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="list the conditions")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_list)
    p = sub.add_parser("render", help="render a 16 kHz WAV/FLAC under one condition (or all)")
    p.add_argument("--condition", required=True, help="condition id, or 'all'")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="output FLAC (a directory when --condition all)")
    p.add_argument("--parent-id", default=None, help="id used for the seed (default: input file name)")
    p.add_argument("--rir-dir", default=None, help="impulse response set (default: the packaged rirs_v1)")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("measure", help="ESTOI, SI-SDR and roll-off of a variant against its clean parent")
    p.add_argument("clean")
    p.add_argument("variant")
    p.set_defaults(fn=cmd_measure)
    p = sub.add_parser("rirs", help="regenerate the impulse response set (needs pyroomacoustics)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_rirs)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
