# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Assemble the Hugging Face dataset folder from the pilot build and the WER evaluation.

Writes into OUT:
  data/<CONDITION>/<split>-00000-of-00001.parquet   audio (HF Audio feature, FLAC bytes) + text + metadata
  pairs.jsonl                 one metadata row per clip (no audio, no text)
  conditions_v1.yaml          the condition definitions used
  rirs_v1/                    the synthetic impulse responses and params.jsonl
  metrics/condition_summary.csv
  eval/                       hyps/, wer_drop.csv, wer_by_clip.csv, summary.json, wer_drop.png/.svg
  meta/                       fleurs_source.json, selection.json, source_checks.json, parents.csv
  MANIFEST.json, SHA256SUMS   counts and checksums (README.md, licences and citation are added separately)

Needs pyarrow, numpy and arabic_channel_sim. Usage:
  python scripts/package_hf.py --pilot PILOT --eval EVAL --fleurs-record fetch.json --out OUT
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil

import numpy as np

VERSION = "0.1.0"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CONDITIONS = ["CLEAN", "BL1", "RS2", "CD1", "CD6", "RV2"]
SPLITS = ["test", "dev"]
ATTRIBUTION = ("Audio derived from Google FLEURS (Conneau et al., 2022), CC BY 4.0; sentences from FLoRes-101 "
               "(Goyal et al., 2022), CC BY-SA 4.0; channel rendering by Bayanat Labs.")

FIELDS = [
    # (name, arrow type, HF feature, source key)
    ("id", "string", {"dtype": "string", "_type": "Value"}),
    ("parent_id", "string", {"dtype": "string", "_type": "Value"}),
    ("condition_id", "string", {"dtype": "string", "_type": "Value"}),
    ("split", "string", {"dtype": "string", "_type": "Value"}),
    ("audio", "audio", {"sampling_rate": 16000, "_type": "Audio"}),
    ("reference_ar", "string", {"dtype": "string", "_type": "Value"}),
    ("reference_fleurs_norm", "string", {"dtype": "string", "_type": "Value"}),
    ("fleurs_sentence_id", "int64", {"dtype": "int64", "_type": "Value"}),
    ("duration_s", "float64", {"dtype": "float64", "_type": "Value"}),
    ("num_samples", "int64", {"dtype": "int64", "_type": "Value"}),
    ("gender_label", "string", {"dtype": "string", "_type": "Value"}),
    ("synthetic", "bool", {"dtype": "bool", "_type": "Value"}),
    ("seed", "int64", {"dtype": "int64", "_type": "Value"}),
    ("delay_samples", "int64", {"dtype": "int64", "_type": "Value"}),
    ("gain_db", "float64", {"dtype": "float64", "_type": "Value"}),
    ("level_capped", "bool", {"dtype": "bool", "_type": "Value"}),
    ("level_diff_db", "float64", {"dtype": "float64", "_type": "Value"}),
    ("clipped_fraction", "float64", {"dtype": "float64", "_type": "Value"}),
    ("rolloff95_hz", "float64", {"dtype": "float64", "_type": "Value"}),
    ("estoi", "float64", {"dtype": "float64", "_type": "Value"}),
    ("si_sdr_db", "float64", {"dtype": "float64", "_type": "Value"}),
    ("rir_id", "string", {"dtype": "string", "_type": "Value"}),
    ("rir_measured_rt60_s", "float64", {"dtype": "float64", "_type": "Value"}),
    ("rir_drr_db", "float64", {"dtype": "float64", "_type": "Value"}),
    ("tail_energy_trimmed_db", "float64", {"dtype": "float64", "_type": "Value"}),
    ("wer_subset", "bool", {"dtype": "bool", "_type": "Value"}),
    ("source_sha256", "string", {"dtype": "string", "_type": "Value"}),
    ("audio_sha256", "string", {"dtype": "string", "_type": "Value"}),
    ("source_url", "string", {"dtype": "string", "_type": "Value"}),
    ("provenance_version", "string", {"dtype": "string", "_type": "Value"}),
    ("attribution", "string", {"dtype": "string", "_type": "Value"}),
]
META_ONLY = {"audio", "reference_ar", "reference_fleurs_norm", "attribution"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_values(r, pilot):
    stem = os.path.splitext(r["file_name"])[0]
    out = {
        "id": r["id"], "parent_id": r["file_name"], "condition_id": r["condition_id"], "split": r["split"],
        "reference_ar": r["raw_transcription"], "reference_fleurs_norm": r["transcription"],
        "fleurs_sentence_id": r["fleurs_sentence_id"], "duration_s": r["duration_s"], "num_samples": r["num_samples"],
        "gender_label": r["gender"], "synthetic": bool(r["synthetic"]), "seed": r.get("seed"),
        "delay_samples": r.get("delay_samples"), "gain_db": r.get("gain_db"), "level_capped": r.get("level_capped"),
        "level_diff_db": r.get("level_diff_db"),
        "clipped_fraction": r["clipped_fraction"], "rolloff95_hz": r["rolloff95_hz"], "estoi": r.get("estoi"),
        "si_sdr_db": r.get("si_sdr_db"), "rir_id": r.get("rir_id"), "rir_measured_rt60_s": r.get("rir_measured_rt60_s"),
        "rir_drr_db": r.get("rir_drr_db"), "tail_energy_trimmed_db": r.get("tail_energy_trimmed_db"),
        "wer_subset": bool(r["wer_subset"]), "source_sha256": r["source_sha256"], "audio_sha256": r["audio_sha256"],
        "source_url": f"https://huggingface.co/datasets/google/fleurs/blob/{REVISION}/data/ar_eg/audio/{r['split']}.tar.gz",
        "provenance_version": f"v{VERSION}+conditions_v1", "attribution": ATTRIBUTION,
    }
    with open(os.path.join(pilot, r["audio_file"]), "rb") as fh:
        data = fh.read()
    if hashlib.sha256(data).hexdigest() != r["audio_sha256"]:
        raise SystemExit(f"{r['id']}: audio checksum mismatch")
    out["audio"] = {"bytes": data, "path": f"{stem}__{r['condition_id']}.flac"}
    return out


def write_parquet(rows, path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    types = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_(),
             "audio": pa.struct([("bytes", pa.binary()), ("path", pa.string())])}
    schema = pa.schema([(n, types[t]) for n, t, _ in FIELDS])
    features = {n: f for n, _, f in FIELDS}
    schema = schema.with_metadata({"huggingface": json.dumps({"info": {"features": features}})})
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, row_group_size=100, compression="zstd")


def summarise(rows):
    out = []
    for c in CONDITIONS:
        for split in SPLITS + ["all"]:
            sel = [r for r in rows if r["condition_id"] == c and (split == "all" or r["split"] == split)]
            if not sel:
                continue

            def q(key, p):
                vals = [r[key] for r in sel if r.get(key) is not None]
                return round(float(np.percentile(vals, p)), 4) if vals else ""

            out.append({"condition_id": c, "split": split, "clips": len(sel),
                        "hours": round(sum(r["duration_s"] for r in sel) / 3600, 4),
                        "estoi_median": q("estoi", 50), "estoi_p25": q("estoi", 25), "estoi_p75": q("estoi", 75),
                        "si_sdr_db_median": q("si_sdr_db", 50), "si_sdr_db_p25": q("si_sdr_db", 25),
                        "si_sdr_db_p75": q("si_sdr_db", 75), "rolloff95_hz_median": q("rolloff95_hz", 50),
                        "rolloff95_hz_max": round(max(r["rolloff95_hz"] for r in sel), 2),
                        "gain_db_median": q("gain_db", 50),
                        "level_capped": sum(1 for r in sel if r.get("level_capped")),
                        "max_clipped_fraction": max(r["clipped_fraction"] for r in sel)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--fleurs-record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import arabic_channel_sim as acs
    from arabic_channel_sim import rirs

    rows = read_jsonl(os.path.join(a.pilot, "rows.jsonl"))
    excluded = read_jsonl(os.path.join(a.pilot, "excluded.jsonl"))
    os.makedirs(a.out, exist_ok=True)
    for c in CONDITIONS:
        for split in SPLITS:
            sel = sorted((r for r in rows if r["condition_id"] == c and r["split"] == split), key=lambda r: r["file_name"])
            d = os.path.join(a.out, "data", c)
            os.makedirs(d, exist_ok=True)
            write_parquet([row_values(r, a.pilot) for r in sel], os.path.join(d, f"{split}-00000-of-00001.parquet"))
            print(c, split, len(sel), flush=True)
    order = {c: i for i, c in enumerate(CONDITIONS)}
    with open(os.path.join(a.out, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: (r["split"] != "test", r["file_name"], order[r["condition_id"]])):
            v = {k: val for k, val in row_values(r, a.pilot).items() if k not in META_ONLY}
            f.write(json.dumps(v, ensure_ascii=False, sort_keys=False) + "\n")
    shutil.copyfile(acs.DEFAULT_CONDITIONS, os.path.join(a.out, "conditions_v1.yaml"))
    if os.path.exists(os.path.join(a.out, "rirs_v1")):
        shutil.rmtree(os.path.join(a.out, "rirs_v1"))
    shutil.copytree(rirs.default_dir(), os.path.join(a.out, "rirs_v1"))
    os.makedirs(os.path.join(a.out, "metrics"), exist_ok=True)
    summ = summarise(rows)
    with open(os.path.join(a.out, "metrics", "condition_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summ)
    ev = os.path.join(a.out, "eval")
    if os.path.exists(ev):
        shutil.rmtree(ev)
    shutil.copytree(a.eval, ev)
    meta = os.path.join(a.out, "meta")
    os.makedirs(meta, exist_ok=True)
    shutil.copyfile(a.fleurs_record, os.path.join(meta, "fleurs_source.json"))
    for name in ("selection.json", "source_checks.json", "parents.csv"):
        shutil.copyfile(os.path.join(a.pilot, name), os.path.join(meta, name))
    with open(os.path.join(meta, "excluded.jsonl"), "w", encoding="utf-8") as f:
        for r in excluded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    parents = {r["file_name"]: r for r in rows if r["condition_id"] == "CLEAN"}
    variants = [r for r in rows if r["condition_id"] != "CLEAN"]
    with open(os.path.join(a.pilot, "selection.json"), encoding="utf-8") as f:
        selection = json.load(f)
    manifest = {
        "name": "arabic-asr-channel-pairs", "version": VERSION, "conditions_file": "conditions_v1.yaml",
        "conditions_sha256": acs.conditions_sha256(), "renderer": f"arabic-channel-sim {acs.__version__}",
        "source": {"dataset": "google/fleurs", "config": "ar_eg", "revision": REVISION},
        "counts": {
            "parents": len(parents),
            "parents_by_split": {s: sum(1 for p in parents.values() if p["split"] == s) for s in SPLITS},
            "sentence_ids": len({p["fleurs_sentence_id"] for p in parents.values()}),
            "sentence_ids_by_split": {s: len({p["fleurs_sentence_id"] for p in parents.values() if p["split"] == s})
                                      for s in SPLITS},
            "parent_hours": round(sum(p["duration_s"] for p in parents.values()) / 3600, 4),
            "parent_hours_by_split": {s: round(sum(p["duration_s"] for p in parents.values() if p["split"] == s) / 3600, 4)
                                      for s in SPLITS},
            "conditions": len(CONDITIONS) - 1,
            "variants": len(variants),
            "variants_by_condition": {c: sum(1 for r in variants if r["condition_id"] == c) for c in CONDITIONS[1:]},
            "rendered_variant_hours": round(sum(r["duration_s"] for r in variants) / 3600, 4),
            "clips_total": len(rows),
            "excluded_variants": len(excluded),
            "level_capped_variants": sum(1 for r in variants if r.get("level_capped")),
            "level_capped_by_condition": {c: sum(1 for r in variants if r["condition_id"] == c and r.get("level_capped"))
                                          for c in CONDITIONS[1:]},
            "wer_subset_parents": sum(1 for p in parents.values() if p["wer_subset"]),
            "rirs": len(rirs.load_params(os.path.join(a.out, "rirs_v1"))),
            "dev_selection_seed": selection["seed"],
            "gender_labels": {g: sum(1 for p in parents.values() if p["gender"] == g)
                              for g in sorted({p["gender"] for p in parents.values()})},
            "min_duration_s": min(p["duration_s"] for p in parents.values()),
            "max_duration_s": max(p["duration_s"] for p in parents.values()),
        },
    }
    with open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(json.dumps(manifest["counts"], indent=1))


if __name__ == "__main__":
    main()
