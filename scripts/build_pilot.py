# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Build the v0.1 pilot: select the parents, render every condition, check and measure every variant.

Input: the folder written by fetch_fleurs.py (TSVs under download/data/ar_eg/, WAVs under wav/<split>/).
Output folder:
  parents.csv          every dev/test parent with duration, SHA-256 and whether it was selected
  selection.json       seed, strata and counts of the selection
  audio/<cond>/<split>/<stem>.flac   CLEAN and one file per condition
  rows.jsonl           one record per released clip (clean and variants), with checks and measurements
  excluded.jsonl       variants that failed a check twice (empty when none failed)
  source_checks.json   duplicate audio, file names in two splits, sentence ids shared across splits

Selection: all test parents, plus DEV_N dev parents drawn with numpy.random.default_rng(SEED), an equal number
from each duration tertile of the dev split (dev parents sorted by file name before drawing).
Usage: python scripts/build_pilot.py --fleurs RAW --out OUT [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SEED = 15
DEV_N = 72
WER_SUBSET_N = 100
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SPLITS = ["test", "dev"]


def read_tsv(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if not r:
                continue
            if len(r) != 7:
                raise ValueError(f"{path}: expected 7 columns, got {len(r)}")
            rows.append({"fleurs_sentence_id": int(r[0]), "file_name": r[1], "raw_transcription": r[2],
                         "transcription": r[3], "num_samples": int(r[5]), "gender": r[6]})
    return rows


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_parents(fleurs_dir):
    import soundfile as sf

    parents = []
    for split in SPLITS:
        for r in read_tsv(os.path.join(fleurs_dir, "download", "data", "ar_eg", f"{split}.tsv")):
            p = os.path.join(fleurs_dir, "wav", split, r["file_name"])
            with open(p, "rb") as fh:
                b = fh.read()
            info = sf.info(p)
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError(f"{p}: {info.samplerate} Hz, {info.channels} channels")
            r["num_samples_tsv"] = r["num_samples"]
            r["num_samples"] = int(info.frames)  # the audio is authoritative; mismatches are listed in source_checks
            r.update({"split": split, "source_sha256": sha256_bytes(b), "source_subtype": info.subtype,
                      "duration_s": round(info.frames / 16000, 4)})
            parents.append(r)
    return parents


def source_checks(parents):
    by_hash, dup = {}, []
    for p in parents:
        if p["source_sha256"] in by_hash:
            dup.append([by_hash[p["source_sha256"]], p["file_name"]])
        else:
            by_hash[p["source_sha256"]] = p["file_name"]
    names = {}
    for p in parents:
        names.setdefault(p["file_name"], set()).add(p["split"])
    sids = {}
    for p in parents:
        sids.setdefault(p["fleurs_sentence_id"], set()).add(p["split"])
    return {"parents_checked": len(parents), "duplicate_audio": dup,
            "file_names_in_two_splits": sorted(n for n, s in names.items() if len(s) > 1),
            "sentence_ids_in_dev_and_test": sum(1 for s in sids.values() if len(s) > 1),
            "length_differs_from_tsv": [{"file_name": p["file_name"], "split": p["split"], "audio_samples": p["num_samples"],
                                         "tsv_num_samples": p["num_samples_tsv"]}
                                        for p in parents if p["num_samples"] != p["num_samples_tsv"]]}


def select(parents):
    test = sorted((p for p in parents if p["split"] == "test"), key=lambda p: p["file_name"])
    dev = sorted((p for p in parents if p["split"] == "dev"), key=lambda p: p["file_name"])
    dur = np.array([p["duration_s"] for p in dev])
    edges = np.quantile(dur, [1 / 3, 2 / 3])
    strata = np.digitize(dur, edges)  # 0, 1, 2
    rng = np.random.default_rng(SEED)
    per = DEV_N // 3
    chosen = []
    for s in range(3):
        idx = np.flatnonzero(strata == s)
        chosen.extend(int(i) for i in rng.choice(idx, size=per, replace=False))
    dev_sel = {dev[i]["file_name"] for i in chosen}
    wer = {p["file_name"] for p in test[:WER_SUBSET_N]}
    for p in parents:
        p["selected"] = p["split"] == "test" or p["file_name"] in dev_sel
        p["wer_subset"] = p["file_name"] in wer
    info = {"seed": SEED, "generator": "numpy.random.default_rng", "dev_drawn": DEV_N,
            "dev_tertile_edges_s": [round(float(e), 4) for e in edges],
            "dev_per_tertile": per, "dev_tertile_sizes": [int(np.sum(strata == s)) for s in range(3)],
            "test_parents": len(test), "dev_parents_total": len(dev),
            "wer_subset": f"first {WER_SUBSET_N} test parents by file name"}
    return info


def _measure(clean, y, mask):
    from arabic_channel_sim import dsp
    from arabic_channel_sim.metrics import estoi, si_sdr

    return {"estoi": round(estoi(clean, y, 16000), 6), "si_sdr_db": round(si_sdr(clean, y), 4),
            "rolloff95_hz": round(dsp.spectral_rolloff(y, 16000, mask=mask), 2),
            "level_diff_db": round(20 * math.log10(max(dsp.active_rms(y, mask), 1e-12) / dsp.active_rms(clean, mask)) + 0.0, 4)}


def work(job):
    """Render, check and measure one parent under every condition. Returns a list of records."""
    import soundfile as sf

    from arabic_channel_sim import dsp, pipeline as R

    p, fleurs_dir, out_dir = job
    doc = R.load_conditions()
    x, _ = sf.read(os.path.join(fleurs_dir, "wav", p["split"], p["file_name"]), dtype="float64")
    x16 = dsp.to_int16(x)
    stem = os.path.splitext(p["file_name"])[0]
    clean_bytes = R.encode_flac(x16)
    clean, _ = R.decode_audio(clean_bytes)
    assert np.array_equal(dsp.to_int16(clean), x16), "FLAC round trip is not lossless"
    mask = dsp.active_mask(clean, 16000, float(doc["common"]["level_match"]["frame_ms"]),
                           float(doc["common"]["level_match"]["active_within_db"]))
    base = {k: p[k] for k in ("file_name", "split", "fleurs_sentence_id", "num_samples", "duration_s", "gender",
                              "raw_transcription", "transcription",
                              "source_sha256", "wer_subset")}
    recs, excluded = [], []

    def save(cond, data):
        rel = os.path.join("audio", cond, p["split"], stem + ".flac")
        path = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return rel

    rel = save("CLEAN", clean_bytes)
    rec = dict(base, id=f"{stem}__CLEAN", condition_id="CLEAN", audio_file=rel, audio_sha256=sha256_bytes(clean_bytes),
               quantisation_max_abs_error=float(np.max(np.abs(clean - x))),
               rolloff95_hz=round(dsp.spectral_rolloff(clean, 16000, mask=mask), 2),
               clipped_fraction=round(float(np.mean((x16 >= 32767) | (x16 <= -32768))), 8), synthetic=False)
    recs.append(rec)
    for c in doc["conditions"]:
        cid = c["id"]
        failure = None
        for attempt in (1, 2):
            try:
                y16, info = R.render(clean, cid, p["file_name"], doc)
                data = R.encode_flac(y16)
                y, _ = R.decode_audio(data)
                m = _measure(clean, y, mask)
                failure = check(c, info, m, len(clean))
            except Exception as e:  # noqa: BLE001 - recorded and counted, never silently dropped
                failure, info, m, data = f"error: {e!r}", {}, {}, None
            if failure is None:
                break
        if failure is not None:
            excluded.append(dict(base, condition_id=cid, reason=failure, attempts=2))
            continue
        rel = save(cid, data)
        recs.append(dict(base, id=f"{stem}__{cid}", audio_file=rel, audio_sha256=sha256_bytes(data), synthetic=True,
                         **info, **m))
    return recs, excluded


def check(cond, info, m, n):
    ck = cond.get("checks", {})
    if info["num_samples"] != n:
        return f"length {info['num_samples']} != {n}"
    if info["clipped_fraction"] > 0.001:
        return f"clipped {info['clipped_fraction']:.5f}"
    if abs(m["level_diff_db"]) > 0.5 and not info.get("level_capped"):
        return f"level differs by {m['level_diff_db']} dB"
    if "rolloff95_max_hz" in ck and m["rolloff95_hz"] >= ck["rolloff95_max_hz"]:
        return f"roll-off {m['rolloff95_hz']} Hz"
    if "rt60_target_s" in ck:
        t = ck["rt60_target_s"]
        if abs(info["rir_measured_rt60_s"] - t) / t > ck["rt60_tolerance"]:
            return f"RT60 {info['rir_measured_rt60_s']}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parents = load_parents(a.fleurs)
    checks = source_checks(parents)
    sel = select(parents)
    with open(os.path.join(a.out, "source_checks.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=1)
    with open(os.path.join(a.out, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)
    with open(os.path.join(a.out, "parents.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        cols = ["file_name", "split", "fleurs_sentence_id", "num_samples", "duration_s", "gender", "source_sha256",
                "source_subtype", "selected", "wer_subset"]
        w.writerow(cols)
        for p in sorted(parents, key=lambda p: (p["split"], p["file_name"])):
            w.writerow([p[c] for c in cols])
    chosen = sorted((p for p in parents if p["selected"]), key=lambda p: (p["split"] != "test", p["file_name"]))
    print(f"parents: {len(parents)}; selected {len(chosen)}", flush=True)
    rows_path = os.path.join(a.out, "rows.jsonl")
    done = set()
    if os.path.exists(rows_path):
        with open(rows_path, encoding="utf-8") as f:
            done = {json.loads(l)["file_name"] for l in f if l.strip()}
    jobs = [(p, a.fleurs, a.out) for p in chosen if p["file_name"] not in done]
    n_done = len(done)
    with open(rows_path, "a", encoding="utf-8") as rf, open(os.path.join(a.out, "excluded.jsonl"), "a", encoding="utf-8") as xf, \
            ProcessPoolExecutor(max_workers=a.workers) as ex:
        for recs, excl in ex.map(work, jobs, chunksize=2):
            for r in recs:
                rf.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            for r in excl:
                xf.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            rf.flush()
            xf.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"{n_done}/{len(chosen)} parents", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
