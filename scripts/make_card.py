# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Write the dataset card (README.md) and SHA256SUMS of the Hugging Face dataset folder.

Every number in the card is read or computed here from files in the folder itself: MANIFEST.json,
pairs.jsonl, metrics/condition_summary.csv, eval/wer_drop.csv, eval/summary.json, conditions_v1.yaml and
rirs_v1/params.jsonl. Usage: python scripts/make_card.py --root HF_FOLDER [--check]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

DATE = "2026-10-15"
HF_ID = "Bayanat-Labs/arabic-asr-channel-pairs"
GITHUB = "https://github.com/Bayanat-Labs/arabic-channel-sim"
ORDER = ["CLEAN", "BL1", "RS2", "CD1", "CD6", "RV2"]
SHORT = {"CLEAN": "Clean", "BL1": "Telephone band", "RS2": "4 kHz resampling", "CD1": "G.711 mu-law",
         "CD6": "Opus 12 kbit/s", "RV2": "Room reverberation"}
SHORT_AR = {"CLEAN": "نظيف", "BL1": "نطاق الهاتف", "RS2": "إعادة أخذ العينات عبر 4 كيلوهرتز", "CD1": "G.711 mu-law",
            "CD6": "Opus بمعدل 12 كيلوبت/ث", "RV2": "صدى غرفة"}
MODEL_LABEL = {"large-v3-turbo": "whisper-large-v3-turbo", "small": "whisper-small"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def n(v):
    return f"{int(v):,}"


def f1(v):
    return f"{float(v):.1f}"


def f2(v):
    return f"{float(v):.2f}"


def signed(v, nd=1):
    v = float(v)
    return f"+{v:.{nd}f}" if v > 0 else f"{v:.{nd}f}"


def step_text(st, doc_rirs):
    op = st["op"]
    if op == "resample":
        return f"resample to {st['to_hz'] / 1000:g} kHz"
    if op == "bandpass":
        return f"band-pass {n(st['low_hz'])}-{n(st['high_hz'])} Hz (linear-phase FIR, {st['numtaps']} taps)"
    if op == "ffmpeg_codec":
        return f"ffmpeg encode `{' '.join(str(v) for v in st['encode'])}` at {st['sample_rate_hz'] / 1000:g} kHz, then decode"
    if op == "reverb":
        return (f"convolve with one of the {doc_rirs} rirs_v1 rooms at RT60 {st['rir_target_s']} s (chosen by seed), "
                "aligned on the direct path")
    raise ValueError(op)


def load(root):
    import yaml

    d = {}
    d["manifest"] = json.load(open(os.path.join(root, "MANIFEST.json"), encoding="utf-8"))
    d["pairs"] = read_jsonl(os.path.join(root, "pairs.jsonl"))
    d["summary_rows"] = read_csv(os.path.join(root, "metrics", "condition_summary.csv"))
    d["wer"] = read_csv(os.path.join(root, "eval", "wer_drop.csv"))
    d["eval"] = json.load(open(os.path.join(root, "eval", "summary.json"), encoding="utf-8"))
    d["conditions"] = yaml.safe_load(open(os.path.join(root, "conditions_v1.yaml"), encoding="utf-8"))
    d["rirs"] = read_jsonl(os.path.join(root, "rirs_v1", "params.jsonl"))
    d["fleurs"] = json.load(open(os.path.join(root, "meta", "fleurs_source.json"), encoding="utf-8"))
    d["selection"] = json.load(open(os.path.join(root, "meta", "selection.json"), encoding="utf-8"))
    d["checks"] = json.load(open(os.path.join(root, "meta", "source_checks.json"), encoding="utf-8"))
    return d


def numbers(d):
    """Every number the card, the hub page and the registry state, computed from the released files."""
    m = d["manifest"]["counts"]
    pairs = d["pairs"]
    parents = [p for p in pairs if p["condition_id"] == "CLEAN"]
    variants = [p for p in pairs if p["condition_id"] != "CLEAN"]
    assert len(parents) == m["parents"] and len(variants) == m["variants"], "MANIFEST and pairs.jsonl disagree"
    by = {}
    for split in ("test", "dev"):
        ps = [p for p in parents if p["split"] == split]
        vs = [p for p in variants if p["split"] == split]
        by[split] = {"parents": len(ps), "sentence_ids": len({p["fleurs_sentence_id"] for p in ps}),
                     "parent_hours": sum(p["duration_s"] for p in ps) / 3600, "variants": len(vs),
                     "variant_hours": sum(p["duration_s"] for p in vs) / 3600}
    by["all"] = {"parents": len(parents), "sentence_ids": len({p["fleurs_sentence_id"] for p in parents}),
                 "parent_hours": sum(p["duration_s"] for p in parents) / 3600, "variants": len(variants),
                 "variant_hours": sum(p["duration_s"] for p in variants) / 3600}
    wer = {}
    for r in d["wer"]:
        wer.setdefault(r["model"], {}).setdefault(r["policy"], {})[r["condition_id"]] = r
    sub = d["eval"]
    capped = sum(1 for p in variants if p["level_capped"])
    clipped_parent = {p["parent_id"] for p in parents if p["clipped_fraction"] > 0}
    capped_from_clipped = sum(1 for p in variants if p["level_capped"] and p["parent_id"] in clipped_parent)
    capped_levels = [p["level_diff_db"] for p in variants if p["level_capped"]]
    return {"by": by, "conditions": len({p["condition_id"] for p in variants}), "clips": len(pairs),
            "excluded": m["excluded_variants"], "capped": capped, "capped_from_clipped": capped_from_clipped,
            "clipped_parents": len(clipped_parent),
            "capped_level_min": min(capped_levels) if capped_levels else 0.0,
            "capped_by": {c: sum(1 for p in variants if p["condition_id"] == c and p["level_capped"]) for c in ORDER[1:]},
            "wer": wer, "wer_parents": sub["parents"], "wer_sentence_ids": sub["sentence_ids"],
            "wer_minutes": sub["audio_seconds_per_condition"] / 60, "rirs": len(d["rirs"]),
            "gender": m["gender_labels"], "seed": d["selection"]["seed"], "dev_total": d["selection"]["dev_parents_total"],
            "dev_per": d["selection"]["dev_per_tertile"]}


def card(d, x):
    c = d["conditions"]
    by = x["by"]
    models = [mm for mm in ("large-v3-turbo", "small") if mm in x["wer"]]
    turbo = x["wer"].get("large-v3-turbo", {}).get("lenient", {})
    L = []
    L.append("---")
    L.append("pretty_name: Arabic ASR Channel Pairs")
    L.append("language:\n- ar")
    L.append("license:\n- cc-by-4.0\n- cc-by-sa-4.0\n- cc0-1.0")
    L.append("task_categories:\n- automatic-speech-recognition")
    L.append("tags:\n- arabic\n- speech\n- asr\n- robustness\n- telephony\n- codec\n- g711\n- opus\n- reverberation\n"
             "- room-impulse-response\n- fleurs\n- synthetic\n- benchmark")
    L.append("size_categories:\n- 1K<n<10K")
    L.append("source_datasets:\n- google/fleurs")
    L.append("configs:")
    for cid in ORDER:
        L.append(f"- config_name: {cid}")
        L.append("  data_files:")
        for split in ("test", "dev"):
            L.append(f"  - split: {split}\n    path: data/{cid}/{split}-*.parquet")
        if cid == "CLEAN":
            L.append("  default: true")
    L.append("---")
    L.append("")
    L.append("# Arabic ASR Channel Pairs")
    L.append("")
    L.append(f"**Version 0.1.0 · {DATE} · [Bayanat Labs](https://bayanatlabs.com)** · English · [العربية](#العربية)")
    L.append("")
    L.append(f"Paired clean and channel-degraded Arabic speech for testing how speech recognition holds up on phone "
             f"lines, codecs and echoey rooms. {n(by['all']['parents'])} read-speech recordings from FLEURS Egyptian "
             f"Arabic (`ar_eg`: {n(by['test']['parents'])} test, {n(by['dev']['parents'])} dev) are each released clean and "
             f"rendered under {x['conditions']} fixed, documented channel conditions, giving {n(by['all']['variants'])} "
             "variants. Every variant keeps its parent's verbatim reference transcript, split and sentence id, and "
             "is aligned and level-matched to the parent, so a change in word error rate can be put down to the "
             "channel alone.")
    L.append("")
    L.append("> **Simulated channel conditions applied to read speech.** These are not real telephone or "
             "call-centre recordings: there is no network jitter, packet loss, background noise or handset "
             "acoustics. Use them to compare systems and to measure sensitivity to a known channel, not to predict "
             "accuracy on real calls.")
    L.append("")
    if turbo:
        cd1, bl1, rv2, rs2 = (turbo[k] for k in ("CD1", "BL1", "RV2", "RS2"))
        L.append(f"On the {x['wer_parents']} test recordings scored, whisper-large-v3-turbo goes from "
                 f"{f1(turbo['CLEAN']['wer_pct'])}% word error rate on clean audio to {f1(cd1['wer_pct'])}% through "
                 f"G.711 ({signed(cd1['delta_wer_points'])} points), {f1(rv2['wer_pct'])}% in a room with RT60 0.6 s "
                 f"({signed(rv2['delta_wer_points'])}) and {f1(rs2['wer_pct'])}% after 4 kHz resampling "
                 f"({signed(rs2['delta_wer_points'])}), scored with the arabicwer `lenient` policy. "
                 "[Full table below](#speech-recognition-under-each-condition).")
        L.append("")
    L.append("## Quick start")
    L.append("")
    L.append("```python\n# pip install datasets soundfile torchcodec   (torchcodec decodes the audio column)\nfrom datasets import load_dataset\n\n"
             f"clean = load_dataset(\"{HF_ID}\", \"CLEAN\", split=\"test\")\n"
             f"g711 = load_dataset(\"{HF_ID}\", \"CD1\", split=\"test\")\n\n"
             "# Rows of every config are sorted by parent_id, so they line up one to one.\n"
             "assert clean[0][\"parent_id\"] == g711[0][\"parent_id\"]\n"
             "print(g711[0][\"reference_ar\"], g711[0][\"estoi\"], g711[0][\"si_sdr_db\"])\n```")
    L.append("")
    L.append("Audio is FLAC, 16 kHz mono, 16-bit, in the `audio` column (Hugging Face `Audio` feature). To read the "
             "bytes without an audio backend, cast the column with `Audio(decode=False)` and decode "
             "`row[\"audio\"][\"bytes\"]` with `soundfile`. `pairs.jsonl` holds the same metadata without audio or "
             "text, one line per clip.")
    L.append("")
    L.append(f"To render other 16 kHz audio under the same conditions, use the open-source renderer "
             f"[arabic-channel-sim]({GITHUB}):")
    L.append("")
    L.append("```bash\npip install \"git+https://github.com/Bayanat-Labs/arabic-channel-sim@v0.1.0\"\n"
             "arabic-channel-sim list\narabic-channel-sim render --condition CD1 --input clean.wav --output g711.flac\n```")
    L.append("")
    L.append("Once the package is published on PyPI, `pip install arabic-channel-sim` will also work. The codec "
             "conditions need an `ffmpeg` binary with `libopus`.")
    L.append("")
    L.append("## Conditions")
    L.append("")
    L.append("| Config | Condition | What is applied |")
    L.append("|---|---|---|")
    L.append("| `CLEAN` | Clean parent | The FLEURS recording, quantised to 16-bit and stored as FLAC |")
    for cond in c["conditions"]:
        steps = "; ".join(step_text(st, x["rirs"] // len(c["rirs"]["rt60_targets_s"])) for st in cond["steps"])
        L.append(f"| `{cond['id']}` | {SHORT[cond['id']]} | {steps} |")
    L.append("")
    L.append("After the steps, every variant is shifted to undo any codec or filter delay (cross-correlation with "
             "the parent within ±100 ms; reverberation is aligned on the direct path instead), cut or padded to the "
             "parent's exact length, set to the parent's active-speech level (RMS over 20 ms frames within 30 dB of the "
             "loudest frame) and written as 16-bit FLAC. If that level would put more than 0.1% of samples at full "
             "scale, the gain is lowered just enough to avoid it and the row is marked `level_capped`. The full "
             "definitions are in [`conditions_v1.yaml`](conditions_v1.yaml); condition ids will not be reused for a "
             "different recipe.")
    L.append("")
    L.append("## In numbers")
    L.append("")
    L.append("| | test | dev | total |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| Parent recordings | {n(by['test']['parents'])} | {n(by['dev']['parents'])} | {n(by['all']['parents'])} |")
    L.append(f"| FLEURS sentence ids | {n(by['test']['sentence_ids'])} | {n(by['dev']['sentence_ids'])} | {n(by['all']['sentence_ids'])} |")
    L.append(f"| Parent audio, hours | {f2(by['test']['parent_hours'])} | {f2(by['dev']['parent_hours'])} | {f2(by['all']['parent_hours'])} |")
    L.append(f"| Channel conditions | {x['conditions']} | {x['conditions']} | {x['conditions']} |")
    L.append(f"| Variants | {n(by['test']['variants'])} | {n(by['dev']['variants'])} | {n(by['all']['variants'])} |")
    L.append(f"| Rendered variant audio, hours | {f2(by['test']['variant_hours'])} | {f2(by['dev']['variant_hours'])} | {f2(by['all']['variant_hours'])} |")
    L.append(f"| Clips in the release (clean + variants) | {n(by['test']['parents'] + by['test']['variants'])} | "
             f"{n(by['dev']['parents'] + by['dev']['variants'])} | {n(x['clips'])} |")
    L.append("")
    L.append("Rendered hours are the same speech heard through a different channel; they are not new speech and "
             f"should not be added to the parent hours. {n(x['excluded'])} variants failed the integrity checks; "
             f"{n(x['capped'])} are `level_capped` (" + ", ".join(f"{k} {v}" for k, v in x["capped_by"].items() if v)
             + "), " + ("all" if x["capped_from_clipped"] == x["capped"] else f"{n(x['capped_from_clipped'])} of them")
             + f" from parents whose FLEURS recording already has samples at full scale ({n(x['clipped_parents'])} of the "
             f"{n(by['all']['parents'])} parents do); the lowest of them sits {f1(abs(x['capped_level_min']))} dB below "
             "its parent's level.")
    L.append("")
    L.append("## Speech recognition under each condition")
    L.append("")
    L.append(f"The first {x['wer_parents']} test recordings by file name ({n(x['wer_sentence_ids'])} sentence ids, "
             f"{f1(x['wer_minutes'])} minutes per condition) were transcribed in every condition by two open Whisper "
             "models and scored with [arabicwer](https://github.com/Bayanat-Labs/arabicwer) 0.1.0 against "
             "`reference_ar`. `raw` compares the text as written, split on white space; `lenient` first removes "
             "diacritics, tatweel and punctuation, maps أ إ آ ٱ to ا, ى to ي and ة to ه, and converts digits to ASCII. "
             "Change is in WER points against `CLEAN`; intervals are "
             "95% bootstrap intervals that resample sentence ids (1,000 resamples, paired for the change).")
    L.append("")
    for pol in ("lenient", "raw"):
        L.append(f"**Policy `{pol}`**")
        L.append("")
        head = "| Condition | " + " | ".join(f"{MODEL_LABEL[mm]} WER (95% CI) | change" for mm in models) + " |"
        L.append(head)
        L.append("|---|" + "---:|---:|" * len(models))
        for cid in ORDER:
            cells = []
            for mm in models:
                r = x["wer"][mm][pol][cid]
                cells.append(f"{f1(r['wer_pct'])}% ({f1(r['wer_ci95_low_pct'])}-{f1(r['wer_ci95_high_pct'])})")
                cells.append("" if cid == "CLEAN" else
                             f"{signed(r['delta_wer_points'])} ({signed(r['delta_ci95_low_points'])} to "
                             f"{signed(r['delta_ci95_high_points'])})")
            L.append(f"| {SHORT[cid]} (`{cid}`) | " + " | ".join(cells) + " |")
        L.append("")
    L.append("![Word error rate of two Whisper models under each channel condition](eval/wer_drop.png)")
    L.append("")
    ev = d["eval"]["models"]
    L.append("| Model | Weights | Runtime | Decoding |")
    L.append("|---|---|---|---|")
    for mm in models:
        e = ev[mm]
        dp = ", ".join(f"{k}={v}" for k, v in e["decode_params"].items())
        L.append(f"| `{e['model_id']}` | `{e['model_weights']}` | {e['runtime']} | {dp} |")
    L.append("")
    L.append("Transcripts are in `eval/hyps/`, per-clip error counts in `eval/wer_by_clip.csv` and the table in "
             "`eval/wer_drop.csv`. The makers of Whisper have not listed its training data, so overlap with FLEURS cannot be ruled "
             "out; read the change against `CLEAN` rather than the absolute WER. These are reference measurements "
             "of two models on a small subset, not a leaderboard.")
    L.append("")
    L.append(f"## Measured degradation (all {n(by['all']['parents'])} variants per condition)")
    L.append("")
    L.append("| Condition | ESTOI, median (IQR) | SI-SDR dB, median (IQR) | 95% spectral roll-off, median Hz | `level_capped` |")
    L.append("|---|---:|---:|---:|---:|")
    srows = {(r["condition_id"], r["split"]): r for r in d["summary_rows"]}
    for cid in ORDER:
        r = srows[(cid, "all")]
        if cid == "CLEAN":
            L.append(f"| {SHORT[cid]} (`{cid}`) | 1 (reference) | reference | {n(float(r['rolloff95_hz_median']))} | |")
            continue
        L.append(f"| {SHORT[cid]} (`{cid}`) | {float(r['estoi_median']):.3f} ({float(r['estoi_p25']):.3f}-"
                 f"{float(r['estoi_p75']):.3f}) | {f1(r['si_sdr_db_median'])} ({f1(r['si_sdr_db_p25'])} to "
                 f"{f1(r['si_sdr_db_p75'])}) | {n(float(r['rolloff95_hz_median']))} | {r['level_capped']} |")
    L.append("")
    L.append("ESTOI (extended short-time objective intelligibility, 0-1, via `pystoi`) and SI-SDR (scale-invariant "
             "signal-to-distortion ratio) are computed between each released variant and its released clean parent. "
             "SI-SDR counts every removed frequency as distortion, so it is low for the band-limited conditions even "
             "where speech stays intelligible. Per-split figures are in `metrics/condition_summary.csv` and every "
             "value is a column of the data. ViSQOL and DNSMOS were not computed for this version: ViSQOL needs its "
             "C++ tool built with Bazel, and DNSMOS needs separately licensed model files.")
    L.append("")
    L.append("## Data fields")
    L.append("")
    fields = [
        ("id", "`<parent file stem>__<condition id>`"),
        ("parent_id", "FLEURS file name of the clean parent (unique per recording)"),
        ("condition_id", "`CLEAN`, `BL1`, `RS2`, `CD1`, `CD6` or `RV2`"),
        ("split", "Official FLEURS split of the parent (`test` or `dev`); variants never change split"),
        ("audio", "FLAC, 16 kHz mono, 16-bit"),
        ("reference_ar", "FLEURS `raw_transcription`, verbatim (CC BY-SA 4.0)"),
        ("reference_fleurs_norm", "FLEURS normalised `transcription`, verbatim (CC BY-SA 4.0)"),
        ("fleurs_sentence_id", "FLoRes sentence id; up to four recordings share one"),
        ("duration_s, num_samples", "Length of the parent, equal for every variant"),
        ("gender_label", "As published by FLEURS (no inference)"),
        ("synthetic", "`true` for rendered variants (the channel is simulated; the speech is the real FLEURS recording), `false` for `CLEAN`"),
        ("seed", "`int(sha256(parent_id + condition_id)[:8], 16)`; chooses the impulse response for `RV2`"),
        ("delay_samples", "Delay removed by alignment (direct-path delay for `RV2`)"),
        ("gain_db, level_capped, level_diff_db", "Level-matching gain, whether the clipping guard lowered it, and the resulting active-speech level against the parent (dB)"),
        ("clipped_fraction", "Share of samples at 16-bit full scale"),
        ("rolloff95_hz", "Frequency below which 95% of the active-speech power lies"),
        ("estoi, si_sdr_db", "Measured against the clean parent (empty for `CLEAN`)"),
        ("rir_id, rir_measured_rt60_s, rir_drr_db, tail_energy_trimmed_db", "Impulse response used by `RV2`, its measured T30 and direct-to-reverberant ratio, and the reverberant tail energy cut at the parent's end relative to the kept signal"),
        ("wer_subset", "`true` for the parents used in the speech recognition table"),
        ("source_sha256, audio_sha256", "SHA-256 of the original FLEURS WAV and of this row's FLAC"),
        ("source_url", "FLEURS archive at the pinned revision that holds the parent"),
        ("provenance_version, attribution", "Release and condition-set version; attribution line"),
    ]
    L.append("| Field | Description |")
    L.append("|---|---|")
    for k, v in fields:
        L.append(f"| `{k}` | {v} |" if "," not in k else "| " + ", ".join(f"`{s.strip()}`" for s in k.split(",")) + f" | {v} |")
    L.append("")
    L.append("## How it was made")
    L.append("")
    fl = d["fleurs"]
    sel = d["selection"]
    ck = d["checks"]
    rir_t = d["conditions"]["rirs"]
    meas = [r["measured_rt60_s"] for r in d["rirs"]]
    L.append(f"- **Source.** [`google/fleurs`](https://huggingface.co/datasets/google/fleurs), config `ar_eg`, revision "
             f"`{fl['revision']}`: the dev and test TSVs and audio archives. Each archive's SHA-256 was checked against "
             "the Hub's LFS object id before use (`meta/fleurs_source.json`).")
    L.append(f"- **Selection.** All {n(sel['test_parents'])} test recordings, plus {n(sel['dev_drawn'])} of the "
             f"{n(sel['dev_parents_total'])} dev recordings drawn with `numpy.random.default_rng({sel['seed']})`, "
             f"{sel['dev_per_tertile']} from each duration tertile (`meta/selection.json`).")
    L.append("- **Parents.** FLEURS stores 32-bit float WAV. Each parent was quantised to 16-bit (round to nearest, no "
             "dither) and stored losslessly as FLAC; that 16-bit signal is the input of every condition and the "
             "reference for every measurement. `source_sha256` identifies the original WAV.")
    L.append(f"- **Rendering.** [arabic-channel-sim]({GITHUB}) 0.1.0: resampling and filters in Python (SciPy), codecs "
             "through ffmpeg 8.1.2 with libopus 1.6.1. Renders are deterministic for a given parent, condition and "
             "library build; Opus output can differ slightly between libopus builds.")
    L.append(f"- **Impulse responses.** {n(len(d['rirs']))} synthetic shoebox rooms made with the pyroomacoustics "
             f"image-source method ({rir_t['rooms_per_target']} each at RT60 "
             + ", ".join(f"{t} s" for t in rir_t["rt60_targets_s"])
             + f"; rooms {rir_t['room_dims_m']['min'][0]:g}x{rir_t['room_dims_m']['min'][1]:g}x{rir_t['room_dims_m']['min'][2]:g} m "
             f"to {rir_t['room_dims_m']['max'][0]:g}x{rir_t['room_dims_m']['max'][1]:g}x{rir_t['room_dims_m']['max'][2]:g} m, "
             f"source-microphone distance {rir_t['source_mic_distance_m'][0]:g}-{rir_t['source_mic_distance_m'][1]:g} m). "
             "Wall absorption starts from the Sabine estimate and is rescaled until the measured T30 is within 5% of "
             f"the target; measured RT60 ranges from {min(meas):.3f} s to {max(meas):.3f} s across the set. Each room's "
             "seed, geometry, absorption, measured RT60 and direct-to-reverberant ratio are in `rirs_v1/params.jsonl`. "
             "No recorded impulse responses are used.")
    L.append("- **Checks on every variant.** Decodes, no NaN or Inf; exact parent length; at most 0.1% of samples at "
             "full scale; active-speech level within 0.5 dB of the parent unless `level_capped`; 95% roll-off below "
             "3.6 kHz (`BL1`, `CD1`), 2.1 kHz (`RS2`) and 8.1 kHz (`CD6`); impulse response RT60 within 10% of the "
             f"target (`RV2`). A failing variant would be rendered again once and then excluded; {n(x['excluded'])} "
             "were excluded.")
    diff = ck["length_differs_from_tsv"]
    L.append(f"- **Source checks.** {len(ck['duplicate_audio'])} duplicate recordings, "
             f"{len(ck['file_names_in_two_splits'])} file names in two splits and {ck['sentence_ids_in_dev_and_test']} "
             "sentence ids shared between dev and test. "
             + (f"{len(diff)} test recording" + (" is" if len(diff) == 1 else "s are") + " shorter than the FLEURS TSV's "
                "`num_samples` (" + "; ".join(f"`{e['file_name']}`: {n(e['audio_samples'])} samples in the audio, "
                                              f"{n(e['tsv_num_samples'])} in the TSV" for e in diff)
                + "); the audio length is used." if diff else ""))
    L.append("")
    L.append("## Limitations")
    L.append("")
    g = x["gender"]
    L.append("- Simulated channels only. Real calls add noise, packet loss, jitter buffers, echo cancellation, "
             "automatic gain control and handset acoustics, none of which is modelled here.")
    L.append("- Read Modern Standard Arabic (FLoRes sentences) by Egyptian speakers. Not conversational, not dialectal.")
    L.append("- FLEURS publishes no speaker ids, and labels every selected recording "
             + ", ".join(f"{k} ({n(v)})" for k, v in g.items())
             + "; robustness by speaker or gender cannot be measured with this release.")
    L.append(f"- The speech recognition table covers {x['wer_parents']} recordings and two models; intervals are wide "
             "and differences of a point or two between conditions are within noise.")
    L.append("")
    L.append("## Licence")
    L.append("")
    L.append("| Part | Licence |")
    L.append("|---|---|")
    L.append("| Audio, clean and rendered (derived from FLEURS) | CC BY 4.0 ([LICENSE](LICENSE)) |")
    L.append("| Text: `reference_ar`, `reference_fleurs_norm`, the transcripts in `eval/hyps/` and the `ref_norm` and `hyp_norm` columns of `eval/wer_by_clip.csv` | CC BY-SA 4.0 ([LICENSE-CC-BY-SA-4.0](LICENSE-CC-BY-SA-4.0)) |")
    L.append("| Metadata, measurements, WER tables and charts by Bayanat Labs | CC BY 4.0 ([LICENSE](LICENSE)) |")
    L.append("| `conditions_v1.yaml` and `rirs_v1/` | CC0 1.0 ([LICENSE-CC0](LICENSE-CC0)) |")
    L.append("")
    L.append("Attribution: *Audio derived from Google FLEURS (Conneau et al., 2022), CC BY 4.0; sentences from "
             "FLoRes-101 (Goyal et al., 2022), CC BY-SA 4.0, translated from Wikinews, Wikijunior and Wikivoyage; "
             "channel rendering by Bayanat Labs.* The underlying recordings are not owned by Bayanat Labs. Anyone "
             "redistributing modified text columns must keep them under CC BY-SA 4.0. The renderer code is Apache-2.0 "
             f"([GitHub]({GITHUB})).")
    L.append("")
    L.append("## Citation")
    L.append("")
    L.append("```bibtex\n@misc{bayanatlabs_arabic_asr_channel_pairs_2026,\n"
             "  title  = {Arabic ASR Channel Pairs},\n  author = {{Bayanat Labs}},\n  year   = {2026},\n"
             "  version = {0.1.0},\n"
             f"  howpublished = {{\\url{{https://huggingface.co/datasets/{HF_ID}}}}},\n"
             "  note   = {Simulated channel conditions on FLEURS Egyptian Arabic. CC BY 4.0 (audio, metadata), CC BY-SA 4.0 (text), CC0 1.0 (conditions, impulse responses)}\n}\n\n"
             "@inproceedings{conneau2022fleurs,\n"
             "  title     = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},\n"
             "  author    = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},\n"
             "  booktitle = {2022 IEEE Spoken Language Technology Workshop (SLT)},\n  year      = {2022}\n}\n\n"
             "@article{goyal2022flores,\n"
             "  title   = {The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation},\n"
             "  author  = {Goyal, Naman and Gao, Cynthia and Chaudhary, Vishrav and Chen, Peng-Jen and Wenzek, Guillaume and Ju, Da and Krishnan, Sanjana and Ranzato, Marc'Aurelio and Guzm{\\'a}n, Francisco and Fan, Angela},\n"
             "  journal = {Transactions of the Association for Computational Linguistics},\n  volume  = {10},\n  pages   = {522--538},\n  year    = {2022}\n}\n```")
    L.append("")
    L.append("## Contact and takedown")
    L.append("")
    L.append("Corrections, questions and removal requests: hello@bayanatlabs.com. Files: `MANIFEST.json` (counts), "
             "`SHA256SUMS` (checksums of every file), `CHANGELOG.md`.")
    L.append("")
    L.append("## العربية")
    L.append("")
    L.append("<div dir=\"rtl\" lang=\"ar\">")
    L.append("")
    L.append("*The Arabic text has not yet been reviewed by a native Arabic writer; the English governs.*")
    L.append("")
    L.append("أزواج من الكلام العربي النظيف ونسخه المتأثرة بقنوات الاتصال، لاختبار ثبات أنظمة التعرّف الآلي على الكلام "
             "عبر خطوط الهاتف والمرمّزات الصوتية والغرف ذات الصدى. أُخذت التسجيلات من مجموعة FLEURS للعربية (الإعداد `ar_eg`)، "
             "وهي كلام مقروء، ونُشر كل تسجيل نظيفًا ثم معالَجًا تحت كل ظرف من ظروف القناة الثابتة الموثّقة. تحتفظ كل نسخة "
             "معالَجة بالنص المرجعي الحرفي لأصلها وبقسمه ورقم جملته، وهي مضبوطة التزامن ومستوى الصوت مع الأصل، بحيث "
             "يُنسب أي تغيّر في معدل خطأ الكلمات إلى القناة وحدها.")
    L.append("")
    L.append("| | العدد |")
    L.append("|---|---:|")
    L.append(f"| التسجيلات الأصلية (قسم الاختبار / قسم التطوير) | {n(by['all']['parents'])} ({n(by['test']['parents'])} / {n(by['dev']['parents'])}) |")
    L.append(f"| ظروف القناة | {x['conditions']} |")
    L.append(f"| النسخ المعالَجة | {n(by['all']['variants'])} |")
    L.append(f"| ساعات الصوت الأصلي | {f2(by['all']['parent_hours'])} |")
    L.append("")
    L.append("**هذه ظروف قناة محاكاة مطبّقة على كلام مقروء، وليست تسجيلات هاتفية أو تسجيلات مراكز اتصال حقيقية.**")
    L.append("")
    L.append("| الإعداد | الظرف |")
    L.append("|---|---|")
    for cid in ORDER:
        L.append(f"| `{cid}` | {SHORT_AR[cid]} |")
    L.append("")
    if turbo:
        L.append(f"على عيّنة من قسم الاختبار عددها {x['wer_parents']}، بلغ معدل خطأ الكلمات لنموذج whisper-large-v3-turbo "
                 f"(سياسة `lenient` في arabicwer) {f1(turbo['CLEAN']['wer_pct'])}% على الصوت النظيف، و{f1(turbo['CD1']['wer_pct'])}% "
                 f"عبر G.711، و{f1(turbo['RV2']['wer_pct'])}% في غرفة زمن صداها 0.6 ثانية، و{f1(turbo['RS2']['wer_pct'])}% بعد "
                 "إعادة أخذ العينات عبر 4 كيلوهرتز. الجداول الكاملة في القسم الإنجليزي أعلاه.")
        L.append("")
    L.append("الترخيص: الصوت والبيانات الوصفية CC BY 4.0، وأعمدة النص CC BY-SA 4.0، وملف الظروف والاستجابات النبضية CC0 1.0. "
             "للتواصل وطلبات الإزالة: hello@bayanatlabs.com")
    L.append("")
    L.append("</div>")
    L.append("")
    return "\n".join(L)


def sha256sums(root):
    lines = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if not d.startswith("."))
        for fn in sorted(fns):
            if fn.startswith(".") or fn == "SHA256SUMS":
                continue
            p = os.path.join(dp, fn)
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {os.path.relpath(p, root)}")
    return "\n".join(sorted(lines, key=lambda s: s.split("  ", 1)[1])) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--check", action="store_true", help="fail if README.md or SHA256SUMS are out of date")
    ap.add_argument("--numbers-json", default=None, help="also write the computed numbers to this file")
    a = ap.parse_args()
    d = load(a.root)
    x = numbers(d)
    text = card(d, x)
    readme = os.path.join(a.root, "README.md")
    if a.check:
        ok = open(readme, encoding="utf-8").read() == text
        ok = ok and open(os.path.join(a.root, "SHA256SUMS"), encoding="utf-8").read() == sha256sums(a.root)
        print("up to date" if ok else "out of date")
        sys.exit(0 if ok else 1)
    with open(readme, "w", encoding="utf-8") as f:
        f.write(text)
    if a.numbers_json:
        with open(a.numbers_json, "w", encoding="utf-8") as f:
            json.dump(x, f, ensure_ascii=False, indent=1, default=float)
            f.write("\n")
    with open(os.path.join(a.root, "SHA256SUMS"), "w", encoding="utf-8") as f:
        f.write(sha256sums(a.root))
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
