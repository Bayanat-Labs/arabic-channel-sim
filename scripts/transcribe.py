# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Transcribe the WER subset (clean parent and every condition) with an open Whisper model (faster-whisper).

Reads rows.jsonl and the FLAC files written by build_pilot.py, keeps the rows with wer_subset = true, and
appends one JSON line per clip to OUT/<model>.jsonl. Re-running resumes after the last finished clip.
Weights are pinned to a revision and loaded on CPU with int8 compute.

Needs faster-whisper==1.2.1 and soundfile.
Usage: python scripts/transcribe.py --pilot PILOT_DIR --out eval/hyps --model large-v3-turbo [--threads 6]
"""
from __future__ import annotations

import argparse
import json
import os
import time

MODELS = {
    # name: (original model, faster-whisper (CTranslate2) conversion, pinned revision)
    # (the large-v3-turbo conversion repository was formerly published as mobiuslabsgmbh/faster-whisper-large-v3-turbo)
    "large-v3-turbo": ("openai/whisper-large-v3-turbo", "dropbox-dash/faster-whisper-large-v3-turbo",
                       "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
    "small": ("openai/whisper-small", "Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120"),
}
DECODE = {"language": "ar", "task": "transcribe", "beam_size": 5, "temperature": 0.0, "vad_filter": False,
          "condition_on_previous_text": False}
CONDITION_ORDER = ["CLEAN", "BL1", "RS2", "CD1", "CD6", "RV2"]


def subset_rows(pilot_dir):
    with open(os.path.join(pilot_dir, "rows.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if r["wer_subset"]]
    order = {c: i for i, c in enumerate(CONDITION_ORDER)}
    return sorted(rows, key=lambda r: (order.get(r["condition_id"], 99), r["file_name"]))


def main():
    import ctranslate2
    import faster_whisper
    import soundfile as sf
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    original, weights, rev = MODELS[a.model]
    path = snapshot_download(weights, revision=rev)
    model = faster_whisper.WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=a.threads)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{a.model}.jsonl")
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    rows = subset_rows(a.pilot)
    runtime = f"faster-whisper {faster_whisper.__version__}, CTranslate2 {ctranslate2.__version__}, CPU, int8"
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            x, fs = sf.read(os.path.join(a.pilot, r["audio_file"]), dtype="float32")
            assert fs == 16000
            segments, _info = model.transcribe(x, **DECODE)
            hyp = " ".join(s.text.strip() for s in segments).strip()
            f.write(json.dumps({"id": r["id"], "file_name": r["file_name"], "condition_id": r["condition_id"],
                                "model_id": original, "model_weights": f"{weights}@{rev}", "runtime": runtime,
                                "decode_params": DECODE, "hypothesis": hyp}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{a.model} {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
