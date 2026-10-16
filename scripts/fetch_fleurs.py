# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Download the FLEURS Egyptian Arabic (ar_eg) dev and test files at a pinned revision and unpack the WAVs.

Files fetched from google/fleurs (dataset repo, CC BY 4.0): data/ar_eg/{dev,test}.tsv and
data/ar_eg/audio/{dev,test}.tar.gz. Each tarball's SHA-256 is checked against the Hub's LFS object id
for the same revision before anything is unpacked. Only regular *.wav members are extracted, flat, into
OUT/wav/<split>/ (no member path is trusted). A JSON record of what was fetched goes to OUT/fetch.json.

Needs huggingface_hub. Usage:
  python scripts/fetch_fleurs.py --out /path/to/raw [--splits dev test] [--keep-tarballs]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tarfile

REPO = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CONFIG = "ar_eg"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lfs_oids(api, revision: str) -> dict:
    out = {}
    for item in api.list_repo_tree(REPO, repo_type="dataset", revision=revision, path_in_repo=f"data/{CONFIG}/audio"):
        lfs = getattr(item, "lfs", None)
        if lfs is not None:
            out[item.path] = {"sha256": lfs.sha256, "size": lfs.size}
    return out


def extract_wavs(tar_path: str, dest: str) -> int:
    os.makedirs(dest, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf:
            if not m.isreg() or not m.name.lower().endswith(".wav"):
                continue
            name = os.path.basename(m.name)
            if not name or name.startswith("."):
                continue
            src = tf.extractfile(m)
            with open(os.path.join(dest, name), "wb") as fh:
                fh.write(src.read())
            n += 1
    return n


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["train", "dev", "test"])
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--keep-tarballs", action="store_true")
    a = ap.parse_args()

    api = HfApi()
    oids = lfs_oids(api, a.revision)
    record = {"repo": REPO, "repo_type": "dataset", "revision": a.revision, "config": CONFIG,
              "retrieved_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "files": []}
    dl = os.path.join(a.out, "download")
    for split in a.splits:
        tsv = hf_hub_download(REPO, f"data/{CONFIG}/{split}.tsv", repo_type="dataset", revision=a.revision, local_dir=dl)
        record["files"].append({"path": f"data/{CONFIG}/{split}.tsv", "sha256": sha256_file(tsv), "bytes": os.path.getsize(tsv)})
        rel = f"data/{CONFIG}/audio/{split}.tar.gz"
        tar = hf_hub_download(REPO, rel, repo_type="dataset", revision=a.revision, local_dir=dl)
        digest = sha256_file(tar)
        expected = oids[rel]["sha256"]
        if digest != expected:
            raise SystemExit(f"{rel}: SHA-256 {digest} does not match the Hub LFS id {expected}")
        n = extract_wavs(tar, os.path.join(a.out, "wav", split))
        record["files"].append({"path": rel, "sha256": digest, "bytes": os.path.getsize(tar), "wav_members": n,
                                "sha256_matches_hub_lfs_oid": True})
        if not a.keep_tarballs:
            os.remove(tar)
        print(f"{split}: {n} WAV files", flush=True)
    with open(os.path.join(a.out, "fetch.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
