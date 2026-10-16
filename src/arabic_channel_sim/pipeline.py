# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Load a conditions file and render a clean 16 kHz parent under one condition."""
from __future__ import annotations

import hashlib
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import dsp, rirs
from .metrics import clipped_fraction

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_CONDITIONS = os.path.join(DATA_DIR, "conditions_v1.yaml")


def load_conditions(path: Optional[str] = None) -> Dict[str, Any]:
    """Read a conditions file. Returns the parsed document with an added ``by_id`` mapping."""
    import yaml

    with open(path or DEFAULT_CONDITIONS, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    ids = [c["id"] for c in doc["conditions"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate condition id in conditions file")
    doc["by_id"] = {c["id"]: c for c in doc["conditions"]}
    return doc


def conditions_sha256(path: Optional[str] = None) -> str:
    with open(path or DEFAULT_CONDITIONS, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def variant_seed(parent_id: str, condition_id: str) -> int:
    return int(hashlib.sha256((parent_id + condition_id).encode("utf-8")).hexdigest()[:8], 16)


def _run_steps(x: np.ndarray, fs: int, steps: List[Dict[str, Any]], common: Dict[str, Any], seed: int,
               rir_dir: Optional[str]) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    info: Dict[str, Any] = {}
    window = tuple(common["resampler"]["window"])
    for st in steps:
        op = st["op"]
        if op == "resample":
            x = dsp.resample(x, fs, int(st["to_hz"]), window=window)
            fs = int(st["to_hz"])
        elif op == "bandpass":
            x = dsp.bandpass(x, fs, float(st["low_hz"]), float(st["high_hz"]), int(st["numtaps"]), tuple(st["window"]))
        elif op == "ffmpeg_codec":
            if int(st["sample_rate_hz"]) != fs:
                raise ValueError(f"codec step expects {st['sample_rate_hz']} Hz input, signal is {fs} Hz")
            x = dsp.ffmpeg_codec(x, fs, [str(v) for v in st["encode"]], [str(v) for v in st["decode_input"]])
        elif op == "reverb":
            ids = rirs.rir_ids(float(st["rir_target_s"]), rir_dir)
            if not ids:
                raise ValueError(f"no impulse responses for RT60 {st['rir_target_s']} s")
            rid = ids[seed % len(ids)]
            h, p = rirs.load_rir(rid, rir_dir)
            x, tail_db = dsp.convolve_rir(x, h, int(p["direct_index"]), len(x))
            info.update({"rir_id": rid, "rir_measured_rt60_s": p["measured_rt60_s"], "rir_drr_db": p["drr_db"],
                         "direct_path_delay_samples": int(p["direct_index"]),
                         "tail_energy_trimmed_db": round(tail_db, 3)})
        else:
            raise ValueError(f"unknown step {op!r}")
    return x, fs, info


def render(parent: np.ndarray, condition_id: str, parent_id: str, conditions: Optional[Dict[str, Any]] = None,
           rir_dir: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Render ``parent`` (float, 16 kHz mono, already on the 16-bit grid) under ``condition_id``.

    Returns (variant as int16, record). The record holds the seed, delay compensation, level-match gain,
    clipping share and, for reverberation, the impulse response used.
    """
    doc = conditions or load_conditions()
    cond = doc["by_id"][condition_id]
    fs = int(doc["sample_rate_hz"])
    common = doc["common"]
    x = np.asarray(parent, dtype=np.float64)
    n = len(x)
    seed = variant_seed(parent_id, condition_id)
    y, out_fs, info = _run_steps(x, fs, cond["steps"], common, seed, rir_dir)
    if out_fs != fs:
        raise ValueError(f"{condition_id}: steps end at {out_fs} Hz, expected {fs} Hz")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{condition_id}: non-finite samples")
    if cond["family"] == "reverb":
        delay = int(info["direct_path_delay_samples"])
        y = dsp.shift_to_length(y, 0, n)
    else:
        max_lag = int(round(float(common["alignment"]["search_ms"]) * fs / 1000.0))
        delay = dsp.estimate_delay(x, y, max_lag)
        y = dsp.shift_to_length(y, delay, n)
    lm = common["level_match"]
    mask = dsp.active_mask(x, fs, float(lm["frame_ms"]), float(lm["active_within_db"]))
    y, gain_db = dsp.level_match(x, y, mask)
    y, cap_db = dsp.clip_guard(y, float(lm["max_clipped_fraction"]))
    gain_db += cap_db
    y16 = dsp.to_int16(y)
    rec = {"condition_id": condition_id, "seed": seed, "delay_samples": int(delay), "gain_db": round(gain_db, 4),
           "level_capped": bool(cap_db < 0),
           "clipped_fraction": round(clipped_fraction(y16), 8), "num_samples": int(len(y16))}
    rec.update(info)
    return y16, rec


def encode_flac(x16: np.ndarray, fs: int = 16000) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(x16, dtype=np.int16), fs, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def decode_audio(data: bytes) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    x, fs = sf.read(io.BytesIO(data), dtype="float64")
    return x, fs
