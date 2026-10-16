# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Synthetic room impulse responses: generation (pyroomacoustics image method), measurement and loading.

The rirs_v1 set ships inside the package (``arabic_channel_sim/data/rirs_v1``): 20 shoebox rooms for each
target reverberation time, with the parameters of every room in ``params.jsonl``. Generation is seeded and
recorded so that the set can be rebuilt with :func:`generate_set` (``pip install "arabic-channel-sim[rir]"``).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SPEED_OF_SOUND = 343.0


def schroeder_rt60(h: np.ndarray, fs: int, start_db: float = -5.0, end_db: float = -35.0) -> float:
    """Reverberation time from the Schroeder backward-integrated energy decay curve.

    A straight line is fitted to the decay between ``start_db`` and ``end_db`` (T30 by default) and
    extrapolated to -60 dB. Returns seconds.
    """
    h = np.asarray(h, dtype=np.float64)
    e = np.cumsum(h[::-1] ** 2)[::-1]
    if e[0] <= 0:
        raise ValueError("impulse response has no energy")
    edc = 10.0 * np.log10(np.maximum(e / e[0], 1e-300))
    i0 = int(np.argmax(edc <= start_db))
    i1 = int(np.argmax(edc <= end_db))
    if i1 <= i0:
        raise ValueError("decay does not reach the end level")
    t = np.arange(i0, i1) / fs
    slope, _ = np.polyfit(t, edc[i0:i1], 1)
    return float(-60.0 / slope)


def direct_index(h: np.ndarray) -> int:
    """Sample index of the direct path (largest absolute value)."""
    return int(np.argmax(np.abs(np.asarray(h))))


def drr_db(h: np.ndarray, fs: int, window_ms: float = 2.5) -> float:
    """Direct-to-reverberant energy ratio: energy within +/- window_ms of the direct path vs the rest."""
    h = np.asarray(h, dtype=np.float64)
    d = direct_index(h)
    w = int(round(window_ms * fs / 1000.0))
    lo, hi = max(0, d - w), min(len(h), d + w + 1)
    direct = float(np.sum(h[lo:hi] ** 2))
    rest = float(np.sum(h ** 2) - direct)
    return 10.0 * math.log10(direct / max(rest, 1e-30))


def _positions(rng: np.random.Generator, dims: np.ndarray, margin: float, dist_range: Tuple[float, float],
               height_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, float]:
    for _ in range(10000):
        src = np.array([rng.uniform(margin, dims[0] - margin), rng.uniform(margin, dims[1] - margin),
                        rng.uniform(*height_range)])
        mic = np.array([rng.uniform(margin, dims[0] - margin), rng.uniform(margin, dims[1] - margin),
                        rng.uniform(*height_range)])
        d = float(np.linalg.norm(src - mic))
        if dist_range[0] <= d <= dist_range[1]:
            return src, mic, d
    raise RuntimeError("could not place source and microphone")


def generate_room(seed: int, rt60_target: float, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Generate one candidate room. Returns the impulse response and its parameters (not yet accepted)."""
    import pyroomacoustics as pra

    fs = int(spec["sample_rate_hz"])
    rng = np.random.default_rng(seed)
    lo, hi = np.array(spec["room_dims_m"]["min"], dtype=float), np.array(spec["room_dims_m"]["max"], dtype=float)
    dims = np.round(rng.uniform(lo, hi), 3)
    src, mic, dist = _positions(rng, dims, float(spec["wall_margin_m"]), tuple(spec["source_mic_distance_m"]),
                                tuple(spec["height_m"]))
    sabine_absorption, max_order = pra.inverse_sabine(rt60_target, dims)
    max_order = int(min(max_order, int(spec["max_order_cap"])))

    def simulate(absorption: float) -> np.ndarray:
        room = pra.ShoeBox(dims, fs=fs, materials=pra.Material(absorption), max_order=max_order,
                           air_absorption=False, ray_tracing=False, use_rand_ism=False)
        room.add_source(src.tolist())
        room.add_microphone(mic.tolist())
        room.compute_rir()
        h = np.asarray(room.rir[0][0], dtype=np.float64)
        e = np.cumsum(h[::-1] ** 2)[::-1]
        keep = int(np.argmax(e / e[0] < 1e-8)) or len(h)  # drop the tail once the decay is 80 dB down
        return h[:keep].astype(np.float32)

    # The Sabine estimate is only a starting point: shoebox rooms are not diffuse, so the absorption is
    # rescaled (RT60 is roughly inversely proportional to absorption) until the measured T30 is within
    # calibration_tolerance of the target, for at most calibration_steps simulations.
    absorption = float(sabine_absorption)
    steps = []
    for _ in range(int(spec["calibration_steps"])):
        h = simulate(absorption)
        measured = schroeder_rt60(h, fs)
        steps.append({"energy_absorption": round(absorption, 6), "measured_rt60_s": round(measured, 4)})
        if abs(measured - rt60_target) / rt60_target <= float(spec["calibration_tolerance"]):
            break
        absorption = float(min(0.99, max(0.01, absorption * measured / rt60_target)))
    return {
        "h": h,
        "seed": int(seed),
        "rt60_target_s": float(rt60_target),
        "room_dims_m": [float(v) for v in dims],
        "energy_absorption": round(float(steps[-1]["energy_absorption"]), 6),
        "sabine_absorption": round(float(sabine_absorption), 6),
        "calibration": steps,
        "max_order": max_order,
        "source_m": [round(float(v), 4) for v in src],
        "mic_m": [round(float(v), 4) for v in mic],
        "source_mic_distance_m": round(dist, 4),
        "measured_rt60_s": round(float(steps[-1]["measured_rt60_s"]), 4),
        "drr_db": round(drr_db(h, fs), 3),
        "direct_index": direct_index(h),
        "length_samples": int(len(h)),
        "sample_rate_hz": fs,
        "pyroomacoustics": pra.__version__,
    }


def generate_set(spec: Dict[str, Any], out_dir: str, log=print) -> List[Dict[str, Any]]:
    """Generate the full set described by ``spec`` (the ``rirs`` block of a conditions file) into out_dir."""
    import soundfile as sf

    fs = int(spec["sample_rate_hz"])
    tol = float(spec["rt60_tolerance"])
    rows = []
    for t_idx, target in enumerate(spec["rt60_targets_s"]):
        sub = os.path.join(out_dir, f"{target:.1f}")
        os.makedirs(sub, exist_ok=True)
        accepted, cand = 0, 0
        while accepted < int(spec["rooms_per_target"]):
            if cand >= int(spec["max_candidates_per_target"]):
                raise RuntimeError(f"RT60 {target}: only {accepted} rooms accepted after {cand} candidates")
            seed = int(spec["seed_base"]) * t_idx + cand
            cand += 1
            r = generate_room(seed, target, spec)
            rel_err = abs(r["measured_rt60_s"] - target) / target
            if rel_err > tol:
                log(f"rt60 {target}: seed {seed} rejected (measured {r['measured_rt60_s']:.3f} s)")
                continue
            name = f"room{accepted:02d}"
            path = os.path.join(sub, name + ".wav")
            sf.write(path, r.pop("h"), fs, subtype="FLOAT")
            with open(path, "rb") as fh:
                r["sha256"] = hashlib.sha256(fh.read()).hexdigest()
            r["rir_id"] = f"rirs_v1/{target:.1f}/{name}"
            r["file"] = f"{target:.1f}/{name}.wav"
            r["candidates_tried_before_accept"] = cand
            r["synthetic"] = True
            rows.append(r)
            accepted += 1
            log(f"rt60 {target}: {r['rir_id']} seed {seed} measured {r['measured_rt60_s']:.3f} s")
    with open(os.path.join(out_dir, "params.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return rows


def default_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rirs_v1")


def load_params(rir_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    rir_dir = rir_dir or default_dir()
    with open(os.path.join(rir_dir, "params.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rir(rir_id: str, rir_dir: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load one impulse response by id (e.g. ``rirs_v1/0.6/room07``). Returns (samples, params)."""
    import soundfile as sf

    rir_dir = rir_dir or default_dir()
    params = {p["rir_id"]: p for p in load_params(rir_dir)}
    if rir_id not in params:
        raise KeyError(f"unknown impulse response {rir_id!r}")
    p = params[rir_id]
    h, fs = sf.read(os.path.join(rir_dir, p["file"]), dtype="float64")
    if fs != p["sample_rate_hz"]:
        raise ValueError(f"{rir_id}: sample rate {fs} != {p['sample_rate_hz']}")
    return h, p


def rir_ids(target_s: float, rir_dir: Optional[str] = None) -> List[str]:
    return sorted(p["rir_id"] for p in load_params(rir_dir) if abs(p["rt60_target_s"] - target_s) < 1e-9)
