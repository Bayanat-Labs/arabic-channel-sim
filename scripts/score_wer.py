# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""Score the WER subset with arabicwer and write the WER-drop table, per-clip scores, summary and chart.

Inputs:  --pairs   pairs.jsonl (or rows.jsonl from build_pilot.py): references and wer_subset flags
         --hyps    folder with <model>.jsonl from transcribe.py
Outputs (--out): wer_drop.csv, wer_by_clip.csv, summary.json, wer_drop.png, wer_drop.svg

Corpus WER = (S + D + I) / N summed over clips. Confidence intervals: 95 % percentile bootstrap that resamples
FLEURS sentence ids (all recordings of a sentence move together); the change against CLEAN is resampled
paired, on the same draw. Needs arabicwer (git+https://github.com/Bayanat-Labs/arabicwer@v0.1.0), numpy and
matplotlib. Usage: python scripts/score_wer.py --pairs pairs.jsonl --hyps eval/hyps --out eval
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

MODELS = ["large-v3-turbo", "small"]
LABEL = {"large-v3-turbo": "whisper-large-v3-turbo", "small": "whisper-small"}
POLICIES = ["raw", "lenient"]
CONDITIONS = ["CLEAN", "BL1", "RS2", "CD1", "CD6", "RV2"]
COND_LABEL = {"CLEAN": "Clean", "BL1": "Telephone band\n(BL1)", "RS2": "4 kHz resample\n(RS2)",
              "CD1": "G.711 mu-law\n(CD1)", "CD6": "Opus 12 kbit/s\n(CD6)", "RV2": "Reverb 0.6 s\n(RV2)"}
RESAMPLES = 1000
SEED = 0


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bootstrap(groups, err, n, base_err=None, resamples=RESAMPLES, seed=SEED):
    """Percentile CI of sum(err)/sum(n) (and of the paired difference to base_err) resampling groups."""
    uniq = sorted(set(groups))
    gi = {g: i for i, g in enumerate(uniq)}
    idx = np.array([gi[g] for g in groups])
    ge = np.bincount(idx, weights=err, minlength=len(uniq))
    gn = np.bincount(idx, weights=n, minlength=len(uniq))
    gb = np.bincount(idx, weights=base_err, minlength=len(uniq)) if base_err is not None else None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(resamples, len(uniq)))
    counts = np.stack([np.bincount(d, minlength=len(uniq)) for d in draws])
    tot_n = counts @ gn
    w = counts @ ge / tot_n
    out = {"wer_ci95": [float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))]}
    if gb is not None:
        d = w - counts @ gb / tot_n
        out["delta_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out


def main():
    import arabicwer
    from arabicwer.report import score_pairs

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    refs = {r["id"]: r for r in read_jsonl(a.pairs) if r.get("wer_subset")}
    parents = sorted({r["file_name"] for r in refs.values()})
    clip_rows, table = [], []
    summary = {"arabicwer_version": arabicwer.__version__, "policies": {}, "parents": len(parents),
               "sentence_ids": len({r["fleurs_sentence_id"] for r in refs.values()}),
               "audio_seconds_per_condition": round(sum(r["duration_s"] for r in refs.values()
                                                        if r["condition_id"] == "CLEAN"), 3),
               "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "unit": "FLEURS sentence id"},
               "models": {}}
    for pol in POLICIES:
        p = arabicwer.load_policy(pol)
        summary["policies"][pol] = p.policy_id if hasattr(p, "policy_id") else pol
    for m in MODELS:
        path = os.path.join(a.hyps, f"{m}.jsonl")
        if not os.path.exists(path):
            continue
        hyps = {h["id"]: h for h in read_jsonl(path)}
        missing = sorted(set(refs) - set(hyps))
        if missing:
            raise SystemExit(f"{m}: {len(missing)} clips have no hypothesis (e.g. {missing[0]})")
        h0 = next(iter(hyps.values()))
        summary["models"][m] = {"model_id": h0["model_id"], "model_weights": h0["model_weights"],
                                "runtime": h0["runtime"], "decode_params": h0["decode_params"], "results": {}}
        for pol in POLICIES:
            per = {}
            for c in CONDITIONS:
                ids = [f"{os.path.splitext(fn)[0]}__{c}" for fn in parents]
                pairs = [(i, refs[i]["raw_transcription"], hyps[i]["hypothesis"]) for i in ids]
                rep = score_pairs(pairs, [arabicwer.load_policy(pol)])
                recs = [row["scores"] for row in rep["pairs"]]
                pid = next(iter(recs[0]))
                vals = [r[pid] for r in recs]
                err = np.array([v["S"] + v["D"] + v["I"] for v in vals], dtype=float)
                n = np.array([v["N"] for v in vals], dtype=float)
                cerr = np.array([v["cS"] + v["cD"] + v["cI"] for v in vals], dtype=float)
                cn = np.array([v["cN"] for v in vals], dtype=float)
                per[c] = {"ids": ids, "err": err, "n": n, "cerr": cerr, "cn": cn,
                          "groups": [refs[i]["fleurs_sentence_id"] for i in ids],
                          "policy_wer": rep["policies"][0]["words"]["wer"]}
                for i, v in zip(ids, vals):
                    clip_rows.append([m, pol, i, c, v["S"], v["D"], v["I"], v["N"], v["cS"], v["cD"], v["cI"], v["cN"],
                                      v["ref_norm"], v["hyp_norm"]])
            base = per["CLEAN"]
            res = {}
            for c in CONDITIONS:
                d = per[c]
                wer = float(d["err"].sum() / d["n"].sum())
                assert abs(wer - d["policy_wer"]) < 1e-9, "corpus WER does not match arabicwer"
                cer = float(d["cerr"].sum() / d["cn"].sum())
                bs = bootstrap(d["groups"], d["err"], d["n"], None if c == "CLEAN" else base["err"])
                clean_wer = float(base["err"].sum() / base["n"].sum())
                r = {"wer": wer, "wer_ci95": bs["wer_ci95"], "cer": cer, "ref_words": int(d["n"].sum()),
                     "delta_points": 100 * (wer - clean_wer),
                     "delta_ci95_points": [100 * v for v in bs.get("delta_ci95", [0.0, 0.0])],
                     "relative_change": (wer - clean_wer) / clean_wer if clean_wer else None}
                res[c] = r
                table.append([m, LABEL[m], pol, c, len(d["ids"]), r["ref_words"], round(100 * wer, 2),
                              round(100 * bs["wer_ci95"][0], 2), round(100 * bs["wer_ci95"][1], 2), round(100 * cer, 2),
                              round(r["delta_points"], 2), round(r["delta_ci95_points"][0], 2),
                              round(r["delta_ci95_points"][1], 2),
                              "" if r["relative_change"] is None else round(100 * r["relative_change"], 1)])
            summary["models"][m]["results"][pol] = res
    with open(os.path.join(a.out, "wer_drop.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "model_label", "policy", "condition_id", "clips", "ref_words", "wer_pct", "wer_ci95_low_pct",
                    "wer_ci95_high_pct", "cer_pct", "delta_wer_points", "delta_ci95_low_points",
                    "delta_ci95_high_points", "relative_change_pct"])
        w.writerows(table)
    with open(os.path.join(a.out, "wer_by_clip.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "policy", "id", "condition_id", "S", "D", "I", "N", "cS", "cD", "cI", "cN", "ref_norm",
                    "hyp_norm"])
        w.writerows(clip_rows)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.write("\n")
    chart(summary, os.path.join(a.out, "wer_drop"))
    for m, d in summary["models"].items():
        print(m, {c: round(100 * d["results"]["lenient"][c]["wer"], 2) for c in CONDITIONS})


def chart(summary, base, policy="lenient"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colours = {"large-v3-turbo": "#2a78d6", "small": "#eb6834"}
    models = [m for m in MODELS if m in summary["models"]]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=100)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    width = 0.36
    ymax = 0.0
    for k, m in enumerate(models):
        res = summary["models"][m]["results"][policy]
        for i, c in enumerate(CONDITIONS):
            r = res[c]
            x = i + (k - (len(models) - 1) / 2) * (width + 0.03)
            v = 100 * r["wer"]
            lo, hi = (100 * t for t in r["wer_ci95"])
            ax.bar(x, v, width=width, color=colours[m], edgecolor=surface, linewidth=2,
                   label=LABEL[m] if i == 0 else None, zorder=2)
            ax.plot([x, x], [lo, hi], color=ink2, linewidth=1.1, zorder=3)
            ax.plot([x - 0.05, x + 0.05], [lo, lo], color=ink2, linewidth=1.1, zorder=3)
            ax.plot([x - 0.05, x + 0.05], [hi, hi], color=ink2, linewidth=1.1, zorder=3)
            ax.text(x, hi + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=ink)
            ymax = max(ymax, hi)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDITIONS], fontsize=9.5, color=ink)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_ylabel(f"Corpus WER (%), arabicwer {policy} policy", color=ink2, fontsize=9.5)
    ax.tick_params(axis="y", colors=ink2, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    fig.suptitle("Arabic speech recognition under simulated channel conditions", x=0.02, y=0.975, ha="left",
                 fontsize=13, color=ink, fontweight="bold")
    sub = (f"FLEURS ar_eg test, the same {summary['parents']} recordings in every condition. "
           f"Whiskers: 95% bootstrap interval over sentence ids ({summary['bootstrap']['resamples']:,} resamples).")
    fig.text(0.02, 0.915, sub, ha="left", va="top", fontsize=9.5, color=ink2)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=ink)
    fig.text(0.02, 0.012, "Bayanat Labs, arabic-asr-channel-pairs v0.1.0, CC BY 4.0. Audio derived from FLEURS (Google, CC BY 4.0).",
             fontsize=8, color=ink2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(base + ".png", dpi=200, facecolor=surface)
    fig.savefig(base + ".svg", facecolor=surface, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
