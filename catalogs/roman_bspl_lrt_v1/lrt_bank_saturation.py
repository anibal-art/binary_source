#!/usr/bin/env python3
"""
Saturation of the Test A statistic with the size of the BSPL start bank.

Reads the runner outputs of an h0a run (data_derived policy). The runner saves
the chi2 of every BSPL start, so nested sub-banks are evaluated offline:

    T_A(bank) = max(0, chi2_PSPL - min_{starts in bank} chi2_BSPL)

Sub-banks always contain the nested start (xi = 1e-4 at the PSPL optimum).
  periods : 1, 2, 4, 8, 16 nodes (every 16th, 8th, ... node; nested subsets)
  phases  : 1 or all, with all periods
  delta_u : each amplitude alone, with all periods and phases
The bank is saturated when the upper quantiles of T_A stop growing as it is
doubled. Also reported: BSPL optimizer failures (raw Delta chi2_A < 0), failed
starts, cost per fit and per event, and the false-positive rate of Wilks
thresholds chi2_k (k = 5, 6, 7) on the full bank (illustrative at pilot N).

Usage:
    python catalogs/roman_bspl_lrt_v1/lrt_bank_saturation.py \
        --tasks results/roman_lrt/test_a_pilot_v1/tasks.txt
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist

from mini_lrt_analyze import chi2, cp_interval, load

LABEL = re.compile(r"bspl_P([0-9.e+]+)_du([0-9.e+-]+)_ph([0-9.e+-]+)")
QUANTILES = (0.5, 0.9, 0.95)


def bank_table(d):
    """(P, du, phase, chi2) of every fitted grid start, chi2 of the nested start."""
    rows, nested, n_fail = [], np.inf, 0
    for s in d["fits"]["BSPL"]["starts"]:
        if s.get("status") != "fitted" or not np.isfinite(s.get("chi2", np.nan)):
            n_fail += 1
            continue
        if s["start_label"] == "bspl_nested":
            nested = float(s["chi2"])
            continue
        m = LABEL.fullmatch(s["start_label"])
        rows.append((float(m[1]), float(m[2]), float(m[3]), float(s["chi2"])))
    return np.array(rows), nested, n_fail


def sub_banks(periods, dus, phases):
    """name -> predicate on (P, du, phase)."""
    banks = {}
    for n in (1, 2, 4, 8, 16):
        if n > len(periods):
            continue
        keep = set(periods[:: len(periods) // n][:n])
        banks[f"P{n:02d}"] = lambda P, du, ph, keep=keep: P in keep
    banks["P16_ph1"] = lambda P, du, ph: np.isclose(ph, phases[0])
    for du_k in dus:
        banks[f"P16_du{du_k:g}"] = lambda P, du, ph, du_k=du_k: np.isclose(du, du_k)
    return banks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--improve-tol", type=float, default=1.0,
                    help="T_A gain counted as 'the larger bank found a better optimum'")
    ap.add_argument("--n-events-full", type=int, default=6300,
                    help="events of the full calibration, for the cost estimate")
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    runs, missing = load(args.tasks)
    runs = [d for d in runs if d["_kind"] == "h0a"]
    print(f"{len(runs)} h0a outputs, {len(missing)} missing")
    if not runs:
        raise SystemExit(1)

    tables = [bank_table(d) for d in runs]
    periods = np.unique(np.concatenate([t[0][:, 0] for t in tables]))
    dus = np.unique(np.concatenate([t[0][:, 1] for t in tables]))
    phases = np.unique(np.concatenate([t[0][:, 2] for t in tables]))
    banks = sub_banks(periods, dus, phases)
    names = list(banks) + ["nested_only"]

    TA = {name: [] for name in names}
    cfg = []
    for d, (tab, nested, _) in zip(runs, tables):
        cP = chi2(d, "PSPL")
        cfg.append(d["catalog_row"]["config_id"])
        for name, pred in banks.items():
            sel = np.array([pred(*r[:3]) for r in tab], bool)
            best = min(nested, tab[sel, 3].min() if sel.any() else np.inf)
            TA[name].append(max(0.0, cP - best))
        TA["nested_only"].append(max(0.0, cP - nested))
    TA = {k: np.array(v) for k, v in TA.items()}
    cfg = np.array(cfg)

    full = "P16"
    if not np.allclose(TA[full], [d["statistics"]["T_A"] for d in runs]):
        print("WARNING: P16 does not reproduce the runner T_A (labels or bank changed?)")

    print("\nT_A quantiles per sub-bank (pooled, then per configuration)")
    groups = [("all", np.ones(len(runs), bool))] + [(c, cfg == c) for c in np.unique(cfg)]
    summary = {}
    for gname, m in groups:
        print(f"\n  [{gname}]  N={m.sum()}")
        print(f"  {'bank':14s}" + "".join(f"  q{int(q*100):02d}" .rjust(10) for q in QUANTILES)
              + "       max   gain>tol vs P16")
        for name in names:
            v = TA[name][m]
            q = np.quantile(v, QUANTILES)
            worse = int(np.sum(TA[full][m] - v > args.improve_tol))
            print(f"  {name:14s}" + "".join(f"{x:10.2f}" for x in q)
                  + f"{v.max():10.2f}   {worse:3d}/{m.sum()}")
            summary.setdefault(gname, {})[name] = {
                "quantiles": dict(zip(map(str, QUANTILES), q.tolist())),
                "max": float(v.max()), "n_full_better": worse}

    raw = np.array([d["statistics"]["delta_chi2_A_raw"] for d in runs])
    n_fail = np.array([t[2] for t in tables])
    print(f"\nBSPL optimizer failures (raw Delta chi2_A < -{args.improve_tol}): "
          f"{int(np.sum(raw < -args.improve_tol))}/{len(runs)}")
    print(f"failed BSPL starts: total {int(n_fail.sum())}, events with any {int(np.sum(n_fail > 0))}")
    best_P = [tab[np.argmin(tab[:, 3]), 0] for tab, _, _ in tables]
    Pv, Pc = np.unique(np.round(best_P, 1), return_counts=True)
    print("period of the best BSPL start: " + ", ".join(f"{p:g} d x{c}" for p, c in zip(Pv, Pc)))
    feats = [d.get("light_curve_features") or {} for d in runs]
    print(f"tE guess from fallback: {sum(f.get('tE_source') == 'fallback' for f in feats)}/{len(runs)}")

    wall = np.array([s.get("wall_seconds", np.nan) for d in runs
                     for fam in ("PSPL", "BSPL") for s in d["fits"][fam]["starts"]])
    per_event = np.array([sum(s.get("wall_seconds", 0.0) for fam in ("PSPL", "BSPL")
                              for s in d["fits"][fam]["starts"]) for d in runs])
    cpu_h = np.median(per_event) * args.n_events_full / 3600.0
    print(f"\ncost: median {np.nanmedian(wall):.1f} s per fit, "
          f"{np.median(per_event)/60:.1f} min per event (max {per_event.max()/60:.1f})")
    print(f"      {args.n_events_full} events ~ {cpu_h:.0f} CPU h "
          f"~ {cpu_h/args.workers:.1f} h wall with {args.workers} workers")

    print("\nWilks false-positive rate on the full bank (68% Clopper-Pearson)")
    wilks = {}
    for k in (5, 6, 7):
        for alpha in (0.05, 0.01, 0.0027):
            c = chi2_dist.isf(alpha, k)
            n_ex = int(np.sum(TA[full] > c))
            lo, hi = cp_interval(n_ex, len(runs))
            print(f"  k={k} alpha={alpha:<6g} threshold={c:6.2f}  "
                  f"FPR={n_ex/len(runs):.3f} [{lo:.3f},{hi:.3f}]")
            wilks[f"k{k}_a{alpha:g}"] = {"threshold": c, "n_exceed": n_ex, "n": len(runs)}

    out = args.tasks.parent / "bank_saturation.json"
    out.write_text(json.dumps({"tasks": str(args.tasks), "n_events": len(runs),
                               "periods": periods.tolist(), "summary": summary,
                               "wilks": wilks, "n_optimizer_failures":
                               int(np.sum(raw < -args.improve_tol))}, indent=2) + "\n")
    np.savez(args.tasks.parent / "bank_saturation_TA.npz", config_id=cfg,
             **{f"TA_{k}": v for k, v in TA.items()})
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
