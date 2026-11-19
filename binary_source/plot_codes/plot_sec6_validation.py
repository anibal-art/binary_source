#!/usr/bin/env python3
"""
Section 6: validation of the noncentral chi2 power model with noisy H1
realizations (sec6_h1_validation_v1).

Inputs (written by binary_source/analysis/roman_sec6_power.py):
    <cells>.json         --select-validation (lambda, c_alpha, k_local per cell)
    <cells>_result.csv   --validate (measured fractions T > c per cell)
    <cells>_k_h1.json    --validate (maximum-likelihood k_H1 and 68% interval)

Output: <out>.{pdf,png}, two panels
    (a) power against lambda.  Points: measured fraction of the 100
        realizations of each cell with T > c (alpha = 5%, filled; 3 sigma,
        open), Wilson 68% intervals.  Curves: P(chi2_k(lambda) > c) for the
        median c of the cells, with k = k_H1 (solid; band: 68% interval of
        k_H1) and k = median H0 k_local (dashed).
    (b) measured against predicted power, cell by cell (each with its own c),
        for k = k_H1 (filled) and k = H0 k_local (open); diagonal = agreement.

Usage:
    python binary_source/plot_codes/plot_sec6_validation.py \
        --cells results/roman_lrt/sec6_h1_validation_v1/cells.json \
        --out figures/current/sec6_validation [--no-usetex]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper_style import apply_paper_style, fig_width, panel_label, save_figure, C_BLUE, C_VERMILION  # noqa: E402
from binary_source.analysis.lrt_gross_vitells import wilson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    apply_paper_style(usetex=not args.no_usetex)

    spec = json.loads(args.cells.read_text())
    cells = {c["config_id"]: c for c in spec["cells"]}
    with open(args.cells.with_name(args.cells.stem + "_result.csv")) as f:
        res = {r["config_id"]: r for r in csv.DictReader(f)}
    kfit = json.loads(args.cells.with_name(args.cells.stem + "_k_h1.json").read_text())
    k1, (k1_lo, k1_hi) = float(kfit["k_h1"]), kfit["k_h1_68"]

    ids = [cid for cid in cells if cid in res]
    lam = np.array([cells[c]["lambda"] for c in ids])
    c05 = np.array([cells[c]["c_alpha05"] for c in ids])
    c3s = np.array([cells[c]["c_3sigma"] for c in ids])
    k0 = np.array([cells[c]["k_local"] for c in ids])
    n = np.array([int(res[c]["n"]) for c in ids])
    m05 = np.array([float(res[c]["power05_meas"]) for c in ids])
    m3s = np.array([float(res[c]["power3s_meas"]) for c in ids])
    ci05 = np.array([wilson(int(round(m * k)), k) for m, k in zip(m05, n)])
    ci3s = np.array([wilson(int(round(m * k)), k) for m, k in zip(m3s, n)])
    # Wilson limits equal to the measured value up to rounding at power 0 or 1
    ci05 = np.column_stack([np.minimum(ci05[:, 0], m05), np.maximum(ci05[:, 1], m05)])
    ci3s = np.column_stack([np.minimum(ci3s[:, 0], m3s), np.maximum(ci3s[:, 1], m3s)])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fig_width(1.0), fig_width(0.42)),
                                 constrained_layout=True)
    lg = np.linspace(0.0, 1.1 * lam.max(), 300)
    for cc, mm, ci, col, mk, lab in ((np.median(c05), m05, ci05, C_BLUE, "o", r"$5\%$"),
                                     (np.median(c3s), m3s, ci3s, C_VERMILION, "s", r"$3\sigma$")):
        ax.fill_between(lg, stats.ncx2.sf(cc, k1_lo, lg), stats.ncx2.sf(cc, k1_hi, lg),
                        color=col, alpha=0.2, lw=0)
        ax.plot(lg, stats.ncx2.sf(cc, k1, lg), color=col, lw=1.2)
        ax.plot(lg, stats.ncx2.sf(cc, np.median(k0), lg), color=col, lw=1.0, ls="--")
        ax.errorbar(lam, mm, yerr=[mm - ci[:, 0], ci[:, 1] - mm], fmt=mk, ms=3.5, color=col,
                    mfc=col if mk == "o" else "white", lw=0.8, capsize=0, label=lab)
    ax.plot([], [], color="0.3", lw=1.2, label=rf"$k = {k1:.1f}$")
    ax.plot([], [], color="0.3", lw=1.0, ls="--", label=rf"$k = {np.median(k0):.1f}$ (PSPL curves)")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Detection efficiency")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, lg[-1])
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "a")

    for k_use, mfc, lab in ((k1, None, rf"$k = {k1:.1f}$"), (k0, "white", r"$k$ (PSPL curves)")):
        p05 = stats.ncx2.sf(c05, k_use, lam)
        p3s = stats.ncx2.sf(c3s, k_use, lam)
        bx.errorbar(p05, m05, yerr=[m05 - ci05[:, 0], ci05[:, 1] - m05], fmt="o", ms=3.5,
                    color=C_BLUE, mfc=mfc or C_BLUE, lw=0.8, capsize=0,
                    label=r"$5\%$, " + lab)
        bx.errorbar(p3s, m3s, yerr=[m3s - ci3s[:, 0], ci3s[:, 1] - m3s], fmt="s", ms=3.5,
                    color=C_VERMILION, mfc=mfc or C_VERMILION, lw=0.8, capsize=0,
                    label=r"$3\sigma$, " + lab)
    bx.plot([0, 1], [0, 1], color="0.5", lw=0.7, ls=":")
    bx.set_xlabel("Predicted efficiency")
    bx.set_ylabel("Measured efficiency")
    bx.set_xlim(-0.02, 1.02)
    bx.set_ylim(-0.02, 1.02)
    bx.legend(frameon=False, loc="lower right", fontsize="small")
    panel_label(bx, "b")
    print(f"k_H1 = {k1:.2f} [{k1_lo:.2f}, {k1_hi:.2f}];  Pearson chi2 k_H1 "
          f"{kfit['pearson_chi2_k_h1']:.1f}/{kfit['pearson_dof_k_h1']}, k_local "
          f"{kfit['pearson_chi2_k_local']:.1f}/{kfit['pearson_dof_k_local']}")
    for f in save_figure(fig, args.out):
        print("saved", f)


if __name__ == "__main__":
    main()
