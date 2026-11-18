#!/usr/bin/env python3
"""
Section 6: power of Test A with Roman for one physical binary-source system
(M1 = 2, M2 = 1 Msun, rEhat = 5 AU, dark companion, tE = 30 d) on the
(u0, P) plane, for three source magnitudes and three blend ratios.

Input : the .npz of binary_source/analysis/roman_sec6_power.py run with
        --k-h1 (power maps of the Asimov grid roman_sec6_asimov_cells.py
        --grid u0P, thresholds from the H0 Gross-Vitells calibration).
Output: <out>.{pdf,png}

Panels: rows g = F_blend/F_source = 0, 1, 4; columns F146 = 19, 21, 23.
Colour: power at alpha = 5% averaged over the 24 isotropic orientations.
White contours: power 0.5 and 0.9 at alpha = 5% (solid) and at 3 sigma
(dashed). Dotted crimson: power 0.5 at alpha = 5% for the reference geometry
of the earlier sections (theta, phi, i) = (0, 0, pi/2).

Usage:
    python binary_source/plot_codes/plot_sec6_power_u0P.py \
        --power results/roman_sec6_asimov_v1/sec6_power_kh1.npz \
        --out figures/current/sec6_power_u0P [--no-usetex]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import apply_paper_style, fig_width, heatmap, save_figure  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    tex = not args.no_usetex
    apply_paper_style(usetex=tex)

    d = np.load(args.power)
    if not np.isfinite(float(d["k_h1"])):
        raise SystemExit("power file made without --k-h1")
    pw05, pw3s = d["POWER_ALPHA05"], d["POWER_3SIGMA"]          # (geo, g, F146, u0, P)
    u0, P_yr = d["u0_grid"], d["P_days"] / 365.25
    blend, mags = d["blend_ratio"], d["F146"]
    iso = slice(1, pw05.shape[0])                                # row 0 = reference geometry

    pct = r"5\%" if tex else "5%"
    alpha05 = (r"$\alpha = 5\%$" if tex else "alpha = 5%")
    fig, axes = plt.subplots(len(blend), len(mags), figsize=(fig_width(1.0), fig_width(0.95)),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for ib, g in enumerate(blend):
        for im, mag in enumerate(mags):
            ax = axes[ib, im]
            m05 = np.nanmean(pw05[iso, ib, im], axis=0).T
            m3s = np.nanmean(pw3s[iso, ib, im], axis=0).T
            ref = pw05[0, ib, im].T
            pcm = heatmap(ax, u0, P_yr, m05, vmin=0.0, vmax=1.0, cmap="viridis")
            ax.contour(u0, P_yr, m05, levels=[0.5, 0.9], colors="w", linewidths=0.9)
            ax.contour(u0, P_yr, m3s, levels=[0.5, 0.9], colors="w", linewidths=0.9,
                       linestyles="--")
            ax.contour(u0, P_yr, ref, levels=[0.5], colors="crimson", linewidths=1.0,
                       linestyles=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if ib == 0:
                ax.set_title(rf"$F_{{146}} = {mag:g}$" if tex else f"F146 = {mag:g}")
            if im == 0:
                ax.set_ylabel(rf"$g = {g:g}$" "\n" r"$P$ [yr]" if tex else f"g = {g:g}\nP [yr]")
            if ib == len(blend) - 1:
                ax.set_xlabel(r"$u_0$" if tex else "u0")
    label = (rf"Power ({alpha05}), mean over orientations" if tex
             else f"Power (alpha = {pct}), mean over orientations")
    fig.colorbar(pcm, ax=axes, label=label, shrink=0.9)
    print("k_H1 =", float(d["k_h1"]))
    for f in save_figure(fig, args.out):
        print("saved", f)


if __name__ == "__main__":
    main()
