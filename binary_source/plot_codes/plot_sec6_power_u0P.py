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
Colour: detection efficiency at the primary false-alarm level (--primary:
3 sigma, the default, or 5%) averaged over the 24 isotropic orientations.
White contours: efficiency 0.5 and 0.9 at the primary level (solid) and at
the other level (dashed). Dotted crimson: efficiency 0.5 at the primary level
for the reference geometry of the earlier sections (theta, phi, i) = (0, 0, pi/2).

Usage:
    python binary_source/plot_codes/plot_sec6_power_u0P.py \
        --power results/roman_sec6_asimov_v1/sec6_power_kh1.npz \
        --out figures/current/sec6_power_u0P [--primary 3sigma|5] [--no-usetex]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import apply_paper_style, fig_width, heatmap, save_figure  # noqa: E402

# false-alarm levels of the efficiency maps of roman_sec6_power.py: npz key, label
LEVELS = {"3sigma": ("POWER_3SIGMA", r"$3\sigma$ false-alarm probability"),
          "5": ("POWER_ALPHA05", r"$5\%$ false-alarm probability")}


def efficiency_levels(d, primary):
    """(efficiency at the primary level, at the other level, label of the primary)."""
    other = "5" if primary == "3sigma" else "3sigma"
    return d[LEVELS[primary][0]], d[LEVELS[other][0]], LEVELS[primary][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--primary", choices=tuple(LEVELS), default="3sigma",
                    help="false-alarm level of the colour map and solid contours")
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    tex = not args.no_usetex
    apply_paper_style(usetex=tex)

    d = np.load(args.power)
    if not np.isfinite(float(d["k_h1"])):
        raise SystemExit("power file made without --k-h1")
    pw, pw_other, lev = efficiency_levels(d, args.primary)      # (geo, g, F146, u0, P)
    u0, P_yr = d["u0_grid"], d["P_days"] / 365.25
    blend, mags = d["blend_ratio"], d["F146"]
    iso = slice(1, pw.shape[0])                                  # row 0 = reference geometry

    fig, axes = plt.subplots(len(blend), len(mags), figsize=(fig_width(1.0), fig_width(0.95)),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for ib, g in enumerate(blend):
        for im, mag in enumerate(mags):
            ax = axes[ib, im]
            m = np.nanmean(pw[iso, ib, im], axis=0).T
            m_other = np.nanmean(pw_other[iso, ib, im], axis=0).T
            ref = pw[0, ib, im].T
            pcm = heatmap(ax, u0, P_yr, m, vmin=0.0, vmax=1.0, cmap="viridis")
            ax.contour(u0, P_yr, m, levels=[0.5, 0.9], colors="w", linewidths=0.9)
            ax.contour(u0, P_yr, m_other, levels=[0.5, 0.9], colors="w", linewidths=0.9,
                       linestyles="--")
            ax.contour(u0, P_yr, ref, levels=[0.5], colors="crimson", linewidths=1.0,
                       linestyles=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if ib == 0:
                ax.set_title(rf"$F_{{146}} = {mag:g}$")
            if im == 0:
                ax.set_ylabel(rf"$g = {g:g}$" "\n" r"$P$ [yr]")
            if ib == len(blend) - 1:
                ax.set_xlabel(r"$u_0$")
    # mathtext labels: the same strings work with and without usetex
    label = f"Detection efficiency ({lev})"
    fig.colorbar(pcm, ax=axes, label=label, shrink=0.9)
    print("k_H1 =", float(d["k_h1"]))
    for f in save_figure(fig, args.out):
        print("saved", f)


if __name__ == "__main__":
    main()
