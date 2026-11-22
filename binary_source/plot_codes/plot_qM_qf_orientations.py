#!/usr/bin/env python3
"""
Section 5: median over random orientations of D_BSPL-PSPL across the
(q_M, q_f) plane (colour), with contours of the median |Delta tE|/tE at
0.1% (dotted) and 1% (solid); one panel per P/tE.

Input : summary_qM_qf.npz written by
        run_final_scans.py --only qmass_qflux with BINARY_SOURCE_QMQF_NGEO > 0
        (orientations of scan_xiE_ptE.geometries, as the (xi_E, P/tE) maps).
Dashed white: q_f = q_M. Delta t0 and Delta u0 are not shown: they depend on
taking star 1 as the reference, tE does not. Key above the panels.

Usage:
    python binary_source/plot_codes/plot_qM_qf_orientations.py \\
        --summary results/<final>/qmass_qflux_tE30_n60_geo24/summary_qM_qf.npz \\
        --out figures/current/qmass_qflux_orientations [--no-usetex]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply_paper_style, heatmap, save_figure,  # noqa: E402
                         d_norm, D_CMAP, D_LABEL)
from run_paper_figures import configure_log_map_axis  # noqa: E402
from plot_sec6_power_xiE_ptE import panels_with_key_row  # noqa: E402

BIAS_LEVELS = (1e-3, 1e-2)          # |Delta tE|/tE contours
BIAS_STYLES = (":", "-")
BIAS_COLOR = "orange"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="figures/current/qmass_qflux_orientations")
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    apply_paper_style(usetex=not args.no_usetex)

    with np.load(args.summary, allow_pickle=False) as d:
        if "D_p16_p50_p84" not in d.files:
            raise SystemExit("summary has no random orientations: run "
                             "run_final_scans.py with BINARY_SOURCE_QMQF_NGEO > 0")
        qM, qf = d["qM_grid"], d["qf_grid"]
        pte = d["P_over_tE_grid"]
        Dm = d["D_p16_p50_p84"][1]                       # (qM, qf, P)
        Bm = d["abs_dtE_over_tE_p16_p50_p84"][1]
        n_geo = int(d["n_random_geometries"])
    pos = qf > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lb = np.where(Bm > 0.0, np.log10(Bm), np.nan)
    print(f"{n_geo} random orientations, P/tE = {pte}")

    fig, lax, axes = panels_with_key_row(1, len(pte), 0.58)
    for c in range(len(pte)):
        ax = axes[0, c]
        pcm = heatmap(ax, qM, qf[pos], Dm[:, pos, c].T, cmap=D_CMAP, norm=d_norm())
        ax.contour(qM, qf[pos], lb[:, pos, c].T, levels=np.log10(BIAS_LEVELS),
                   colors=BIAS_COLOR, linestyles=BIAS_STYLES, linewidths=1.1)
        configure_log_map_axis(ax, qM, qf[pos], show_xlabel=True, show_ylabel=(c == 0),
                               diag_color="w", hide_last_xlabel=(c < len(pte) - 1))
        ax.set_title(rf"$P/t_E={pte[c]:g}$", fontsize="small")
    fig.colorbar(pcm, ax=list(axes.ravel()), shrink=0.9,
                 label=D_LABEL + ", median over orientations")

    outline = [pe.Stroke(linewidth=2.6, foreground="k"), pe.Normal()]
    handles = [Line2D([], [], color="w", lw=1.4, ls="--", path_effects=outline),
               Line2D([], [], color=BIAS_COLOR, lw=1.4, ls=BIAS_STYLES[0]),
               Line2D([], [], color=BIAS_COLOR, lw=1.4, ls=BIAS_STYLES[1])]
    names = [r"$q_f=q_M$", r"$|\Delta t_E|/t_E=0.1\%$", r"$|\Delta t_E|/t_E=1\%$"]
    lax.legend(handles, names, loc="center", ncol=3, frameon=False,
               handlelength=2.2, columnspacing=1.5)
    for f in save_figure(fig, args.out):
        print("saved", f)


if __name__ == "__main__":
    main()
