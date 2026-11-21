#!/usr/bin/env python3
"""
Section 5: median over random orientations of D_BSPL-PSPL and of
|Delta tE|/tE across the (q_M, q_f) plane, one row per P/tE.

Input : summary_qM_qf.npz written by
        run_final_scans.py --only qmass_qflux with BINARY_SOURCE_QMQF_NGEO > 0
        (orientations of scan_xiE_ptE.geometries, as the (xi_E, P/tE) maps).
|Delta tE|/tE is shown only where the median D < BIAS_D_MAX, with the colour
map of Figure 3 (plot_xiE_ptE_maps). Dashed: q_f = q_M. Delta t0 and Delta u0
are not shown: they depend on taking star 1 as the reference, tE does not.

Usage:
    python binary_source/plot_codes/plot_qM_qf_orientations.py \\
        --summary results/<final>/qmass_qflux_tE30_n60_geo24/summary_qM_qf.npz \\
        --out figures/current/qmass_qflux_orientations [--no-usetex]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply_paper_style, fig_width, heatmap, save_figure,  # noqa: E402
                         d_norm, D_CMAP, D_LABEL)
from plot_xiE_ptE_maps import BIAS_CMAP, BIAS_D_MAX  # noqa: E402
from run_paper_figures import configure_log_map_axis  # noqa: E402

BIAS_NORM = (1e-4, 1e-1)


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
    Bm = np.where(Dm < BIAS_D_MAX, Bm, np.nan)
    print(f"{n_geo} random orientations, P/tE = {pte}")

    fig, axes = plt.subplots(len(pte), 2, squeeze=False, constrained_layout=True,
                             figsize=(fig_width(1.0), fig_width(0.42 * len(pte) + 0.1)))
    for r in range(len(pte)):
        pcm_d = heatmap(axes[r, 0], qM, qf[pos], Dm[:, pos, r].T, cmap=D_CMAP,
                        norm=d_norm())
        pcm_b = heatmap(axes[r, 1], qM, qf[pos], Bm[:, pos, r].T, cmap=BIAS_CMAP,
                        norm=LogNorm(*BIAS_NORM))
        for c, ax in enumerate(axes[r]):
            configure_log_map_axis(ax, qM, qf[pos], show_xlabel=(r == len(pte) - 1),
                                   show_ylabel=(c == 0), diag_color="w",
                                   hide_last_xlabel=(c == 0))
            ax.set_title(rf"$P/t_E={pte[r]:g}$", fontsize="small")
    fig.colorbar(pcm_d, ax=list(axes[:, 0]), shrink=0.9,
                 label=D_LABEL + ", median over orientations")
    fig.colorbar(pcm_b, ax=list(axes[:, 1]), shrink=0.9, extend="both",
                 label=r"$|\Delta t_E|/t_E$, median over orientations")
    for f in save_figure(fig, args.out):
        print("saved", f)


if __name__ == "__main__":
    main()
