#!/usr/bin/env python3
"""
Intrinsic BSPL--PSPL mismatch for TWO luminous sources in (xi_rel, P/tE):
one row per (q_M, q_f), one column per u0, with the maps drawn exactly as in
plot_xiE_ptE_maps.draw_map (median D over the random orientations, crimson
contours D = 1e-3 dashed and 1e-2 solid, grey unreachable region).

Comparison with one luminous source (white dotted contours, same levels): to
first order in the orbital displacement the two-source magnification equals
that of ONE luminous source moving with the photocentre amplitude

    xi_ph = |B - beta| xi_rel,   B = q_M/(1+q_M),  beta = q_f/(1+q_f),

so the prediction is D_one(xi_E = xi_ph, P/tE), interpolated in log xi_E from
the one-luminous summary (same u0 and P/tE grid). On q_f = q_M the prediction
is zero and no white contour is drawn. The console prints, per panel, the
median |log10 D - log10 D_pred| over the cells where both are >= 1e-4.

Inputs : summaries of scan_xiE_ptE.py --qM --qf (one per row, in row order)
         and the one-luminous summary with the same --u0 and --refine.
Usage:
    python binary_source/plot_codes/plot_xirel_ptE_two_luminous.py \
        --summaries S1.npz S2.npz S3.npz --one-luminous S0.npz
Output (--out): D_xirel_ptE_two_luminous.{pdf,png}
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply_paper_style, fig_width, panel_label, save_figure,  # noqa: E402
                         D_LABEL, FS_ANNOT, LW_THIN)
import plot_xiE_ptE_maps as base  # noqa: E402

D_FLOOR = 1e-4          # cells used in the printed comparison


def photocentre_factor(qm, qf):
    return abs(qm / (1.0 + qm) - qf / (1.0 + qf))


def predicted_from_one(D1, xiE1, xi_rel, factor):
    """D_one(xi_E = factor * xi_rel, P/tE); NaN outside the one-luminous grid."""
    out = np.full((len(xi_rel), D1.shape[1]), np.nan)
    if factor == 0.0:
        return out
    x = np.log10(factor * np.asarray(xi_rel))
    lx1 = np.log10(xiE1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lD1 = np.log10(D1)
    for j in range(D1.shape[1]):
        ok = np.isfinite(lD1[:, j])
        if ok.sum() >= 2:
            out[:, j] = 10 ** np.interp(x, lx1[ok], lD1[ok, j], left=np.nan, right=np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="two-luminous summaries, one per row")
    ap.add_argument("--one-luminous", required=True,
                    help="one-luminous summary (same u0 and P/tE grid)")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"),
                                                  "binary_source", "figures", "current"))
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    apply_paper_style(usetex=not args.no_usetex)

    S1 = np.load(args.one_luminous, allow_pickle=False)
    u0s, xiE1, pte = S1["u0"], S1["xi_E"], S1["P_over_tE"]
    D1 = S1["D_p16_p50_p84"][:, 1]
    print("one luminous:", args.one_luminous, "| commit", S1["code_commit"],
          "dirty", bool(S1["code_dirty"]), "| u0", u0s)

    rows = []
    for path in args.summaries:
        S = np.load(path, allow_pickle=False)
        assert np.allclose(S["u0"], u0s), f"{path}: u0 differs from the one-luminous summary"
        assert np.allclose(S["P_over_tE"], pte), f"{path}: P/tE grid differs"
        qm, qf = float(S["q_mass_used"]), float(S["q_flux_used"])
        rows.append((qm, qf, S["xi_rel"], S["D_p16_p50_p84"][:, 1]))
        n_nan = int(np.isnan(S["D_all"]).sum())
        print(f"q_M={qm:g} q_f={qf:g}:", path, "| commit", S["code_commit"],
              "dirty", bool(S["code_dirty"]), f"| failed fits (NaN) {n_nan}")

    nr, nc = len(rows), len(u0s)
    fig = plt.figure(figsize=(fig_width(1.0), 1.0 + 2.15 * nr))
    left, right, gap, bottom = 0.085, 0.985, 0.022, 0.065
    top = 1.0 - 0.95 / (1.0 + 2.15 * nr)
    w = (right - left - (nc - 1) * gap) / nc
    h = (top - bottom - (nr - 1) * gap) / nr
    cax = fig.add_axes([left, top + 0.45 / (1.0 + 2.15 * nr), right - left, 0.012])
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for r, (qm, qf, xi_rel, Dmed) in enumerate(rows):
        fac = photocentre_factor(qm, qf)
        for k, u0 in enumerate(u0s):
            ax = fig.add_axes([left + k * (w + gap), top - (r + 1) * h - r * gap, w, h])
            pcm = base.draw_map(ax, xi_rel, pte, Dmed[k], label_contours=(r == 0 and k == 0),
                                period_labels=None, label_unreachable=False,
                                show_families=False)
            Dp = predicted_from_one(D1[k], xiE1, xi_rel, fac)
            if np.isfinite(Dp).any():
                X, Y = np.meshgrid(xi_rel, pte, indexing="ij")
                ax.contour(X, Y, np.log10(Dp), levels=np.log10(base.D_LEVELS), colors="white",
                           linewidths=LW_THIN + 0.3, linestyles=":", zorder=11)
                good = (Dmed[k] >= D_FLOOR) & (Dp >= D_FLOOR)
                dev = np.nanmedian(np.abs(np.log10(Dmed[k][good]) - np.log10(Dp[good])))
                print(f"  q_M={qm:g} q_f={qf:g} u0={u0:g}: |B-beta|={fac:.3f}, "
                      f"median |dlog10 D| = {dev:.3f} over {int(good.sum())} cells")
            else:
                print(f"  q_M={qm:g} q_f={qf:g} u0={u0:g}: |B-beta|={fac:.3f}, "
                      "no first-order prediction")
            if r == nr - 1:
                ax.set_xlabel(r"$\xi_{\rm rel}$")
            else:
                ax.set_xticklabels([])
            if k == 0:
                ax.set_ylabel(r"$P/t_E$")
            else:
                ax.set_yticklabels([])
            panel_label(ax, next(letters), color="white")
            ax.text(0.19, 0.95, rf"$u_0={u0:g}$", transform=ax.transAxes, ha="left",
                    va="top", color="white", fontsize=FS_ANNOT, zorder=30)
            ax.text(0.97, 0.05, rf"$q_M={qm:g}$, $q_f={qf:g}$", transform=ax.transAxes,
                    ha="right", va="bottom", color="white", fontsize=FS_ANNOT - 1, zorder=30)
    base.top_colorbar(fig, cax, pcm, D_LABEL)
    handles = [Line2D([], [], color="crimson", lw=1.2),
               Line2D([], [], color="white", lw=1.2, ls=":")]
    leg = fig.legend(handles, ["two luminous sources",
                               r"one luminous source at $\xi_E=|B-\beta|\,\xi_{\rm rel}$"],
                     loc="lower center", ncol=2, fontsize=FS_ANNOT - 1, frameon=True,
                     bbox_to_anchor=(0.5, top + 0.03 / (1.0 + 2.15 * nr)),
                     facecolor="0.3", labelcolor="white", edgecolor="none")
    leg.set_zorder(40)
    out = save_figure(fig, os.path.join(args.out, "D_xirel_ptE_two_luminous"))
    plt.close(fig)
    print("saved:", *out)


if __name__ == "__main__":
    main()
