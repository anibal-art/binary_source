#!/usr/bin/env python3
"""
Variant of plot_xiE_ptE_maps.py with physical scales, to compare with it.

Same maps (median D_BSPL-PSPL over the random orientations, crimson contours
D = 1e-3 and 1e-2, grey unreachable region; all drawn by
plot_xiE_ptE_maps.draw_map). Differences:

1. Secondary right axis: P for tE = REF_TE (labels on the last panel), with
   ticks at the periods of PERIOD_TICKS.
2. Instead of one binary at two tE, one white line per dark companion
   (labelled with its mass M2) of a luminous star of LUMINOUS_M1, all for tE = REF_TE and
   rEhat = REF_REHAT:  xi_E = M2/(M1+M2) * a_rel / rEhat, a_rel from Kepler.
   Since all lines share tE, a period is read as a horizontal row on the
   right axis.
Other tE or rEhat shift every line parallel to itself, by
(tE/REF_TE)^(2/3) (REF_REHAT/rEhat) in xi_E at fixed P/tE.

Usage:
    python binary_source/plot_codes/plot_xiE_ptE_maps_physical.py --summary PATH [PATH ...]

Several summaries (same xi_E and P/tE grids and geometries, e.g. the u0 = 0.01,
0.3, 1 run and a later u0 = 3 run) are joined along u0, one panel per u0.
With more than three panels the figure is wider (full page in landscape).
Output (--out, default as plot_xiE_ptE_maps.py): D_xiE_ptE_three_u0_physical.{pdf,png}
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply_paper_style, fig_width, panel_label, save_figure,  # noqa: E402
                         halo, D_LABEL, FS_ANNOT, LW_MAIN)
import plot_xiE_ptE_maps as base  # noqa: E402

REF_TE = 30.0            # d
REF_REHAT = 5.0          # AU
LUMINOUS_M1 = 1.0        # Msun
LABEL_PTE = 1.5          # all M2 labels at the same P/tE, above their lines
COMPANIONS = (           # kind (console only), M2 [Msun], P/tE of the M2 label, label above line
    ("brown dwarf", 0.05, LABEL_PTE, True),     # contour labels are placed away from lines
    ("white dwarf", 0.6, LABEL_PTE, True),      # and labels (base.label_contours_avoiding)
    ("neutron star", 1.4, LABEL_PTE, True),
    ("black hole", 10.0, LABEL_PTE, True),
    ("black hole", 60.0, LABEL_PTE, True),      # massive BHs, as found in GW mergers
)
PERIOD_TICKS = ((10.0, "10 d"), (100.0, "100 d"), (base.YEAR_D, "1 yr"),   # plain text:
                (10 * base.YEAR_D, "10 yr"), (100 * base.YEAR_D, "100 yr"))  # works without usetex


def companion_xi(pte, m2, tE=REF_TE, rehat=REF_REHAT, m1=LUMINOUS_M1):
    return base.xi_family(pte, tE, mtot=m1 + m2, qM=m2 / m1, rehat=rehat)


def draw_companions(ax, pte):
    """White companion lines with their M2 labels. Returns the lines and the label
    positions (data coordinates), which the contour labels must avoid."""
    p_line = np.logspace(np.log10(pte[0]) - 0.1, np.log10(pte[-1]) + 0.1, 400)
    avoid = []
    for _, m2, p_lab, above in COMPANIONS:
        x = companion_xi(p_line, m2)
        x_lab = float(companion_xi(p_lab, m2))
        ax.plot(x, p_line, color="white", lw=LW_MAIN, zorder=15)
        # smaller font: the 0.6 and 1.4 Msun lines are ~8 pt apart at a common P/tE
        text = rf"${m2:g}\,M_\odot$"
        base.label_on_curve(ax, x, p_line, x_lab, text, "white", above=above, dark=True,
                            fontsize=FS_ANNOT - 2)
        avoid.append(np.c_[x, p_line])
        avoid.append(np.array([[x_lab / 1.5, p_lab], [x_lab * 1.5, p_lab]]))
    return avoid


CROSS_LEVELS = (1e-3, 1e-2)
CROSS_PTE = (0.3, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


def _log_interp(xi, pte, Dmed):
    from scipy.interpolate import RegularGridInterpolator
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log10(np.where(Dmed > 0, Dmed, np.nan))
    return RegularGridInterpolator((np.log10(xi), np.log10(pte)), L, bounds_error=False)


def print_crossings(u0s, xi, pte, Dmed):
    """Numbers for the text, read from the median maps (not from the figure):
    1. along each companion line, the longest P/tE (and P for tE = REF_TE) with
       median D >= level;
    2. at fixed P/tE, the smallest xi_E with median D >= level."""
    p_line = np.logspace(np.log10(pte[0]), np.log10(pte[-1]), 2000)
    x_line = np.logspace(np.log10(xi[0]), np.log10(xi[-1]), 2000)
    for k, u0 in enumerate(u0s):
        f = _log_interp(xi, pte, Dmed[k])
        print(f"\n== u0 = {u0:g}: longest P/tE on each line with D >= level")
        for label, m2, *_ in COMPANIONS:
            x = companion_xi(p_line, m2)
            inside = (x >= xi[0]) & (x <= xi[-1])
            logD = f(np.c_[np.log10(x[inside]), np.log10(p_line[inside])])
            cells = []
            for lev in CROSS_LEVELS:
                ok = np.flatnonzero(logD >= np.log10(lev))
                if ok.size == 0:
                    cells.append(f"D={lev:g}: never")
                    continue
                pmax = p_line[inside][ok[-1]]
                tag = " (to the edge of the map)" if ok[-1] == inside.sum() - 1 else ""
                cells.append(f"D={lev:g}: P/tE {pmax:7.3g} = {pmax * REF_TE:8.4g} d{tag}")
            print(f"  {label:12s} M2={m2:5.2f}:  " + ";  ".join(cells))
        print(f"   smallest xi_E with D >= level at fixed P/tE")
        for pt in CROSS_PTE:
            logD = f(np.c_[np.log10(x_line), np.full_like(x_line, np.log10(pt))])
            cells = []
            for lev in CROSS_LEVELS:
                ok = np.flatnonzero(logD >= np.log10(lev))
                cells.append(f"D={lev:g}: " + (f"xi_E {x_line[ok[0]]:.3g}" if ok.size else "never"))
            print(f"  P/tE = {pt:6g}:  " + ";  ".join(cells))


def add_period_axis(ax, right):
    ax.tick_params(which="both", right=False)
    if right:
        sec = ax.secondary_yaxis("right", functions=(lambda p: p * REF_TE,
                                                     lambda P: P / REF_TE))
        sec.set_yticks([P for P, _ in PERIOD_TICKS])          # matplotlib < 3.5: no labels=
        sec.set_yticklabels([t for _, t in PERIOD_TICKS])
        sec.yaxis.set_minor_locator(NullLocator())
        sec.tick_params(direction="in")
        sec.set_ylabel(rf"$P$ for $t_E={REF_TE:g}\,$d")
    else:
        sec = ax.secondary_yaxis("right", functions=(lambda p: p, lambda p: p))
        sec.set_yticks([P / REF_TE for P, _ in PERIOD_TICKS])
        sec.yaxis.set_minor_locator(NullLocator())
        sec.tick_params(direction="in", labelright=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", nargs="+", default=None,
                    help="one or more summary_xiE_ptE.npz, joined along u0")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"),
                                                  "binary_source", "figures", "current"))
    ap.add_argument("--no-usetex", action="store_true")
    ap.add_argument("--crossings", action="store_true",
                    help="print where the D = 1e-3, 1e-2 contours cross the companion lines "
                         "and fixed P/tE (numbers for the text), then exit")
    args = ap.parse_args()
    apply_paper_style(usetex=not args.no_usetex)

    paths = args.summary or [base.find_summary()]
    u0_list, D_list, xi, pte, geo = [], [], None, None, None
    for path in paths:
        S = np.load(path, allow_pickle=False)
        if xi is None:
            xi, pte = S["xi_E"], S["P_over_tE"]
            geo = S["geometries"] if "geometries" in S.files else None
        elif not (np.allclose(S["xi_E"], xi) and np.allclose(S["P_over_tE"], pte)):
            raise SystemExit(f"{path}: xi_E or P/tE grid differs from {paths[0]}")
        elif geo is not None and "geometries" in S.files and not np.allclose(S["geometries"], geo):
            raise SystemExit(f"{path}: orbital geometries differ from {paths[0]}")
        u0_list.append(np.atleast_1d(S["u0"]))
        # D_p16_p50_p84 is (percentile, u0, xi, P/tE): np.nanpercentile puts the percentile axis first
        D_list.append(S["D_p16_p50_p84"][1])
        print("summary:", path, "| commit", S["code_commit"], "dirty", bool(S["code_dirty"]),
              "| u0", S["u0"], "| grid", len(xi), "x", len(pte))
    u0s = np.concatenate(u0_list)
    Dmed = np.concatenate(D_list, axis=0)
    order = np.argsort(u0s)
    u0s, Dmed = u0s[order], Dmed[order]
    if args.crossings:
        print_crossings(u0s, xi, pte, Dmed)
        return
    for label, m2, *_ in COMPANIONS:
        print(f"  {label:13s} M2={m2:5.2f}: xi_E(P=1 yr) = {companion_xi(base.YEAR_D / REF_TE, m2):.3f}")

    n = len(u0s)
    width = fig_width(1.0) * max(1.0, n / 3.0)      # 4 panels: landscape, full page
    fig = plt.figure(figsize=(width, 3.35))
    scale = fig_width(1.0) / width                   # keep the margins in inches
    left, right, gap, bottom, top = 0.075 * scale, 1 - 0.10 * scale, 0.025 * scale, 0.125, 0.83
    w = (right - left - (n - 1) * gap) / n
    axes = [fig.add_axes([left + k * (w + gap), bottom, w, top - bottom]) for k in range(n)]
    cax = fig.add_axes([left, 0.875, right - left, 0.025])
    for k, (ax, u0) in enumerate(zip(axes, u0s)):
        pcm, cs = base.draw_map(ax, xi, pte, Dmed[k], label_contours=False, period_labels=None,
                                label_unreachable=False, show_families=False,
                                return_contours=True)
        avoid = draw_companions(ax, pte)
        base.label_contours_avoiding(ax, cs, avoid_lines=avoid,
                                     box=(0.10, 0.92, 0.38, 0.85))   # above the M2 labels
        add_period_axis(ax, right=(k == n - 1))
        ax.set_xlabel(r"$\xi_E$")
        if k == 0:
            ax.set_ylabel(r"$P/t_E$")
        else:
            ax.set_yticklabels([])
        panel_label(ax, "abcdef"[k], color="white")
        ax.text(0.19, 0.95, rf"$u_0={u0:g}$", transform=ax.transAxes, ha="left", va="top",
                color="white", fontsize=FS_ANNOT, zorder=30, path_effects=halo(1.5, "black"))
    base.top_colorbar(fig, cax, pcm, D_LABEL)
    out = save_figure(fig, os.path.join(args.out, "D_xiE_ptE_three_u0_physical"))
    plt.close(fig)
    print("saved:", *out)


if __name__ == "__main__":
    main()
