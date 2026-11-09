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
    python binary_source/plot_codes/plot_xiE_ptE_maps_physical.py --summary PATH
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
COMPANIONS = (           # kind (console only), M2 [Msun], xi_E of the M2 label, label above line
    ("brown dwarf", 0.05, 0.0018, False),   # labels near P/tE ~ 1, between the 10 d
    ("white dwarf", 0.6, 0.017, True),      # and 100 d rows of circles, away from the
    ("neutron star", 1.4, 0.030, False),    # contour labels at the right edge
    ("black hole", 10.0, 0.076, False),
)
PERIOD_TICKS = ((10.0, r"10\,d"), (100.0, r"100\,d"), (base.YEAR_D, r"1\,yr"),
                (10 * base.YEAR_D, r"10\,yr"), (100 * base.YEAR_D, r"100\,yr"))


def companion_xi(pte, m2, tE=REF_TE, rehat=REF_REHAT, m1=LUMINOUS_M1):
    return base.xi_family(pte, tE, mtot=m1 + m2, qM=m2 / m1, rehat=rehat)


def draw_companions(ax, pte):
    p_line = np.logspace(np.log10(pte[0]) - 0.1, np.log10(pte[-1]) + 0.1, 400)
    for _, m2, x_lab, above in COMPANIONS:
        x = companion_xi(p_line, m2)
        ax.plot(x, p_line, color="white", lw=LW_MAIN, zorder=15)
        base.label_on_curve(ax, x, p_line, x_lab, rf"${m2:g}\,M_\odot$", "white",
                            above=above, dark=True)


def add_period_axis(ax, right):
    ax.tick_params(which="both", right=False)
    if right:
        sec = ax.secondary_yaxis("right", functions=(lambda p: p * REF_TE,
                                                     lambda P: P / REF_TE))
        sec.set_yticks([P for P, _ in PERIOD_TICKS], labels=[t for _, t in PERIOD_TICKS])
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
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"),
                                                  "binary_source", "figures", "current"))
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    apply_paper_style(usetex=not args.no_usetex)

    path = args.summary or base.find_summary()
    S = np.load(path, allow_pickle=False)
    u0s, xi, pte = S["u0"], S["xi_E"], S["P_over_tE"]
    Dmed = S["D_p16_p50_p84"][:, 1]
    print("summary:", path, "| commit", S["code_commit"], "dirty", bool(S["code_dirty"]),
          "| u0", u0s, "| grid", len(xi), "x", len(pte))
    for label, m2, *_ in COMPANIONS:
        print(f"  {label:13s} M2={m2:5.2f}: xi_E(P=1 yr) = {companion_xi(base.YEAR_D / REF_TE, m2):.3f}")

    n = len(u0s)
    fig = plt.figure(figsize=(fig_width(1.0), 3.35))
    left, right, gap, bottom, top = 0.075, 0.90, 0.025, 0.125, 0.83
    w = (right - left - (n - 1) * gap) / n
    axes = [fig.add_axes([left + k * (w + gap), bottom, w, top - bottom]) for k in range(n)]
    cax = fig.add_axes([left, 0.875, right - left, 0.025])
    for k, (ax, u0) in enumerate(zip(axes, u0s)):
        pcm = base.draw_map(ax, xi, pte, Dmed[k], label_contours=True, period_labels=None,
                            label_unreachable=False, show_families=False)
        draw_companions(ax, pte)
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
