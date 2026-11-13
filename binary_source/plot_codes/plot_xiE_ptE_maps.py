#!/usr/bin/env python3
"""
Intrinsic BSPL--PSPL mismatch in the exact variables (xi_E, P/tE) for one
luminous source, marginalised over isotropic orbital orientations.

Input : results/final_<commit>/xiE_ptE_maps/summary_xiE_ptE.npz
        (binary_source/scan_codes/scan_xiE_ptE.py)
Output (figures/current/):
    D_xiE_ptE_three_u0.{pdf,png}   one panel per u0 (full text width)
    D_xiE_ptE_u0_0p1.{pdf,png}     u0 = 0.1 only (same size as Fig. 2a)

Colour: median of D_BSPL-PSPL over the random orientations (viridis, same
LogNorm as Fig. 2). Crimson contours: median D = 1e-2 (solid) and 1e-3
(dashed). White curves: physical families with M1 + M2 = 3 Msun,
q_M = 0.5, rEhat = 5 AU (the family of Fig. 2), for tE = 30 d (solid) and
tE = 150 d (dashed); each curve is xi_E(P/tE) from Kepler's law.
Points on the white curves mark orbital periods (10 d ... 100 yr); on the
tE = 30 d curve they are labelled with P and the relative separation a_rel
(which depends only on P and the total mass, so it is the same on both curves).
Grey region: xi_E larger than any system with M1 + M2 <= MTOT_MAX,
tE <= TE_MAX and rEhat >= REHAT_MIN can reach (q_M -> infinity).

With --interacting, an extra single-panel figure (D_xiE_ptE_u0_0p1_interacting)
shows two families of a luminous 1 Msun giant with a dark compact companion
(white dwarf 0.6 Msun, neutron star 1.4 Msun), rEhat = 5 AU, tE = 30 d, and
marks for giant radii GIANT_RADII the period at which the giant fills its
Roche lobe (Eggleton 1983); shorter periods correspond to Roche-lobe overflow.
Open diamonds: rho = R/rEhat >= u0/2, where finite-source effects (not in the
point-source map) are important.

With --biases, a figure of the best-fitting PSPL parameter offsets over the
random orientations (PSPL_biases_one_luminous), replacing the reference-geometry
bias figures of plot_intrinsic_biases_paper.py (legacy). A PSPL fit absorbs the
almost constant orbital velocity of the luminous star during the event: the
trajectory stays straight, with another speed and direction. With
A = 2 pi xi_E tE/P (orbital speed in theta_E per tE) and v_par, v_perp the
projected unit-velocity components along and across the motion, to first order
    |Delta tE/tE| = A |v_par|,  |Delta t0/tE| = u0 A |v_perp|,  Delta u0/u0 = O(A^2),
so Delta tE/(tE A) and Delta t0/(tE u0 A) lie in [-1, 1] (uniform for isotropic
orbits). Panels: (a) median |Delta tE/tE| over orientations at u0 = U0_BIAS_MAP,
only where the median D < BIAS_D_MAX, with lines of constant A; (b) for each
orientation, the median of Delta tE/(tE A) and Delta t0/(tE u0 A) over the
first-order cells (A <= BIAS_A_MAX, P/tE >= BIAS_PTE_MIN, D < BIAS_D_MAX, all
u0), ranked, against the uniform distribution; (c) median |Delta u0/u0| against A.

Usage:
    python binary_source/plot_codes/plot_xiE_ptE_maps.py [--summary PATH] [--biases]
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply_paper_style, fig_width, heatmap, panel_label,  # noqa: E402
                         save_figure, d_norm, halo,
                         D_CMAP, D_LABEL, FS_ANNOT, LW_MAIN, LW_THIN,
                         C_BLACK, C_BLUE, C_VERMILION)

REPO_ROOT = Path(__file__).resolve().parents[2]
YEAR_D = 365.25

# physical family of Fig. 2
FAMILY_MTOT, FAMILY_QM, FAMILY_REHAT = 3.0, 0.5, 5.0
FAMILY_TE = (30.0, 150.0)

# limits for the reachable region (hatched outside)
MTOT_MAX, TE_MAX, REHAT_MIN = 20.0, 300.0, 1.0

PERIOD_MARKS = ((10.0, r"10\,d"), (100.0, r"100\,d"), (YEAR_D, r"1\,yr"),
                (10 * YEAR_D, r"10\,yr"), (100 * YEAR_D, r"100\,yr"))

# interacting-binary overlay: a 1 Msun giant (the luminous source) with a dark
# compact companion. Roche-lobe filling from Eggleton (1983).
GIANT_M1 = 1.0
GIANT_FAMILIES = (                          # label, companion mass, colour, linestyle
    (r"giant $+$ white dwarf ($0.6\,M_\odot$)", 0.6, "#E69F00", "-"),
    (r"giant $+$ neutron star ($1.4\,M_\odot$)", 1.4, "#CC79A7", "-"),
)
GIANT_RADII = (10.0, 30.0, 100.0)           # R_sun
RSUN_AU = 1.0 / 215.032

D_LEVELS = (1e-3, 1e-2)
U0_SINGLE = 0.1

# PSPL bias figure (--biases)
U0_BIAS_MAP = 0.3
BIAS_D_MAX = 1e-2          # biases only where the PSPL fit is acceptable
BIAS_A_MAX = 0.1           # first-order (constant-velocity) regime
BIAS_PTE_MIN = 10.0
BIAS_CMAP = "magma"
A_LINES = (1e-3, 1e-2, 1e-1)

PANEL_W = fig_width(0.48)          # same printed size as Fig. 2a
PANEL_H = 2.75
AX_RECT = [0.175, 0.135, 0.785, 0.640]
CAX_RECT = [0.175, 0.845, 0.785, 0.028]


def log_edges(x):
    lx = np.log10(np.asarray(x, float))
    mid = 0.5 * (lx[1:] + lx[:-1])
    return 10 ** np.concatenate([[lx[0] - (mid[0] - lx[0])], mid,
                                 [lx[-1] + (lx[-1] - mid[-1])]])


def xi_family(pte, tE, mtot=FAMILY_MTOT, qM=FAMILY_QM, rehat=FAMILY_REHAT):
    a_au = (mtot * (pte * tE / YEAR_D) ** 2) ** (1.0 / 3.0)
    return qM / (1.0 + qM) * a_au / rehat


def a_rel_au(P_days, mtot):
    return (mtot * (P_days / YEAR_D) ** 2) ** (1.0 / 3.0)


def roche_fraction(q):
    """Eggleton (1983): R_L / a for a donor with mass ratio q = M_donor / M_other."""
    return 0.49 * q ** (2 / 3) / (0.6 * q ** (2 / 3) + np.log(1.0 + q ** (1 / 3)))


def fmt_au(a):
    return rf"{a:.2g}\,AU" if a < 10 else rf"{a:.0f}\,AU"


def xi_reachable_max(pte):
    return (MTOT_MAX * (pte * TE_MAX / YEAR_D) ** 2) ** (1.0 / 3.0) / REHAT_MIN


def find_summary():
    files = sorted(glob.glob(str(REPO_ROOT / "results" / "*" / "xiE_ptE_maps"
                                 / "summary_xiE_ptE.npz")), key=os.path.getmtime)
    if not files:
        raise SystemExit("no summary_xiE_ptE.npz found; pass --summary")
    return files[-1]


def top_colorbar(fig, cax, mappable, label):
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(which="both", direction="in", pad=1.5)
    cbar.set_label(label, labelpad=3)
    cbar.ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
    return cbar


def label_on_curve(ax, x, y, x_at, text, color, dy_pts=1.5, above=True, dark=False):
    """Text aligned with a log-log curve at abscissa x_at (as in Fig. 2)."""
    import matplotlib.transforms as mtransforms
    good = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[good], np.asarray(y)[good]
    i = int(np.argmin(np.abs(np.log10(x) - np.log10(x_at))))
    j, k = min(i + 3, len(x) - 1), max(i - 3, 0)
    p1, p2 = ax.transData.transform((x[k], y[k])), ax.transData.transform((x[j], y[j]))
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    off = mtransforms.offset_copy(ax.transData, fig=ax.figure, x=0.0,
                                  y=(dy_pts if above else -dy_pts), units="points")
    return ax.text(x[i], y[i], text, color=color, fontsize=FS_ANNOT, rotation=angle,
                   rotation_mode="anchor", ha="center", va="bottom" if above else "top",
                   transform=off, zorder=30,
                   path_effects=halo(1.8, "black" if dark else "white"))


def contour_xy(cs):
    segs = [s for s in cs.allsegs[0] if len(s) > 1]
    seg = max(segs, key=len)
    return seg[:, 0], seg[:, 1]


def family_legend(ax, y0=0.84):
    """System of the white curves, with line styles for the two tE values."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="white", lw=LW_MAIN, ls=ls) for ls in ("-", "--")]
    labels = [rf"$t_E={tE:.0f}$\,d" for tE in FAMILY_TE]
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, y0 + 0.02),
                    fontsize=FS_ANNOT - 1.5, labelcolor="white", handlelength=1.6,
                    borderaxespad=0.3)
    leg.set_zorder(30)


def draw_map(ax, xi, pte, Dmed, label_family=False, label_contours=True,
             period_labels="full", label_unreachable=True, show_families=True,
             max_label_P=np.inf):
    pcm = heatmap(ax, log_edges(xi), log_edges(pte), np.ma.masked_invalid(Dmed).T,
                  norm=d_norm(), cmap=D_CMAP)
    X, Y = np.meshgrid(xi, pte, indexing="ij")
    logD = np.log10(np.where(np.isfinite(Dmed), Dmed, np.nan))
    cs = []
    for lev, ls in zip(D_LEVELS, ("--", "-")):
        c = ax.contour(X, Y, logD, levels=[np.log10(lev)], colors="crimson",
                       linewidths=LW_MAIN, linestyles=ls, zorder=10)
        cs.append(c)
    if label_contours:
        for c, txt in zip(cs, (r"$10^{-3}$", r"$10^{-2}$")):
            cx, cy = contour_xy(c)
            label_on_curve(ax, cx, cy, 2.2, txt, "crimson", above=True)

    # unreachable region
    p_line = np.logspace(np.log10(pte[0]) - 0.1, np.log10(pte[-1]) + 0.1, 400)
    xmax = xi_reachable_max(p_line)
    ax.fill_betweenx(p_line, xmax, xi[-1] * 10, facecolor="0.82", edgecolor="none",
                     zorder=12)
    ax.plot(xmax, p_line, color="0.45", lw=LW_THIN, zorder=12)
    if label_unreachable:
        ax.text(0.97, 0.03, "not\nreachable", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=FS_ANNOT - 1.5, color="0.25", zorder=13,
                linespacing=0.9)

    # physical families
    for tE, ls in (zip(FAMILY_TE, ("-", "--")) if show_families else ()):
        ax.plot(xi_family(p_line, tE), p_line, color="white", lw=LW_MAIN, ls=ls, zorder=15)
    if label_family and show_families:
        family_legend(ax)
    for k_te, tE in enumerate(FAMILY_TE if show_families else ()):
        for P, ptxt in PERIOD_MARKS:
            pt = P / tE
            if not (pte[0] <= pt <= pte[-1]):
                continue
            x = xi_family(pt, tE)
            ax.plot(x, pt, "o", ms=3.2, mfc="white", mec="black", mew=0.5, zorder=16)
            if k_te == 0 and period_labels and P <= max_label_P:
                txt = ptxt if period_labels == "period" else \
                    ptxt + ", " + fmt_au(a_rel_au(P, FAMILY_MTOT))
                ax.text(x / 1.35, pt, txt, ha="right", va="center", color="white",
                        fontsize=FS_ANNOT - 1.5, zorder=30, path_effects=halo(1.5, "black"))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(log_edges(xi)[0], log_edges(xi)[-1])
    ax.set_ylim(log_edges(pte)[0], log_edges(pte)[-1])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=12))
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    return pcm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"),
                                                  "binary_source", "figures", "current"))
    ap.add_argument("--no-usetex", action="store_true")
    ap.add_argument("--interacting", action="store_true",
                    help="also draw the u0=0.1 figure with a giant-donor family")
    ap.add_argument("--biases", action="store_true",
                    help="also draw the PSPL parameter-offset figure over orientations")
    args = ap.parse_args()
    apply_paper_style(usetex=not args.no_usetex)

    path = args.summary or find_summary()
    S = np.load(path, allow_pickle=False)
    u0s, xi, pte = S["u0"], S["xi_E"], S["P_over_tE"]
    Dmed = S["D_p16_p50_p84"][:, 1]                    # (u0, xi, P/tE)
    print("summary:", path, "| commit", S["code_commit"], "dirty", bool(S["code_dirty"]),
          "| random orientations:", int(S["n_random_geometries"]))

    # ---------------- three panels ----------------
    fig = plt.figure(figsize=(fig_width(1.0), 2.85))
    left, right, gap, bottom, top = 0.085, 0.985, 0.025, 0.145, 0.735
    w = (right - left - 2 * gap) / 3
    axes = [fig.add_axes([left + k * (w + gap), bottom, w, top - bottom]) for k in range(3)]
    cax = fig.add_axes([left, 0.815, right - left, 0.025])
    for k, (ax, u0) in enumerate(zip(axes, u0s)):
        pcm = draw_map(ax, xi, pte, Dmed[k], label_family=(k == 0), label_contours=True,
                       period_labels=("period" if k == 0 else None), label_unreachable=False,
                       max_label_P=20 * YEAR_D)
        ax.set_xlabel(r"$\xi_E$")
        if k == 0:
            ax.set_ylabel(r"$P/t_E$")
        else:
            ax.set_yticklabels([])
        panel_label(ax, "abc"[k], color="white")
        ax.text(0.19, 0.95, rf"$u_0={u0:g}$", transform=ax.transAxes, ha="left", va="top",
                color="white", fontsize=FS_ANNOT, zorder=30)
    top_colorbar(fig, cax, pcm, D_LABEL)
    out3 = save_figure(fig, os.path.join(args.out, "D_xiE_ptE_three_u0"))
    plt.close(fig)

    # ---------------- single u0 ----------------
    k = int(np.argmin(np.abs(np.log(u0s / U0_SINGLE))))
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes(AX_RECT)
    cax = fig.add_axes(CAX_RECT)
    out1 = single_panel(xi, pte, Dmed[k], u0s[k], args.out, interacting=False)
    outs = [*out3, *out1]
    if args.interacting:
        outs += single_panel(xi, pte, Dmed[k], u0s[k], args.out, interacting=True)
    if args.biases:
        outs += bias_figure(S, args.out)
    print("saved:", *outs)


def single_panel(xi, pte, Dmed, u0, outdir, interacting=False):
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes(AX_RECT)
    cax = fig.add_axes(CAX_RECT)
    pcm = draw_map(ax, xi, pte, Dmed, label_family=not interacting,
                   show_families=not interacting)
    ax.set_xlabel(r"$\xi_E$"); ax.set_ylabel(r"$P/t_E$")
    ax.text(0.04, 0.95, rf"$u_0={u0:g}$", transform=ax.transAxes, ha="left", va="top",
            color="white", fontsize=FS_ANNOT, zorder=30)
    stem = f"D_xiE_ptE_u0_{u0:g}".replace(".", "p")
    if interacting:
        from matplotlib.lines import Line2D
        stem += "_interacting"
        tE = FAMILY_TE[0]
        p_line = np.logspace(np.log10(pte[0]), np.log10(pte[-1]), 400)
        handles = []
        for label, m2, col, ls in GIANT_FAMILIES:
            mtot, qM = GIANT_M1 + m2, m2 / GIANT_M1
            ax.plot(xi_family(p_line, tE, mtot=mtot, qM=qM), p_line, color=col,
                    lw=LW_MAIN, ls=ls, zorder=17)
            rl = roche_fraction(GIANT_M1 / m2)          # donor = giant
            for R in GIANT_RADII:
                a = R * RSUN_AU / rl
                P = YEAR_D * np.sqrt(a ** 3 / mtot)
                pt = P / tE
                if not (pte[0] <= pt <= pte[-1]):
                    continue
                x = xi_family(pt, tE, mtot=mtot, qM=qM)
                finite = R * RSUN_AU / FAMILY_REHAT >= 0.5 * u0
                ax.plot(x, pt, "D", ms=4.0, mfc=("white" if finite else col), mec=col,
                        mew=1.0, zorder=18)
            handles.append(Line2D([], [], color=col, lw=LW_MAIN, ls=ls, marker="D",
                                  ms=3.5, mfc=col, mec=col))
        for R, dx in zip(GIANT_RADII, (1.35, 1.35, 1.35)):
            # label each radius once, next to the neutron-star family
            m2 = GIANT_FAMILIES[1][1]
            mtot, qM = GIANT_M1 + m2, m2 / GIANT_M1
            a = R * RSUN_AU / roche_fraction(GIANT_M1 / m2)
            P = YEAR_D * np.sqrt(a ** 3 / mtot)
            ax.text(xi_family(P / tE, tE, mtot=mtot, qM=qM) * dx, P / tE,
                    rf"${R:.0f}\,R_\odot$", ha="left", va="center", color="white",
                    fontsize=FS_ANNOT - 1.5, zorder=30, path_effects=halo(1.5, "black"))
        leg = ax.legend(handles, [f[0] for f in GIANT_FAMILIES], loc="upper left",
                        bbox_to_anchor=(0.0, 0.86), fontsize=FS_ANNOT - 2.5,
                        labelcolor="white", handlelength=1.6, borderaxespad=0.3)
        leg.set_zorder(30)
    top_colorbar(fig, cax, pcm, D_LABEL)
    out = save_figure(fig, os.path.join(outdir, stem))
    plt.close(fig)
    return out


def velocity_amplitude(xi, pte, factor=1.0):
    """A = 2 pi factor xi tE/P on the (xi, P/tE) grid (factor = |B - beta| for xi_rel)."""
    return 2.0 * np.pi * factor * np.asarray(xi)[:, None] / np.asarray(pte)[None, :]


def bias_cells(S, xi, pte, factor=1.0, d_max=BIAS_D_MAX, a_max=BIAS_A_MAX,
               pte_min=BIAS_PTE_MIN):
    """
    Offsets over the random orientations (geometry 0, the reference one, is
    excluded), as arrays (u0, xi, orientation, P/tE): r_tE = Delta tE/(tE A),
    r_t0 = Delta t0/(tE u0 A), |Delta u0/u0|, A, and the mask of first-order
    cells (finite offsets, D < d_max, 0 < A <= a_max, P/tE >= pte_min).
    """
    u0 = S["u0"]
    A = velocity_amplitude(xi, pte, factor)[None, :, None, :]
    D = S["D_all"][:, :, 1:, :]
    dte = S["dtE_over_tE_all"][:, :, 1:, :]
    dt0 = S["dt0_over_tE_all"][:, :, 1:, :]
    du0 = S["du0_over_u0_all"][:, :, 1:, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_tE = dte / A
        r_t0 = dt0 / (u0[:, None, None, None] * A)
    ok = (np.isfinite(dte) & np.isfinite(dt0) & np.isfinite(du0) & (D < d_max)
          & (A > 0) & (A <= a_max) & (np.asarray(pte)[None, None, None, :] >= pte_min))
    return dict(r_tE=r_tE, r_t0=r_t0, du0=np.abs(du0),
                A=np.broadcast_to(A, dte.shape), ok=ok)


def ranked_orientation_medians(r, ok):
    """Per orientation: median and 16-84 percentiles of r over the valid cells, ranked."""
    n_geo = r.shape[2]
    med, lo, hi = (np.full(n_geo, np.nan) for _ in range(3))
    for g in range(n_geo):
        v = r[:, :, g, :][ok[:, :, g, :]]
        if v.size:
            lo[g], med[g], hi[g] = np.percentile(v, [16, 50, 84])
    order = np.argsort(med)
    return med[order], lo[order], hi[order]


def bias_figure(S, outdir):
    from matplotlib.colors import LogNorm
    xi, pte, u0s = S["xi_E"], S["P_over_tE"], S["u0"]
    B = bias_cells(S, xi, pte)
    n_geo = B["r_tE"].shape[2]
    fig = plt.figure(figsize=(fig_width(1.0), 2.75))
    ax_a = fig.add_axes([0.075, 0.15, 0.27, 0.62])
    cax = fig.add_axes([0.075, 0.845, 0.27, 0.028])
    ax_b = fig.add_axes([0.435, 0.15, 0.235, 0.70])
    ax_c = fig.add_axes([0.755, 0.15, 0.215, 0.70])

    # (a) median |Delta tE/tE| at one u0, where the PSPL fit is acceptable
    k = int(np.argmin(np.abs(np.log(u0s / U0_BIAS_MAP))))
    D = S["D_all"][k, :, 1:, :]
    dte = np.abs(S["dtE_over_tE_all"][k, :, 1:, :])
    dte = np.where(D < BIAS_D_MAX, dte, np.nan)
    import warnings
    with warnings.catch_warnings():                       # all-NaN cells stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        med = 100.0 * np.nanmedian(dte, axis=1)          # percent, (xi, P/tE)
    Dmed = S["D_p16_p50_p84"][k, 1]
    med = np.where(Dmed < BIAS_D_MAX, med, np.nan)
    ax_a.set_facecolor("0.82")
    pcm = heatmap(ax_a, log_edges(xi), log_edges(pte), np.ma.masked_invalid(med).T,
                  norm=LogNorm(1e-3, 10.0), cmap=BIAS_CMAP)
    X, Y = np.meshgrid(xi, pte, indexing="ij")
    ax_a.contour(X, Y, np.log10(Dmed), levels=[np.log10(BIAS_D_MAX)], colors="crimson",
                 linewidths=LW_MAIN, linestyles="-", zorder=10)
    x_line = np.logspace(np.log10(xi[0]), np.log10(xi[-1]), 50)
    for a_val in A_LINES:
        y_line = 2.0 * np.pi * x_line / a_val
        ax_a.plot(x_line, y_line, color="white", lw=LW_THIN, ls="--", zorder=11)
        x_at = np.clip(a_val * 300.0 / (2.0 * np.pi), xi[2], xi[-8])   # label near P/tE ~ 300
        label_on_curve(ax_a, x_line, y_line, x_at, rf"$A={a_val:g}$", "white",
                       above=True, dark=True)
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_xlim(log_edges(xi)[0], log_edges(xi)[-1])
    ax_a.set_ylim(log_edges(pte)[0], log_edges(pte)[-1])
    ax_a.set_xlabel(r"$\xi_E$"); ax_a.set_ylabel(r"$P/t_E$")
    ax_a.text(0.96, 0.05, rf"$u_0={u0s[k]:g}$", transform=ax_a.transAxes, ha="right",
              va="bottom", color="white", fontsize=FS_ANNOT, zorder=30,
              path_effects=halo(1.5, "black"))
    panel_label(ax_a, "a", color="white")
    cbar = fig.colorbar(pcm, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top"); cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(which="both", direction="in", pad=1.5)
    cbar.set_label(r"median $|\Delta t_E|/t_E$ (\%)", labelpad=3)

    # (b) ranked per-orientation medians against the uniform distribution
    cdf = (np.arange(n_geo) + 0.5) / n_geo
    rline = np.linspace(-1.0, 1.0, 2)
    ax_b.plot(rline, 0.5 * (rline + 1.0), color="0.5", lw=LW_THIN, ls="--", zorder=1,
              label=r"uniform in $[-1,1]$")
    spreads = {}
    for key, col, lab, mk in (("r_tE", C_BLUE, r"$\Delta t_E/(t_E A)$", "o"),
                              ("r_t0", C_VERMILION, r"$\Delta t_0/(t_E u_0 A)$", "s")):
        m, lo, hi = ranked_orientation_medians(B[key], B["ok"])
        ax_b.errorbar(m, cdf, xerr=[m - lo, hi - m], fmt=mk, ms=3.0, color=col, mfc=col,
                      elinewidth=0.6, capsize=0, zorder=3, label=lab)
        spreads[key] = np.nanmax(hi - lo)
    ax_b.set_xlim(-1.15, 1.15); ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("offset / first-order amplitude")
    ax_b.set_ylabel("fraction of orientations")
    ax_b.legend(loc="upper left", fontsize=FS_ANNOT - 1.5, handlelength=1.4)
    panel_label(ax_b, "b", x=0.88, y=0.12)

    # (c) |Delta u0/u0| against A: second order
    edges = np.logspace(np.log10(BIAS_A_MAX) - 3.0, np.log10(BIAS_A_MAX), 13)
    mid = np.sqrt(edges[1:] * edges[:-1])
    colours = (C_BLACK, C_BLUE, C_VERMILION)
    for iu, u0 in enumerate(u0s):
        a = B["A"][iu][B["ok"][iu]]
        v = B["du0"][iu][B["ok"][iu]]
        y = np.full(len(mid), np.nan)
        for j in range(len(mid)):
            sel = (a >= edges[j]) & (a < edges[j + 1])
            if sel.sum() >= 10:
                y[j] = np.median(v[sel])
        ax_c.plot(mid, y, "o-", ms=3.0, lw=LW_THIN, color=colours[iu % 3],
                  label=rf"$u_0={u0:g}$")
    ax_c.plot(mid, mid, color="0.5", lw=LW_THIN, ls=":", label=r"$\propto A$")
    ref = np.nanmedian([np.nanmedian(B["du0"][iu][B["ok"][iu] & (B["A"][iu] > 0.03)])
                        for iu in range(len(u0s))])
    ax_c.plot(mid, ref * (mid / 0.05) ** 2, color="0.5", lw=LW_THIN, ls="--",
              label=r"$\propto A^2$")
    ax_c.set_xscale("log"); ax_c.set_yscale("log")
    ax_c.set_xlabel(r"$A = 2\pi\,\xi_E\,t_E/P$")
    ax_c.set_ylabel(r"median $|\Delta u_0|/u_0$")
    ax_c.legend(loc="upper left", fontsize=FS_ANNOT - 1.5, handlelength=1.4)
    panel_label(ax_c, "c", x=0.88, y=0.12)

    print(f"biases: {int(B['ok'].sum())} first-order cells over {n_geo} orientations; "
          f"largest 16-84 spread of a per-orientation median: r_tE {spreads['r_tE']:.3f}, "
          f"r_t0 {spreads['r_t0']:.3f}")
    for key in ("r_tE", "r_t0"):
        v = B[key][B["ok"]]
        print(f"  {key}: min {v.min():+.3f} max {v.max():+.3f} "
              f"median |.| {np.median(np.abs(v)):.3f} (uniform: 0.5)")
    out = save_figure(fig, os.path.join(outdir, "PSPL_biases_one_luminous"))
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
