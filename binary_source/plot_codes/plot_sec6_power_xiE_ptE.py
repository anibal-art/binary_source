#!/usr/bin/env python3
"""
Section 6 on the (xi_E, P/tE) plane of the intrinsic maps (tE = 30 d).

Inputs
    --power   : .npz of binary_source/analysis/roman_sec6_power.py on the
                --grid xiP Asimov summary (run with --k-h1);
    --summary : summary_xiE_ptE.npz of binary_source/scan_codes/scan_xiE_ptE.py
                (intrinsic maps, same u0, xi_E, P/tE nodes and orientations);
                default: the newest results/*/xiE_ptE_maps/summary_xiE_ptE.npz.

Outputs (<outdir>/)
    sec6_Droman_xiE_ptE_g<g>.{pdf,png}     (main figure; --droman-g)
        rows u0, columns F146. Colour: D_Roman = sqrt(lambda)/SNR, median over
        the random orientations, on the colour scale of the intrinsic D
        maps (paper_style.d_norm). White: Roman detection, orientation-mean
        efficiency 0.5 (thick) and 0.9 (thin) at the primary false-alarm level
        (--primary: 3 sigma, the default, or 5%). Crimson: intrinsic
        median D = 1e-2 (solid) and 1e-3 (dashed). Dash-dotted: physical family.
    sec6_power_xiE_ptE_u0_<u0>.{pdf,png}   (only with --power-maps)
        rows g = 0, 1, 4, columns F146 = 19, 21, 23, at one u0. Colour:
        efficiency at the primary level averaged over the 24 isotropic
        orientations. White: 0.5 and 0.9 at the primary level (solid), 0.5 at
        the other level (dashed).
        Dash-dotted: the physical family of the paper (M1 + M2 = 3 Msun,
        q_M = 0.5, rEhat = 5 AU) at tE = 30 d.
    sec6_regions_xiE_ptE_g<g>.{pdf,png}
        columns F146, rows u0: each cell is "detected" (orientation-mean
        efficiency >= 0.5 at the primary level), "confused, biased"
        (efficiency < 0.5 and the intrinsic
        median |Delta tE/tE| over orientations >= --bias-threshold) or
        "confused, unbiased".
    sec6_D_ratio.{pdf,png}
        median over orientations of D_Roman / D_intrinsic against P/tE, per
        F146 (g = 0), one panel per u0; only cells with
        D_intrinsic in [1e-4, 1e-2] (D_Roman = sqrt(lambda)/SNR, eq. 22).

With real sampling the problem is not scale free: the maps hold at tE = 30 d
for any companion mass and source distance, not for another tE.

Usage (paper figures; the defaults of --power and --summary are the paper
inputs listed at PAPER_POWER / PAPER_SUMMARY below):
    python binary_source/plot_codes/plot_sec6_power_xiE_ptE.py --outdir figures/current [--no-usetex]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply_paper_style, fig_width, heatmap, save_figure,  # noqa: E402
                         d_norm, C_BLUE, C_VERMILION, D_CMAP)
from plot_xiE_ptE_maps import REPO_ROOT, log_edges, xi_family  # noqa: E402
from plot_sec6_power_u0P import LEVELS, efficiency_levels  # noqa: E402

D_RATIO_RANGE = (1e-4, 1e-2)

# Inputs of the paper figures (Section 6 and Appendix E):
#   power   : roman_sec6_power.py --k-h1 6.21 on the xiPpaper grid
#             (roman_sec6_asimov_cells.py --grid xiPpaper: u0 = 0.01, 0.1, 0.3, 1;
#             results/roman_sec6_asimov_xi_v2/roman_sec6_asimov_xi_paper.npz)
#             with the H0 thresholds of results/roman_lrt/gv_sec6_h0_final;
#   summary : scan_xiE_ptE.py --u0 0.01 0.3 1 --refine 2 (the maps of Figure 2).
# The rows are the u0 present in both files: 0.01, 0.3, 1.
PAPER_POWER = REPO_ROOT / "results/roman_sec6_final/sec6_power_xi_kh1.npz"
PAPER_SUMMARY = (REPO_ROOT / "results/final_52468b04987a"
                 / "xiE_ptE_maps_u0_0p01_0p3_1_refine2/summary_xiE_ptE.npz")
PAPER_U0 = (0.01, 0.3, 1.0)


def match_index(full, sub, what):
    idx = []
    for v in sub:
        j = np.flatnonzero(np.isclose(full, v, rtol=1e-9, atol=0.0))
        if len(j) != 1:
            raise SystemExit(f"{what} = {v:g} of the Roman grid is not a node of the intrinsic grid")
        idx.append(int(j[0]))
    return np.array(idx)


def labels(tex):
    """Mathtext labels: the same strings work with and without usetex."""
    return dict(xi=r"$\xi_E$", pte=r"$P/t_E$",
                f146=(lambda m: rf"$F_{{146}} = {m:g}$"),
                g=(lambda g: rf"$g = {g:g}$"),
                u0=(lambda u: rf"$u_0 = {u:g}$"))


def power_figure(d, iu, outdir, tex, primary):
    L = labels(tex)
    xi, pte, blend, mags = d["xiE_grid"], d["PtE_grid"], d["blend_ratio"], d["F146"]
    tE = float(d["tE"])
    pw, pw_other, lev = efficiency_levels(d, primary)        # (geo, g, F146, u0, xi, P)
    iso = slice(1, pw.shape[0])
    fig, axes = plt.subplots(len(blend), len(mags), figsize=(fig_width(1.0), fig_width(0.95)),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    pline = np.geomspace(pte[0], pte[-1], 200)
    for ib, g in enumerate(blend):
        for im, mag in enumerate(mags):
            ax = axes[ib, im]
            m = np.nanmean(pw[iso, ib, im, iu], axis=0).T          # (P, xi)
            m_other = np.nanmean(pw_other[iso, ib, im, iu], axis=0).T
            pcm = heatmap(ax, log_edges(xi), log_edges(pte), m, vmin=0.0, vmax=1.0,
                          cmap="viridis")
            ax.contour(xi, pte, m, levels=[0.5, 0.9], colors="w", linewidths=0.9)
            ax.contour(xi, pte, m_other, levels=[0.5], colors="w", linewidths=0.9,
                       linestyles="--")
            ax.plot(xi_family(pline, tE), pline, color="w", lw=0.8, ls="-.")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(xi[0], xi[-1])
            ax.set_ylim(pte[0], pte[-1])
            if ib == 0:
                ax.set_title(L["f146"](mag))
            if im == 0:
                ax.set_ylabel(L["g"](g) + "\n" + L["pte"])
            if ib == len(blend) - 1:
                ax.set_xlabel(L["xi"])
    fig.colorbar(pcm, ax=axes, shrink=0.9,
                 label=f"Detection efficiency ({lev}), mean over orientations, "
                       + L["u0"](d["u0_grid"][iu]))
    return save_figure(fig, Path(outdir) / f"sec6_power_xiE_ptE_u0_{d['u0_grid'][iu]:g}".replace(".", "p"))


def intrinsic(summaries, u0, xi, pte):
    """
    Intrinsic maps at one u0 from the first summary that has it: median over
    orientations of |Delta tE/tE| (xi, P) and D for every orientation
    (geo, xi, P), on the Roman (xi_E, P/tE) nodes.  None if no summary has u0.
    """
    for path, S in summaries:
        j = np.flatnonzero(np.isclose(S["u0"], u0, rtol=1e-9, atol=0.0))
        if len(j) != 1:
            continue
        jx = match_index(S["xi_E"], xi, "xi_E")
        jp = match_index(S["P_over_tE"], pte, "P/tE")
        bias = S["abs_dtE_over_tE_p16_p50_p84"][1][j[0]][np.ix_(jx, jp)]
        D = S["D_all"][j[0]][np.ix_(jx, np.arange(S["D_all"].shape[2]), jp)]
        return path, bias, np.moveaxis(D, 1, 0)
    return None


def droman_figure(d, summaries, ib, outdir, tex, primary, bias_thr=0.01):
    """
    D_Roman = sqrt(lambda)/SNR (median over the random orientations) on the
    axes and colour scale of the intrinsic D maps, with the Roman detection
    contour (orientation-mean efficiency 0.5 and 0.9 at the primary level) and the
    intrinsic contours D = 1e-2 (solid) and 1e-3 (dashed). Hatched: undetected
    (efficiency < 0.5) with intrinsic median |Delta tE/tE| >= bias_thr.
    """
    L = labels(tex)
    xi, pte, mags = d["xiE_grid"], d["PtE_grid"], d["F146"]
    tE = float(d["tE"])
    print(f"\nD_Roman map, g = {d['blend_ratio'][ib]:g}:")
    rows = rows_with_intrinsic(d, summaries)
    Drom = d["D_ROMAN_EFF"]                                  # (geo, g, F146, u0, xi, P)
    pw = efficiency_levels(d, primary)[0]
    iso = slice(1, Drom.shape[0])
    pline = np.geomspace(pte[0], pte[-1], 200)
    fig, axes = plt.subplots(len(rows), len(mags), figsize=(fig_width(1.0), fig_width(0.95)),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for r, (iu, (_, b, di)) in enumerate(rows):               # b: (xi, P); di: (geo, xi, P)
        dint = np.nanmedian(di[iso], axis=0)                  # intrinsic, (xi, P)
        for im, mag in enumerate(mags):
            ax = axes[r, im]
            dr = np.nanmedian(Drom[iso, ib, im, iu], axis=0)  # (xi, P)
            m = np.nanmean(pw[iso, ib, im, iu], axis=0)
            pcm = heatmap(ax, log_edges(xi), log_edges(pte), dr.T, norm=d_norm(), cmap=D_CMAP)
            ax.contour(xi, pte, m.T, levels=[0.5], colors="w", linewidths=1.1)
            ax.contour(xi, pte, m.T, levels=[0.9], colors="w", linewidths=0.7)
            with np.errstate(divide="ignore", invalid="ignore"):
                ld = np.log10(dint.T)
            ax.contour(xi, pte, ld, levels=[-2.0], colors="crimson", linewidths=1.0,
                       linestyles="solid")
            ax.contour(xi, pte, ld, levels=[-3.0], colors="crimson", linewidths=1.0,
                       linestyles="--")
            hidden = ((m < 0.5) & (b >= bias_thr)).astype(float)
            hidden[~np.isfinite(m) | ~np.isfinite(b)] = 0.0
            if hidden.any():
                with plt.rc_context({"hatch.color": "w", "hatch.linewidth": 0.5}):
                    ax.contourf(xi, pte, hidden.T, levels=[0.5, 1.5], colors="none",
                                hatches=["////"])
            ax.plot(xi_family(pline, tE), pline, color="w", lw=0.8, ls="-.")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(xi[0], xi[-1])
            ax.set_ylim(pte[0], pte[-1])
            if r == 0:
                ax.set_title(L["f146"](mag))
            if im == 0:
                ax.set_ylabel(L["u0"](d["u0_grid"][iu]) + "\n" + L["pte"])
            if r == len(rows) - 1:
                ax.set_xlabel(L["xi"])
    fig.colorbar(pcm, ax=axes, shrink=0.9,
                 label=r"$D_{\rm Roman}=\sqrt{\lambda}/{\rm SNR}$, median over orientations, "
                       + L["g"](d["blend_ratio"][ib]))
    return save_figure(fig, Path(outdir) / f"sec6_Droman_xiE_ptE_g{d['blend_ratio'][ib]:g}")


def rows_with_intrinsic(d, summaries):
    rows = []
    for iu, u in enumerate(d["u0_grid"]):
        r = intrinsic(summaries, u, d["xiE_grid"], d["PtE_grid"])
        if r is None:
            print(f"  u0 = {u:g}: no intrinsic summary with this u0, skipped")
        else:
            print(f"  u0 = {u:g}: intrinsic maps from {r[0]}")
            rows.append((iu, r))
    if not rows:
        raise SystemExit("no u0 of the Roman grid is in the intrinsic summaries")
    return rows


def regions_figure(d, summaries, ib, bias_thr, outdir, tex, primary):
    L = labels(tex)
    xi, pte, mags = d["xiE_grid"], d["PtE_grid"], d["F146"]
    rows = rows_with_intrinsic(d, summaries)
    u0 = np.array([d["u0_grid"][iu] for iu, _ in rows])
    pw = efficiency_levels(d, primary)[0]
    iso = slice(1, pw.shape[0])
    cmap = ListedColormap(["0.85", C_VERMILION, C_BLUE])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig = plt.figure(figsize=(fig_width(1.0), fig_width(1.02)), constrained_layout=True)
    gs = fig.add_gridspec(len(u0) + 1, len(mags), height_ratios=[0.13] + [1.0] * len(u0))
    lax = fig.add_subplot(gs[0, :])
    lax.axis("off")
    axes = np.empty((len(u0), len(mags)), dtype=object)
    for r in range(len(u0)):
        for c in range(len(mags)):
            axes[r, c] = fig.add_subplot(gs[r + 1, c],
                                         sharex=axes[0, 0] if (r or c) else None,
                                         sharey=axes[0, 0] if (r or c) else None)
            if r < len(u0) - 1:
                axes[r, c].tick_params(labelbottom=False)
            if c > 0:
                axes[r, c].tick_params(labelleft=False)
    print(f"\nregions, g = {d['blend_ratio'][ib]:g}, |Delta tE/tE| threshold {bias_thr:g}:")
    for r, (iu, (_, b, _)) in enumerate(rows):                # b: (xi, P)
        for im, mag in enumerate(mags):
            ax = axes[r, im]
            m = np.nanmean(pw[iso, ib, im, iu], axis=0)       # (xi, P)
            cls = np.where(m >= 0.5, 2, np.where(b >= bias_thr, 1, 0)).astype(float)
            cls[~np.isfinite(m) | ~np.isfinite(b)] = np.nan
            heatmap(ax, log_edges(xi), log_edges(pte), cls.T, cmap=cmap, norm=norm)
            ax.set_xscale("log")
            ax.set_yscale("log")
            from matplotlib.ticker import LogLocator     # same ticks in the PDF and the PNG
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
            fin = np.isfinite(cls)
            print(f"  u0 {u0[r]:g} F146 {mag:g}: detected {np.mean(cls[fin] == 2):.3f}  "
                  f"confused+biased {np.mean(cls[fin] == 1):.3f}  "
                  f"confused+unbiased {np.mean(cls[fin] == 0):.3f}")
            if r == 0:
                ax.set_title(L["f146"](mag))
            if im == 0:
                ax.set_ylabel(L["u0"](u0[r]) + "\n" + L["pte"])
            if r == len(u0) - 1:
                ax.set_xlabel(L["xi"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_BLUE, C_VERMILION, "0.85")]
    thr = rf"$|\Delta t_E/t_E| \geq {bias_thr:g}$"
    lax.legend(handles, [r"$\varepsilon \geq 0.5$", r"$\varepsilon < 0.5$, " + thr,
                         r"$\varepsilon < 0.5$, $|\Delta t_E/t_E| <$ " + f"{bias_thr:g}"],
               loc="center", ncol=3, frameon=False, handlelength=1.2, columnspacing=1.2)
    return save_figure(fig, Path(outdir) / f"sec6_regions_xiE_ptE_g{d['blend_ratio'][ib]:g}")


def d_ratio_figure(d, summaries, outdir, tex):
    L = labels(tex)
    pte, mags = d["PtE_grid"], d["F146"]
    print("\nD_Roman / D_intrinsic (median over orientations and xi_E cells, g = 0):")
    rows = rows_with_intrinsic(d, summaries)
    Drom = d["D_ROMAN_EFF"]                                  # (geo, g, F146, u0, xi, P)
    ib = int(np.flatnonzero(np.isclose(d["blend_ratio"], 0.0))[0])
    fig, axes = plt.subplots(1, len(rows), figsize=(fig_width(1.0), fig_width(0.36)),
                             sharey=True, squeeze=False, constrained_layout=True)
    u0 = d["u0_grid"]
    for r, (iu, (_, _, di)) in enumerate(rows):               # di: (geo, xi, P)
        ax = axes[0, r]
        for im, (mag, col) in enumerate(zip(mags, ("k", C_BLUE, C_VERMILION))):
            dr = Drom[:, ib, im, iu]                                  # (geo, xi, P)
            ok = (di >= D_RATIO_RANGE[0]) & (di <= D_RATIO_RANGE[1]) & np.isfinite(dr) & (di > 0)
            ok[0] = False                                             # random-orientation rows only
            ratio = np.where(ok, dr / np.where(di > 0, di, np.nan), np.nan)
            med = np.array([np.nanmedian(ratio[..., j]) if ok[..., j].any() else np.nan
                            for j in range(len(pte))])
            ax.plot(pte, med, color=col, marker="o", ms=2.5, lw=1.0, label=L["f146"](mag))
            print(f"  u0 {u0[iu]:g} F146 {mag:g}: " + " ".join(f"{v:.2f}" for v in med[::4]))
        ax.axhline(1.0, color="0.5", lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(L["pte"])
        ax.set_title(L["u0"](u0[iu]))
    axes[0, 0].set_ylabel(r"$D_{\rm Roman}/D$")
    axes[0, 0].legend(frameon=False)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    yax = axes[0, 0].yaxis                                   # shared by all panels
    yax.set_major_locator(FixedLocator([0.3, 0.5, 1.0, 2.0, 5.0, 10.0]))
    yax.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    yax.set_minor_formatter(NullFormatter())
    return save_figure(fig, Path(outdir) / "sec6_D_ratio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", default=str(PAPER_POWER))
    ap.add_argument("--summary", nargs="+", default=[str(PAPER_SUMMARY)],
                    help="one or more summary_xiE_ptE.npz; each u0 of the Roman grid is "
                         "taken from the first summary that has it")
    ap.add_argument("--outdir", default="figures/current")
    ap.add_argument("--u0", type=float, default=0.1, help="u0 of the power figure")
    ap.add_argument("--region-g", type=float, default=0.0, help="blend ratio of the region figure")
    ap.add_argument("--bias-threshold", type=float, default=0.01)
    ap.add_argument("--droman-g", type=float, nargs="+", default=[0.0],
                    help="blend ratios of the D_Roman maps (one figure each)")
    ap.add_argument("--power-maps", action="store_true",
                    help="also the power maps (rows g) at --u0")
    ap.add_argument("--primary", choices=tuple(LEVELS), default="3sigma",
                    help="false-alarm level of the detection contours and regions")
    ap.add_argument("--any-u0", action="store_true",
                    help="accept rows other than the paper's u0 = 0.01, 0.3, 1")
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    tex = not args.no_usetex
    apply_paper_style(usetex=tex)

    d = np.load(args.power)
    if "xiE_grid" not in d.files:
        raise SystemExit("--power is not a --grid xiP power file")
    if not np.isfinite(float(d["k_h1"])):
        print("warning: power file made without --k-h1 (H0 k_local used)")
    summaries = []
    for spath in args.summary:
        S = np.load(spath)
        if not np.allclose(S["geometries"], d["geometry"]):
            raise SystemExit(f"orientations of {spath} differ from the Roman grid")
        # one luminous source: the xi axis is xi_E (two luminous sources: xi_rel);
        # q_flux_used is not stored in older summaries
        if "xi_E" not in S.files or ("q_flux_used" in S.files and float(S["q_flux_used"]) != 0.0):
            raise SystemExit(f"{spath} is not a one-luminous-source summary (xi_E grid)")
        summaries.append((spath, S))
    rows_u0 = [float(u) for u in d["u0_grid"]
               if any(np.isclose(S["u0"], u, rtol=1e-9, atol=0.0).any() for _, S in summaries)]
    if not args.any_u0 and (len(rows_u0) != len(PAPER_U0)
                            or not np.allclose(rows_u0, PAPER_U0)):
        raise SystemExit(f"rows u0 = {rows_u0}, not the paper's {list(PAPER_U0)}: "
                         "check --power and --summary, or pass --any-u0")
    iu = int(np.argmin(np.abs(np.log(d["u0_grid"] / args.u0))))
    ib = int(np.argmin(np.abs(d["blend_ratio"] - args.region_g)))
    out = []
    for g in args.droman_g:
        out += droman_figure(d, summaries, int(np.argmin(np.abs(d["blend_ratio"] - g))),
                             args.outdir, tex, args.primary, args.bias_threshold)
    if args.power_maps:
        out += power_figure(d, iu, args.outdir, tex, args.primary)
    out += regions_figure(d, summaries, ib, args.bias_threshold, args.outdir, tex, args.primary)
    out += d_ratio_figure(d, summaries, args.outdir, tex)
    for f in out:
        print("saved", f)


if __name__ == "__main__":
    main()
