#!/usr/bin/env python3
"""
Quantitative checks of the two-luminous-source maps of scan_xiE_ptE.py --qM --qf.

Expansion of the two-source magnification around the light centroid
(Han & Gould 1997, ApJ 480, 196, Sect. 2.2):
  * first order: one luminous source moving with the photocentre amplitude
        xi_ph = |B - beta| xi_rel,  B = q_M/(1+q_M), beta = q_f/(1+q_f);
  * second order: a separation term proportional to beta(1-beta) xi_rel^2,
    which does not average out and, for P >> tE, is a STATIC binary source
    that the PSPL fit cannot absorb.

Checks (all on the median D over the random orientations):
 1. First order, by bands of P/tE: median |log10 D - log10 D_one(xi_ph)|
    (only rows with q_f != q_M; D_one from the one-luminous summary).
 2. Long-period floor, P/tE >= --pte-static: D_floor(xi_rel) = median of D
    over those P/tE.
    a. spread of log10 D over those P/tE at fixed xi_rel (P-independence);
    b. power-law slope of D_floor vs xi_rel in the perturbative window
       (expected 2), least squares in log-log (analyze_photocenter_scaling.py
       runs its analysis at import time, so its fit_power_law is not imported);
    c. collapse: C = D_floor / [beta(1-beta) xi_rel^2] in the window; if the
       floor is the second-order term, C depends on u0 only, not on (q_M, q_f);
    d. xi_rel where D_floor = 1e-3 and 1e-2, and the ratio to the row with the
       largest beta(1-beta), compared with [beta(1-beta)]^(-1/2).
 3. Validation (--validate S.npz, a (q_M, q_f) pair NOT in --summaries):
    C(u0) is calibrated as the mean of check 2c over the --summaries rows and
    the two-term model
        D_model = sqrt( D_one(xi_ph)^2 + [C(u0) beta(1-beta) xi_rel^2]^2 ),
    restricted to xi_rel <= --xi-pert, is compared with the measured map
    (median |log10 D - log10 D_model| by bands of P/tE), together with the
    long-period floor alone and the xi_rel where it reaches 1e-3. Adding the
    two terms in quadrature is an assumption of the model, tested here.
Optional figure (--figure): D_floor / beta(1-beta) vs xi_rel, one panel per
u0, one curve per (q_M, q_f), with a xi^2 reference; with --validate also the
measured map of the validation pair with the model contours.

Usage:
    python binary_source/analysis/two_luminous_scaling.py \
        --summaries S1.npz S2.npz S3.npz --one-luminous S0.npz [--figure] \
        [--validate S4.npz]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parents[1]
for sub in ("analysis", "plot_codes"):
    if str(SOURCE_DIR / sub) not in sys.path:
        sys.path.insert(0, str(SOURCE_DIR / sub))

from plot_xirel_ptE_two_luminous import (            # noqa: E402
    photocentre_factor, predicted_from_one)

PTE_BANDS = ((0.0, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, np.inf))
LEVELS = (1e-3, 1e-2)


def crossing(x, y, level):
    """First x where y rises through level (log-log interpolation); NaN if none."""
    lx, ly = np.log10(x), np.log10(y)
    ok = np.isfinite(ly)
    lx, ly = lx[ok], ly[ok]
    above = ly >= np.log10(level)
    idx = np.flatnonzero(above[1:] & ~above[:-1])
    if not idx.size:
        return np.nan
    i = idx[0]
    f = (np.log10(level) - ly[i]) / (ly[i + 1] - ly[i])
    return 10 ** (lx[i] + f * (lx[i + 1] - lx[i]))


def band_dev(D, Dp, pte, lo, hi, floor=1e-4):
    """Median |log10 D - log10 Dp| over the cells of lo <= P/tE < hi with both >= floor."""
    band = (pte >= lo) & (pte < hi)
    d, p = D[:, band], Dp[:, band]
    good = (d >= floor) & (p >= floor)
    if not good.any():
        return np.nan
    return np.nanmedian(np.abs(np.log10(d[good]) - np.log10(p[good])))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--one-luminous", required=True)
    ap.add_argument("--pte-static", type=float, default=100.0,
                    help="P/tE above which the floor is measured")
    ap.add_argument("--xi-window", type=float, nargs=2, default=(1e-2, 1e-1),
                    help="xi_rel range of the perturbative fit")
    ap.add_argument("--d-min", type=float, default=2e-5,
                    help="ignore D below this value in the fit (numerical floor)")
    ap.add_argument("--xi-pert", type=float, default=0.2,
                    help="largest xi_rel where the two-term model is applied")
    ap.add_argument("--validate", default=None,
                    help="summary of a (q_M, q_f) pair not used for the calibration")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"),
                                                  "binary_source", "figures", "current"))
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()

    S1 = np.load(args.one_luminous, allow_pickle=False)
    u0s, xiE1, pte = S1["u0"], S1["xi_E"], S1["P_over_tE"]
    D1 = S1["D_p16_p50_p84"][:, 1]
    rows = []
    for path in args.summaries:
        S = np.load(path, allow_pickle=False)
        assert np.allclose(S["u0"], u0s) and np.allclose(S["P_over_tE"], pte)
        qm, qf = float(S["q_mass_used"]), float(S["q_flux_used"])
        beta = qf / (1.0 + qf)
        rows.append(dict(qm=qm, qf=qf, beta=beta, b1b=beta * (1.0 - beta),
                         fac=photocentre_factor(qm, qf), xi=S["xi_rel"],
                         D=S["D_p16_p50_p84"][:, 1], commit=str(S["code_commit"])))
    xi = rows[0]["xi"]
    assert all(np.allclose(r["xi"], xi) for r in rows)
    print(f"one luminous: commit {S1['code_commit']}; two luminous: commit "
          f"{rows[0]['commit']}; u0 = {u0s}")

    # ---------------- 1. first order by P/tE band ----------------
    print("\n1. First order: median |log10 D - log10 D_one(xi_ph)| (cells with both >= 1e-4)")
    print("   q_M   q_f   u0    " + "  ".join(f"P/tE {lo:g}-{hi:g}" for lo, hi in PTE_BANDS))
    for r in rows:
        if r["fac"] == 0.0:
            continue
        for k, u0 in enumerate(u0s):
            Dp = predicted_from_one(D1[k], xiE1, xi, r["fac"])
            cells = [band_dev(r["D"][k], Dp, pte, lo, hi) for lo, hi in PTE_BANDS]
            print(f"   {r['qm']:<5g} {r['qf']:<5g} {u0:<5g} " +
                  "  ".join(f"{c:12.3f}" for c in cells))

    # ---------------- 2. long-period floor ----------------
    long = pte >= args.pte_static
    lo_xi, hi_xi = args.xi_window
    print(f"\n2. Long-period floor, P/tE >= {args.pte_static:g} ({int(long.sum())} values); "
          f"fit window xi_rel = {lo_xi:g}-{hi_xi:g}, D >= {args.d_min:g}")
    print("   q_M   q_f   b(1-b)  u0    spread  slope  rms    C=D/[b(1-b)xi^2]  "
          "xi(1e-3)  xi(1e-2)")
    floors, Cs = {}, {}
    for r in rows:
        for k, u0 in enumerate(u0s):
            Dl = r["D"][k][:, long]
            with np.errstate(invalid="ignore", divide="ignore"):
                floor = np.nanmedian(Dl, axis=1)
                spread = np.nanmedian(np.nanstd(np.log10(Dl), axis=1)[
                    (xi >= lo_xi) & (xi <= hi_xi)])
            w = (xi >= lo_xi) & (xi <= hi_xi) & (floor >= args.d_min)
            slope = rms = np.nan
            if w.sum() >= 5:
                lx, ly = np.log10(xi[w]), np.log10(floor[w])
                slope, icpt = np.polyfit(lx, ly, 1)
                rms = np.sqrt(np.mean((ly - slope * lx - icpt) ** 2))
            C = np.nanmedian(floor[w] / (r["b1b"] * xi[w] ** 2)) if w.any() else np.nan
            xc = [crossing(xi, floor, L) for L in LEVELS]
            floors[(r["qm"], r["qf"], k)] = (floor, xc)
            Cs[(r["qm"], r["qf"], k)] = C
            print(f"   {r['qm']:<5g} {r['qf']:<5g} {r['b1b']:<7.4f} {u0:<5g} {spread:6.3f} "
                  f"{slope:6.2f} {rms:6.3f} {C:12.4g}       {xc[0]:8.4f}  {xc[1]:8.4f}")

    ref = max(rows, key=lambda r: r["b1b"])
    print(f"\n   Ratios xi(level) / xi_ref(level), reference q_M={ref['qm']:g}, "
          f"q_f={ref['qf']:g}; prediction [b(1-b)_ref / b(1-b)]^(1/2)")
    for r in rows:
        pred = np.sqrt(ref["b1b"] / r["b1b"])
        for k, u0 in enumerate(u0s):
            xc = floors[(r["qm"], r["qf"], k)][1]
            xr = floors[(ref["qm"], ref["qf"], k)][1]
            ratios = "  ".join(f"{a / b:6.2f}" for a, b in zip(xc, xr))
            print(f"   q_M={r['qm']:<4g} q_f={r['qf']:<5g} u0={u0:<5g} "
                  f"measured (1e-3, 1e-2): {ratios}   predicted: {pred:5.2f}")

    # ---------------- 3. validation on an independent pair ----------------
    models = None
    if args.validate:
        V = np.load(args.validate, allow_pickle=False)
        assert np.allclose(V["u0"], u0s) and np.allclose(V["P_over_tE"], pte)
        assert np.allclose(V["xi_rel"], xi)
        vqm, vqf = float(V["q_mass_used"]), float(V["q_flux_used"])
        assert all((vqm, vqf) != (r["qm"], r["qf"]) for r in rows), \
            "the validation pair is also a calibration row"
        vbeta = vqf / (1.0 + vqf)
        vb1b, vfac = vbeta * (1.0 - vbeta), photocentre_factor(vqm, vqf)
        Dv = V["D_p16_p50_p84"][:, 1]
        C_cal = [np.mean([Cs[(r["qm"], r["qf"], k)] for r in rows]) for k in range(len(u0s))]
        print(f"\n3. Validation: q_M={vqm:g}, q_f={vqf:g} (commit {V['code_commit']}, "
              f"dirty {bool(V['code_dirty'])}, failed fits {int(np.isnan(V['D_all']).sum())}); "
              f"|B-beta|={vfac:.3f}, b(1-b)={vb1b:.4f}")
        print("   C(u0) calibrated on the rows above: " +
              ", ".join(f"u0={u0:g}: {c:.3g}" for u0, c in zip(u0s, C_cal)))
        print(f"   model: D = sqrt(D_one(xi_ph)^2 + (C b(1-b) xi^2)^2) for xi_rel <= "
              f"{args.xi_pert:g}; median |log10 D - log10 D_model| (cells >= 1e-4)")
        print("   u0    " + "  ".join(f"P/tE {lo:g}-{hi:g}" for lo, hi in PTE_BANDS) +
              "   floor alone   xi(1e-3) measured / predicted")
        models = []
        pert = xi <= args.xi_pert
        for k, u0 in enumerate(u0s):
            D_one = np.nan_to_num(predicted_from_one(D1[k], xiE1, xi, vfac))
            D_sep = C_cal[k] * vb1b * xi[:, None] ** 2
            model = np.sqrt(D_one ** 2 + D_sep ** 2)
            model[~pert] = np.nan
            models.append(model)
            cells = [band_dev(Dv[k], model, pte, lo, hi) for lo, hi in PTE_BANDS]
            with np.errstate(invalid="ignore", divide="ignore"):
                floor = np.nanmedian(Dv[k][:, long], axis=1)
            w = (xi >= lo_xi) & pert & (floor >= args.d_min)
            pred_floor = C_cal[k] * vb1b * xi ** 2
            fdev = np.nanmedian(np.abs(np.log10(floor[w]) - np.log10(pred_floor[w])))
            x_meas = crossing(xi, floor, 1e-3)
            x_pred = np.sqrt(1e-3 / (C_cal[k] * vb1b))
            print(f"   {u0:<5g} " + "  ".join(f"{c:12.3f}" for c in cells) +
                  f"   {fdev:11.3f}   {x_meas:.4f} / {x_pred:.4f}")

    if not args.figure:
        return
    import matplotlib.pyplot as plt
    from paper_style import apply_paper_style, fig_width, panel_label, save_figure
    apply_paper_style(usetex=not args.no_usetex)
    fig, axes = plt.subplots(1, len(u0s), figsize=(fig_width(1.0), 2.6), sharey=True)
    for k, (ax, u0) in enumerate(zip(np.atleast_1d(axes), u0s)):
        for r in rows:
            floor = floors[(r["qm"], r["qf"], k)][0]
            ax.loglog(xi, floor / r["b1b"], lw=1.2,
                      label=rf"$q_M={r['qm']:g}$, $q_f={r['qf']:g}$")
        w = (xi >= lo_xi) & (xi <= hi_xi)
        ref_floor = floors[(ref["qm"], ref["qf"], k)][0] / ref["b1b"]
        x0 = xi[w][len(xi[w]) // 2]
        y0 = np.interp(np.log10(x0), np.log10(xi), np.log10(ref_floor))
        ax.loglog(xi, 10 ** y0 * (xi / x0) ** 2, color="0.5", ls="--", lw=0.8,
                  label=r"$\propto\xi_{\rm rel}^2$")
        ax.axvspan(lo_xi, hi_xi, color="0.9", zorder=0)
        ax.set_xlabel(r"$\xi_{\rm rel}$")
        ax.set_title(rf"$u_0={u0:g}$, $P/t_E\geq{args.pte_static:g}$", fontsize=9)
        panel_label(ax, "abcdef"[k])
    np.atleast_1d(axes)[0].set_ylabel(r"$D_{\rm floor}/[\beta(1-\beta)]$")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower right")
    out = save_figure(fig, os.path.join(args.out, "two_luminous_static_floor"))
    plt.close(fig)
    print("saved:", *out)

    if models is None:
        return
    import plot_xiE_ptE_maps as base
    from paper_style import D_LABEL, FS_ANNOT, LW_THIN
    fig = plt.figure(figsize=(fig_width(1.0), 3.2))
    left, right, gap, bottom, top = 0.085, 0.985, 0.025, 0.14, 0.74
    n = len(u0s)
    w = (right - left - (n - 1) * gap) / n
    cax = fig.add_axes([left, 0.88, right - left, 0.025])
    X, Y = np.meshgrid(xi, pte, indexing="ij")
    for k, u0 in enumerate(u0s):
        ax = fig.add_axes([left + k * (w + gap), bottom, w, top - bottom])
        pcm = base.draw_map(ax, xi, pte, Dv[k], label_contours=(k == 0), period_labels=None,
                            label_unreachable=False, show_families=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.contour(X, Y, np.log10(models[k]), levels=np.log10(base.D_LEVELS),
                       colors="white", linewidths=LW_THIN + 0.3, linestyles=":", zorder=11)
        ax.axvline(args.xi_pert, color="white", lw=0.6, ls="-", alpha=0.6, zorder=10)
        ax.set_xlabel(r"$\xi_{\rm rel}$")
        if k == 0:
            ax.set_ylabel(r"$P/t_E$")
        else:
            ax.set_yticklabels([])
        panel_label(ax, "abcdef"[k], color="white")
        ax.text(0.19, 0.95, rf"$u_0={u0:g}$", transform=ax.transAxes, ha="left", va="top",
                color="white", fontsize=FS_ANNOT, zorder=30)
    fig.text(0.5, 0.775, rf"$q_M={vqm:g}$, $q_f={vqf:g}$: measured (crimson) and "
             r"two-term model (white dotted, $\xi_{\rm rel}\leq" + f"{args.xi_pert:g}" + "$)",
             ha="center", fontsize=FS_ANNOT)
    base.top_colorbar(fig, cax, pcm, D_LABEL)
    out = save_figure(fig, os.path.join(args.out, "two_luminous_validation"))
    plt.close(fig)
    print("saved:", *out)


if __name__ == "__main__":
    main()
