#!/usr/bin/env python3
"""
Diagnostic figure of one LRT runner event (not a manuscript figure).

Inputs: the runner JSON and the arrays written with --save-arrays (fitted light
curve and noiseless truth). The best PSPL and BSPL curves are rebuilt from the
stored parameter vectors with the runner's build_fit_model / model_magnitude,
i.e. with the same roman_rubin + pyLIMA model that was fitted. Needs
ROMAN_RUBIN_REPO and PYTHONPATH=<binary_source repo>.

Panels
  (a) whole Roman light curve with the noiseless truth
  (b) zoom t0 +/- N tE: binned data, truth, best PSPL, best BSPL
  (c) residuals relative to the best PSPL
  (d) linearized xallarap periodogram Delta chi2(P) of the PSPL residuals,
      with the period of the BSPL start and the final T_A
A table with the start, the final value and the covariance error of every
parameter is printed.

Usage:
    python binary_source/plot_codes/plot_lrt_event_fits.py \
        --json h0a_h0_118900.json --arrays h0a_h0_118900.npz --out event.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from binary_source.analysis import roman_bspl_lrt_catalog_runner as runner


def binned(t, y, w, width):
    """Inverse-variance weighted means in bins of `width` days."""
    edges = np.arange(t.min(), t.max() + width, width)
    idx = np.digitize(t, edges) - 1
    sw = np.bincount(idx, weights=w, minlength=len(edges))
    swy = np.bincount(idx, weights=w * y, minlength=len(edges))
    ok = sw > 0
    return (edges + 0.5 * width)[ok], swy[ok] / sw[ok], 1.0 / np.sqrt(sw[ok])


def print_table(name, best):
    names = best["fit_parameter_names"]
    x = best["best_model"]
    cov = best.get("covariance_matrix")
    err = np.sqrt(np.diag(cov)) if cov is not None else [np.nan] * len(x)
    guess = best["initial_guess"]
    print(f"\n{name}  chi2 = {best['chi2']:.3f}  start {best['start_label']}")
    print(f"  {'parameter':22s} {'start':>14s} {'final':>14s} {'sigma':>12s}")
    for n, v, e in zip(names, x, err):
        g = guess.get(n, guess.get(n.replace("_Roman", "_W149"), np.nan))
        print(f"  {n:22s} {g:14.6g} {v:14.6g} {e:12.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--zoom-tE", type=float, default=3.0)
    ap.add_argument("--bin-days", type=float, default=0.1)
    ap.add_argument("--resid-bin-days", type=float, default=0.5)
    args = ap.parse_args()

    d = json.loads(args.json.read_text())
    a = np.load(args.arrays)
    lc = np.column_stack([a["time"], a["mag"], a["err_mag"]])
    t, m, e = lc.T
    w = 1.0 / e ** 2
    s = d["statistics"]
    ra, dec, ref = d["event_ra_deg"], d["event_dec_deg"], d["fit_reference"]

    curves = {}
    for fam, double in (("PSPL", False), ("BSPL", True)):
        best = d["fits"][fam]["best"]
        model = runner.build_fit_model(lc, ra, dec, ref, double)
        curves[fam] = runner.model_magnitude(model, best["best_model"])
        print_table(fam, best)
    print(f"\nT_A = {s['T_A']:.3f}   (chi2_PSPL {s['chi2_PSPL']:.3f}, "
          f"chi2_BSPL {s['chi2_BSPL']:.3f}, N = {d['n_roman_points']})")
    pg = d.get("bspl_start_periodogram") or {}
    for c in pg.get("candidates", []):
        print(f"  start candidate alpha = {c['alpha']:<5g} chi2 improvement over PSPL "
              f"(full model, linear fluxes) = {c['improvement_over_pspl']:9.3f}")

    best_pspl = d["fits"]["PSPL"]["best"]
    names = best_pspl["fit_parameter_names"]
    t0 = best_pspl["best_model"][names.index("t0")]
    tE = abs(best_pspl["best_model"][names.index("tE")])
    off = 2_460_000.0
    z = np.abs(t - t0) < args.zoom_tE * tE
    o = np.argsort(t)

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.4, 0.9], hspace=0.35, wspace=0.25)

    # (a) whole light curve
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t - off, m, ",", color="0.6", rasterized=True)
    if "truth_mag" in a:
        ot = np.argsort(a["truth_time"])
        ax.plot(a["truth_time"][ot] - off, a["truth_mag"][ot], ".", ms=0.5, color="k")
    ax.invert_yaxis()
    ax.set_xlabel("JD - 2460000")
    ax.set_ylabel("magnitude (roman_rubin)")
    ax.set_title(f"{d['catalog_row']['config_id']}  seed {d['simulation_seed']}  "
                 f"N = {d['n_roman_points']}", fontsize=10)

    # (b) zoom with the models
    ax = fig.add_subplot(gs[1, 0])
    tb, mb, eb = binned(t[z], m[z], w[z], args.bin_days)
    ax.plot(t[z] - off, m[z], ",", color="0.75", rasterized=True)
    ax.errorbar(tb - off, mb, eb, fmt=".", ms=2, color="k", lw=0.5,
                label=f"data, {args.bin_days:g} d bins")
    if "truth_mag" in a:
        ot = np.argsort(a["truth_time"])
        zt = np.abs(a["truth_time"][ot] - t0) < args.zoom_tE * tE
        ax.plot(a["truth_time"][ot][zt] - off, a["truth_mag"][ot][zt], "-", color="C2",
                lw=1, label="truth")
    oz = o[z[o]]
    ax.plot(t[oz] - off, curves["PSPL"][oz], "--", color="C0", lw=1, label="best PSPL")
    ax.plot(t[oz] - off, curves["BSPL"][oz], ":", color="C3", lw=1.2, label="best BSPL")
    ax.invert_yaxis()
    ax.set_xlabel("JD - 2460000")
    ax.set_ylabel("magnitude")
    ax.legend(fontsize=8)

    # (c) residuals relative to the best PSPL
    ax = fig.add_subplot(gs[2, 0])
    r = m - curves["PSPL"]
    tr, rb, rbe = binned(t[z], r[z], w[z], args.resid_bin_days)
    ax.errorbar(tr - off, 1e3 * rb, 1e3 * rbe, fmt=".", ms=2, color="k", lw=0.5,
                label=f"data - PSPL, {args.resid_bin_days:g} d bins")
    ax.plot(t[oz] - off, 1e3 * (curves["BSPL"] - curves["PSPL"])[oz], ":", color="C3",
            lw=1.2, label="BSPL - PSPL")
    if "truth_mag" in a and np.array_equal(a["truth_time"], t):
        ax.plot(t[oz] - off, 1e3 * (a["truth_mag"] - curves["PSPL"])[oz], "-",
                color="C2", lw=1, label="truth - PSPL")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("JD - 2460000")
    ax.set_ylabel("residual [mmag]")
    ax.legend(fontsize=7, ncol=3, loc="upper center")

    # (d) linearized periodogram of the PSPL residuals
    ax = fig.add_subplot(gs[1:, 1])
    pg = d.get("bspl_start_periodogram")
    if pg is not None:
        ax.plot(pg["period_days"], pg["delta_chi2"], "-", color="k", lw=0.8)
        ax.axvline(pg["peak_period_days"], color="C3", ls=":", lw=1,
                   label=f"BSPL start: P = {pg['peak_period_days']:.4g} d, "
                         f"linear Delta chi2 = {pg['peak_delta_chi2']:.2f}")
    ax.axhline(s["T_A"], color="C0", ls="--", lw=1,
               label=f"T_A after the BSPL fit = {s['T_A']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("period [d]")
    ax.set_ylabel("Delta chi2 (linearized, PSPL residuals)")
    ax.set_title("linearized xallarap periodogram", fontsize=10)
    ax.legend(fontsize=8)

    out = args.out or args.json.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
