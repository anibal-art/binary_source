#!/usr/bin/env python3
"""
Section 6 (detectability with Roman): power, confusion, false-positive rate
and p-values of Test A (PSPL vs PSPL + xallarap with a dark companion).

Inputs
    --gv-summary : <out>_summary.json of
                   lrt_gross_vitells.py --group-by-config --extra-alphas 0.0027
                   on the gv_sec6_h0_v1 run (one row per H0 configuration
                   (u0 node, F146, g): critical values, k_local, N(c0)).
    --asimov     : roman_sec6_asimov.npz of roman_sec6_asimov_cells.py
                   --summarize (lambda = Asimov Delta chi2 on
                   geometry x g x F146 x u0 x P).

Model
    Under H0 the critical value c_alpha of T = max_omega q(omega) comes from
    the Gross-Vitells fit of the H0 run (look-elsewhere effect over the
    period included).  Under H1, T is approximately noncentral
    chi2_k(lambda), with lambda the Asimov Delta chi2 (Cowan, Cranmer,
    Gross & Vitells 2011, EPJC 71, 1554) and k = k_local of the H0 node:
        power     = P(chi2_k(lambda) > c_alpha)
        confusion = 1 - power   (a binary source accepted as PSPL)
        median p  = P_H0(T > median of chi2_k(lambda)), Gross-Vitells tail.
    c_alpha, k_local and N(c0) are interpolated linearly in log u0 between
    the H0 u0 nodes, which are exact points of the H1 grid (no
    extrapolation).  The 16-84% bootstrap band of c_alpha gives a band on
    the power.  The validity of the noncentral approximation is checked
    separately with noisy H1 realizations.

    Reference rules, for comparison with the draft:
        Delta chi2 >= 100: false-positive rate P_H0(T > 100) per node and
                           fraction of cells with lambda >= 100;
        Wilks chi2_5     : false-positive rate of 11.07 (nominal 5%).

Outputs (<out>.*)
    .npz  power maps (alpha = 5% and 3 sigma, with band), median p-value,
          interpolated thresholds, and the inputs' axes;
    _nodes.csv   per H0 node: c_alpha, k_local, N(c0), FPR of the reference rules;
    _table.csv   per (g, F146): fractions of cells with power >= 0.5, 0.9
                 (isotropic orientations and the reference geometry), fraction
                 with lambda >= 100, and the longest period with median power
                 >= 0.5 for some u0;
    .pdf/.png    orientation-averaged power maps (unless --no-plot).

Usage (from the repository root, PYTHONPATH=$PWD):
    python binary_source/analysis/roman_sec6_power.py \
        --gv-summary results/roman_lrt/gv_sec6_h0_v1/gross_vitells_summary.json \
        --asimov results/roman_sec6_asimov_v1/roman_sec6_asimov.npz \
        --out results/roman_sec6_asimov_v1/sec6_power --no-usetex

Validation of the noncentral chi2 power with noisy H1 realizations
    1. --select-validation CELLS.json [--n-cells 12 --n-validation 100]
       picks isotropic-geometry cells with lambda spread over the transition
       (targets 6 ... 45), cycling over g and F146, and stores for each its
       truth, lambda, k_local, c_alpha and predicted power;
    2. build_h1_bspl_catalog.py --sec6-validation CELLS.json -> H1 parquet,
       runner with --truth-kind h1a and the H0 fit policy, lrt_omega_profile.py
       (identical to the H0 calibration);
    3. --validate CELLS.json --profiles .../omega_profile/*_omega_profile.json
       compares, per cell, the measured fraction T > c_alpha (Wilson 68%) with
       the prediction, and the mean of T with k + lambda.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent.parent
for p in (REPO_ROOT, ANALYSIS_DIR.parent / "plot_codes"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from binary_source.analysis.lrt_gross_vitells import gv_sf  # noqa: E402

H0_U0_NODES = np.geomspace(0.005, 2.0, 7)       # build_h0_catalog.U0_NODES
ALPHA_3SIGMA = 0.0027
CONFIG_RE = re.compile(r":m(\d+):u(\d+):pi0(?::g([0-9.]+))?$")


def parse_config(cid):
    m = CONFIG_RE.search(str(cid))
    if m is None:
        raise SystemExit(f"cannot parse config_id {cid!r}")
    return float(m.group(1)), int(m.group(2)), float(m.group(3) or 0.0)


def node_tables(summary, blend, mags):
    """Arrays (g, F146, u0 node) of the H0 quantities; every node must be present."""
    a3 = f"{ALPHA_3SIGMA:g}"
    keys = {"c05": "c_crit_gv_klocal", "c05_lo": "c_crit_gv_klocal_16",
            "c05_hi": "c_crit_gv_klocal_84", "c3s": f"c_crit_gv_klocal_alpha{a3}",
            "c3s_lo": f"c_crit_gv_klocal_alpha{a3}_16", "c3s_hi": f"c_crit_gv_klocal_alpha{a3}_84",
            "k": "k_local", "N0": "N_c0", "c05_emp": "c_crit_empirical", "n": "n_realizations"}
    rows = summary["rows"]
    iu_used = sorted({parse_config(r["config_id"])[1] for r in rows})
    out = {k: np.full((len(blend), len(mags), len(iu_used)), np.nan) for k in keys}
    for r in rows:
        mag, iu, g = parse_config(r["config_id"])
        ib, im = np.flatnonzero(np.isclose(blend, g)), np.flatnonzero(np.isclose(mags, mag))
        if not (len(ib) and len(im)):
            continue
        for k, col in keys.items():
            if col not in r:
                raise SystemExit(f"{col} missing in the summary (run with --extra-alphas {a3})")
            out[k][ib[0], im[0], iu_used.index(iu)] = r[col]
    if not np.isfinite(out["c05"]).all():
        raise SystemExit("H0 summary misses some (g, F146, u0) nodes of the H1 grid")
    return out, H0_U0_NODES[iu_used]


def interp_log_u0(values, nodes, u0):
    """Linear in log u0 along the last axis; u0 must lie inside the node range."""
    if u0.min() < nodes.min() * (1 - 1e-9) or u0.max() > nodes.max() * (1 + 1e-9):
        raise SystemExit("H1 u0 grid extends beyond the H0 nodes")
    x, xp = np.log(u0), np.log(nodes)
    flat = values.reshape(-1, values.shape[-1])
    return np.array([np.interp(x, xp, v) for v in flat]).reshape(values.shape[:-1] + (len(u0),))


def power(c, k, lam):
    lam = np.where(np.isfinite(lam), np.maximum(lam, 0.0), np.nan)
    return stats.ncx2.sf(c, k, lam)


def longest_period(pw_med, P):
    """Largest P with pw_med >= 0.5 for some u0 (pw_med: u0 x P); NaN if none."""
    ok = (pw_med >= 0.5).any(axis=0)
    return float(P[ok].max()) if ok.any() else np.nan


VALIDATION_LAMBDA_TARGETS = (6.0, 10.0, 15.0, 22.0, 32.0, 45.0)
VALIDATION_SEED = 20261117


def select_validation(args, lam, I, A, blend, mags, u0, P):
    """Cells of the isotropic geometries with lambda near the targets."""
    rng = np.random.default_rng(VALIDATION_SEED)
    geo = np.asarray(A["geometry"], float)
    cells = []
    for j in range(args.n_cells):
        ib, im = j % len(blend), (j // len(blend)) % len(mags)
        target = VALIDATION_LAMBDA_TARGETS[j % len(VALIDATION_LAMBDA_TARGETS)]
        sub = lam[1:, ib, im]                                   # isotropic rows only
        dist = np.abs(np.log(np.maximum(sub, 1e-12) / target))
        cand = np.argwhere(dist <= np.log(1.1))                 # within 10% of the target
        if len(cand) == 0:
            cand = np.argwhere(dist == np.nanmin(dist))
        ig1, iu, ip = cand[rng.integers(len(cand))]
        ig = int(ig1) + 1
        th, ph, inc = geo[ig]
        k, c05, c3s = (float(I[key][ib, im, iu]) for key in ("k", "c05", "c3s"))
        lv = float(lam[ig, ib, im, iu, ip])
        cells.append({
            "config_id": f"sec6_validation:c{j:02d}", "geometry_index": ig,
            "theta": float(th), "phi": float(ph), "inclination": float(inc),
            "u0": float(u0[iu]), "u0_index": int(iu), "P_days": float(P[ip]), "P_index": int(ip),
            "F146": float(mags[im]), "blend_ratio": float(blend[ib]),
            "lambda": lv, "lambda_target": target, "k_local": k,
            "c_alpha05": c05, "c_3sigma": c3s,
            "power05_pred": float(stats.ncx2.sf(c05, k, lv)),
            "power3s_pred": float(stats.ncx2.sf(c3s, k, lv))})
    spec = {"n_realizations": args.n_validation, "tE": float(A["tE"]), "t0": float(A["t0"]),
            "q_mass": float(A["q_mass"]),
            "Mtot_Msun": 3.0,                     # M1 + M2 of roman_bspl_roman_rubin (2 + 1)
            "rEhat_AU": float(A["rEhat_AU"]),
            "asimov": str(args.asimov), "gv_summary": str(args.gv_summary),
            "seed": VALIDATION_SEED, "cells": cells}
    args.select_validation.parent.mkdir(parents=True, exist_ok=True)
    args.select_validation.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"{'cell':>18} {'g':>2} {'F146':>4} {'u0':>8} {'P[d]':>8} {'geo':>3} "
          f"{'lambda':>7} {'k':>5} {'c05':>6} {'pow05':>6} {'pow3s':>6}")
    for c in cells:
        print(f"{c['config_id']:>18} {c['blend_ratio']:2g} {c['F146']:4g} {c['u0']:8.4g} "
              f"{c['P_days']:8.1f} {c['geometry_index']:3d} {c['lambda']:7.2f} {c['k_local']:5.2f} "
              f"{c['c_alpha05']:6.2f} {c['power05_pred']:6.3f} {c['power3s_pred']:6.3f}")
    print(f"saved {args.select_validation}  ({len(cells)} cells x {args.n_validation} realizations)")


def validate(args):
    from binary_source.analysis.lrt_gross_vitells import wilson
    spec = json.loads(args.validate.read_text())
    cells = {c["config_id"]: c for c in spec["cells"]}
    T = {cid: [] for cid in cells}
    for f in args.profiles:
        d = json.loads(Path(f).read_text())
        if d.get("config_id") in T:
            T[d["config_id"]].append(float(d["T_profile"]))
    lines = ["config_id,n,lambda,k_local,c_alpha05,power05_pred,power05_meas,power05_lo,power05_hi,"
             "power3s_pred,power3s_meas,mean_T,k_plus_lambda,median_T,median_T_pred"]
    print(f"{'cell':>18} {'n':>4} {'lambda':>7} | {'pow05 pred':>10} {'meas [68%]':>20} | "
          f"{'pow3s pred':>10} {'meas':>6} | {'<T>':>7} {'k+lam':>7}")
    for cid, c in cells.items():
        t = np.asarray(T[cid])
        n = len(t)
        if n == 0:
            print(f"{cid:>18}    0  (no profiles)")
            continue
        n05, n3s = int((t > c["c_alpha05"]).sum()), int((t > c["c_3sigma"]).sum())
        lo, hi = wilson(n05, n)
        med_pred = float(stats.ncx2.ppf(0.5, c["k_local"], c["lambda"]))
        print(f"{cid:>18} {n:4d} {c['lambda']:7.2f} | {c['power05_pred']:10.3f} "
              f"{n05 / n:6.3f} [{lo:.3f}, {hi:.3f}] | {c['power3s_pred']:10.3f} {n3s / n:6.3f} | "
              f"{t.mean():7.2f} {c['k_local'] + c['lambda']:7.2f}")
        lines.append(f"{cid},{n},{c['lambda']:.4f},{c['k_local']:.4f},{c['c_alpha05']:.4f},"
                     f"{c['power05_pred']:.4f},{n05 / n:.4f},{lo:.4f},{hi:.4f},"
                     f"{c['power3s_pred']:.4f},{n3s / n:.4f},{t.mean():.4f},"
                     f"{c['k_local'] + c['lambda']:.4f},{np.median(t):.4f},{med_pred:.4f}")
    out = args.validate.with_name(args.validate.stem + "_result.csv")
    out.write_text("\n".join(lines) + "\n")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gv-summary", type=Path)
    ap.add_argument("--asimov", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-usetex", action="store_true")
    ap.add_argument("--select-validation", type=Path, default=None,
                    help="write the validation cells JSON and stop")
    ap.add_argument("--n-cells", type=int, default=12)
    ap.add_argument("--n-validation", type=int, default=100)
    ap.add_argument("--validate", type=Path, default=None,
                    help="cells JSON: compare predicted and measured power (needs --profiles)")
    ap.add_argument("--profiles", nargs="*", default=[])
    args = ap.parse_args()

    if args.validate is not None:
        validate(args)
        return
    if args.gv_summary is None or args.asimov is None or (
            args.out is None and args.select_validation is None):
        ap.error("--gv-summary, --asimov and --out (or --select-validation) are required")

    summary = json.loads(args.gv_summary.read_text())
    c0 = float(summary["c0"])
    if abs(float(summary["alpha"]) - 0.05) > 1e-12 or ALPHA_3SIGMA not in summary["extra_alphas"]:
        raise SystemExit("expected --alpha 0.05 --extra-alphas 0.0027 in the H0 summary")
    A = np.load(args.asimov, allow_pickle=False)
    blend, mags, u0, P = (np.asarray(A[k], float) for k in ("blend_ratio", "F146", "u0_grid", "P_days"))
    lam = np.asarray(A["DELTA_CHI2_PSPL"], float)            # (geo, g, F146, u0, P)
    n_geo = lam.shape[0]

    nodes, u0_nodes = node_tables(summary, blend, mags)
    I = {k: interp_log_u0(v, u0_nodes, u0) for k, v in nodes.items()}
    if args.select_validation is not None:
        select_validation(args, lam, I, A, blend, mags, u0, P)
        return

    def b(x):                                                 # (g, F146, u0) -> broadcast
        return x[None, :, :, :, None]

    pw05 = power(b(I["c05"]), b(I["k"]), lam)
    pw05_lo = power(b(I["c05_hi"]), b(I["k"]), lam)            # higher threshold, lower power
    pw05_hi = power(b(I["c05_lo"]), b(I["k"]), lam)
    pw3s = power(b(I["c3s"]), b(I["k"]), lam)
    pw3s_lo = power(b(I["c3s_hi"]), b(I["k"]), lam)
    pw3s_hi = power(b(I["c3s_lo"]), b(I["k"]), lam)
    kk = np.broadcast_to(b(I["k"]), lam.shape)
    lam_ok = np.where(np.isfinite(lam), np.maximum(lam, 0.0), np.nan)
    t_med = stats.ncx2.ppf(0.5, kk, lam_ok)
    p_med = gv_sf(t_med, np.broadcast_to(b(I["N0"]), lam.shape), kk, c0)

    np.savez_compressed(
        str(args.out) + ".npz",
        POWER_ALPHA05=pw05, POWER_ALPHA05_LO=pw05_lo, POWER_ALPHA05_HI=pw05_hi,
        POWER_3SIGMA=pw3s, POWER_3SIGMA_LO=pw3s_lo, POWER_3SIGMA_HI=pw3s_hi,
        PVALUE_MEDIAN=p_med, LAMBDA=lam,
        C_ALPHA05=I["c05"], C_3SIGMA=I["c3s"], K_LOCAL=I["k"], N_C0=I["N0"],
        blend_ratio=blend, F146=mags, u0_grid=u0, P_days=P, geometry=A["geometry"],
        u0_nodes=u0_nodes, c0=c0, gv_summary=str(args.gv_summary), asimov=str(args.asimov))

    # ---- per H0 node: thresholds and false-positive rate of the reference rules
    c_wilks = float(stats.chi2.isf(0.05, 5))
    lines = ["g,F146,u0,n_realizations,k_local,N_c0,c_alpha05,c_alpha05_16,c_alpha05_84,"
             "c_alpha05_empirical,c_3sigma,FPR_wilks_11.07,FPR_dchi2_100"]
    for ib, g in enumerate(blend):
        for im, mag in enumerate(mags):
            for iu, un in enumerate(u0_nodes):
                v = {k: nodes[k][ib, im, iu] for k in nodes}
                fw = float(gv_sf(c_wilks, v["N0"], v["k"], c0))
                f100 = float(gv_sf(100.0, v["N0"], v["k"], c0))
                lines.append(f"{g:g},{mag:g},{un:.6g},{int(v['n'])},{v['k']:.3f},{v['N0']:.3f},"
                             f"{v['c05']:.3f},{v['c05_lo']:.3f},{v['c05_hi']:.3f},"
                             f"{v['c05_emp']:.3f},{v['c3s']:.3f},{fw:.4f},{f100:.3e}")
    Path(str(args.out) + "_nodes.csv").write_text("\n".join(lines) + "\n")

    # ---- per (g, F146): fractions of the (log u0, log P) grid
    iso = slice(1, n_geo)                                     # row 0 = reference geometry
    lines = ["g,F146,frac_pow05_ge0.5,frac_pow05_ge0.9,frac_pow3s_ge0.5,frac_pow3s_ge0.9,"
             "ref_frac_pow05_ge0.5,ref_frac_pow05_ge0.9,frac_lambda_ge100,ref_frac_lambda_ge100,"
             "P50_alpha05_yr,P50_3sigma_yr,n_cells_finite"]
    print(f"\n{'g':>3} {'F146':>4} | iso: pow>=0.5  pow>=0.9 (5%) | pow>=0.5 pow>=0.9 (3s) | "
          f"ref: pow>=0.5 pow>=0.9 | lam>=100 iso ref | P50 5% 3s [yr]")
    for ib, g in enumerate(blend):
        for im, mag in enumerate(mags):
            def frac(x, thr, sl):
                v = x[sl, ib, im]
                ok = np.isfinite(v)
                return float((v[ok] >= thr).mean()) if ok.any() else np.nan
            row = [frac(pw05, 0.5, iso), frac(pw05, 0.9, iso), frac(pw3s, 0.5, iso),
                   frac(pw3s, 0.9, iso), frac(pw05, 0.5, slice(0, 1)), frac(pw05, 0.9, slice(0, 1)),
                   frac(lam, 100.0, iso), frac(lam, 100.0, slice(0, 1))]
            p50 = [longest_period(np.nanmedian(x[iso, ib, im], axis=0), P) / 365.25
                   for x in (pw05, pw3s)]
            n_fin = int(np.isfinite(lam[:, ib, im]).sum())
            lines.append(f"{g:g},{mag:g}," + ",".join(f"{x:.4f}" for x in row + p50) + f",{n_fin}")
            print(f"{g:3g} {mag:4g} | {row[0]:9.3f} {row[1]:9.3f} | {row[2]:8.3f} {row[3]:8.3f} | "
                  f"{row[4]:9.3f} {row[5]:9.3f} | {row[6]:7.3f} {row[7]:5.3f} | "
                  f"{p50[0]:6.2f} {p50[1]:5.2f}")
    Path(str(args.out) + "_table.csv").write_text("\n".join(lines) + "\n")
    print("\nsaved", str(args.out) + ".npz", str(args.out) + "_nodes.csv", str(args.out) + "_table.csv")

    if not args.no_plot:
        plot(pw05, pw3s, blend, mags, u0, P, args)


def plot(pw05, pw3s, blend, mags, u0, P, args):
    import matplotlib.pyplot as plt
    from paper_style import apply_paper_style, fig_width, heatmap, save_figure
    apply_paper_style(usetex=not args.no_usetex)
    iso = slice(1, pw05.shape[0])
    fig, axes = plt.subplots(len(blend), len(mags), figsize=(fig_width(1.0), fig_width(1.0)),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for ib, g in enumerate(blend):
        for im, mag in enumerate(mags):
            ax = axes[ib, im]
            m05 = np.nanmean(pw05[iso, ib, im], axis=0)
            m3s = np.nanmean(pw3s[iso, ib, im], axis=0)
            pcm = heatmap(ax, u0, P / 365.25, m05.T, vmin=0, vmax=1, cmap="viridis")
            ax.contour(u0, P / 365.25, m05.T, levels=[0.5, 0.9], colors="w", linewidths=0.8)
            ax.contour(u0, P / 365.25, m3s.T, levels=[0.5, 0.9], colors="w",
                       linewidths=0.8, linestyles="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if ib == 0:
                ax.set_title(f"F146 = {mag:g}")
            if im == 0:
                ax.set_ylabel(f"g = {g:g}\nP [yr]")
            if ib == len(blend) - 1:
                ax.set_xlabel("u0")
    pct = r"5\%" if not args.no_usetex else "5%"          # % starts a LaTeX comment
    fig.colorbar(pcm, ax=axes, label=f"power (alpha = {pct}), mean over orientations")
    out = save_figure(fig, args.out)
    print("saved", *out)


if __name__ == "__main__":
    main()
