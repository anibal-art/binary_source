#!/usr/bin/env python3
"""
Gross & Vitells (2010) check of the Test A null distribution.

Statistic: T = max_omega q(omega), q(omega) = chi2_PSPL - min chi2_BSPL(omega),
the BSPL profile likelihood in omega (lrt_omega_profile.py), computed on H0
(PSPL truth) realizations of ONE catalog configuration.

Davies (1987) bound, used by Gross & Vitells for the look-elsewhere effect: if
q(omega) is a chi2_k process in omega,
    P(T > c) ~= P(chi2_k > c) + E[N(c)],
    E[N(c)] = E[N(c0)] (c / c0)^((k-1)/2) exp(-(c - c0)/2),
with N(c) the number of upcrossings of the level c by q(omega). E[N(c0)] at a
low level c0 is estimated from a few tens of realizations and the formula
gives the tail at high c (e.g. the 95% quantile), to be compared with Wilks
(chi2 with 5 degrees of freedom: 11.07 at 95%, the threshold of Rahvar &
Dominik 2009 used by Miyazaki et al. 2021).

k is estimated two ways:
  k_local : mean of q over all nodes and realizations (chi2_k has mean k);
  k_N     : fit of log E[N(c)] at several levels to the formula above.
The empirical survival of T is shown next to both predictions; where the
sample still has events above c the two must agree.

Usage:
    python binary_source/analysis/lrt_gross_vitells.py \
        --profiles results/roman_lrt/gv_pilot_v1/omega_profile/*_omega_profile.json \
        --out results/roman_lrt/gv_pilot_v1/gross_vitells

--group-by-config analyses the profiles of every configuration_id separately
(one run directory with several configurations), writes <out>_<config>.json /
.png for each and <out>_summary.json / .csv with one row per configuration.
--extra-alphas adds critical values at other levels (e.g. 0.0027 = 3 sigma),
extrapolated with the same Gross-Vitells fit.

--validate (with --group-by-config) checks the Gross-Vitells critical values
against the realizations themselves: (1) pooled fraction of realizations with
T above the critical value of their configuration, in-sample; (2) the same
with split halves (critical value fitted on one half of each configuration,
counted on the other, and vice versa); (3) chi2 homogeneity of the 5%
critical value across configurations, with the bootstrap 16-84 half-width as
uncertainty.  Writes <out>_validation.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats


def load(paths):
    curves, periods, T, names = [], [], [], []
    for p in paths:
        d = json.loads(Path(p).read_text())
        om = np.array([x["omega"] for x in d["profile"]])
        q = np.array([np.nan if x["delta_chi2"] is None else x["delta_chi2"]
                      for x in d["profile"]])
        order = np.argsort(om)
        om, q = om[order], q[order]
        bad = ~np.isfinite(q)
        if bad.sum() > 0.1 * len(q):
            print(f"[skip] {p}: {bad.sum()} of {len(q)} nodes failed")
            continue
        if bad.any():                                   # few failed nodes: interpolate
            q[bad] = np.interp(om[bad], om[~bad], q[~bad])
        curves.append(np.clip(q, 0.0, None))            # nested models: q >= 0
        periods.append(2 * np.pi / om)
        T.append(max(0.0, float(d["T_profile"])))
        names.append(d.get("config_id"))
    if len(set(names)) > 1:
        raise SystemExit(f"profiles of more than one configuration: {sorted(set(names))}")
    return curves, periods, np.array(T), names[0] if names else None


def upcrossings(q, c):
    return int(np.sum((q[:-1] < c) & (q[1:] >= c)))


def gv_sf(c, N0, k, c0):
    """P(T > c): Davies / Gross-Vitells, local chi2_k term plus upcrossings."""
    c = np.asarray(c, dtype=float)
    k = np.maximum(np.asarray(k, dtype=float), 0.1)      # guard a degenerate k_N fit
    N0 = np.asarray(N0, dtype=float)                     # scalars or broadcastable arrays
    return stats.chi2.sf(c, k) + N0 * (c / c0) ** ((k - 1) / 2) * np.exp(-(c - c0) / 2)


def gv_quantile(alpha, N0, k, c0):
    f = lambda c: gv_sf(c, N0, k, c0) - alpha
    lo = 1e-3 if f(1e-3) > 0 else None
    return optimize.brentq(f, lo, 500.0) if lo is not None else 0.0


def fit_levels(curves, levels, c0):
    """E[N(c)] per level, and (N0, k) from log E[N] linear in (1, log(c/c0))."""
    N = np.array([[upcrossings(q, c) for c in levels] for q in curves], dtype=float)
    mean = N.mean(axis=0)
    sem = N.std(axis=0, ddof=1) / np.sqrt(len(curves))
    ok = mean > 0
    y = np.log(mean[ok]) + (levels[ok] - c0) / 2
    x = np.log(levels[ok] / c0)
    w = np.sqrt(N.sum(axis=0)[ok])                       # 1 / sigma(log N), Poisson
    A = np.c_[np.ones(ok.sum()), x] * w[:, None]
    (lnN0, slope), *_ = np.linalg.lstsq(A, y * w, rcond=None)
    return mean, sem, float(np.exp(lnN0)), float(2 * slope + 1)


def analyse(curves, T, levels, c0, alpha, extra_alphas=()):
    pooled = np.concatenate(curves)
    k_local = float(pooled.mean())
    mean, sem, N0_fit, k_N = fit_levels(curves, levels, c0)
    N0_c0 = float(np.mean([upcrossings(q, c0) for q in curves]))
    return {
        "k_local": k_local, "k_local_from_variance": float(pooled.var() / 2),
        "N_c0": N0_c0, "N0_fit": N0_fit, "k_N": k_N,
        "N_mean": mean.tolist(), "N_sem": sem.tolist(),
        "c_crit_gv_klocal": gv_quantile(alpha, N0_c0, k_local, c0),
        "c_crit_gv_kN": gv_quantile(alpha, N0_fit, k_N, c0),
        "c_crit_empirical": float(np.quantile(T, 1 - alpha)),
        "c_crit_extra": {f"{a:g}": {"gv_klocal": gv_quantile(a, N0_c0, k_local, c0),
                                    "gv_kN": gv_quantile(a, N0_fit, k_N, c0)}
                         for a in extra_alphas},
    }


def wilson(k, n, z=1.0):
    p = k / n
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def split_half_counts(curves, T, levels, c0, alphas, rng):
    """Exceedances of T over the GV k_local critical value fitted on the other half."""
    n = len(T)
    idx = rng.permutation(n)
    halves = (idx[: n // 2], idx[n // 2:])
    counts = {a: [0, 0] for a in alphas}
    for fit_i, test_i in ((0, 1), (1, 0)):
        f, t = halves[fit_i], halves[test_i]
        res = analyse([curves[i] for i in f], T[f], levels, c0, alphas[0], alphas[1:])
        crit = {alphas[0]: res["c_crit_gv_klocal"]}
        crit.update({a: res["c_crit_extra"][f"{a:g}"]["gv_klocal"] for a in alphas[1:]})
        for a in alphas:
            counts[a][0] += int(np.sum(T[t] > crit[a]))
            counts[a][1] += len(t)
    return counts


def validation(rows, args):
    """Pooled empirical false-alarm rate of the GV critical values, and homogeneity."""
    alphas = [args.alpha] + list(args.extra_alphas)
    levels = np.array(args.levels, dtype=float)
    rng = np.random.default_rng(1)
    insample = {a: [0, 0] for a in alphas}
    split = {a: [0, 0] for a in alphas}
    for o in rows:
        T = np.array(o["T"])
        crit = {args.alpha: o["c_crit_gv_klocal"]}
        crit.update({a: o["c_crit_extra"][f"{a:g}"]["gv_klocal"] for a in args.extra_alphas})
        for a in alphas:
            insample[a][0] += int(np.sum(T > crit[a]))
            insample[a][1] += len(T)
        curves, _, T2, _ = load(o["profiles"])
        for a, (k, m) in split_half_counts(curves, T2, levels, args.c0, alphas, rng).items():
            split[a][0] += k
            split[a][1] += m
    report = {"c0": args.c0, "n_configurations": len(rows), "levels": {}}
    print("\nvalidation of the Gross-Vitells critical values (k_local):")
    for a in alphas:
        r = {}
        for name, (k, m) in (("in_sample", insample[a]), ("split_half", split[a])):
            lo, hi = wilson(k, m)
            r[name] = {"n_above": k, "n": m, "rate": k / m, "rate_1sigma": [lo, hi]}
            print(f"  alpha = {a:g} ({name}): {k}/{m} = {k / m:.4f} "
                  f"[{lo:.4f}, {hi:.4f}]  (expected {a:g})")
        report["levels"][f"{a:g}"] = r
    c = np.array([o["c_crit_gv_klocal"] for o in rows])
    b = np.array([o["bootstrap_16_84"]["c_crit_gv_klocal"] for o in rows])
    sig = (b[:, 1] - b[:, 0]) / 2
    ok = sig > 0
    w = 1.0 / sig[ok] ** 2
    cbar = float(np.sum(w * c[ok]) / np.sum(w))
    chi2 = float(np.sum(((c[ok] - cbar) / sig[ok]) ** 2))
    dof = int(ok.sum() - 1)
    pval = float(stats.chi2.sf(chi2, dof))
    report["homogeneity"] = {"weighted_mean": cbar, "chi2": chi2, "dof": dof, "p": pval,
                             "min": float(c.min()), "max": float(c.max())}
    print(f"  homogeneity of the {100 * (1 - args.alpha):g}% critical value: weighted mean "
          f"{cbar:.2f}, chi2 = {chi2:.1f} for {dof} dof, p = {pval:.3g} "
          f"(range {c.min():.1f}-{c.max():.1f})")
    out = Path(str(args.out) + "_validation.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("saved", out)


def config_of(path):
    return json.loads(Path(path).read_text()).get("config_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output prefix (.json, .png)")
    ap.add_argument("--c0", type=float, default=1.0, help="reference level for E[N(c0)]")
    ap.add_argument("--levels", type=float, nargs="+", default=[0.5, 1, 1.5, 2, 3, 4, 5],
                    help="levels where upcrossings are counted (fit of k_N)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--k-wilks", type=int, default=5)
    ap.add_argument("--band-edges", type=float, nargs="+", default=[30.0, 150.0],
                    help="period band edges [d] for the mean of q per band")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--extra-alphas", type=float, nargs="*", default=[],
                    help="further levels for the critical value, e.g. 0.0027 (3 sigma)")
    ap.add_argument("--group-by-config", action="store_true",
                    help="analyse each config_id of --profiles separately and write a summary")
    ap.add_argument("--validate", action="store_true",
                    help="with --group-by-config: empirical false-alarm rate of the critical "
                         "values (in-sample and split halves) and homogeneity test")
    args = ap.parse_args()
    if args.validate and not args.group_by_config:
        raise SystemExit("--validate needs --group-by-config")

    if not args.group_by_config:
        run(args.profiles, args.out, args)
        return
    groups = {}
    for p in args.profiles:
        groups.setdefault(config_of(p), []).append(p)
    rows = []
    for cid in sorted(groups, key=str):
        tag = str(cid).replace(":", "_").replace("/", "_")
        print(f"\n==================== {cid}")
        rows.append(run(groups[cid], Path(f"{args.out}_{tag}"), args))
    keys = ["config_id", "n_realizations", "k_local", "N_c0", "c_crit_gv_klocal",
            "c_crit_gv_klocal_16", "c_crit_gv_klocal_84", "c_crit_empirical", "c_crit_wilks"]
    for a in args.extra_alphas:
        keys += [f"c_crit_gv_klocal_alpha{a:g}", f"c_crit_gv_klocal_alpha{a:g}_16",
                 f"c_crit_gv_klocal_alpha{a:g}_84"]
    table = []
    for o in rows:
        r = {"config_id": o["config_id"], "n_realizations": o["n_realizations"],
             "k_local": o["k_local"], "N_c0": o["N_c0"],
             "c_crit_gv_klocal": o["c_crit_gv_klocal"],
             "c_crit_gv_klocal_16": o["bootstrap_16_84"]["c_crit_gv_klocal"][0],
             "c_crit_gv_klocal_84": o["bootstrap_16_84"]["c_crit_gv_klocal"][1],
             "c_crit_empirical": o["c_crit_empirical"], "c_crit_wilks": o["c_crit_wilks"]}
        for a in args.extra_alphas:
            k_ = f"{a:g}"
            r[f"c_crit_gv_klocal_alpha{k_}"] = o["c_crit_extra"][k_]["gv_klocal"]
            r[f"c_crit_gv_klocal_alpha{k_}_16"], r[f"c_crit_gv_klocal_alpha{k_}_84"] = \
                o["bootstrap_16_84"]["extra_klocal"][k_]
        table.append(r)
    Path(str(args.out) + "_summary.json").write_text(json.dumps(
        {"alpha": args.alpha, "extra_alphas": args.extra_alphas, "c0": args.c0,
         "rows": table}, indent=2) + "\n")
    with open(str(args.out) + "_summary.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in table:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print("\nsummary:")
    for r in table:
        print("  " + "  ".join(f"{k}={r[k]:.3g}" if isinstance(r[k], float) else f"{k}={r[k]}"
                              for k in keys))
    print("saved", str(args.out) + "_summary.json", str(args.out) + "_summary.csv")
    if args.validate:
        validation(rows, args)


def run(profiles, out_prefix, args):
    """Analyse the profiles of one configuration; returns the output dict."""
    curves, periods, T, config_id = load(profiles)
    n = len(T)
    levels = np.array(args.levels, dtype=float)
    res = analyse(curves, T, levels, args.c0, args.alpha, args.extra_alphas)

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, n)
        try:
            b = analyse([curves[i] for i in idx], T[idx], levels, args.c0, args.alpha,
                        args.extra_alphas)
            boot.append([b["c_crit_gv_klocal"], b["c_crit_gv_kN"], b["k_local"], b["k_N"]]
                        + [b["c_crit_extra"][f"{a:g}"]["gv_klocal"] for a in args.extra_alphas])
        except Exception:
            pass
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [16, 84], axis=0)

    edges = [0.0] + sorted(args.band_edges) + [np.inf]
    P_all, q_all = np.concatenate(periods), np.concatenate(curves)
    bands = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        m = (P_all >= a_) & (P_all < b_)
        if m.any():
            bands.append({"P_lo": a_, "P_hi": b_, "n_nodes": int(m.sum()),
                          "mean_q": float(q_all[m].mean())})

    c_wilks = float(stats.chi2.isf(args.alpha, args.k_wilks))
    checkpoints = [2.0, 4.0, 6.0, 8.0, 10.0, c_wilks, 15.0]
    surv = []
    for c in checkpoints:
        k_ = int(np.sum(T > c))
        surv.append({"c": c, "n_above": k_, "empirical": k_ / n,
                     "empirical_1sigma": list(wilson(k_, n)),
                     "gv_klocal": float(gv_sf(c, res["N_c0"], res["k_local"], args.c0)),
                     "gv_kN": float(gv_sf(c, res["N0_fit"], res["k_N"], args.c0)),
                     "wilks": float(stats.chi2.sf(c, args.k_wilks))})

    print(f"configuration {config_id}: {n} H0 realizations, "
          f"{np.mean([len(q) for q in curves]):.0f} omega nodes per profile")
    print(f"mean of q per period band (chi2_k process: mean k, constant in omega):")
    for b in bands:
        print(f"    P in [{b['P_lo']:g}, {b['P_hi']:g}) d: mean q = {b['mean_q']:.3f}"
              f"  ({b['n_nodes']} nodes)")
    print(f"k_local = {res['k_local']:.3f} [{lo[2]:.3f}, {hi[2]:.3f}]"
          f"   (from variance/2: {res['k_local_from_variance']:.3f})")
    print(f"k_N     = {res['k_N']:.3f} [{lo[3]:.3f}, {hi[3]:.3f}]"
          f"   E[N(c0={args.c0:g})] = {res['N_c0']:.2f}")
    print("upcrossings E[N(c)]: " + "  ".join(
        f"{c:g}: {m:.2f}+/-{s:.2f}" for c, m, s in zip(levels, res["N_mean"], res["N_sem"])))
    print(f"\n{100 * (1 - args.alpha):g}% critical value of T:")
    print(f"    Gross-Vitells, k_local : {res['c_crit_gv_klocal']:.2f} "
          f"[{lo[0]:.2f}, {hi[0]:.2f}]")
    print(f"    Gross-Vitells, k_N     : {res['c_crit_gv_kN']:.2f} [{lo[1]:.2f}, {hi[1]:.2f}]")
    print(f"    empirical quantile     : {res['c_crit_empirical']:.2f}"
          f"   ({n} realizations: {int(round(args.alpha * n))} expected above)")
    print(f"    Wilks chi2_{args.k_wilks}         : {c_wilks:.2f}")
    for j, a in enumerate(args.extra_alphas):
        e = res["c_crit_extra"][f"{a:g}"]
        print(f"  alpha = {a:g}: Gross-Vitells k_local {e['gv_klocal']:.2f} "
              f"[{lo[4 + j]:.2f}, {hi[4 + j]:.2f}]   k_N {e['gv_kN']:.2f}   "
              f"Wilks {stats.chi2.isf(a, args.k_wilks):.2f}")
    print(f"\nP(T > c):  c    n_above  empirical [1 sigma]     GV k_local  GV k_N   Wilks")
    for s in surv:
        print(f"        {s['c']:6.2f}  {s['n_above']:5d}   {s['empirical']:.3f} "
              f"[{s['empirical_1sigma'][0]:.3f},{s['empirical_1sigma'][1]:.3f}]"
              f"   {s['gv_klocal']:.3f}      {s['gv_kN']:.3f}   {s['wilks']:.3f}")

    out = dict(res, config_id=config_id, n_realizations=n, c0=args.c0,
               levels=levels.tolist(), alpha=args.alpha, k_wilks=args.k_wilks,
               c_crit_wilks=c_wilks, bootstrap_16_84={
                   "c_crit_gv_klocal": [lo[0], hi[0]], "c_crit_gv_kN": [lo[1], hi[1]],
                   "k_local": [lo[2], hi[2]], "k_N": [lo[3], hi[3]],
                   "extra_klocal": {f"{a:g}": [lo[4 + j], hi[4 + j]]
                                    for j, a in enumerate(args.extra_alphas)}},
               bands=bands, survival=surv, T=T.tolist(),
               profiles=[str(p) for p in profiles])
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out_prefix) + ".json").write_text(json.dumps(out, indent=2) + "\n")
    plot(out, curves, periods, levels, args, str(out_prefix) + ".png")
    print("saved", str(out_prefix) + ".json", str(out_prefix) + ".png")
    return out


def plot(out, curves, periods, levels, args, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    for P, q in zip(periods, curves):
        ax[0].plot(P, q, "-", color="0.5", lw=0.5, alpha=0.5)
    ax[0].axhline(out["k_local"], color="C3", lw=1, label=f"k_local = {out['k_local']:.2f}")
    ax[0].axhline(args.c0, color="C0", ls=":", lw=1, label=f"c0 = {args.c0:g}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("period [d]")
    ax[0].set_ylabel("q = chi2_PSPL - chi2_BSPL(omega)")
    ax[0].set_title("H0 profiles", fontsize=10)
    ax[0].legend(fontsize=8)

    c = np.linspace(0.3, 20, 300)
    m, s = np.array(out["N_mean"]), np.array(out["N_sem"])
    ax[1].errorbar(levels, m, s, fmt="o", color="k", ms=4, label="measured")
    for N0, k, col, lab in ((out["N_c0"], out["k_local"], "C3", "k_local"),
                            (out["N0_fit"], out["k_N"], "C2", "k_N")):
        ax[1].plot(c, N0 * (c / args.c0) ** ((k - 1) / 2) * np.exp(-(c - args.c0) / 2),
                   color=col, lw=1, label=f"Davies, {lab} = {k:.2f}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("level c")
    ax[1].set_ylabel("E[N(c)] upcrossings")
    ax[1].legend(fontsize=8)

    T = np.sort(np.array(out["T"]))
    n = len(T)
    ax[2].step(T, 1 - np.arange(1, n + 1) / n, where="post", color="k",
               label=f"empirical ({n})")
    ax[2].plot(c, gv_sf(c, out["N_c0"], out["k_local"], args.c0), color="C3",
               label=f"Gross-Vitells k_local: c95 = {out['c_crit_gv_klocal']:.1f}")
    ax[2].plot(c, gv_sf(c, out["N0_fit"], out["k_N"], args.c0), color="C2",
               label=f"Gross-Vitells k_N: c95 = {out['c_crit_gv_kN']:.1f}")
    ax[2].plot(c, stats.chi2.sf(c, args.k_wilks), color="C0", ls="--",
               label=f"Wilks chi2_{args.k_wilks}: c95 = {out['c_crit_wilks']:.2f}")
    ax[2].axhline(args.alpha, color="0.6", ls=":", lw=1)
    ax[2].set_yscale("log")
    ax[2].set_ylim(1e-3, 1.5)
    ax[2].set_xlabel("c")
    ax[2].set_ylabel("P(T > c)")
    ax[2].legend(fontsize=8)

    fig.suptitle(f"Test A H0, {out['config_id']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
