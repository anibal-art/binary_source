#!/usr/bin/env python3
"""
Analyze a MINI end-to-end LRT (pipeline test, low statistics).

Runner v2 fits only what each truth kind needs:
    h0a -> PSPL, BSPL          (T_A)
    h0b -> PSPL_PI, BSPL       (T_B)
    h1  -> PSPL, PSPL_PI, BSPL (T_A and T_B)
The pi_E = 0 null rows are run as h0a AND h0b (same data, checked by hash).

Pipeline checks
  * every task produced a JSON with the fits its truth kind requires
  * T_A == max(0, chi2_PSPL - chi2_BSPL), T_B == chi2_PSPL_PI - chi2_BSPL
  * h0a/h0b pair of the same row: identical data hash
  * data hashes otherwise unique, runner seed = catalogue seed
Fit-quality check (nesting)
  * PSPL is the xi = 0, q_flux = 0 special case of BSPL, so a converged BSPL
    fit must have chi2_BSPL <= chi2_PSPL.  raw Delta chi2_A < 0 therefore
    measures BSPL optimizer failure, not physics.
Statistics (illustrative only with these N)
  * Test A threshold from H0-A
  * Test B threshold = worst case (max) over pi_E = 0 and |pi_E| = 0.17
  * order statistic with 95% confidence that the true FPR <= alpha + eps
  * power per H1 cell with Clopper-Pearson 68% intervals
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

REQUIRED = {"h0a": ("PSPL", "BSPL"), "h0b": ("PSPL_PI", "BSPL"),
            "h1": ("PSPL", "PSPL_PI", "BSPL")}


def cp_interval(k, n, cl=0.68):
    a = (1 - cl) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return lo, hi


def guaranteed_threshold(values, alpha, eps, conf=0.95):
    """r-th largest value s.t. P(true FPR <= alpha+eps) >= conf."""
    v = np.sort(np.asarray(values, float))[::-1]
    n = len(v)
    if n == 0:
        return np.nan, 0, 0
    p = alpha + eps
    r = 0
    for cand in range(1, n + 1):
        if binom.sf(cand - 1, n, p) >= conf:
            r = cand
    if r == 0:
        return v[0] + 1e-9, 0, n          # too few samples: above the maximum
    return v[r - 1], r, n


def load(tasks):
    runs, missing = [], []
    for line in Path(tasks).read_text().split("\n"):
        if not line.strip():
            continue
        cat, idx, kind, out = line.split()
        p = Path(out)
        if not p.is_file():
            missing.append(out)
            continue
        d = json.loads(p.read_text())
        d["_kind"], d["_idx"], d["_cat"] = kind, int(idx), cat
        row = d.get("catalog_row", {})
        d["_zero_pi"] = (float(row.get("piEN", 0.0)) == 0.0
                         and float(row.get("piEE", 0.0)) == 0.0)
        runs.append(d)
    return runs, missing


def chi2(d, fam):
    blk = (d.get("fits") or {}).get(fam)
    if not blk:
        return None
    c = blk.get("best", {}).get("chi2")
    return float(c) if c is not None else None


def best_params(fit_block):
    best = (fit_block or {}).get("best", {})
    for key in ("parameters", "best_parameters", "params"):
        if isinstance(best.get(key), dict):
            return best[key]
    return {k: v for k, v in best.items() if str(k).startswith("piE")}


def describe(name, v):
    v = np.asarray(v, float)
    if len(v):
        q = np.quantile(v, [0.5, 0.9])
        print(f"  {name:28s} N={len(v):3d}  median={q[0]:10.3f}  q90={q[1]:10.3f}  "
              f"min={v.min():10.3f}  max={v.max():10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="/tmp/mini_lrt/tasks.txt")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="nominal level (0.10 because N0 is tiny)")
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--nest-tol", type=float, default=1.0,
                    help="tolerated chi2_BSPL - chi2_PSPL before calling it a BSPL failure")
    args = ap.parse_args()

    runs, missing = load(args.tasks)
    if not runs:
        print(f"No runner outputs found ({len(missing)} missing). Check the logs.")
        raise SystemExit(1)
    fails = []
    print("=" * 80)
    print(f"MINI LRT: {len(runs)} runs loaded, {len(missing)} missing")
    print("=" * 80)
    if missing:
        fails.append(f"{len(missing)} runs missing (e.g. {missing[0]})")

    # ------------------------------------------------------------ pipeline checks
    by_row = {}
    for d in runs:
        tag = f"{d['_kind']} row {d['_idx']}"
        s = d["statistics"]
        for fam in REQUIRED[d["_kind"]]:
            c = chi2(d, fam)
            if c is None or not np.isfinite(c):
                fails.append(f"{tag}: {fam} fit missing/non-finite")
        cP, cPI, cB = chi2(d, "PSPL"), chi2(d, "PSPL_PI"), chi2(d, "BSPL")
        if cP is not None and cB is not None:
            if not np.isclose(s["T_A"], max(0.0, cP - cB)):
                fails.append(f"{tag}: T_A inconsistent")
        if cPI is not None and cB is not None:
            if not np.isclose(s["T_B"], cPI - cB):
                fails.append(f"{tag}: T_B inconsistent")
        row = d.get("catalog_row", {})
        if row and int(row.get("simulation_seed", -1)) != int(d["simulation_seed"]):
            fails.append(f"{tag}: runner seed != catalogue seed")
        by_row.setdefault((d["_cat"], d["_idx"]), []).append(d)

    n_pairs, hashes = 0, []
    for key, ds in by_row.items():
        hs = {d["data_sha256"] for d in ds}
        if len(hs) != 1:
            fails.append(f"row {key[1]}: h0a/h0b data differ")
        if len(ds) > 1:
            n_pairs += 1
        hashes.append(ds[0]["data_sha256"])
    if len(set(hashes)) != len(hashes):
        fails.append(f"{len(hashes) - len(set(hashes))} duplicated data hashes across rows")
    print(f"h0a/h0b pairs with identical data : {n_pairs}")
    print(f"unique data hashes (one per row)  : {len(set(hashes))}/{len(hashes)}")

    # ------------------------------------------------------------ nesting
    print()
    print("FIT QUALITY: chi2_BSPL - chi2_PSPL (must be <= 0 up to optimizer noise)")
    nest_bad = []
    for d in runs:
        cP, cB = chi2(d, "PSPL"), chi2(d, "BSPL")
        if cP is None or cB is None:
            continue
        excess = cB - cP
        if excess > args.nest_tol:
            nest_bad.append((f"{d['_kind']}:{d['_idx']}", round(excess, 1)))
    n_nest = sum(1 for d in runs if chi2(d, "PSPL") is not None and chi2(d, "BSPL") is not None)
    print(f"  BSPL worse than PSPL by > {args.nest_tol}: {len(nest_bad)}/{n_nest}  {nest_bad[:8]}")
    if nest_bad:
        fails.append(f"BSPL fit did not reach the PSPL solution in {len(nest_bad)}/{n_nest} runs "
                     f"(optimizer/starts problem: thresholds and power are not meaningful)")

    # truth-family fit must reach chi2 ~ N (it contains the true model)
    truth_fam = {"h0a": "PSPL", "h0b": "PSPL_PI", "h1": "BSPL"}
    print("FIT QUALITY: chi2 of the TRUE family vs N (expected N +- sqrt(2N))")
    bad_truth = []
    for d in runs:
        fam, n = truth_fam[d["_kind"]], int(d.get("n_roman_points", 0))
        c = chi2(d, fam)
        if c is None or n == 0:
            continue
        z = (c - n) / np.sqrt(2.0 * n)
        if z > 5.0:
            bad_truth.append((f"{d['_kind']}:{d['_idx']}", fam, round(float(z), 1)))
    print(f"  true family > N + 5 sqrt(2N): {len(bad_truth)}/{len(runs)}  {bad_truth[:8]}")
    if bad_truth:
        fails.append(f"true-family fit far above N in {len(bad_truth)}/{len(runs)} runs "
                     f"(fit did not find the true solution)")

    # ------------------------------------------------------------ thresholds
    def stat(ds, key):
        return np.array([d["statistics"][key] for d in ds
                         if d["statistics"].get(key) is not None], float)

    h0a = [d for d in runs if d["_kind"] == "h0a"]
    h0b_zero = [d for d in runs if d["_kind"] == "h0b" and d["_zero_pi"]]
    h0b_pi = [d for d in runs if d["_kind"] == "h0b" and not d["_zero_pi"]]
    h1 = [d for d in runs if d["_kind"] == "h1"]

    print()
    print(f"alpha = {args.alpha}, eps = {args.eps}  (illustrative, tiny N)")
    TA0, TBz, TBp = stat(h0a, "T_A"), stat(h0b_zero, "T_B"), stat(h0b_pi, "T_B")
    describe("raw dchi2_A | H0-A", stat(h0a, "delta_chi2_A_raw"))
    describe("T_A | H0-A", TA0)
    describe("T_B | H0, pi_E=0", TBz)
    describe("T_B | H0, |pi_E|=0.17", TBp)

    tA, rA, nA = guaranteed_threshold(TA0, args.alpha, args.eps)
    tBz, _, _ = guaranteed_threshold(TBz, args.alpha, args.eps)
    tBp, _, _ = guaranteed_threshold(TBp, args.alpha, args.eps)
    tB = np.nanmax([tBz, tBp]) if np.isfinite([tBz, tBp]).any() else np.nan
    print(f"\n  threshold A = {tA:.3f}  (r={rA} largest of {nA})")
    print(f"  threshold B = max({tBz:.3f}, {tBp:.3f}) = {tB:.3f}  (worst case)")

    # ------------------------------------------------------------ power
    print()
    print("POWER per H1 cell (68% Clopper-Pearson)")
    cells = {}
    for d in h1:
        cells.setdefault(d.get("catalog_row", {}).get("config_id", "?"), []).append(d)
    for cfg, ds in sorted(cells.items(),
                          key=lambda kv: kv[1][0].get("catalog_row", {}).get("P_days", 0)):
        P = ds[0].get("catalog_row", {}).get("P_days", np.nan)
        TA, TB = stat(ds, "T_A"), stat(ds, "T_B")
        n = len(ds)
        kA, kB = int((TA > tA).sum()), int((TB > tB).sum())
        loA, hiA = cp_interval(kA, n)
        loB, hiB = cp_interval(kB, n)
        print(f"  P={P:9.4g} d  n={n:2d}  power A={kA/n:.2f} [{loA:.2f},{hiA:.2f}]  "
              f"power B={kB/n:.2f} [{loB:.2f},{hiB:.2f}]  "
              f"T_A={np.round(TA, 1).tolist()}  T_B={np.round(TB, 1).tolist()}")
        piE = [np.hypot(*(float(best_params(d['fits'].get('PSPL_PI')).get(k, np.nan))
                          for k in ("piEN", "piEE"))) for d in ds]
        if np.isfinite(piE).any():
            print(f"      spurious |pi_E| from PSPL+parallax: {np.round(piE, 3).tolist()}")

    print()
    print("=" * 80)
    if fails:
        print(f"MINI LRT PIPELINE: FAIL ({len(fails)})")
        for f in fails[:20]:
            print("  -", f)
    else:
        print("MINI LRT PIPELINE: PASS")
    print("(power and thresholds are illustrative: N0 and N1 are far too small)")


if __name__ == "__main__":
    main()
