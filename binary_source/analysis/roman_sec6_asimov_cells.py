#!/usr/bin/env python3
"""
Section 6 (detectability with Roman), alternative hypothesis H1.

Asimov noncentrality of the Test A statistic for dark-companion binary
sources,

    lambda = chi2_PSPL,fit - chi2_truth      (Asimov: chi2_truth = 0),

on the grid

    geometry x blend ratio g x F146 x u0 x P

with tE = 30 d and the physical constants of roman_bspl_roman_rubin
(M1 = 2, M2 = 1 Msun, rEhat = 5 AU, q_flux = 0).  Under H1 the statistic
T = max_omega [chi2_PSPL - chi2_BSPL(omega)] is approximately noncentral
chi2_k(lambda), so the power at level alpha is P(chi2_k(lambda) > c_alpha),
with c_alpha and k from the H0 calibration (lrt_gross_vitells.py on the
gv_sec6_h0_v1 run).

The light curve is defined exactly as in the H0 calibration
(roman_bspl_lrt_catalog_runner + lrt_catalog_io):
    * whole Roman survey, no time window (time_window = None);
    * roman_rubin photometric filter on;
    * t0 = 2461849.0 (season 3 midpoint), event at the fixed Roman field;
    * W149 = source magnitude, fsource = F(W149), ftotal = fsource (1 + g).
Periods span the period range of the BSPL profile of the fit policy
(period_min_days, period_max_days of BSPL.start_rule).

Axes
    geometry: row 0 = reference (theta, phi, i) = (0, 0, pi/2) of the paper,
              rows 1..24 = isotropic orientations, scan_xiE_ptE.geometries
              (fixed seed, same orientations as the intrinsic maps)
    g       : 0, 1, 4
    F146 AB : 19, 21, 23
    u0      : geomspace(0.005, 2, 25)[:21], i.e. 0.005 .. 0.737 with the H0
              u0 nodes (geomspace(0.005, 2, 7)) at every 4th point, so the
              H0 critical value is interpolated in log u0 between them
    P       : geomspace(period_min_days, period_max_days, 60)

Every case is run_roman_bspl_asimov.run_case (roman_rubin simulation, pyLIMA
TRF fits of PSPL and PSPL+parallax).  One task = (geometry, g, F146, u0),
one NPZ with the P axis; existing valid NPZ are skipped.

Usage (from the repository root, PYTHONPATH=$PWD):
    python binary_source/analysis/roman_sec6_asimov_cells.py --pilot
    python binary_source/analysis/roman_sec6_asimov_cells.py --list
    python binary_source/analysis/roman_sec6_asimov_cells.py --shard 0 --n-shards 8 --workers 40
    python binary_source/analysis/roman_sec6_asimov_cells.py --summarize
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ANALYSIS_DIR.parent
REPO_ROOT = SOURCE_DIR.parent
SCAN_DIR = SOURCE_DIR / "scan_codes"
for p in (REPO_ROOT, SOURCE_DIR, SCAN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scan_xiE_ptE import N_GEOMETRIES, geometries  # noqa: E402
from binary_source.analysis import roman_bspl_roman_rubin as bs  # noqa: E402
from binary_source.analysis.lrt_catalog_io import roman_rubin_commit  # noqa: E402
from binary_source.analysis.run_roman_bspl_asimov import T0_DEFAULT, plain, run_case  # noqa: E402

# ============================================================
# Grid
# ============================================================

TE = 30.0
MAGS = np.array([19.0, 21.0, 23.0])                 # F146 AB
BLEND_RATIOS = np.array([0.0, 1.0, 4.0])            # g = F_blend / F_source
U0_GRID = np.geomspace(0.005, 2.0, 25)[:21]         # H0 u0 nodes at [::4]
N_P = 60
GEO = geometries(N_GEOMETRIES)                      # row 0 = reference geometry
FIT_WINDOW_TE = None                                # whole survey, as H0
PHOTOMETRIC_FILTER = True                           # as H0
SEED_BASE = 4_000_000_000                           # disjoint from the H0/H1 catalogue seeds
DEFAULT_POLICY = REPO_ROOT / "catalogs/roman_bspl_lrt_v1/fit_policy_test_a_dark_v4.json"
DEFAULT_OUT = REPO_ROOT / "results/roman_sec6_asimov_v1"

KEYS = [
    "delta_chi2_pspl", "delta_chi2_pspl_parallax",
    "chi2_truth", "chi2_truth_fixed_flux", "chi2_pspl", "chi2_pspl_parallax", "n_data",
    "snr_event", "D_roman_eff", "xi_rel",
    "best_t0", "best_u0", "best_tE", "best_parallax_tE", "best_piE",
    "max_asimov_flux_difference", "h1_nested_ok", "h1_attempts", "seconds",
]

assert np.allclose(U0_GRID[::4], np.geomspace(0.005, 2.0, 7)[:6]), "H0 u0 nodes not on the grid"


def period_grid(policy_path):
    rule = json.loads(Path(policy_path).read_text())["BSPL"]["start_rule"]
    return np.geomspace(float(rule["period_min_days"]), float(rule["period_max_days"]), N_P)


def tasks():
    """(geometry, blend, magnitude, u0) indices, in a fixed order."""
    return [(ig, ib, im, iu)
            for ig in range(len(GEO))
            for ib in range(len(BLEND_RATIOS))
            for im in range(len(MAGS))
            for iu in range(len(U0_GRID))]


def flat_index(ig, ib, im, iu, ip):
    return ((((ig * len(BLEND_RATIOS) + ib) * len(MAGS) + im) * len(U0_GRID) + iu) * N_P + ip)


def outfile(directory, ig, ib, im, iu):
    return Path(directory) / "cells" / f"geo{ig:02d}_g{ib}_m{im}_u{iu:02d}.npz"


def valid(path, P):
    try:
        with np.load(path, allow_pickle=False) as d:
            return (d["delta_chi2_pspl"].shape == (len(P),) and np.allclose(d["P_days"], P))
    except Exception:
        return False


def one_case(ig, ib, im, iu, ip, P):
    theta, phi, inc = GEO[ig]
    t1 = time.time()
    r = run_case(
        event_index=SEED_BASE + flat_index(ig, ib, im, iu, ip),
        P_days=float(P[ip]),
        u0=float(U0_GRID[iu]),
        tE=TE,
        source_mag_ab=float(MAGS[im]),
        fit_window_tE=FIT_WINDOW_TE,
        phi=float(phi),
        inclination=float(inc),
        theta=float(theta),
        asimov=True,
        photometric_filter=PHOTOMETRIC_FILTER,
        blend_ratio=float(BLEND_RATIOS[ib]),
    )
    r["seconds"] = time.time() - t1
    return r


def worker(job):
    ig, ib, im, iu, directory, policy, commits = job
    P = period_grid(policy)
    out = outfile(directory, ig, ib, im, iu)
    if valid(out, P):
        return "skip"
    res = {k: np.full(len(P), np.nan) for k in KEYS}
    err = np.full(len(P), "", dtype="<U300")
    for ip in range(len(P)):
        try:
            r = one_case(ig, ib, im, iu, ip, P)
            for k in KEYS:
                res[k][ip] = float(r[k])
        except Exception as exc:
            err[ip] = f"{type(exc).__name__}: {exc}"[:300]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **res, error=err, P_days=P,
        u0=U0_GRID[iu], tE=TE, F146=MAGS[im], blend_ratio=BLEND_RATIOS[ib],
        geometry=GEO[ig], t0=T0_DEFAULT, q_mass=bs.M2_MSUN / bs.M1_MSUN, q_flux=0.0,
        rEhat_AU=bs.REHAT_AU, fit_window="whole survey", photometric_filter=PHOTOMETRIC_FILTER,
        asimov=True, fit_policy=str(policy), **commits,
    )
    return "done"


def provenance():
    return {"code_commit": roman_rubin_commit(REPO_ROOT),
            "roman_rubin_commit": roman_rubin_commit(bs.get_roman_rubin_root()),
            "roman_rubin_repo": str(bs.get_roman_rubin_root())}


def run(args):
    todo = tasks()[args.shard::args.n_shards]
    commits = provenance()
    print(f"provenance: {commits}")
    print(f"shard {args.shard}/{args.n_shards}: {len(todo)} tasks x {N_P} periods")
    jobs = [(*t, str(args.out), str(args.fit_policy), commits) for t in todo]
    n_done = n_skip = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, j) for j in jobs]
        for f in as_completed(futs):
            status = f.result()
            n_done += status == "done"
            n_skip += status == "skip"
            if (n_done + n_skip) % 10 == 0 or n_done + n_skip == len(jobs):
                print(f"{n_done + n_skip}/{len(jobs)} tasks (skipped {n_skip}) | "
                      f"{time.time() - t0:.0f} s", flush=True)


def pilot(args):
    """Timing and light-curve check on a few cases before production."""
    P = period_grid(args.fit_policy)
    print(f"provenance: {provenance()}")
    print(f"u0 grid ({len(U0_GRID)}): {np.round(U0_GRID, 5).tolist()}")
    print(f"P grid ({len(P)}): {P[0]:.1f} .. {P[-1]:.1f} d")
    print(f"tasks = {len(tasks())}, cases = {len(tasks()) * N_P}")
    cases = [(0, 0, 1, 8, 30), (0, 2, 1, 8, 30), (1, 0, 2, 12, 45), (5, 1, 0, 4, 10)]
    secs = []
    for ig, ib, im, iu, ip in cases:
        r = one_case(ig, ib, im, iu, ip, P)
        t = np.asarray(plain(r["simulation"]["model"].event.telescopes[0].lightcurve["time"]), float)
        secs.append(r["seconds"])
        print(f"geo {ig:2d} g {BLEND_RATIOS[ib]:g} F146 {MAGS[im]:g} u0 {U0_GRID[iu]:.4g} "
              f"P {P[ip]:.1f} d | n_data {int(r['n_data'])} span {t.min():.1f}-{t.max():.1f} "
              f"({(t.max() - t.min()) / 365.25:.2f} yr) | lambda {r['delta_chi2_pspl']:.2f} "
              f"chi2_truth {r['chi2_truth']:.2e} | max Asimov diff "
              f"{r['max_asimov_flux_difference']:.1e} | {r['seconds']:.0f} s")
    s = float(np.mean(secs))
    n = len(tasks()) * N_P
    print(f"mean {s:.0f} s/case -> {n * s / 3600:.0f} CPU h for {n} cases")


def summarize(args):
    P = period_grid(args.fit_policy)
    shape = (len(GEO), len(BLEND_RATIOS), len(MAGS), len(U0_GRID), len(P))
    out = {k: np.full(shape, np.nan) for k in KEYS}
    n_err = np.zeros(shape, dtype=int)
    missing = []
    commits = set()
    for ig, ib, im, iu in tasks():
        f = outfile(args.out, ig, ib, im, iu)
        if not valid(f, P):
            missing.append(f.name)
            continue
        with np.load(f, allow_pickle=False) as d:
            for k in KEYS:
                out[k][ig, ib, im, iu] = d[k]
            n_err[ig, ib, im, iu] = d["error"] != ""
            commits.add((str(d["code_commit"]), str(d["roman_rubin_commit"])))
    dest = Path(args.out) / "roman_sec6_asimov.npz"
    np.savez_compressed(
        dest, **{k.upper(): v for k, v in out.items()}, N_ERROR=n_err,
        geometry=GEO, blend_ratio=BLEND_RATIOS, F146=MAGS, u0_grid=U0_GRID, P_days=P,
        tE=TE, t0=T0_DEFAULT, q_mass=bs.M2_MSUN / bs.M1_MSUN, q_flux=0.0, rEhat_AU=bs.REHAT_AU,
        fit_window="whole survey", photometric_filter=PHOTOMETRIC_FILTER, asimov=True,
        fit_policy=str(args.fit_policy), commits=np.array(sorted(commits), dtype="<U64"),
    )
    lam = out["delta_chi2_pspl"]
    print(f"tasks missing: {len(missing)} / {len(tasks())}" + (f"  e.g. {missing[:5]}" if missing else ""))
    print(f"cases with error: {int(n_err.sum())}; finite lambda: {int(np.isfinite(lam).sum())} / {lam.size}")
    print(f"commits (binary_source, roman_rubin): {sorted(commits)}")
    print(f"saved {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fit-policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--list", action="store_true")
    g.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.pilot:
        pilot(args)
    elif args.list:
        P = period_grid(args.fit_policy)
        print(f"{len(GEO)} geometries x {len(BLEND_RATIOS)} g x {len(MAGS)} F146 x "
              f"{len(U0_GRID)} u0 = {len(tasks())} tasks, x {len(P)} P = {len(tasks()) * len(P)} cases")
    elif args.summarize:
        summarize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
