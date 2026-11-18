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

Second grid, --grid xiP (the (P/tE, xi_E) maps of the paper at tE = 30 d,
same conventions as the intrinsic maps of scan_xiE_ptE.py):
    u0      : scan_xiE_ptE.U0_VALUES (0.03, 0.1, 0.3)
    xi_E    : scan_xiE_ptE.XI_E_GRID[::2] (amplitude of the luminous star,
              xi_rel = xi_E (1 + q)/q passed to run_case, as scan_xiE_ptE)
    P/tE    : the scan_xiE_ptE.PTE_GRID nodes with P = tE P/tE inside the
              period range of the profile
    geometry, g, F146 as above.  With real sampling the problem is not scale
    free: the map holds for any companion mass and source distance at
    tE = 30 d, not for another tE.

Every case is run_roman_bspl_asimov.run_case (roman_rubin simulation, pyLIMA
TRF fits of PSPL and PSPL+parallax).  One task = (geometry, g, F146, u0),
one NPZ with the P axis; existing valid NPZ are skipped.

Usage (from the repository root, PYTHONPATH=$PWD):
    python binary_source/analysis/roman_sec6_asimov_cells.py --pilot
    python binary_source/analysis/roman_sec6_asimov_cells.py --list
    python binary_source/analysis/roman_sec6_asimov_cells.py --shard 0 --n-shards 8 --workers 40
    python binary_source/analysis/roman_sec6_asimov_cells.py --summarize
    (add --grid xiP to any of them for the (P/tE, xi_E) grid)
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

from scan_xiE_ptE import N_GEOMETRIES, PTE_GRID, U0_VALUES, XI_E_GRID, geometries  # noqa: E402
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
DEFAULT_POLICY = REPO_ROOT / "catalogs/roman_bspl_lrt_v1/fit_policy_test_a_dark_v4.json"
Q_MASS = bs.M2_MSUN / bs.M1_MSUN

# grid name -> u0 axis, xi_E axis (None: physical xi from Kepler), seeds, output
GRIDS = {
    "u0P": {"u0": U0_GRID, "xi": None, "seed_base": 4_000_000_000,
            "out": REPO_ROOT / "results/roman_sec6_asimov_v1", "summary": "roman_sec6_asimov.npz"},
    "xiP": {"u0": U0_VALUES, "xi": XI_E_GRID[::2], "seed_base": 4_100_000_000,
            "out": REPO_ROOT / "results/roman_sec6_asimov_xi_v1",
            "summary": "roman_sec6_asimov_xi.npz"},
}

KEYS = [
    "delta_chi2_pspl", "delta_chi2_pspl_parallax",
    "chi2_truth", "chi2_truth_fixed_flux", "chi2_pspl", "chi2_pspl_parallax", "n_data",
    "snr_event", "D_roman_eff", "xi_rel",
    "best_t0", "best_u0", "best_tE", "best_parallax_tE", "best_piE",
    "max_asimov_flux_difference", "h1_nested_ok", "h1_attempts", "seconds",
]

assert np.allclose(U0_GRID[::4], np.geomspace(0.005, 2.0, 7)[:6]), "H0 u0 nodes not on the grid"


def period_grid(policy_path, grid="u0P"):
    rule = json.loads(Path(policy_path).read_text())["BSPL"]["start_rule"]
    pmin, pmax = float(rule["period_min_days"]), float(rule["period_max_days"])
    if grid == "u0P":
        return np.geomspace(pmin, pmax, N_P)
    P = TE * PTE_GRID
    return P[(P >= pmin * (1 - 1e-12)) & (P <= pmax * (1 + 1e-12))]


def n_xi(grid):
    return 1 if GRIDS[grid]["xi"] is None else len(GRIDS[grid]["xi"])


def tasks(grid="u0P"):
    """(geometry, blend, magnitude, u0, xi) indices, in a fixed order (xi = 0 for u0P)."""
    return [(ig, ib, im, iu, ix)
            for ig in range(len(GEO))
            for ib in range(len(BLEND_RATIOS))
            for im in range(len(MAGS))
            for iu in range(len(GRIDS[grid]["u0"]))
            for ix in range(n_xi(grid))]


def flat_index(grid, ig, ib, im, iu, ix, ip, n_p):
    nu = len(GRIDS[grid]["u0"])
    return (((((ig * len(BLEND_RATIOS) + ib) * len(MAGS) + im) * nu + iu) * n_xi(grid) + ix)
            * n_p + ip)


def outfile(directory, grid, ig, ib, im, iu, ix):
    tail = "" if grid == "u0P" else f"_x{ix:02d}"          # u0P keeps its original names
    return Path(directory) / "cells" / f"geo{ig:02d}_g{ib}_m{im}_u{iu:02d}{tail}.npz"


def valid(path, P):
    try:
        with np.load(path, allow_pickle=False) as d:
            return (d["delta_chi2_pspl"].shape == (len(P),) and np.allclose(d["P_days"], P))
    except Exception:
        return False


def one_case(grid, ig, ib, im, iu, ix, ip, P):
    theta, phi, inc = GEO[ig]
    xi = GRIDS[grid]["xi"]
    t1 = time.time()
    r = run_case(
        event_index=GRIDS[grid]["seed_base"] + flat_index(grid, ig, ib, im, iu, ix, ip, len(P)),
        P_days=float(P[ip]),
        u0=float(GRIDS[grid]["u0"][iu]),
        tE=TE,
        source_mag_ab=float(MAGS[im]),
        fit_window_tE=FIT_WINDOW_TE,
        phi=float(phi),
        inclination=float(inc),
        theta=float(theta),
        asimov=True,
        photometric_filter=PHOTOMETRIC_FILTER,
        blend_ratio=float(BLEND_RATIOS[ib]),
        xi_rel=(None if xi is None else float(xi[ix]) * (1.0 + Q_MASS) / Q_MASS),
    )
    r["seconds"] = time.time() - t1
    return r


def worker(job):
    grid, ig, ib, im, iu, ix, directory, policy, commits = job
    P = period_grid(policy, grid)
    out = outfile(directory, grid, ig, ib, im, iu, ix)
    if valid(out, P):
        return "skip"
    res = {k: np.full(len(P), np.nan) for k in KEYS}
    err = np.full(len(P), "", dtype="<U300")
    for ip in range(len(P)):
        try:
            r = one_case(grid, ig, ib, im, iu, ix, ip, P)
            for k in KEYS:
                res[k][ip] = float(r[k])
        except Exception as exc:
            err[ip] = f"{type(exc).__name__}: {exc}"[:300]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **res, error=err, P_days=P,
        u0=GRIDS[grid]["u0"][iu], tE=TE, F146=MAGS[im], blend_ratio=BLEND_RATIOS[ib],
        xi_E=(np.nan if GRIDS[grid]["xi"] is None else GRIDS[grid]["xi"][ix]), grid=grid,
        geometry=GEO[ig], t0=T0_DEFAULT, q_mass=Q_MASS, q_flux=0.0,
        rEhat_AU=bs.REHAT_AU, fit_window="whole survey", photometric_filter=PHOTOMETRIC_FILTER,
        asimov=True, fit_policy=str(policy), **commits,
    )
    return "done"


def provenance():
    return {"code_commit": roman_rubin_commit(REPO_ROOT),
            "roman_rubin_commit": roman_rubin_commit(bs.get_roman_rubin_root()),
            "roman_rubin_repo": str(bs.get_roman_rubin_root())}


def run(args):
    todo = tasks(args.grid)[args.shard::args.n_shards]
    commits = provenance()
    n_p = len(period_grid(args.fit_policy, args.grid))
    print(f"provenance: {commits}")
    print(f"grid {args.grid}, shard {args.shard}/{args.n_shards}: {len(todo)} tasks x {n_p} periods")
    jobs = [(args.grid, *t, str(args.out), str(args.fit_policy), commits) for t in todo]
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
    grid = args.grid
    P = period_grid(args.fit_policy, grid)
    u0s, xis = GRIDS[grid]["u0"], GRIDS[grid]["xi"]
    print(f"provenance: {provenance()}")
    print(f"grid {grid}: u0 ({len(u0s)}): {np.round(u0s, 5).tolist()}")
    if xis is not None:
        print(f"xi_E ({len(xis)}): {xis[0]:.4g} .. {xis[-1]:.4g};  P/tE {P[0] / TE:.3g} .. {P[-1] / TE:.3g}")
    print(f"P grid ({len(P)}): {P[0]:.1f} .. {P[-1]:.1f} d")
    print(f"tasks = {len(tasks(grid))}, cases = {len(tasks(grid)) * len(P)}")
    if grid == "u0P":
        cases = [(0, 0, 1, 8, 0, 30), (0, 2, 1, 8, 0, 30), (1, 0, 2, 12, 0, 45), (5, 1, 0, 4, 0, 10)]
    else:
        cases = [(0, 0, 1, 1, 10, 14), (0, 2, 1, 1, 10, 14), (1, 0, 2, 2, 6, 5), (5, 1, 0, 0, 14, 25)]
    secs = []
    for ig, ib, im, iu, ix, ip in cases:
        r = one_case(grid, ig, ib, im, iu, ix, ip, P)
        t = np.asarray(plain(r["simulation"]["model"].event.telescopes[0].lightcurve["time"]), float)
        secs.append(r["seconds"])
        xs = "" if xis is None else f"xi_E {xis[ix]:.4g} "
        print(f"geo {ig:2d} g {BLEND_RATIOS[ib]:g} F146 {MAGS[im]:g} u0 {u0s[iu]:.4g} {xs}"
              f"P {P[ip]:.1f} d | n_data {int(r['n_data'])} span {t.min():.1f}-{t.max():.1f} "
              f"({(t.max() - t.min()) / 365.25:.2f} yr) | lambda {r['delta_chi2_pspl']:.2f} "
              f"chi2_truth {r['chi2_truth']:.2e} | max Asimov diff "
              f"{r['max_asimov_flux_difference']:.1e} | {r['seconds']:.0f} s")
    s = float(np.mean(secs))
    n = len(tasks(grid)) * len(P)
    print(f"mean {s:.0f} s/case -> {n * s / 3600:.0f} CPU h for {n} cases")


def summarize(args):
    grid = args.grid
    P = period_grid(args.fit_policy, grid)
    u0s, xis = GRIDS[grid]["u0"], GRIDS[grid]["xi"]
    # u0P: (geo, g, F146, u0, P), as before; xiP: (geo, g, F146, u0, xi_E, P)
    shape = (len(GEO), len(BLEND_RATIOS), len(MAGS), len(u0s)) + \
        (() if xis is None else (len(xis),)) + (len(P),)
    out = {k: np.full(shape, np.nan) for k in KEYS}
    n_err = np.zeros(shape, dtype=int)
    missing = []
    commits = set()
    for ig, ib, im, iu, ix in tasks(grid):
        f = outfile(args.out, grid, ig, ib, im, iu, ix)
        idx = (ig, ib, im, iu) if xis is None else (ig, ib, im, iu, ix)
        if not valid(f, P):
            missing.append(f.name)
            continue
        with np.load(f, allow_pickle=False) as d:
            for k in KEYS:
                out[k][idx] = d[k]
            n_err[idx] = d["error"] != ""
            commits.add((str(d["code_commit"]), str(d["roman_rubin_commit"])))
    dest = Path(args.out) / GRIDS[grid]["summary"]
    extra = {} if xis is None else {"xiE_grid": xis, "PtE_grid": P / TE}
    np.savez_compressed(
        dest, **{k.upper(): v for k, v in out.items()}, N_ERROR=n_err, grid=grid, **extra,
        geometry=GEO, blend_ratio=BLEND_RATIOS, F146=MAGS, u0_grid=u0s, P_days=P,
        tE=TE, t0=T0_DEFAULT, q_mass=Q_MASS, q_flux=0.0, rEhat_AU=bs.REHAT_AU,
        fit_window="whole survey", photometric_filter=PHOTOMETRIC_FILTER, asimov=True,
        fit_policy=str(args.fit_policy), commits=np.array(sorted(commits), dtype="<U64"),
    )
    lam = out["delta_chi2_pspl"]
    print(f"tasks missing: {len(missing)} / {len(tasks(grid))}" + (f"  e.g. {missing[:5]}" if missing else ""))
    print(f"cases with error: {int(n_err.sum())}; finite lambda: {int(np.isfinite(lam).sum())} / {lam.size}")
    print(f"commits (binary_source, roman_rubin): {sorted(commits)}")
    print(f"saved {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--grid", choices=sorted(GRIDS), default="u0P")
    ap.add_argument("--out", type=Path, default=None, help="default: per grid (GRIDS)")
    ap.add_argument("--fit-policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--list", action="store_true")
    g.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = GRIDS[args.grid]["out"]
    if args.pilot:
        pilot(args)
    elif args.list:
        P = period_grid(args.fit_policy, args.grid)
        nt = len(tasks(args.grid))
        print(f"grid {args.grid}: {len(GEO)} geometries x {len(BLEND_RATIOS)} g x {len(MAGS)} F146 x "
              f"{len(GRIDS[args.grid]['u0'])} u0 x {n_xi(args.grid)} xi = {nt} tasks, "
              f"x {len(P)} P = {nt * len(P)} cases")
    elif args.summarize:
        summarize(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
