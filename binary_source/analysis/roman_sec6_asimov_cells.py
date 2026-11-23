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

Grids xiP2lum_* (Section 6 experiment with two luminous sources): the
(xi_rel, P/tE) maps of Section 5 (scan_xiE_ptE.py --qM --qf, Figure 4) for the
same pairs (q_M, q_f) = (0.5, 0.05), (0.5, 0.5), (1, 1) and u0 = 0.01, 0.3, 1:
    xi      : XI_E_GRID[::2] read as the RELATIVE amplitude xi_rel (as Figure 4)
    F146    : 21, the TOTAL magnitude of the two stars (what Roman measures;
              source 1 gets F146 + 2.5 log10(1 + q_f))
    g       : 0
    geometry: the same 25 rows.
lambda is the chi2 of the PSPL fit to the noiseless light curve, the Asimov
noncentrality of T when the alternative model has q_f and q_M free
(fit_policy_test_a_luminous_v1.json): the true model is then nested in it.

Every case is run_roman_bspl_asimov.run_case (roman_rubin simulation, pyLIMA
TRF fits of PSPL and PSPL+parallax).  One task = (geometry, g, F146, u0),
one NPZ with the P axis; existing valid NPZ are skipped.

Usage (from the repository root, PYTHONPATH=$PWD):
    python binary_source/analysis/roman_sec6_asimov_cells.py --pilot
    python binary_source/analysis/roman_sec6_asimov_cells.py --list
    python binary_source/analysis/roman_sec6_asimov_cells.py --shard 0 --n-shards 8 --workers 40
    python binary_source/analysis/roman_sec6_asimov_cells.py --summarize
    (add --grid xiP to any of them for the (P/tE, xi_E) grid)
    python binary_source/analysis/roman_sec6_asimov_cells.py --grid xiPpaper --umin
    (finite-source check: closest approach of the luminous source, no fits)
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
    # the u0 of the paper maps (Fig. 2: scan_xiE_ptE.py --u0 0.01 0.3 1 --refine 2;
    # Fig. 3: u0 = 0.1).  0.1 and 0.3 sit at the same indices as in xiP, so those
    # cells can be linked from roman_sec6_asimov_xi_v1 (the check of the stored
    # u0 and xi_E in valid() guards the reuse).
    "xiPpaper": {"u0": np.array([0.01, 0.1, 0.3, 1.0]), "xi": XI_E_GRID[::2],
                 "seed_base": 4_150_000_000,
                 "out": REPO_ROOT / "results/roman_sec6_asimov_xi_v2",
                 "summary": "roman_sec6_asimov_xi_paper.npz"},
}

# Two luminous sources: the pairs and u0 of Section 5 (Figure 4).  Optional keys
# of a grid: mags, blends (default MAGS, BLEND_RATIOS), q_mass, q_flux (default
# Q_MASS, 0), xi_kind ("E": amplitude of the luminous star, converted to xi_rel;
# "rel": xi_rel itself), mag_kind ("source1" or "total").
for _name, (_qm, _qf), _seed in (("xiP2lum_q0p5_f0p05", (0.5, 0.05), 4_200_000_000),
                                 ("xiP2lum_q0p5_f0p5", (0.5, 0.5), 4_250_000_000),
                                 ("xiP2lum_q1_f1", (1.0, 1.0), 4_300_000_000)):
    GRIDS[_name] = {"u0": np.array([0.01, 0.3, 1.0]), "xi": XI_E_GRID[::2],
                    "seed_base": _seed, "mags": np.array([21.0]), "blends": np.array([0.0]),
                    "q_mass": _qm, "q_flux": _qf, "xi_kind": "rel", "mag_kind": "total",
                    "out": REPO_ROOT / "results/roman_sec6_asimov_2lum_v1" / _name,
                    "summary": f"roman_sec6_asimov_{_name}.npz"}


def mags(grid):
    return GRIDS[grid].get("mags", MAGS)


def blends(grid):
    return GRIDS[grid].get("blends", BLEND_RATIOS)


def q_pair(grid):
    return float(GRIDS[grid].get("q_mass", Q_MASS)), float(GRIDS[grid].get("q_flux", 0.0))


def source1_mag(grid, mag):
    """Magnitude passed to the simulation (source 1) for the grid's convention."""
    if GRIDS[grid].get("mag_kind", "source1") == "total":
        return float(mag) + 2.5 * np.log10(1.0 + q_pair(grid)[1])
    return float(mag)


def xi_rel_of(grid, xi):
    """xi_rel passed to run_case for a node of the grid's xi axis."""
    if GRIDS[grid].get("xi_kind", "E") == "rel":
        return float(xi)
    qm = q_pair(grid)[0]
    return float(xi) * (1.0 + qm) / qm

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
            for ib in range(len(blends(grid)))
            for im in range(len(mags(grid)))
            for iu in range(len(GRIDS[grid]["u0"]))
            for ix in range(n_xi(grid))]


def flat_index(grid, ig, ib, im, iu, ix, ip, n_p):
    nu = len(GRIDS[grid]["u0"])
    return (((((ig * len(blends(grid)) + ib) * len(mags(grid)) + im) * nu + iu) * n_xi(grid) + ix)
            * n_p + ip)


def outfile(directory, grid, ig, ib, im, iu, ix):
    tail = "" if grid == "u0P" else f"_x{ix:02d}"          # u0P keeps its original names
    return Path(directory) / "cells" / f"geo{ig:02d}_g{ib}_m{im}_u{iu:02d}{tail}.npz"


def valid(path, P, u0=None, xi=None):
    """Complete cell file for this P grid (and, when given, this u0 and xi_E)."""
    try:
        with np.load(path, allow_pickle=False) as d:
            ok = d["delta_chi2_pspl"].shape == (len(P),) and np.allclose(d["P_days"], P)
            if u0 is not None:
                ok = ok and np.isclose(float(d["u0"]), u0, rtol=1e-9, atol=0.0)
            if xi is not None:
                ok = ok and np.isclose(float(d["xi_E"]), xi, rtol=1e-9, atol=0.0)
            return bool(ok)
    except Exception:
        return False


def cell_values(grid, iu, ix):
    xi = GRIDS[grid]["xi"]
    return float(GRIDS[grid]["u0"][iu]), (None if xi is None else float(xi[ix]))


def one_case(grid, ig, ib, im, iu, ix, ip, P):
    theta, phi, inc = GEO[ig]
    xi = GRIDS[grid]["xi"]
    t1 = time.time()
    r = run_case(
        event_index=GRIDS[grid]["seed_base"] + flat_index(grid, ig, ib, im, iu, ix, ip, len(P)),
        P_days=float(P[ip]),
        u0=float(GRIDS[grid]["u0"][iu]),
        tE=TE,
        source_mag_ab=source1_mag(grid, mags(grid)[im]),
        fit_window_tE=FIT_WINDOW_TE,
        phi=float(phi),
        inclination=float(inc),
        theta=float(theta),
        asimov=True,
        photometric_filter=PHOTOMETRIC_FILTER,
        blend_ratio=float(blends(grid)[ib]),
        xi_rel=(None if xi is None else xi_rel_of(grid, xi[ix])),
        q_flux=q_pair(grid)[1],
        q_mass=q_pair(grid)[0],
    )
    r["seconds"] = time.time() - t1
    return r


def worker(job):
    grid, ig, ib, im, iu, ix, directory, policy, commits = job
    P = period_grid(policy, grid)
    out = outfile(directory, grid, ig, ib, im, iu, ix)
    if valid(out, P, *cell_values(grid, iu, ix)):
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
        u0=GRIDS[grid]["u0"][iu], tE=TE, F146=mags(grid)[im], blend_ratio=blends(grid)[ib],
        xi_E=(np.nan if GRIDS[grid]["xi"] is None else GRIDS[grid]["xi"][ix]), grid=grid,
        geometry=GEO[ig], t0=T0_DEFAULT, q_mass=q_pair(grid)[0], q_flux=q_pair(grid)[1],
        xi_kind=GRIDS[grid].get("xi_kind", "E"), mag_kind=GRIDS[grid].get("mag_kind", "source1"),
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
    elif grid.startswith("xiP2lum"):
        # (geometry, g, F146, u0, xi_rel, P) indices: every u0, short and long periods
        cases = [(1, 0, 0, 0, 10, 14), (1, 0, 0, 1, 12, 5), (5, 0, 0, 2, 14, 25), (2, 0, 0, 1, 16, 20)]
    else:
        cases = [(0, 0, 1, 1, 10, 14), (0, 2, 1, 1, 10, 14), (1, 0, 2, 2, 6, 5), (5, 1, 0, 0, 14, 25)]
        if len(u0s) > 3:                                       # also the largest u0 (u0 = 1)
            cases.append((2, 0, 2, len(u0s) - 1, 8, 20))
    secs = []
    for ig, ib, im, iu, ix, ip in cases:
        r = one_case(grid, ig, ib, im, iu, ix, ip, P)
        t = np.asarray(plain(r["simulation"]["model"].event.telescopes[0].lightcurve["time"]), float)
        secs.append(r["seconds"])
        xs = "" if xis is None else f"xi_E {xis[ix]:.4g} "
        print(f"geo {ig:2d} g {blends(grid)[ib]:g} F146 {mags(grid)[im]:g} u0 {u0s[iu]:.4g} {xs}"
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
    shape = (len(GEO), len(blends(grid)), len(mags(grid)), len(u0s)) + \
        (() if xis is None else (len(xis),)) + (len(P),)
    out = {k: np.full(shape, np.nan) for k in KEYS}
    n_err = np.zeros(shape, dtype=int)
    missing = []
    commits = set()
    for ig, ib, im, iu, ix in tasks(grid):
        f = outfile(args.out, grid, ig, ib, im, iu, ix)
        idx = (ig, ib, im, iu) if xis is None else (ig, ib, im, iu, ix)
        if not valid(f, P, *cell_values(grid, iu, ix)):
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
        geometry=GEO, blend_ratio=blends(grid), F146=mags(grid), u0_grid=u0s, P_days=P,
        tE=TE, t0=T0_DEFAULT, q_mass=q_pair(grid)[0], q_flux=q_pair(grid)[1],
        xi_kind=GRIDS[grid].get("xi_kind", "E"), mag_kind=GRIDS[grid].get("mag_kind", "source1"),
        rEhat_AU=bs.REHAT_AU,
        fit_window="whole survey", photometric_filter=PHOTOMETRIC_FILTER, asimov=True,
        fit_policy=str(args.fit_policy), commits=np.array(sorted(commits), dtype="<U64"),
    )
    lam = out["delta_chi2_pspl"]
    print(f"tasks missing: {len(missing)} / {len(tasks(grid))}" + (f"  e.g. {missing[:5]}" if missing else ""))
    print(f"cases with error: {int(n_err.sum())}; finite lambda: {int(np.isfinite(lam).sum())} / {lam.size}")
    print(f"commits (binary_source, roman_rubin): {sorted(commits)}")
    print(f"saved {dest}")


# ============================================================
# Finite-source check, stage 1 (--umin): closest approach of the luminous source
# ============================================================
# u0 is the lens - luminous-source separation at t0, but with the orbital motion
# the luminous source can pass closer to the lens at other epochs.  Finite-source
# effects matter when this closest approach u_min is ~< 3 rho.  For grid nodes
# near P/tE = UMIN_PTE and xi_E = UMIN_XI_OVER_U0 * u0, all geometries and the
# three F146, g = 0, u_min is measured over the observed Roman epochs of the
# same simulated light curve as the cells (simulate_roman_bspl, photometric
# filter on; saturated peak epochs are removed), from the pyLIMA source
# trajectory (trajectory_diagnostics of sanity_check_finite_source).  No fits.
# The lambda of the corresponding cell is attached when the cell exists.

UMIN_RHO = np.array([5e-4, 1e-3, 1.5e-3])            # bulge sources, F146 = 19-23
UMIN_PTE = np.array([0.5, 1.0, 3.0, 10.0, 30.0])     # nearest P/tE nodes of the grid
UMIN_XI_OVER_U0 = np.array([0.5, 2.0, 10.0])         # nearest xi_E nodes of the grid


def umin_axes(grid, policy):
    """Period grid, P indices, and xi_E indices per u0 used by --umin."""
    P = period_grid(policy, grid)
    ips = sorted({int(np.argmin(np.abs(np.log(P / TE / r)))) for r in UMIN_PTE})
    xis = GRIDS[grid]["xi"]
    ixs = [sorted({int(np.argmin(np.abs(np.log(xis / (u0 * r))))) for r in UMIN_XI_OVER_U0})
           for u0 in GRIDS[grid]["u0"]]
    return P, ips, ixs


def umin_case(job):
    """(u_min, t(u_min) - t0, u at the epoch nearest t0, that epoch - t0) of the luminous source."""
    from binary_source.analysis.sanity_check_finite_source import trajectory_diagnostics
    grid, ig, im, iu, ix, P_days = job
    theta, phi, inc = GEO[ig]
    u0, xi = cell_values(grid, iu, ix)
    sim = bs.simulate_roman_bspl(
        event_index=0, P_days=float(P_days), u0=u0, tE=TE, t0=T0_DEFAULT,
        source_mag_ab=float(MAGS[im]), asimov=True, truth_parallax=False, q_flux=0.0,
        time_window_te=FIT_WINDOW_TE, apply_photometric_filter=PHOTOMETRIC_FILTER,
        phi=float(phi), inclination=float(inc), theta=float(theta),
        xi_rel=xi * (1.0 + Q_MASS) / Q_MASS, blend_ratio=0.0,
    )
    model, pyp = sim["model"], sim["py_params"]
    tel = model.event.telescopes[0]
    t = plain(tel.lightcurve["time"])
    # q_flux = 0: source 1 is the luminous star, source 2 the dark companion
    geo = trajectory_diagnostics(t, model.sources_trajectory(tel, pyp, data_type="photometry"))
    i0 = int(np.argmin(np.abs(t - T0_DEFAULT)))
    return (geo["u1_min"], geo["t_u1_min"] - T0_DEFAULT,
            float(geo["u1"][i0]), float(t[i0] - T0_DEFAULT))


def cell_lambda(directory, grid, ig, im, iu, ix, ip, P):
    """lambda of the (g = 0) cell at this period, NaN if the cell is not there."""
    f = outfile(directory, grid, ig, 0, im, iu, ix)
    if not valid(f, P, *cell_values(grid, iu, ix)):
        return np.nan
    with np.load(f, allow_pickle=False) as d:
        return float(d["delta_chi2_pspl"][ip])


def umin(args):
    if "q_flux" in GRIDS[args.grid]:
        raise SystemExit("--umin is the dark-companion check (one luminous source)")
    grid = args.grid
    if GRIDS[grid]["xi"] is None:
        raise SystemExit("--umin needs a grid with a xi_E axis (xiP, xiPpaper)")
    P, ips, ixs = umin_axes(grid, args.fit_policy)
    u0s, xis = GRIDS[grid]["u0"], GRIDS[grid]["xi"]
    idx = [(ig, im, iu, ix, ip)
           for im in range(len(MAGS)) for iu in range(len(u0s)) for ix in ixs[iu]
           for ip in ips for ig in range(len(GEO))]
    if args.umin_limit is not None:
        idx = idx[:args.umin_limit]
    print(f"provenance: {provenance()}")
    print(f"--umin, grid {grid}: P/tE nodes {np.round(P[ips] / TE, 3).tolist()}, "
          f"{len(idx)} simulations (no fits)")
    out = np.full((len(idx), 4), np.nan)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(umin_case, (grid, ig, im, iu, ix, float(P[ip]))): n
                for n, (ig, im, iu, ix, ip) in enumerate(idx)}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                out[futs[f]] = f.result()
            except Exception as exc:
                print(f"error {idx[futs[f]]}: {type(exc).__name__}: {exc}"[:300])
            if k % 200 == 0 or k == len(idx):
                print(f"{k}/{len(idx)}", flush=True)
    a = np.array(idx)
    lam = np.array([cell_lambda(args.out, grid, ig, im, iu, ix, ip, P) for ig, im, iu, ix, ip in idx])
    rec = dict(geometry_index=a[:, 0], F146=MAGS[a[:, 1]], u0=u0s[a[:, 2]], xi_E=xis[a[:, 3]],
               PtE=P[a[:, 4]] / TE, u_min=out[:, 0], t_u_min=out[:, 1], u_t0=out[:, 2],
               dt_t0=out[:, 3], lambda_cell=lam)
    dest = Path(args.out) / f"umin_{grid}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **rec, rho=UMIN_RHO, tE=TE, t0=T0_DEFAULT, blend_ratio=0.0,
                        photometric_filter=PHOTOMETRIC_FILTER, **provenance())

    # u at the epoch nearest t0 must be u0 (convention check; epochs within 0.01 d)
    near = np.isfinite(rec["u_t0"]) & (np.abs(rec["dt_t0"]) < 0.01)
    if near.any():
        dev = np.abs(rec["u_t0"][near] / rec["u0"][near] - 1.0)
        print(f"convention check, |u(t0)/u0 - 1| over {int(near.sum())} cases: max {dev.max():.2e}")
    print("\nF146 u0     xi_E    P/tE  | med u_min/u0  min u_min | frac(u_min < 3 rho), rho = "
          + " ".join(f"{r:.1e}" for r in UMIN_RHO) + " | med lambda")
    for m in MAGS:
        for u0 in u0s:
            for xi in np.unique(rec["xi_E"][rec["u0"] == u0]):
                for pte in np.unique(rec["PtE"]):
                    s = ((rec["F146"] == m) & (rec["u0"] == u0) & (rec["xi_E"] == xi)
                         & (rec["PtE"] == pte) & np.isfinite(rec["u_min"]))
                    if not s.any():
                        continue
                    um = rec["u_min"][s]
                    fr = " ".join(f"{np.mean(um < 3 * r):7.2f}" for r in UMIN_RHO)
                    print(f"{m:4.0f} {u0:<6.3g} {xi:<7.3g} {pte:<6.3g}| {np.median(um / u0):12.3f}"
                          f"  {um.min():9.2e} | {fr} | {np.nanmedian(rec['lambda_cell'][s]):9.3g}")
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
    g.add_argument("--umin", action="store_true",
                   help="finite-source check: closest approach of the luminous source (no fits)")
    ap.add_argument("--umin-limit", type=int, default=None,
                    help="--umin: only the first N simulations (quick test)")
    args = ap.parse_args()
    if args.out is None:
        args.out = GRIDS[args.grid]["out"]
    if args.pilot:
        pilot(args)
    elif args.list:
        P = period_grid(args.fit_policy, args.grid)
        nt = len(tasks(args.grid))
        print(f"grid {args.grid}: {len(GEO)} geometries x {len(blends(args.grid))} g x "
              f"{len(mags(args.grid))} F146 x "
              f"{len(GRIDS[args.grid]['u0'])} u0 x {n_xi(args.grid)} xi = {nt} tasks, "
              f"x {len(P)} P = {nt * len(P)} cases")
    elif args.summarize:
        summarize(args)
    elif args.umin:
        umin(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
