#!/usr/bin/env python3
"""
Roman F146 maps in the (xi_E, P/t_E) plane for one luminous source,
with PSPL and PSPL+parallax fits to the SAME simulated light curve.

Grid and orientations as in scan_xiE_ptE.py (every second xi_E and P/t_E
node, so the maps align with the intrinsic D maps). Simulation and fits:
run_roman_bspl_asimov.run_case, i.e. roman_rubin (Roman cadence and
seasons, ephemerides, F146/Pandeia uncertainties) and pyLIMA TRF for

    A: PSPL, no parallax
    B: PSPL + annual parallax

Production light curves carry photometric noise: ONE realization per
(cell, orientation, magnitude), with a deterministic seed. The 24 isotropic
orientations of a cell are 24 independent events, so each cell gives the
fraction of events above a Delta chi2 level. Delta chi2 is computed with
respect to the TRUE BSPL model (not fitted), a lower bound of the BSPL-fit
likelihood-ratio statistic. Uncertainties are pyLIMA's covariance, which is
the appropriate estimate for noisy data.

Roman results depend on t_E through the cadence and the seasons, so t_E is
a grid axis.

Usage
-----
    python scan_roman_xiE_ptE.py --pilot --workers 40
    python scan_roman_xiE_ptE.py --estimate --seconds-per-case 20
    python scan_roman_xiE_ptE.py --shard $SLURM_ARRAY_TASK_ID --n-shards 8 --workers 40
    python scan_roman_xiE_ptE.py --summarize
"""

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCAN_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCAN_DIR.parent
REPO_ROOT = SOURCE_DIR.parent
for p in (REPO_ROOT, SOURCE_DIR, SCAN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from final_config import CODE_COMMIT, CODE_DIRTY, final_output_dir  # noqa: E402
from scan_xiE_ptE import (  # noqa: E402
    N_GEOMETRIES, PTE_GRID, Q_M, XI_E_GRID, geometries,
)
from binary_source.analysis import roman_bspl_roman_rubin as bs  # noqa: E402
from binary_source.analysis.run_roman_bspl_asimov import run_case  # noqa: E402

# ============================================================
# Grid
# ============================================================

U0_VALUES = np.array([0.03, 0.1, 0.3])
TE_VALUES = np.array([30.0, 100.0, 200.0])
MAG_VALUES = np.array([19.0, 21.0, 23.0])        # F146 (AB)
XI_GRID = XI_E_GRID[::2]                          # 20 nodes
PT_GRID = PTE_GRID[::2]                           # 20 nodes
GEO = geometries(N_GEOMETRIES)                    # row 0 = reference geometry
FIT_WINDOW_TE = 3.5
DCHI2_LEVELS = np.array([25.0, 50.0, 100.0])      # reference levels until H0 calibration

OUTPUT_NAME = "roman_xiE_ptE_maps"
PILOT_NAME = "roman_xiE_ptE_pilot"

KEYS = [
    "delta_chi2_pspl", "delta_chi2_pspl_parallax",
    "chi2_truth", "chi2_pspl", "chi2_pspl_parallax", "n_data",
    "snr_event", "D_roman_eff",
    "best_t0", "best_u0", "best_tE", "sigma_t0", "sigma_u0", "sigma_tE",
    "best_parallax_tE", "sigma_parallax_tE",
    "best_piEN", "best_piEE", "best_piE", "sigma_piE",
    "max_asimov_flux_difference", "seconds",
]

assert np.isclose(bs.M2_MSUN / bs.M1_MSUN, Q_M), "q_M differs from the intrinsic scan"


def reachable(xi_e, p_te):
    """Same limit as the grey region of the intrinsic maps:
    M1+M2 <= 20 Msun, t_E <= 300 d, rEhat >= 1 AU."""
    a_max = (20.0 * (p_te * 300.0 / 365.25) ** 2) ** (1.0 / 3.0)
    return xi_e <= a_max / 1.0


def task_list():
    return [(iu, it, ix, ig)
            for iu in range(len(U0_VALUES))
            for it in range(len(TE_VALUES))
            for ix in range(len(XI_GRID))
            for ig in range(len(GEO))]


def noise_seed(iu, it, ix, ig, ip, im):
    """Deterministic, unique and < 2**32 (np.random.seed in roman_rubin)."""
    return ((((iu * 10 + it) * 100 + ix) * 100 + ig) * 100 + ip) * 10 + im


def outfile(directory, iu, it, ix, ig):
    return Path(directory) / f"u0_{iu}" / f"tE_{it}" / f"xi_{ix:02d}_geo_{ig:02d}.npz"


def valid(path):
    if not Path(path).exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            return (str(d["code_commit"]) == CODE_COMMIT
                    and not bool(d["code_dirty"])
                    and d["delta_chi2_pspl"].shape == (len(PT_GRID), len(MAG_VALUES)))
    except Exception:
        return False


def one_case(u0, tE, xi_e, p_te, geo_row, mag, seed, asimov=False):
    theta, phi, inc = geo_row
    t1 = time.time()
    r = run_case(
        event_index=int(seed),
        P_days=float(p_te * tE),
        u0=float(u0),
        tE=float(tE),
        source_mag_ab=float(mag),
        fit_window_tE=FIT_WINDOW_TE,
        phi=float(phi),
        inclination=float(inc),
        theta=float(theta),
        xi_rel=None if xi_e is None else float(xi_e) * (1.0 + Q_M) / Q_M,
        asimov=bool(asimov),
    )
    r["seconds"] = time.time() - t1
    return r


def worker(args):
    iu, it, ix, ig, directory = args
    out = outfile(directory, iu, it, ix, ig)
    if valid(out):
        return "skip"
    u0, tE, xi_e = U0_VALUES[iu], TE_VALUES[it], XI_GRID[ix]
    shape = (len(PT_GRID), len(MAG_VALUES))
    res = {k: np.full(shape, np.nan) for k in KEYS}
    seeds = np.zeros(shape, dtype=np.int64)
    err = np.full(shape, "", dtype="<U300")
    for ip, p_te in enumerate(PT_GRID):
        if not reachable(xi_e, p_te):
            continue
        for im, mag in enumerate(MAG_VALUES):
            seeds[ip, im] = noise_seed(iu, it, ix, ig, ip, im)
            try:
                r = one_case(u0, tE, xi_e, p_te, GEO[ig], mag, seeds[ip, im])
                for k in KEYS:
                    res[k][ip, im] = float(r[k])
            except Exception as exc:
                err[ip, im] = f"{type(exc).__name__}: {exc}"[:300]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, **res, error=err, noise_seed=seeds,
        u0=u0, tE=tE, xi_E=xi_e, P_over_tE=PT_GRID, F146=MAG_VALUES,
        geometry=GEO[ig], q_mass=Q_M, fit_window_tE=FIT_WINDOW_TE, asimov=False,
        code_commit=CODE_COMMIT, code_dirty=CODE_DIRTY,
    )
    return "done"


def run_tasks(tasks, directory, workers):
    jobs = [(*t, str(directory)) for t in tasks]
    n_done = n_skip = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, j) for j in jobs]
        for f in as_completed(futs):
            status = f.result()
            n_done += status == "done"
            n_skip += status == "skip"
            if (n_done + n_skip) % 25 == 0 or n_done + n_skip == len(jobs):
                print(f"{n_done + n_skip}/{len(jobs)} tasks "
                      f"(skipped {n_skip}) | {time.time() - t0:.0f} s", flush=True)


def pilot(workers):
    """Sanity checks and timing before the production grid.

    (1) Continuity with the current Roman figure: reference geometry, Kepler
        xi (M1=2, M2=1, rEhat=5 AU), u0=0.1, t_E=30 d, three periods, Asimov.
    (2) The same cases with noise: chi2_truth/N ~ 1, and Delta chi2 close to
        the Asimov value (within a few sqrt(Delta chi2)).
    (3) Timing on production nodes (noise, one isotropic geometry, all t_E).
    """
    print(f"commit {CODE_COMMIT} dirty={CODE_DIRTY}", flush=True)
    cases = []
    for asimov in (True, False):
        cases += [(0.1, 30.0, None, P / 30.0, GEO[0], m, asimov)
                  for P in (30.0, 300.0, 3000.0) for m in MAG_VALUES]
    cases += [(0.1, tE, XI_GRID[10], PT_GRID[k], GEO[1], 21.0, False)
              for tE in TE_VALUES for k in (4, 10, 16)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one_case, *c[:6], 1000 + (n % 9), c[6]): c
                for n, c in enumerate(cases)}
        for f in as_completed(futs):
            c = futs[f]
            try:
                rows.append((c, f.result()))
            except Exception:
                print("FAILED", c, traceback.format_exc(), flush=True)
    rows.sort(key=lambda x: (x[0][2] is not None, not x[0][6], x[0][1], x[0][3], x[0][5]))
    print("\nasimov u0   tE   xi_E     P/tE   F146  dchi2_A    dchi2_B    "
          "chi2_true/N  dtE/tE    sig_tE/tE  piE      sig_piE   sec")
    for c, r in rows:
        u0, tE, xi, pte, _, m, asimov = c
        print(f"{'yes' if asimov else 'no ':>6} {u0:4.2f} {tE:4.0f} "
              f"{('kepler' if xi is None else f'{xi:.3g}'):>7} {pte:7.3g} {m:5.1f} "
              f"{r['delta_chi2_pspl']:10.4g} {r['delta_chi2_pspl_parallax']:10.4g} "
              f"{r['chi2_truth'] / r['n_data']:11.4f} {(r['best_tE'] - tE) / tE:9.3g} "
              f"{r['sigma_tE'] / tE:10.3g} {r['best_piE']:8.3g} {r['sigma_piE']:8.3g} "
              f"{r['seconds']:5.0f}")
    t = np.array([r["seconds"] for c, r in rows if not c[6]])
    print(f"\nseconds per noisy case (simulation + two fits): "
          f"median {np.median(t):.1f}, max {t.max():.1f}")
    d = final_output_dir(PILOT_NAME)
    np.save(Path(d) / "pilot_rows.npy",
            np.array([(c[0], c[1], np.nan if c[2] is None else c[2], c[3], c[5], c[6],
                       *[r[k] for k in KEYS]) for c, r in rows]))
    print(f"saved {d}/pilot_rows.npy")


def estimate(seconds_per_case):
    n = sum(1 for (iu, it, ix, ig) in task_list()
            for p in PT_GRID if reachable(XI_GRID[ix], p)) * len(MAG_VALUES)
    print(f"tasks {len(task_list())}, cases {n} (each: simulation + two fits)")
    print(f"CPU-hours at {seconds_per_case:.0f} s/case: {n * seconds_per_case / 3600:.0f}")


def summarize():
    directory = final_output_dir(OUTPUT_NAME)
    shape = (len(U0_VALUES), len(TE_VALUES), len(MAG_VALUES),
             len(XI_GRID), len(GEO), len(PT_GRID))
    S = {k: np.full(shape, np.nan) for k in KEYS}
    missing = 0
    for iu, it, ix, ig in task_list():
        f = outfile(directory, iu, it, ix, ig)
        if not valid(f):
            missing += 1
            continue
        with np.load(f) as d:
            for k in KEYS:
                S[k][iu, it, :, ix, ig, :] = d[k].T
    print(f"missing task files: {missing}")
    rnd = slice(1, None)                          # isotropic orientations only
    dA = S["delta_chi2_pspl"][:, :, :, :, rnd, :]
    dB = S["delta_chi2_pspl_parallax"][:, :, :, :, rnd, :]
    tE_true = TE_VALUES[None, :, None, None, None, None]
    z_tE = ((S["best_tE"] - tE_true) / S["sigma_tE"])[:, :, :, :, rnd, :]
    z_pi = (S["best_piE"] / S["sigma_piE"])[:, :, :, :, rnd, :]
    frac = {f"frac_dchi2_A_above_{int(L)}": np.nanmean(dA > L, axis=4) for L in DCHI2_LEVELS}
    frac.update({f"frac_dchi2_B_above_{int(L)}": np.nanmean(dB > L, axis=4) for L in DCHI2_LEVELS})
    out = Path(directory) / "summary_roman_xiE_ptE.npz"
    np.savez_compressed(
        out, **{f"{k}_all": v for k, v in S.items()}, **frac,
        delta_chi2_pspl_median=np.nanmedian(dA, axis=4),
        delta_chi2_pspl_parallax_median=np.nanmedian(dB, axis=4),
        # mean of z^2 - 1 over orientations estimates <(bias/sigma)^2>
        tE_bias2_over_sigma2=np.nanmean(z_tE ** 2, axis=4) - 1.0,
        # |piE|/sigma has two degrees of freedom: <z^2> - 2 estimates <(piE/sigma)^2>
        piE2_over_sigma2=np.nanmean(z_pi ** 2, axis=4) - 2.0,
        u0=U0_VALUES, tE=TE_VALUES, F146=MAG_VALUES, xi_E=XI_GRID, P_over_tE=PT_GRID,
        geometries=GEO, dchi2_levels=DCHI2_LEVELS,
        code_commit=CODE_COMMIT, code_dirty=CODE_DIRTY,
        axes="u0, tE, F146, xi_E, geometry, P_over_tE",
    )
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--seconds-per-case", type=float, default=20.0)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.pilot:
        return pilot(a.workers)
    if a.estimate:
        return estimate(a.seconds_per_case)
    if a.summarize:
        return summarize()
    if CODE_DIRTY:
        sys.exit("Refusing production run from a dirty working tree (commit first).")
    tasks = task_list()[a.shard::a.n_shards]
    print(f"commit {CODE_COMMIT} | shard {a.shard}/{a.n_shards} | {len(tasks)} tasks "
          f"| workers {a.workers}", flush=True)
    run_tasks(tasks, final_output_dir(OUTPUT_NAME), a.workers)


if __name__ == "__main__":
    main()
