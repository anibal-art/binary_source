#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intrinsic BSPL--PSPL maps in the exact variables of the one-luminous-source
problem: (u0, xi_E, P/tE), marginalised over the projected orbital geometry.

For one luminous source (q_f = 0) the magnification, written as a function of
tau = (t - t0)/tE, depends only on u0, xi_E, P/tE and the orientation
(alpha, phi, i). Masses, q_M, rEhat and tE enter only through

    xi_E = q_M / (1 + q_M) * a_rel / rEhat .

Every fit is done with degeneracy_fit.run_grid_and_save_npz_kepler, the same
routine used for all other intrinsic maps of the paper. That routine takes
the RELATIVE amplitude xi_rel through ``override_xiE``; we pass

    override_xiE = xi_E * (1 + q_M) / q_M

so that the luminous star moves with amplitude xi_E. Because only xi_E
matters for q_f = 0, the choice of (M1, M2) is irrelevant; we keep the
paper's M1 = 2, M2 = 1 Msun. tE is also irrelevant; we use tE = 30 d.

One call of run_grid_and_save_npz_kepler scans P/tE for a fixed
(u0, xi_E, geometry) and writes one NPZ. ``--summarize`` collects all NPZ
files into one summary with medians and 16/84th percentiles over geometry.

Geometry: index 0 is the reference geometry of the main text,
(theta, phi, i) = (0, 0, pi/2). Indices 1..N are isotropic random
orientations: theta, phi uniform in [0, 2 pi), sin(i) uniform in [-1, 1]
(pyLIMA's i is the complement of the usual inclination, so uniform sin(i)
is uniform cos(inclination)). The random geometries are drawn once from a
fixed seed and stored in the output.

Usage (from the repository root):
    python binary_source/scan_codes/scan_xiE_ptE.py --estimate
    python binary_source/scan_codes/scan_xiE_ptE.py --smoke
    python binary_source/scan_codes/scan_xiE_ptE.py
    python binary_source/scan_codes/scan_xiE_ptE.py --summarize
"""

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SCAN_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCAN_DIR.parent
for p in (SOURCE_DIR, SCAN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from degeneracy_fit import FIT_OBJECTIVE, run_grid_and_save_npz_kepler  # noqa: E402
from final_config import (  # noqa: E402
    CODE_COMMIT, CODE_DIRTY, FINAL_N_TIME, MAX_WORKERS, final_output_dir,
)

# ============================================================
# Configuration (same physical defaults as run_final_scans.py)
# ============================================================

T0 = 50.0
TE = 30.0                      # irrelevant for the dimensionless maps
WINDOW_TE = 3.5
M1, M2 = 2.0, 1.0              # irrelevant for q_f = 0 once xi_E is fixed
Q_M = M2 / M1
REHAT_AU = 5.0                 # irrelevant once xi_E is fixed
MSOURCE = 24.0
MTOTAL = 24.0

U0_VALUES = np.array([0.03, 0.1, 0.3])
N_XI = 40
N_PTE = 40
XI_E_GRID = np.logspace(-3.0, 0.5, N_XI)        # amplitude of the luminous star
PTE_GRID = np.logspace(np.log10(0.3), np.log10(3000.0), N_PTE)
N_GEOMETRIES = 24                                # random, plus the reference one
GEOMETRY_SEED = 20261030

SMOKE = dict(u0=np.array([0.1]), xi=np.logspace(-2, 0, 3),
             pte=np.logspace(0, 2, 4), n_geo=2)

OUTPUT_NAME = "xiE_ptE_maps"


def geometries(n_random, seed=GEOMETRY_SEED):
    """Rows (theta, phi, i). Row 0 is the main-text reference geometry."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_random)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_random)
    inc = np.arcsin(rng.uniform(-1.0, 1.0, n_random))
    ref = np.array([[0.0, 0.0, np.pi / 2.0]])
    return np.vstack([ref, np.c_[theta, phi, inc]])


def time_grid():
    return np.linspace(T0 - WINDOW_TE * TE, T0 + WINDOW_TE * TE, FINAL_N_TIME)


def outfile_for(directory, i_u0, i_xi, i_geo):
    return Path(directory) / f"u0_{i_u0:02d}" / f"xi_{i_xi:03d}_geo_{i_geo:03d}.npz"


def valid_npz(path, expected_P):
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            return (np.allclose(d["P_grid"], expected_P, rtol=1e-12, atol=0.0)
                    and str(d["code_commit"].item()) == CODE_COMMIT
                    and not bool(d["code_dirty"].item())
                    and str(d["fit_objective"].item()) == FIT_OBJECTIVE)
    except Exception:
        return False


def worker(task):
    i_u0, u0, i_xi, xi_e, i_geo, (theta, phi, inc), pte, directory = task
    out = outfile_for(directory, i_u0, i_xi, i_geo)
    P_grid = np.asarray(pte) * TE
    if valid_npz(out, P_grid):
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    run_grid_and_save_npz_kepler(
        out_npz_path=str(out),
        t=time_grid(),
        t0_true=T0,
        u0_true=float(u0),
        tE_true=TE,
        phi_true=float(phi),
        i_true=float(inc),
        qflux_true=0.0,
        theta_true=float(theta),
        M1_Msun=M1,
        M2_Msun=M2,
        rEhat_AU=REHAT_AU,
        P_grid=P_grid,
        msource_true=MSOURCE,
        mtotal_true=MTOTAL,
        override_xiE=float(xi_e) * (1.0 + Q_M) / Q_M,    # xi_rel giving xi_E
        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,
        store_curves=False,
    )
    return "done"


def build_tasks(u0_values, xi_grid, pte_grid, geo, directory):
    return [(i_u0, float(u0), i_xi, float(xi), i_g, tuple(g), pte_grid, str(directory))
            for i_u0, u0 in enumerate(u0_values)
            for i_xi, xi in enumerate(xi_grid)
            for i_g, g in enumerate(geo)]


def run(tasks, max_workers):
    done = skipped = failed = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for k, f in enumerate(as_completed(futures), start=1):
            try:
                if f.result() == "done":
                    done += 1
                else:
                    skipped += 1
            except Exception as error:      # noqa: BLE001
                failed += 1
                print("ERROR:", repr(error))
            if k % 50 == 0 or k == len(tasks):
                el = time.time() - t_start
                print(f"[{k}/{len(tasks)}] done={done} skipped={skipped} "
                      f"failed={failed} elapsed={el/60:.1f} min", flush=True)
    if failed:
        raise RuntimeError(f"{failed} tasks failed")


def summarize(u0_values, xi_grid, pte_grid, geo, directory):
    shape = (len(u0_values), len(xi_grid), len(geo), len(pte_grid))
    D = np.full(shape, np.nan)
    DTE = np.full(shape, np.nan)
    DU0 = np.full(shape, np.nan)
    DT0 = np.full(shape, np.nan)
    missing = 0
    for i_u0, u0 in enumerate(u0_values):
        for i_xi in range(len(xi_grid)):
            for i_g in range(len(geo)):
                f = outfile_for(directory, i_u0, i_xi, i_g)
                if not f.exists():
                    missing += 1
                    continue
                with np.load(f, allow_pickle=False) as d:
                    ok = d["SUCCESS"]
                    D[i_u0, i_xi, i_g] = np.where(ok, d["D"], np.nan)
                    DTE[i_u0, i_xi, i_g] = np.where(ok, d["DTE"] / TE, np.nan)
                    DU0[i_u0, i_xi, i_g] = np.where(ok, d["DU0"] / u0, np.nan)
                    DT0[i_u0, i_xi, i_g] = np.where(ok, d["DT0"] / TE, np.nan)
    rnd = slice(1, None)                    # random geometries only
    pct = lambda a: np.nanpercentile(a[:, :, rnd], [16, 50, 84], axis=2)  # noqa: E731
    out = Path(directory) / "summary_xiE_ptE.npz"
    np.savez(
        out,
        u0=u0_values, xi_E=xi_grid, P_over_tE=pte_grid, geometries=geo,
        D_all=D, dtE_over_tE_all=DTE, du0_over_u0_all=DU0, dt0_over_tE_all=DT0,
        D_p16_p50_p84=pct(D),
        abs_dtE_over_tE_p16_p50_p84=pct(np.abs(DTE)),
        D_reference_geometry=D[:, :, 0], dtE_over_tE_reference_geometry=DTE[:, :, 0],
        q_mass_used=Q_M, tE_used=TE, n_random_geometries=len(geo) - 1,
        code_commit=np.array(CODE_COMMIT), code_dirty=np.bool_(CODE_DIRTY),
        fit_objective=np.array(FIT_OBJECTIVE),
    )
    print(f"summary written: {out}  (missing files: {missing})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny grid to test the pipeline")
    ap.add_argument("--estimate", action="store_true",
                    help="time a few tasks and estimate the full run")
    ap.add_argument("--summarize", action="store_true", help="collect NPZ files")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--u0", type=float, nargs="+", default=None,
                    help="u0 values instead of U0_VALUES; written to a separate "
                         "directory xiE_ptE_maps_u0_<values> (files are indexed by "
                         "the position of u0 in the list)")
    ap.add_argument("--refine", type=int, default=1,
                    help="k > 1 inserts k-1 nodes between the nodes of XI_E_GRID and "
                         "PTE_GRID (same range; every original node is kept)")
    args = ap.parse_args()

    if args.smoke:
        u0s, xis, ptes, geo = SMOKE["u0"], SMOKE["xi"], SMOKE["pte"], geometries(SMOKE["n_geo"])
        directory = final_output_dir(OUTPUT_NAME + "_smoke")
    elif args.u0:
        u0s, xis, ptes, geo = (np.array(args.u0, dtype=float), XI_E_GRID, PTE_GRID,
                               geometries(N_GEOMETRIES))
        tag = "_".join(f"{u:g}".replace(".", "p") for u in u0s)
        if args.refine > 1:
            k = args.refine
            xis = np.logspace(np.log10(XI_E_GRID[0]), np.log10(XI_E_GRID[-1]),
                              k * (len(XI_E_GRID) - 1) + 1)
            ptes = np.logspace(np.log10(PTE_GRID[0]), np.log10(PTE_GRID[-1]),
                               k * (len(PTE_GRID) - 1) + 1)
            assert np.allclose(xis[::k], XI_E_GRID) and np.allclose(ptes[::k], PTE_GRID)
            tag += f"_refine{k}"
        directory = final_output_dir(f"{OUTPUT_NAME}_u0_{tag}")
    else:
        u0s, xis, ptes, geo = U0_VALUES, XI_E_GRID, PTE_GRID, geometries(N_GEOMETRIES)
        directory = final_output_dir(OUTPUT_NAME)
    if CODE_DIRTY and not args.smoke:
        print("WARNING: repository has uncommitted changes; outputs will be "
              "marked dirty and not reused on a later run.")

    if args.summarize:
        summarize(u0s, xis, ptes, geo, directory)
        return

    tasks = build_tasks(u0s, xis, ptes, geo, directory)
    print(f"output   : {directory}")
    print(f"grid     : {len(u0s)} u0 x {len(xis)} xi_E x {len(geo)} geometries "
          f"x {len(ptes)} P/tE = {len(tasks) * len(ptes)} PSPL fits")
    print(f"commit   : {CODE_COMMIT} (dirty={CODE_DIRTY})  workers={args.workers}")

    if args.estimate:
        rng = np.random.default_rng(0)
        sample = [tasks[i] for i in rng.choice(len(tasks), size=min(4, len(tasks)),
                                               replace=False)]
        est_dir = final_output_dir(OUTPUT_NAME + "_estimate")
        sample = [t[:-1] + (str(est_dir),) for t in sample]
        t1 = time.time()
        for t in sample:
            worker(t)
        per_task = (time.time() - t1) / len(sample)
        total_h = per_task * len(tasks) / 3600.0
        print(f"time per task ({len(ptes)} fits): {per_task:.1f} s")
        print(f"estimated total: {total_h:.1f} CPU-h, "
              f"{total_h / args.workers:.1f} h with {args.workers} workers")
        return

    run(tasks, args.workers)
    summarize(u0s, xis, ptes, geo, directory)


if __name__ == "__main__":
    main()
