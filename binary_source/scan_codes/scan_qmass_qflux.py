#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Joint scan in source mass ratio q_M and flux ratio q_f.

We keep fixed:

    Mtot = M1 + M2

and vary

    q_M = M2 / M1
    q_f = F2 / F1

For each (q_M, q_f), the code evaluates several P/tE values and fits
each BSPL light curve with a PSPL model.

Main output:
    D[q_M, q_f, P]
    RMS
    MAXABS
    TDEV
    DT0
    DU0
    DTE
    Q_A

The scan is parallelized over the (q_M, q_f) pairs.
"""

# ============================================================
# Imports
# ============================================================

import os
import sys
import time
import traceback
import multiprocessing as mp

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# ============================================================
# Import project function
# ============================================================


import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from degeneracy_fit import run_grid_and_save_npz_kepler


# ============================================================
# Base event
# ============================================================

t0_true = 50.0
u0_true = 0.1
tE_true = 150.0


# ============================================================
# Orbital geometry
#
# Fixed in this experiment.
# Geometry can later be marginalized independently.
# ============================================================

phi_true = 0.0

i_true = np.pi / 2.0

theta_true = 0.0


# ============================================================
# Physical binary-source configuration
# ============================================================

# Fixed total source mass.
#
# This is important because:
#
#       a_rel^3 ∝ Mtot P^2
#
# so at fixed P, varying q_M does not change a_rel.

Mtot_source = 3.0      # Msun

rEhat_AU = 5.0


# ============================================================
# q_M grid
# ============================================================

N_qM = 41

qM_grid = np.logspace(
    -4,
    0,
    N_qM,
)


# ============================================================
# q_f grid
#
# Include q_f = 0 explicitly.
#
# The remaining grid is identical to q_M so that q_f=q_M
# is sampled exactly.
# ============================================================

qf_grid = np.concatenate(
    [
        np.array([0.0]),
        qM_grid.copy(),
    ]
)

N_qf = len(qf_grid)


# ============================================================
# Periods
#
# Initial experiment:
# three representative temporal regimes.
# ============================================================

P_over_tE_grid = np.array(
    [
        0.3,
        1.0,
        3.0,
    ],
    dtype=float,
)

P_grid = (
    P_over_tE_grid
    * tE_true
)

N_P = len(P_grid)


# ============================================================
# Time sampling
# ============================================================

N_time = 5000

window_tE = 5.0

t = np.linspace(
    t0_true - window_tE * tE_true,
    t0_true + window_tE * tE_true,
    N_time,
)


# ============================================================
# Photometric parameters
#
# These are retained for compatibility with degeneracy_fit.
#
# The main quantity of interest in this experiment is D,
# which is computed in magnification.
# ============================================================

msource_true = 24.0
mtotal_true = 24.0


# ============================================================
# Output directory
# ============================================================

home = Path.home()

output_dir = (
    home
    / "binary_source"
    / "results"
    / f"scan_qM_qf_Mtotfixed_tE{int(tE_true)}"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Parallelization
# ============================================================

N_cpu = os.cpu_count() or 1

N_workers = max(
    1,
    N_cpu - 1,
)

overwrite = False


# ============================================================
# Functions
# ============================================================

def masses_from_qM(qM):
    """
    Convert q_M and fixed Mtot into M1 and M2.

        q_M = M2/M1
        Mtot = M1 + M2
    """

    M1 = (
        Mtot_source
        / (1.0 + qM)
    )

    M2 = (
        qM
        * M1
    )

    return M1, M2


def output_filename(i_qM, i_qf):
    """
    One NPZ per (q_M, q_f).
    """

    return (
        output_dir
        / (
            f"scan_qM_{i_qM:03d}"
            f"_qf_{i_qf:03d}.npz"
        )
    )


def is_valid_file(filename):
    """
    Check whether an existing file can be reused.
    """

    if not filename.exists():
        return False

    try:

        with np.load(
            filename,
            allow_pickle=False,
        ) as d:

            required = [
                "P_grid",
                "D",
                "RMS",
                "SUCCESS",
                "truth",
            ]

            for key in required:

                if key not in d.files:
                    return False

            if len(d["P_grid"]) != N_P:
                return False

        return True

    except Exception:

        return False


def load_results(filename):
    """
    Extract the quantities needed for the summary cube.
    """

    with np.load(
        filename,
        allow_pickle=False,
    ) as d:

        result = {

            "D":
                np.asarray(
                    d["D"],
                    dtype=float,
                ),

            "RMS":
                np.asarray(
                    d["RMS"],
                    dtype=float,
                ),

            "MAXABS":
                np.asarray(
                    d["MAXABS"],
                    dtype=float,
                ),

            "TDEV":
                np.asarray(
                    d["TDEV"],
                    dtype=float,
                ),

            "DT0":
                np.asarray(
                    d["DT0"],
                    dtype=float,
                ),

            "DU0":
                np.asarray(
                    d["DU0"],
                    dtype=float,
                ),

            "DTE":
                np.asarray(
                    d["DTE"],
                    dtype=float,
                ),

            "Q_A":
                np.asarray(
                    d["Q_A"],
                    dtype=float,
                ),

            "SUCCESS":
                np.asarray(
                    d["SUCCESS"],
                    dtype=bool,
                ),

            # IMPORTANT:
            # Despite its historical name, this is
            #
            #       a_rel / rEhat = xi_rel
            #
            # in the current degeneracy_fit implementation.
            "XI_REL":
                np.asarray(
                    d["xiE_of_P"],
                    dtype=float,
                ),
        }

    return result


def run_one_pair(task):
    """
    Worker for one pair (q_M, q_f).
    """

    (
        i_qM,
        qM,
        i_qf,
        qf,
    ) = task

    qM = float(qM)
    qf = float(qf)

    filename = output_filename(
        i_qM,
        i_qf,
    )

    # --------------------------------------------------------
    # Resume previous run
    # --------------------------------------------------------

    if (
        not overwrite
        and is_valid_file(filename)
    ):

        result = load_results(
            filename
        )

        return {
            "status": "SKIPPED",
            "i_qM": i_qM,
            "i_qf": i_qf,
            "qM": qM,
            "qf": qf,
            "file": str(filename),
            **result,
        }

    # --------------------------------------------------------
    # Convert q_M -> physical component masses
    # --------------------------------------------------------

    M1, M2 = masses_from_qM(
        qM
    )

    try:

        run_grid_and_save_npz_kepler(

            out_npz_path=str(
                filename
            ),

            t=t,

            # -----------------------------------------------
            # base PSPL trajectory
            # -----------------------------------------------

            t0_true=t0_true,

            u0_true=u0_true,

            tE_true=tE_true,

            # -----------------------------------------------
            # geometry
            # -----------------------------------------------

            phi_true=phi_true,

            i_true=i_true,

            theta_true=theta_true,

            # -----------------------------------------------
            # flux ratio
            # -----------------------------------------------

            qflux_true=qf,

            # -----------------------------------------------
            # masses
            # -----------------------------------------------

            M1_Msun=M1,

            M2_Msun=M2,

            rEhat_AU=rEhat_AU,

            # -----------------------------------------------
            # period cases
            # -----------------------------------------------

            P_grid=P_grid,

            # -----------------------------------------------
            # photometric wrapper
            # -----------------------------------------------

            msource_true=msource_true,

            mtotal_true=mtotal_true,

            # -----------------------------------------------
            # Kepler-consistent xi_rel
            # -----------------------------------------------

            override_xiE=None,

            set_flux_from_truth_photometry=True,

            rms_on_magnification=True,

            # No need to store all curves for thousands
            # of grid points.
            store_curves=False,
        )

        result = load_results(
            filename
        )

        return {
            "status": "DONE",
            "i_qM": i_qM,
            "i_qf": i_qf,
            "qM": qM,
            "qf": qf,
            "M1": M1,
            "M2": M2,
            "file": str(filename),
            **result,
        }

    except Exception as error:

        return {
            "status": "FAILED",
            "i_qM": i_qM,
            "i_qf": i_qf,
            "qM": qM,
            "qf": qf,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "file": str(filename),
        }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    try:

        mp.set_start_method(
            "spawn"
        )

    except RuntimeError:

        pass

    # ========================================================
    # Allocate summary cubes
    # ========================================================

    shape = (
        N_qM,
        N_qf,
        N_P,
    )

    D_cube = np.full(
        shape,
        np.nan,
    )

    RMS_cube = np.full(
        shape,
        np.nan,
    )

    MAXABS_cube = np.full(
        shape,
        np.nan,
    )

    TDEV_cube = np.full(
        shape,
        np.nan,
    )

    DT0_cube = np.full(
        shape,
        np.nan,
    )

    DU0_cube = np.full(
        shape,
        np.nan,
    )

    DTE_cube = np.full(
        shape,
        np.nan,
    )

    QA_cube = np.full(
        shape,
        np.nan,
    )

    SUCCESS_cube = np.zeros(
        shape,
        dtype=bool,
    )

    XI_REL_cube = np.full(
        shape,
        np.nan,
    )


    # ========================================================
    # Build jobs
    # ========================================================

    tasks = [

        (
            i_qM,
            qM,
            i_qf,
            qf,
        )

        for i_qM, qM
        in enumerate(qM_grid)

        for i_qf, qf
        in enumerate(qf_grid)
    ]

    total_jobs = len(tasks)

    total_fits = (
        total_jobs
        * N_P
    )


    # ========================================================
    # Print configuration
    # ========================================================

    print()
    print("=" * 72)
    print("JOINT q_M -- q_f SCAN")
    print("=" * 72)

    print(
        f"Mtot source = "
        f"{Mtot_source:.3f} Msun"
    )

    print(
        f"u0          = "
        f"{u0_true:.4f}"
    )

    print(
        f"tE          = "
        f"{tE_true:.1f} d"
    )

    print(
        f"rEhat       = "
        f"{rEhat_AU:.3f} AU"
    )

    print()

    print(
        f"N(q_M)      = "
        f"{N_qM}"
    )

    print(
        f"N(q_f)      = "
        f"{N_qf}"
    )

    print(
        "P/tE        = ",
        P_over_tE_grid,
    )

    print()

    print(
        f"pair jobs   = "
        f"{total_jobs}"
    )

    print(
        f"total fits  = "
        f"{total_fits}"
    )

    print(
        f"workers     = "
        f"{N_workers}"
    )

    print(
        f"output      = "
        f"{output_dir}"
    )

    print("=" * 72)
    print()


    # ========================================================
    # Run
    # ========================================================

    start_time = time.time()

    done = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=N_workers
    ) as executor:

        futures = {

            executor.submit(
                run_one_pair,
                task,
            ): task

            for task in tasks
        }

        for future in as_completed(
            futures
        ):

            result = future.result()

            status = result[
                "status"
            ]

            i_qM = result[
                "i_qM"
            ]

            i_qf = result[
                "i_qf"
            ]

            qM = result[
                "qM"
            ]

            qf = result[
                "qf"
            ]

            if status in (
                "DONE",
                "SKIPPED",
            ):

                D_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["D"]

                RMS_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["RMS"]

                MAXABS_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["MAXABS"]

                TDEV_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["TDEV"]

                DT0_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["DT0"]

                DU0_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["DU0"]

                DTE_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["DTE"]

                QA_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["Q_A"]

                SUCCESS_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["SUCCESS"]

                XI_REL_cube[
                    i_qM,
                    i_qf,
                    :
                ] = result["XI_REL"]

                if status == "DONE":
                    done += 1

                else:
                    skipped += 1

            else:

                failed += 1

                print()
                print(
                    "FAILED:"
                )

                print(
                    result[
                        "traceback"
                    ]
                )

            completed = (
                done
                + skipped
                + failed
            )

            print(
                f"[{completed:4d}/"
                f"{total_jobs:4d}] "
                f"{status:7s} "
                f"qM={qM:9.3e} "
                f"qf={qf:9.3e}"
            )


    # ========================================================
    # Derived physical quantities
    # ========================================================

    # --------------------------------------------------------
    # Component masses
    # --------------------------------------------------------

    M1_grid = (
        Mtot_source
        / (1.0 + qM_grid)
    )

    M2_grid = (
        qM_grid
        * M1_grid
    )


    # --------------------------------------------------------
    # xi_rel
    #
    # Since Mtot and P are fixed for every qM/qf pair,
    # xi_rel should be identical across the qM-qf plane.
    #
    # Median makes this robust to occasional failed points.
    # --------------------------------------------------------

    xi_rel_of_P = np.nanmedian(
        XI_REL_cube,
        axis=(0, 1),
    )


    # --------------------------------------------------------
    # Individual barycentric amplitudes
    #
    # xi_E1 = qM/(1+qM) * xi_rel
    #
    # xi_E2 = 1/(1+qM) * xi_rel
    # --------------------------------------------------------

    xiE1 = (
        qM_grid[:, None, None]
        / (
            1.0
            + qM_grid[:, None, None]
        )
        * xi_rel_of_P[
            None,
            None,
            :
        ]
    )

    xiE2 = (
        1.0
        / (
            1.0
            + qM_grid[:, None, None]
        )
        * xi_rel_of_P[
            None,
            None,
            :
        ]
    )


    # --------------------------------------------------------
    # First-order photocenter orbital amplitude
    #
    # xi_phot =
    #
    # |q_M - q_f|
    # ----------------------------  xi_rel
    # (1+q_M)(1+q_f)
    #
    # This vanishes on q_f = q_M.
    # --------------------------------------------------------

    QM = qM_grid[
        :,
        None,
        None,
    ]

    QF = qf_grid[
        None,
        :,
        None,
    ]

    XI = xi_rel_of_P[
        None,
        None,
        :,
    ]

    xi_phot_cube = (
        np.abs(
            QM - QF
        )
        / (
            (1.0 + QM)
            * (1.0 + QF)
        )
        * XI
    )


    # ========================================================
    # Save one compact summary file
    # ========================================================

    summary_file = (
        output_dir
        / "summary_qM_qf.npz"
    )

    np.savez_compressed(

        summary_file,

        # grids
        qM_grid=qM_grid,
        qf_grid=qf_grid,

        P_grid=P_grid,
        P_over_tE_grid=P_over_tE_grid,

        # physical masses
        Mtot_source=Mtot_source,
        M1_grid=M1_grid,
        M2_grid=M2_grid,

        # fixed event
        t0_true=t0_true,
        u0_true=u0_true,
        tE_true=tE_true,

        rEhat_AU=rEhat_AU,

        phi_true=phi_true,
        i_true=i_true,
        theta_true=theta_true,

        # orbital scales
        xi_rel_of_P=xi_rel_of_P,
        xiE1=xiE1,
        xiE2=xiE2,
        xi_phot=xi_phot_cube,

        # metrics
        D=D_cube,
        RMS=RMS_cube,
        MAXABS=MAXABS_cube,
        TDEV=TDEV_cube,

        # biases
        DT0=DT0_cube,
        DU0=DU0_cube,
        DTE=DTE_cube,

        # Roman-like quantity already stored
        Q_A=QA_cube,

        SUCCESS=SUCCESS_cube,
    )


    # ========================================================
    # End
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 72)
    print("FINISHED")
    print("=" * 72)

    print(
        f"DONE     = {done}"
    )

    print(
        f"SKIPPED  = {skipped}"
    )

    print(
        f"FAILED   = {failed}"
    )

    print(
        f"time      = "
        f"{elapsed / 60.0:.1f} min"
    )

    print(
        f"summary   = "
        f"{summary_file}"
    )

    print("=" * 72)
