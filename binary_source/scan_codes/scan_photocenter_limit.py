#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import multiprocessing as mp

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# ============================================================
# Import
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

P_over_tE = 1.0

P = (
    P_over_tE
    * tE_true
)

P_grid = np.array(
    [P],
    dtype=float,
)


# ============================================================
# Time sampling
# ============================================================

N_time = 5000

t = np.linspace(
    t0_true - 5.0 * tE_true,
    t0_true + 5.0 * tE_true,
    N_time,
)


# ============================================================
# Geometry
# ============================================================

phi_true = 0.0
i_true = np.pi / 2.0
theta_true = 0.0


# ============================================================
# Fixed total source mass
#
# Only q_M matters dynamically here because xi_rel is explicitly
# overridden.
# ============================================================

Mtot_source = 3.0

rEhat_AU = 5.0


# ============================================================
# q_M grid
# ============================================================

N_qM = 61

qM_grid = np.logspace(
    -4,
    0,
    N_qM,
)


# ============================================================
# Controlled xi_rel / u0 values
# ============================================================

xi_over_u0_grid = np.array(
    [
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
    ],
    dtype=float,
)

N_xi = len(
    xi_over_u0_grid
)


# ============================================================
# Only two q_f families are required
# ============================================================

families = [
    "dark",
    "photocenter_cancel",
]

N_family = len(
    families
)


# ============================================================
# Output
# ============================================================

home = Path.home()

output_dir = (
    home
    / "binary_source"
    / "results"
    / "scan_photocenter_small_xi"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Parallel
# ============================================================

N_cpu = os.cpu_count() or 1

N_workers = max(
    1,
    N_cpu - 1,
)

overwrite = False


# ============================================================
# Masses
# ============================================================

def masses_from_qM(qM):

    M1 = (
        Mtot_source
        / (1.0 + qM)
    )

    M2 = (
        qM
        * M1
    )

    return M1, M2


# ============================================================
# Worker
# ============================================================

def run_one(task):

    (
        i_xi,
        xi_over_u0,
        i_qM,
        qM,
        i_family,
        family,
    ) = task

    qM = float(qM)

    M1, M2 = masses_from_qM(
        qM
    )


    # --------------------------------------------------------
    # q_f
    # --------------------------------------------------------

    if family == "dark":

        qf = 0.0

    elif family == "photocenter_cancel":

        qf = qM

    else:

        raise ValueError(
            family
        )


    # --------------------------------------------------------
    # Controlled relative orbital amplitude
    #
    # xi_rel / u0 = requested value
    # --------------------------------------------------------

    xi_rel = (
        xi_over_u0
        * u0_true
    )


    # --------------------------------------------------------
    # File
    # --------------------------------------------------------

    outfile = (
        output_dir
        / (
            f"xi_{i_xi:02d}"
            f"_qM_{i_qM:03d}"
            f"_{family}.npz"
        )
    )


    if (
        outfile.exists()
        and not overwrite
    ):

        with np.load(
            outfile,
            allow_pickle=False,
        ) as d:

            return {
                "status": "SKIPPED",
                "i_xi": i_xi,
                "i_qM": i_qM,
                "i_family": i_family,
                "D": float(
                    d["D"][0]
                ),
                "RMS": float(
                    d["RMS"][0]
                ),
                "DTE": float(
                    d["DTE"][0]
                ),
                "DU0": float(
                    d["DU0"][0]
                ),
                "SUCCESS": bool(
                    d["SUCCESS"][0]
                ),
            }


    try:

        run_grid_and_save_npz_kepler(

            out_npz_path=str(
                outfile
            ),

            t=t,

            t0_true=t0_true,

            u0_true=u0_true,

            tE_true=tE_true,

            phi_true=phi_true,

            i_true=i_true,

            theta_true=theta_true,

            qflux_true=qf,

            M1_Msun=M1,

            M2_Msun=M2,

            rEhat_AU=rEhat_AU,

            P_grid=P_grid,

            msource_true=24.0,

            mtotal_true=24.0,

            # ================================================
            # KEY CONTROLLED PARAMETER
            # ================================================

            override_xiE=xi_rel,

            set_flux_from_truth_photometry=True,

            rms_on_magnification=True,

            store_curves=False,
        )


        with np.load(
            outfile,
            allow_pickle=False,
        ) as d:

            return {
                "status": "DONE",
                "i_xi": i_xi,
                "i_qM": i_qM,
                "i_family": i_family,
                "D": float(
                    d["D"][0]
                ),
                "RMS": float(
                    d["RMS"][0]
                ),
                "DTE": float(
                    d["DTE"][0]
                ),
                "DU0": float(
                    d["DU0"][0]
                ),
                "SUCCESS": bool(
                    d["SUCCESS"][0]
                ),
            }


    except Exception as error:

        return {
            "status": "FAILED",
            "i_xi": i_xi,
            "i_qM": i_qM,
            "i_family": i_family,
            "error": repr(error),
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


    # --------------------------------------------------------
    # Arrays
    #
    # [xi/u0, qM, family]
    # --------------------------------------------------------

    shape = (
        N_xi,
        N_qM,
        N_family,
    )

    D = np.full(
        shape,
        np.nan,
    )

    RMS = np.full(
        shape,
        np.nan,
    )

    DTE = np.full(
        shape,
        np.nan,
    )

    DU0 = np.full(
        shape,
        np.nan,
    )

    SUCCESS = np.zeros(
        shape,
        dtype=bool,
    )


    # --------------------------------------------------------
    # Tasks
    # --------------------------------------------------------

    tasks = []

    for i_xi, xi_over_u0 in enumerate(
        xi_over_u0_grid
    ):

        for i_qM, qM in enumerate(
            qM_grid
        ):

            for i_family, family in enumerate(
                families
            ):

                tasks.append(
                    (
                        i_xi,
                        xi_over_u0,
                        i_qM,
                        qM,
                        i_family,
                        family,
                    )
                )


    print(
        f"Total fits = {len(tasks)}"
    )

    print(
        f"Workers = {N_workers}"
    )


    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    start = time.time()

    with ProcessPoolExecutor(
        max_workers=N_workers
    ) as executor:

        futures = [
            executor.submit(
                run_one,
                task,
            )
            for task in tasks
        ]

        for k, future in enumerate(
            as_completed(
                futures
            ),
            start=1,
        ):

            result = future.result()

            if result[
                "status"
            ] != "FAILED":

                i_xi = result[
                    "i_xi"
                ]

                i_qM = result[
                    "i_qM"
                ]

                i_family = result[
                    "i_family"
                ]

                D[
                    i_xi,
                    i_qM,
                    i_family,
                ] = result[
                    "D"
                ]

                RMS[
                    i_xi,
                    i_qM,
                    i_family,
                ] = result[
                    "RMS"
                ]

                DTE[
                    i_xi,
                    i_qM,
                    i_family,
                ] = result[
                    "DTE"
                ]

                DU0[
                    i_xi,
                    i_qM,
                    i_family,
                ] = result[
                    "DU0"
                ]

                SUCCESS[
                    i_xi,
                    i_qM,
                    i_family,
                ] = result[
                    "SUCCESS"
                ]


            print(
                f"[{k:4d}/{len(tasks):4d}] "
                f"{result['status']}"
            )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary_file = (
        output_dir
        / "summary_photocenter_small_xi.npz"
    )

    np.savez_compressed(

        summary_file,

        xi_over_u0_grid=xi_over_u0_grid,

        qM_grid=qM_grid,

        families=np.array(
            families
        ),

        D=D,

        RMS=RMS,

        DTE=DTE,

        DU0=DU0,

        SUCCESS=SUCCESS,

        u0_true=u0_true,

        tE_true=tE_true,

        P=P,

        P_over_tE=P_over_tE,

        Mtot_source=Mtot_source,

        rEhat_AU=rEhat_AU,
    )


    print()
    print(
        "Saved:",
        summary_file,
    )

    print(
        f"Elapsed = "
        f"{(time.time() - start)/60:.1f} min"
    )
