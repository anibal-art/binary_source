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
# Import degeneracy code
# ============================================================


import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1]

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from degeneracy_fit import run_grid_and_save_npz_kepler


# ============================================================
# Base microlensing event
# ============================================================

t0_true = 50.0
u0_true = 0.1
tE_true = 150.0

N_time = 5000
window_tE = 5.0

t = np.linspace(
    t0_true - window_tE * tE_true,
    t0_true + window_tE * tE_true,
    N_time,
)


# ============================================================
# Geometry
# ============================================================

phi_true = 0.0
i_true = np.pi / 2.0
theta_true = 0.0


# ============================================================
# Physical primary
# ============================================================

M1_fixed = 1.0       # Msun
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
# Periods
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
# Photometric wrapper
# ============================================================

msource_true = 24.0
mtotal_true = 24.0


# ============================================================
# Approximate luminosity functions
# ============================================================

def main_sequence_luminosity(M):
    """
    Approximate bolometric main-sequence mass-luminosity relation.

    IMPORTANT:
    This is only a physical sanity check.
    It is NOT a Roman-band isochrone.

    M in Msun.
    L returned in Lsun.
    """

    M = np.asarray(
        M,
        dtype=float,
    )

    L = np.zeros_like(
        M,
        dtype=float,
    )

    # --------------------------------------------------------
    # Planet / brown-dwarf regime
    #
    # We neglect their light here.
    # --------------------------------------------------------

    mask_dark = (
        M < 0.08
    )

    L[
        mask_dark
    ] = 0.0


    # --------------------------------------------------------
    # Low-mass main sequence
    # --------------------------------------------------------

    mask_low = (
        (M >= 0.08)
        &
        (M < 0.43)
    )

    L[
        mask_low
    ] = (
        0.23
        * M[
            mask_low
        ] ** 2.3
    )


    # --------------------------------------------------------
    # Solar-like main sequence
    # --------------------------------------------------------

    mask_mid = (
        (M >= 0.43)
        &
        (M < 2.0)
    )

    L[
        mask_mid
    ] = (
        M[
            mask_mid
        ] ** 4.0
    )


    # --------------------------------------------------------
    # Higher masses
    # --------------------------------------------------------

    mask_high = (
        M >= 2.0
    )

    L[
        mask_high
    ] = (
        1.4
        * M[
            mask_high
        ] ** 3.5
    )

    return L


def qflux_piecewise(qM):
    """
    q_f = L2/L1 using the approximate luminosity law.
    """

    M1 = M1_fixed
    M2 = qM * M1

    L1 = float(
        main_sequence_luminosity(
            np.array([M1])
        )[0]
    )

    L2 = float(
        main_sequence_luminosity(
            np.array([M2])
        )[0]
    )

    if L1 <= 0:
        return 0.0

    return L2 / L1


def qflux_powerlaw(qM):
    """
    Toy relation L ~ M^4.
    """

    return float(
        qM**4
    )


# ============================================================
# Flux tracks
# ============================================================

TRACKS = {
    "dark": lambda q: 0.0,
    "qM4": qflux_powerlaw,
    "piecewise_MS": qflux_piecewise,
    "qf_eq_qM": lambda q: float(q),
}

track_names = list(
    TRACKS.keys()
)

N_tracks = len(
    track_names
)


# ============================================================
# Output
# ============================================================

home = Path.home()

output_dir = (
    home
    / "binary_source"
    / "results"
    / "scan_physical_mass_luminosity"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Parallel configuration
# ============================================================

N_cpu = os.cpu_count() or 1

N_workers = max(
    1,
    N_cpu - 1,
)

overwrite = False


# ============================================================
# Worker
# ============================================================

def run_one(task):

    i_track, track_name, i_qM, qM = task

    qM = float(qM)

    M1 = M1_fixed
    M2 = qM * M1

    qf = float(
        TRACKS[
            track_name
        ](
            qM
        )
    )

    outfile = (
        output_dir
        / (
            f"{track_name}"
            f"_qM_{i_qM:03d}.npz"
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
                "track": track_name,
                "i_track": i_track,
                "i_qM": i_qM,
                "qM": qM,
                "qf": qf,
                "D": d["D"].astype(float),
                "RMS": d["RMS"].astype(float),
                "DTE": d["DTE"].astype(float),
                "DU0": d["DU0"].astype(float),
                "SUCCESS": d["SUCCESS"].astype(bool),
                "XI_REL": d["xiE_of_P"].astype(float),
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

            msource_true=msource_true,
            mtotal_true=mtotal_true,

            override_xiE=None,

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
                "track": track_name,
                "i_track": i_track,
                "i_qM": i_qM,
                "qM": qM,
                "qf": qf,
                "D": d["D"].astype(float),
                "RMS": d["RMS"].astype(float),
                "DTE": d["DTE"].astype(float),
                "DU0": d["DU0"].astype(float),
                "SUCCESS": d["SUCCESS"].astype(bool),
                "XI_REL": d["xiE_of_P"].astype(float),
            }

    except Exception as error:

        return {
            "status": "FAILED",
            "track": track_name,
            "i_track": i_track,
            "i_qM": i_qM,
            "qM": qM,
            "qf": qf,
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
    # [track, qM, P]
    # --------------------------------------------------------

    shape = (
        N_tracks,
        N_qM,
        N_P,
    )

    D_all = np.full(
        shape,
        np.nan,
    )

    RMS_all = np.full(
        shape,
        np.nan,
    )

    DTE_all = np.full(
        shape,
        np.nan,
    )

    DU0_all = np.full(
        shape,
        np.nan,
    )

    SUCCESS_all = np.zeros(
        shape,
        dtype=bool,
    )

    qf_all = np.full(
        (
            N_tracks,
            N_qM,
        ),
        np.nan,
    )

    xi_rel_all = np.full(
        shape,
        np.nan,
    )


    # --------------------------------------------------------
    # Tasks
    # --------------------------------------------------------

    tasks = []

    for i_track, track_name in enumerate(
        track_names
    ):

        for i_qM, qM in enumerate(
            qM_grid
        ):

            tasks.append(
                (
                    i_track,
                    track_name,
                    i_qM,
                    qM,
                )
            )


    print(
        f"Number of jobs = {len(tasks)}"
    )

    print(
        f"Number of fits = "
        f"{len(tasks) * N_P}"
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
            as_completed(futures),
            start=1,
        ):

            result = future.result()

            status = result[
                "status"
            ]

            i_track = result[
                "i_track"
            ]

            i_qM = result[
                "i_qM"
            ]

            if status != "FAILED":

                D_all[
                    i_track,
                    i_qM,
                    :
                ] = result["D"]

                RMS_all[
                    i_track,
                    i_qM,
                    :
                ] = result["RMS"]

                DTE_all[
                    i_track,
                    i_qM,
                    :
                ] = result["DTE"]

                DU0_all[
                    i_track,
                    i_qM,
                    :
                ] = result["DU0"]

                SUCCESS_all[
                    i_track,
                    i_qM,
                    :
                ] = result["SUCCESS"]

                xi_rel_all[
                    i_track,
                    i_qM,
                    :
                ] = result["XI_REL"]

                qf_all[
                    i_track,
                    i_qM,
                ] = result["qf"]

            print(
                f"[{k:4d}/{len(tasks):4d}] "
                f"{status:7s} "
                f"{result['track']:12s} "
                f"qM={result['qM']:.3e} "
                f"qf={result['qf']:.3e}"
            )


    # --------------------------------------------------------
    # Physical masses
    # --------------------------------------------------------

    M2_grid = (
        M1_fixed
        * qM_grid
    )


    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_file = (
        output_dir
        / "summary_mass_luminosity.npz"
    )

    np.savez_compressed(

        summary_file,

        track_names=np.array(
            track_names
        ),

        qM_grid=qM_grid,

        qf=qf_all,

        M1_fixed=M1_fixed,

        M2_grid=M2_grid,

        P_grid=P_grid,

        P_over_tE_grid=P_over_tE_grid,

        D=D_all,

        RMS=RMS_all,

        DTE=DTE_all,

        DU0=DU0_all,

        SUCCESS=SUCCESS_all,

        xi_rel=xi_rel_all,

        u0_true=u0_true,

        tE_true=tE_true,

        rEhat_AU=rEhat_AU,
    )

    elapsed = (
        time.time()
        - start
    )

    print()
    print(
        "Saved:",
        summary_file,
    )

    print(
        f"Elapsed = {elapsed / 60:.1f} min"
    )
