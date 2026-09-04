#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final publication production for the intrinsic BSPL--PSPL study.

Experiments implemented here:

    1. u0 versus P/tE, for several tE
    2. qM versus P/tE at fixed total source mass
    3. qM versus qf for representative P/tE values

All outputs:
    - use the intrinsic magnification-space objective
    - use a common temporal window
    - contain Git provenance
    - are stored below results/final_<commit>/
    - can resume interrupted runs
"""

import argparse
import multiprocessing as mp
import subprocess
import sys
import time
import os

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np


# ============================================================
# Paths / project imports
# ============================================================

SCAN_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCAN_DIR.parent
REPO_ROOT = SOURCE_DIR.parent

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

if str(SCAN_DIR) not in sys.path:
    sys.path.insert(0, str(SCAN_DIR))

from degeneracy_fit import (
    FIT_OBJECTIVE,
    run_grid_and_save_npz_kepler,
)

from final_config import (
    CODE_COMMIT,
    CODE_DIRTY,
    RESULTS_ROOT,
    FINAL_N_2D,
    FINAL_N_TIME,
    MAX_WORKERS,
    final_output_dir,
)


from final_remaining_experiments import (
    run_mass_luminosity,
    run_photocenter,
)


# ============================================================
# Common physical/numerical configuration
# ============================================================

T0 = 50.0

PHI = 0.0
INCLINATION = np.pi / 2.0
THETA = 0.0

REHAT_AU = 5.0

MSOURCE = 24.0
MTOTAL = 24.0

WINDOW_TE = 3.5

P_MIN_DAYS = 10.0
P_MAX_DAYS = 100000.0


# ============================================================
# Final tE sampling
# ============================================================

TE_VALUES = np.array(
    [
        10.0,
        20.0,
        30.0,
        50.0,
        75.0,
        100.0,
        150.0,
        200.0,
        300.0,
        500.0,
        750.0,
        1000.0,
    ],
    dtype=float,
)


# ============================================================
# qM-qf scan
#
# 200x201 instead of 300x300 because every point is a complete
# pyLIMA fit and we also require qf=0 explicitly.
#
# qf_grid = [0] + qM_grid ensures qf=qM is sampled exactly.
# ============================================================

N_QMQF = 200

P_OVER_TE_QMQF = np.array(
    [
        0.3,
        1.0,
        3.0,
    ],
    dtype=float,
)


# ============================================================
# Helpers
# ============================================================

def format_number(x):
    return f"{float(x):g}".replace(".", "p")


def time_grid(tE):
    return np.linspace(
        T0 - WINDOW_TE * float(tE),
        T0 + WINDOW_TE * float(tE),
        FINAL_N_TIME,
    )


def repository_is_clean():

    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
        ],
        cwd=REPO_ROOT,
        text=True,
    )

    return not bool(
        status.strip()
    )


def valid_npz(
    filename,
    expected_P,
):
    """
    Only reuse files produced by the current clean commit
    with the current intrinsic objective.
    """

    filename = Path(filename)

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
                "SUCCESS",
                "code_commit",
                "code_dirty",
                "fit_objective",
            ]

            if any(
                key not in d.files
                for key in required
            ):
                return False

            P = np.asarray(
                d["P_grid"],
                dtype=float,
            )

            if (
                P.shape
                != np.asarray(expected_P).shape
            ):
                return False

            if not np.allclose(
                P,
                expected_P,
                rtol=1e-12,
                atol=0.0,
            ):
                return False

            commit = str(
                d["code_commit"].item()
            )

            dirty = bool(
                d["code_dirty"].item()
            )

            objective = str(
                d["fit_objective"].item()
            )

            if commit != CODE_COMMIT:
                return False

            if dirty:
                return False

            if objective != FIT_OBJECTIVE:
                return False

        return True

    except Exception:
        return False


def masses_fixed_mtot(
    qM,
    Mtot=3.0,
):

    M1 = (
        float(Mtot)
        / (1.0 + float(qM))
    )

    M2 = (
        float(qM)
        * M1
    )

    return M1, M2


# ============================================================
# u0 x P scan
# ============================================================

U0_GRID = np.logspace(
    -2,
    1,
    FINAL_N_2D,
)

P_GRID_MAIN = np.logspace(
    np.log10(P_MIN_DAYS),
    np.log10(P_MAX_DAYS),
    FINAL_N_2D,
)


def worker_u0(task):

    (
        i_u0,
        u0,
        tE,
        directory,
    ) = task

    directory = Path(directory)

    outfile = (
        directory
        / f"scan_kepler_u0_{i_u0:03d}.npz"
    )

    if valid_npz(
        outfile,
        P_GRID_MAIN,
    ):
        return (
            "skip",
            str(outfile),
        )

    run_grid_and_save_npz_kepler(
        out_npz_path=str(outfile),

        t=time_grid(tE),

        t0_true=T0,
        u0_true=float(u0),
        tE_true=float(tE),

        phi_true=PHI,
        i_true=INCLINATION,
        qflux_true=0.0,
        theta_true=THETA,

        M1_Msun=2.0,
        M2_Msun=1.0,

        rEhat_AU=REHAT_AU,

        P_grid=P_GRID_MAIN,

        msource_true=MSOURCE,
        mtotal_true=MTOTAL,

        override_xiE=None,

        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,

        store_curves=False,
    )

    return (
        "done",
        str(outfile),
    )


def run_u0():

    root = final_output_dir(
        "u0_multi_tE"
    )

    print()
    print("=" * 80)
    print("FINAL u0 x P/tE SCANS")
    print("=" * 80)

    for tE in TE_VALUES:

        label = format_number(
            tE
        )

        directory = (
            root
            / f"scan_u0_tE{label}"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        tasks = [
            (
                i,
                float(u0),
                float(tE),
                str(directory),
            )
            for i, u0
            in enumerate(U0_GRID)
        ]

        print()
        print(
            f"tE={tE:g} d | "
            f"{FINAL_N_2D} x {FINAL_N_2D} | "
            f"output={directory}"
        )

        done = 0
        skipped = 0
        failed = 0

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            mp_context=mp.get_context(
                "spawn"
            ),
        ) as executor:

            futures = [
                executor.submit(
                    worker_u0,
                    task,
                )
                for task in tasks
            ]

            for future in as_completed(
                futures
            ):

                try:

                    status, _ = (
                        future.result()
                    )

                    if status == "done":
                        done += 1
                    else:
                        skipped += 1

                except Exception as error:

                    failed += 1

                    print(
                        "ERROR u0:",
                        repr(error),
                    )

        print(
            f"finished tE={tE:g}: "
            f"done={done}, "
            f"skipped={skipped}, "
            f"failed={failed}"
        )

        if failed:
            raise RuntimeError(
                f"{failed} u0 workers failed "
                f"for tE={tE}"
            )


# ============================================================
# qM x P scan, fixed Mtot
# ============================================================

QM_GRID = np.logspace(
    -4,
    0,
    FINAL_N_2D,
)

QMASS_TE = float(
    os.environ.get(
        "BINARY_SOURCE_QMASS_TE",
        "150.0",
    )
)


def worker_qmass(task):

    (
        i_q,
        qM,
        directory,
    ) = task

    tE = QMASS_TE

    outfile = (
        Path(directory)
        / f"scan_kepler_q_{i_q:03d}.npz"
    )

    if valid_npz(
        outfile,
        P_GRID_MAIN,
    ):
        return (
            "skip",
            str(outfile),
        )

    M1, M2 = masses_fixed_mtot(
        qM,
        Mtot=3.0,
    )

    run_grid_and_save_npz_kepler(
        out_npz_path=str(outfile),

        t=time_grid(tE),

        t0_true=T0,
        u0_true=0.1,
        tE_true=tE,

        phi_true=PHI,
        i_true=INCLINATION,
        qflux_true=0.0,
        theta_true=THETA,

        M1_Msun=M1,
        M2_Msun=M2,

        rEhat_AU=REHAT_AU,

        P_grid=P_GRID_MAIN,

        msource_true=MSOURCE,
        mtotal_true=MTOTAL,

        override_xiE=None,

        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,

        store_curves=False,
    )

    return (
        "done",
        str(outfile),
    )


def run_qmass():

    directory = final_output_dir(
        f"qmass_fixed_mtot_tE{format_number(QMASS_TE)}"
    )

    tasks = [
        (
            i,
            float(qM),
            str(directory),
        )
        for i, qM
        in enumerate(QM_GRID)
    ]

    print()
    print("=" * 80)
    print("FINAL qM x P/tE SCAN")
    print("=" * 80)

    print(
        f"grid       = "
        f"{FINAL_N_2D} x {FINAL_N_2D}"
    )

    print(
        f"Mtot       = 3 Msun"
    )

    print(
        f"output     = {directory}"
    )

    done = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=mp.get_context(
            "spawn"
        ),
    ) as executor:

        futures = [
            executor.submit(
                worker_qmass,
                task,
            )
            for task in tasks
        ]

        for future in as_completed(
            futures
        ):

            try:

                status, _ = (
                    future.result()
                )

                if status == "done":
                    done += 1
                else:
                    skipped += 1

            except Exception as error:

                failed += 1

                print(
                    "ERROR qM:",
                    repr(error),
                )

    print(
        f"done={done}, "
        f"skipped={skipped}, "
        f"failed={failed}"
    )

    if failed:
        raise RuntimeError(
            f"{failed} qM workers failed"
        )


# ============================================================
# qM x qf scan
# ============================================================

QMQF_QM_GRID = np.logspace(
    -4,
    0,
    N_QMQF,
)

QMQF_QF_GRID = np.concatenate(
    [
        np.array(
            [0.0]
        ),
        QMQF_QM_GRID.copy(),
    ]
)

QMQF_TE = float(os.environ.get("BINARY_SOURCE_QMQF_TE", "150.0"))

QMQF_P_GRID = (
    P_OVER_TE_QMQF
    * QMQF_TE
)


def worker_qmqf(task):

    (
        i_qM,
        qM,
        i_qf,
        qf,
        directory,
    ) = task

    outfile = (
        Path(directory)
        / (
            f"scan_qM_{i_qM:03d}"
            f"_qf_{i_qf:03d}.npz"
        )
    )

    if valid_npz(
        outfile,
        QMQF_P_GRID,
    ):
        return (
            "skip",
            str(outfile),
        )

    M1, M2 = masses_fixed_mtot(
        qM,
        Mtot=3.0,
    )

    run_grid_and_save_npz_kepler(
        out_npz_path=str(outfile),

        t=time_grid(
            QMQF_TE
        ),

        t0_true=T0,
        u0_true=0.1,
        tE_true=QMQF_TE,

        phi_true=PHI,
        i_true=INCLINATION,
        qflux_true=float(qf),
        theta_true=THETA,

        M1_Msun=M1,
        M2_Msun=M2,

        rEhat_AU=REHAT_AU,

        P_grid=QMQF_P_GRID,

        msource_true=MSOURCE,
        mtotal_true=MTOTAL,

        override_xiE=None,

        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,

        store_curves=False,
    )

    return (
        "done",
        str(outfile),
    )


def build_qmqf_summary(
    directory,
):

    directory = Path(
        directory
    )

    shape = (
        len(QMQF_QM_GRID),
        len(QMQF_QF_GRID),
        len(QMQF_P_GRID),
    )

    metrics = {
        key: np.full(
            shape,
            np.nan,
            dtype=float,
        )
        for key in [
            "D",
            "RMS",
            "MAXABS",
            "TDEV",
            "DT0",
            "DU0",
            "DTE",
            "Q_A",
        ]
    }

    success = np.zeros(
        shape,
        dtype=bool,
    )

    xi_rel_of_P = None

    for i_qM in range(
        len(QMQF_QM_GRID)
    ):

        for i_qf in range(
            len(QMQF_QF_GRID)
        ):

            filename = (
                directory
                / (
                    f"scan_qM_{i_qM:03d}"
                    f"_qf_{i_qf:03d}.npz"
                )
            )

            if not valid_npz(
                filename,
                QMQF_P_GRID,
            ):
                continue

            with np.load(
                filename,
                allow_pickle=False,
            ) as d:

                for key in metrics:

                    metrics[key][
                        i_qM,
                        i_qf,
                        :,
                    ] = np.asarray(
                        d[key],
                        dtype=float,
                    )

                success[
                    i_qM,
                    i_qf,
                    :,
                ] = np.asarray(
                    d["SUCCESS"],
                    dtype=bool,
                )

                if xi_rel_of_P is None:

                    xi_rel_of_P = np.asarray(
                        d["xiE_of_P"],
                        dtype=float,
                    )


    if xi_rel_of_P is None:

        raise RuntimeError(
            "Could not construct qM-qf summary."
        )


    M1_grid = (
        3.0
        / (
            1.0
            + QMQF_QM_GRID
        )
    )

    M2_grid = (
        QMQF_QM_GRID
        * M1_grid
    )


    QM = (
        QMQF_QM_GRID[
            :,
            None,
            None,
        ]
    )

    QF = (
        QMQF_QF_GRID[
            None,
            :,
            None,
        ]
    )

    XI = (
        xi_rel_of_P[
            None,
            None,
            :,
        ]
    )


    xi_phot = (
        np.abs(
            QM - QF
        )
        / (
            (1.0 + QM)
            * (1.0 + QF)
        )
        * XI
    )


    summary = (
        directory
        / "summary_qM_qf.npz"
    )

    np.savez_compressed(
        summary,

        qM_grid=QMQF_QM_GRID,
        qf_grid=QMQF_QF_GRID,

        P_grid=QMQF_P_GRID,
        P_over_tE_grid=(
            P_OVER_TE_QMQF
        ),

        Mtot_source=3.0,
        M1_grid=M1_grid,
        M2_grid=M2_grid,

        t0_true=T0,
        u0_true=0.1,
        tE_true=QMQF_TE,

        rEhat_AU=REHAT_AU,

        phi_true=PHI,
        i_true=INCLINATION,
        theta_true=THETA,

        xi_rel_of_P=xi_rel_of_P,
        xi_phot=xi_phot,

        D=metrics["D"],
        RMS=metrics["RMS"],
        MAXABS=metrics["MAXABS"],
        TDEV=metrics["TDEV"],

        DT0=metrics["DT0"],
        DU0=metrics["DU0"],
        DTE=metrics["DTE"],

        Q_A=metrics["Q_A"],

        SUCCESS=success,

        fit_objective=np.array(
            FIT_OBJECTIVE
        ),

        code_commit=np.array(
            CODE_COMMIT
        ),

        code_dirty=np.bool_(
            CODE_DIRTY
        ),

        window_tE=np.float64(
            WINDOW_TE
        ),

        n_time=np.int64(
            FINAL_N_TIME
        ),
    )

    print(
        f"Summary saved: {summary}"
    )


def run_qmass_qflux():

    directory = final_output_dir(
        f"qmass_qflux_tE{format_number(QMQF_TE)}"
    )

    tasks = [
        (
            i_qM,
            float(qM),
            i_qf,
            float(qf),
            str(directory),
        )
        for i_qM, qM
        in enumerate(
            QMQF_QM_GRID
        )
        for i_qf, qf
        in enumerate(
            QMQF_QF_GRID
        )
    ]

    print()
    print("=" * 80)
    print("FINAL qM x qf SCAN")
    print("=" * 80)

    print(
        f"qM points   = "
        f"{len(QMQF_QM_GRID)}"
    )

    print(
        f"qf points   = "
        f"{len(QMQF_QF_GRID)}"
    )

    print(
        f"pairs       = "
        f"{len(tasks)}"
    )

    print(
        f"P/tE        = "
        f"{P_OVER_TE_QMQF}"
    )

    print(
        f"output      = {directory}"
    )

    done = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=mp.get_context(
            "spawn"
        ),
    ) as executor:

        futures = [
            executor.submit(
                worker_qmqf,
                task,
            )
            for task in tasks
        ]

        for number, future in enumerate(
            as_completed(
                futures
            ),
            start=1,
        ):

            try:

                status, _ = (
                    future.result()
                )

                if status == "done":
                    done += 1
                else:
                    skipped += 1

            except Exception as error:

                failed += 1

                print(
                    "ERROR qM-qf:",
                    repr(error),
                )

            if (
                number % 1000 == 0
                or number == len(tasks)
            ):

                print(
                    f"{number}/{len(tasks)} | "
                    f"done={done} "
                    f"skip={skipped} "
                    f"fail={failed}"
                )

    if failed:

        raise RuntimeError(
            f"{failed} qM-qf workers failed"
        )

    build_qmqf_summary(
        directory
    )


# ============================================================
# Production registry
# ============================================================

EXPERIMENTS = {
    "u0":
        run_u0,

    "qmass":
        run_qmass,

    "qmass_qflux":
        run_qmass_qflux,

    "mass_luminosity":
        run_mass_luminosity,

    "photocenter":
        run_photocenter,
}


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(
            EXPERIMENTS
        ),
        default=None,
        help=(
            "Run only selected experiments."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()


    if args.list:

        print(
            "\n".join(
                EXPERIMENTS
            )
        )

        return


    selected = (
        args.only
        if args.only
        else list(
            EXPERIMENTS
        )
    )


    print()
    print("=" * 80)
    print("FINAL BSPL--PSPL PRODUCTION")
    print("=" * 80)

    print(
        f"commit       = {CODE_COMMIT}"
    )

    print(
        f"dirty        = {CODE_DIRTY}"
    )

    print(
        f"objective    = {FIT_OBJECTIVE}"
    )

    print(
        f"results root = {RESULTS_ROOT}"
    )

    print(
        f"2D grid      = {FINAL_N_2D}"
    )

    print(
        f"N time       = {FINAL_N_TIME}"
    )

    print(
        f"window       = +/-{WINDOW_TE} tE"
    )

    print(
        f"workers      = {MAX_WORKERS}"
    )

    print(
        f"experiments  = {selected}"
    )

    print("=" * 80)


    if args.dry_run:
        return


    if CODE_DIRTY:

        raise RuntimeError(
            "final_config reports a dirty repository."
        )


    if not repository_is_clean():

        raise RuntimeError(
            "Git working tree is not clean."
        )


    start = time.time()


    for name in selected:

        print()
        print(
            "#" * 80
        )

        print(
            f"STARTING: {name}"
        )

        print(
            "#" * 80
        )

        EXPERIMENTS[
            name
        ]()


    elapsed = (
        time.time()
        - start
    )


    print()
    print("=" * 80)
    print("FINAL PRODUCTION FINISHED")
    print("=" * 80)

    print(
        f"elapsed = "
        f"{elapsed / 60.0:.2f} min"
    )

    print(
        f"results = {RESULTS_ROOT}"
    )


if __name__ == "__main__":
    main()
