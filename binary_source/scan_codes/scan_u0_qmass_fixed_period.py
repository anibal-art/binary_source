#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic scan in (u0, qM) at fixed P/tE and qf.

Scientific purpose
------------------
Investigate the approximately vertical structure around qM ~ 0.1
seen in the qM-qf map for P/tE ~ 1.

Default experiment
------------------
tE       = 150 d
P/tE     = 1
P        = 150 d
qf       = 0
Mtot     = 3 Msun
rEhat    = 5 AU
phi      = 0
i        = pi/2
theta    = 0

Grid
----
u0       = logspace(1e-2, 1, 200)
qM       = logspace(1e-4, 1, 200)

For every point we save:
    D, RMS, MAXABS, TDEV
    DT0, DU0, DTE, Q_A
    xi_rel, xi1/u0, xi2/u0

and exact pyLIMA trajectory diagnostics:
    min u1(t), min u2(t)
    time of those minima
    max A1, max A2

No observational cadence/noise is introduced.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import sys
import tempfile

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

SCRIPT = Path(__file__).resolve()

SCAN_DIR = SCRIPT.parent
SOURCE_DIR = SCAN_DIR.parent
REPO_ROOT = SOURCE_DIR.parent

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_DIR),
    )

if str(SCAN_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCAN_DIR),
    )


# ============================================================
# Project imports
# ============================================================

from degeneracy_fit import (
    FIT_OBJECTIVE,
    CODE_COMMIT,
    CODE_DIRTY,
    run_grid_and_save_npz_kepler,
)

from functions_aux import (
    build_sim_event,
    mag_to_flux,
)

from pyLIMA.models import PSPL_model


# ============================================================
# Physical / numerical configuration
# ============================================================

T0 = 50.0
TE = 150.0

P_OVER_TE = 1.0
P_DAYS = P_OVER_TE * TE

QFLUX = 0.0

MTOT_MSUN = 3.0
REHAT_AU = 5.0

PHI = 0.0
INCLINATION = np.pi / 2.0
THETA = 0.0

MSOURCE = 24.0
MTOTAL_FLUX_MAG = 24.0

WINDOW_TE = 3.5
N_TIME = 10000

DEFAULT_N_U0 = 200
DEFAULT_N_QM = 200

U0_MIN = 1e-2
U0_MAX = 1.0

QM_MIN = 1e-4
QM_MAX = 1.0


# ============================================================
# Helpers
# ============================================================

def masses_fixed_mtot(
    qM,
    Mtot=MTOT_MSUN,
):

    qM = float(qM)

    M1 = (
        float(Mtot)
        / (1.0 + qM)
    )

    M2 = (
        qM
        * M1
    )

    return M1, M2


def time_grid():

    return np.linspace(
        T0 - WINDOW_TE * TE,
        T0 + WINDOW_TE * TE,
        N_TIME,
    )


def pspl_A_from_u(
    u,
):

    u = np.asarray(
        u,
        dtype=float,
    )

    u_safe = np.maximum(
        u,
        np.finfo(float).tiny,
    )

    return (
        (
            u_safe**2
            + 2.0
        )
        /
        (
            u_safe
            * np.sqrt(
                u_safe**2
                + 4.0
            )
        )
    )


# ============================================================
# Exact source trajectories from the same pyLIMA model
# ============================================================

def trajectory_diagnostics(
    t,
    u0,
    qM,
    xi_rel,
):

    omega = (
        2.0
        * np.pi
        / P_DAYS
    )


    xi_para = (
        xi_rel
        * np.cos(
            THETA
        )
    )

    xi_perp = (
        xi_rel
        * np.sin(
            THETA
        )
    )


    ev = build_sim_event(
        t,
        mag0=19.0,
        emag=0.01,
        filt="G",
    )


    model = PSPL_model.PSPLmodel(
        ev,
        parallax=[
            "None",
            0.0,
        ],
        double_source=[
            "Circular",
            T0,
        ],
    )


    model.define_model_parameters()


    zp = 27.615

    fs = mag_to_flux(
        MSOURCE,
        zp=zp,
    )

    ftotal = mag_to_flux(
        MTOTAL_FLUX_MAG,
        zp=zp,
    )


    params = [
        T0,
        float(u0),
        TE,
        float(xi_para),
        float(xi_perp),
        float(omega),
        PHI,
        INCLINATION,
        float(qM),
        QFLUX,
        fs,
        ftotal,
    ]


    py_params = (
        model.compute_pyLIMA_parameters(
            params
        )
    )


    trajectories = (
        model.sources_trajectory(
            ev.telescopes[0],
            py_params,
            data_type="photometry",
        )
    )


    (
        source1_x,
        source1_y,
        source2_x,
        source2_y,
        _,
        _,
    ) = trajectories


    source1_x = np.asarray(
        source1_x,
        dtype=float,
    )

    source1_y = np.asarray(
        source1_y,
        dtype=float,
    )

    source2_x = np.asarray(
        source2_x,
        dtype=float,
    )

    source2_y = np.asarray(
        source2_y,
        dtype=float,
    )


    u1 = np.hypot(
        source1_x,
        source1_y,
    )

    u2 = np.hypot(
        source2_x,
        source2_y,
    )


    finite1 = np.isfinite(
        u1
    )

    finite2 = np.isfinite(
        u2
    )


    if not np.any(finite1):
        raise RuntimeError(
            "No finite source-1 trajectory."
        )

    if not np.any(finite2):
        raise RuntimeError(
            "No finite source-2 trajectory."
        )


    i1 = np.flatnonzero(
        finite1
    )[
        np.argmin(
            u1[finite1]
        )
    ]

    i2 = np.flatnonzero(
        finite2
    )[
        np.argmin(
            u2[finite2]
        )
    ]


    u1_min = float(
        u1[i1]
    )

    u2_min = float(
        u2[i2]
    )


    t_u1_min = float(
        t[i1]
    )

    t_u2_min = float(
        t[i2]
    )


    A1 = pspl_A_from_u(
        u1
    )

    A2 = pspl_A_from_u(
        u2
    )


    i_t0 = int(
        np.argmin(
            np.abs(
                t - T0
            )
        )
    )


    return {

        "U1MIN":
            u1_min,

        "U2MIN":
            u2_min,

        "T_U1MIN":
            t_u1_min,

        "T_U2MIN":
            t_u2_min,

        "DT_U1MIN_OVER_TE":
            (
                t_u1_min
                - T0
            )
            / TE,

        "DT_U2MIN_OVER_TE":
            (
                t_u2_min
                - T0
            )
            / TE,

        "A1MAX":
            float(
                np.nanmax(
                    A1
                )
            ),

        "A2MAX":
            float(
                np.nanmax(
                    A2
                )
            ),

        "U1_AT_T0":
            float(
                u1[i_t0]
            ),

        "U2_AT_T0":
            float(
                u2[i_t0]
            ),
    }


# ============================================================
# Per-qM file validation
# ============================================================

def valid_qmass_file(
    filename,
    qM,
    u0_grid,
):

    filename = Path(
        filename
    )

    if not filename.exists():
        return False

    # Dirty-code products are intentionally never resumed.
    if CODE_DIRTY:
        return False


    try:

        with np.load(
            filename,
            allow_pickle=False,
        ) as d:


            required = [
                "u0_grid",
                "qM",
                "D",
                "SUCCESS",
                "U1MIN",
                "U2MIN",
                "code_commit",
                "code_dirty",
                "fit_objective",
            ]


            if any(
                key not in d.files
                for key in required
            ):
                return False


            if str(
                d[
                    "code_commit"
                ].item()
            ) != CODE_COMMIT:
                return False


            if bool(
                d[
                    "code_dirty"
                ].item()
            ):
                return False


            if str(
                d[
                    "fit_objective"
                ].item()
            ) != FIT_OBJECTIVE:
                return False


            if not np.isclose(
                float(
                    d[
                        "qM"
                    ].item()
                ),
                float(qM),
                rtol=1e-13,
                atol=0.0,
            ):
                return False


            old_u0 = np.asarray(
                d[
                    "u0_grid"
                ],
                dtype=float,
            )


            if (
                old_u0.shape
                != u0_grid.shape
            ):
                return False


            if not np.allclose(
                old_u0,
                u0_grid,
                rtol=1e-13,
                atol=0.0,
            ):
                return False


    except Exception:

        return False


    return True


# ============================================================
# Worker: one qM, all u0
# ============================================================

def worker_qmass(
    task,
):

    (
        i_q,
        qM,
        u0_grid,
        directory,
    ) = task


    directory = Path(
        directory
    )


    outfile = (
        directory
        / f"qM_{i_q:04d}.npz"
    )


    if valid_qmass_file(
        outfile,
        qM,
        u0_grid,
    ):

        with np.load(
            outfile,
            allow_pickle=False,
        ) as d:

            n_failed = int(
                np.count_nonzero(
                    ~d[
                        "SUCCESS"
                    ].astype(bool)
                )
            )


        return (
            "skip",
            str(outfile),
            n_failed,
        )


    t = time_grid()


    M1, M2 = masses_fixed_mtot(
        qM
    )


    n_u = len(
        u0_grid
    )


    metric_names = (
        "D",
        "RMS",
        "MAXABS",
        "TDEV",
        "DT0",
        "DU0",
        "DTE",
        "Q_A",
    )


    diagnostic_names = (
        "U1MIN",
        "U2MIN",
        "T_U1MIN",
        "T_U2MIN",
        "DT_U1MIN_OVER_TE",
        "DT_U2MIN_OVER_TE",
        "A1MAX",
        "A2MAX",
        "U1_AT_T0",
        "U2_AT_T0",
    )


    metrics = {
        key:
            np.full(
                n_u,
                np.nan,
                dtype=float,
            )
        for key
        in metric_names
    }


    diagnostics = {
        key:
            np.full(
                n_u,
                np.nan,
                dtype=float,
            )
        for key
        in diagnostic_names
    }


    success = np.zeros(
        n_u,
        dtype=bool,
    )


    best = np.full(
        (
            n_u,
            3,
        ),
        np.nan,
        dtype=float,
    )


    xi_rel_arr = np.full(
        n_u,
        np.nan,
        dtype=float,
    )


    a_AU_arr = np.full(
        n_u,
        np.nan,
        dtype=float,
    )


    with tempfile.TemporaryDirectory(
        prefix=f"bspl_u0qm_{i_q:04d}_"
    ) as tmpdir:


        tmpdir = Path(
            tmpdir
        )


        for i_u, u0 in enumerate(
            u0_grid
        ):


            tmpfile = (
                tmpdir
                / f"u0_{i_u:04d}.npz"
            )


            try:


                # -------------------------------------------
                # Exact intrinsic BSPL -> PSPL fit.
                # -------------------------------------------

                with contextlib.redirect_stdout(
                    io.StringIO()
                ):


                    run_grid_and_save_npz_kepler(

                        out_npz_path=str(
                            tmpfile
                        ),

                        t=t,

                        t0_true=T0,
                        u0_true=float(
                            u0
                        ),
                        tE_true=TE,

                        phi_true=PHI,
                        i_true=INCLINATION,
                        theta_true=THETA,

                        qflux_true=QFLUX,

                        M1_Msun=M1,
                        M2_Msun=M2,

                        rEhat_AU=REHAT_AU,

                        P_grid=np.array(
                            [
                                P_DAYS
                            ],
                            dtype=float,
                        ),

                        msource_true=MSOURCE,
                        mtotal_true=MTOTAL_FLUX_MAG,

                        override_xiE=None,

                        set_flux_from_truth_photometry=True,

                        rms_on_magnification=True,

                        store_curves=False,
                    )


                with np.load(
                    tmpfile,
                    allow_pickle=False,
                ) as d:


                    fit_success = bool(
                        d[
                            "SUCCESS"
                        ][0]
                    )


                    if not fit_success:
                        continue


                    for key in metric_names:

                        metrics[
                            key
                        ][
                            i_u
                        ] = float(
                            d[
                                key
                            ][0]
                        )


                    best[
                        i_u,
                        :,
                    ] = np.asarray(
                        d[
                            "BEST_T0U0TE"
                        ][0],
                        dtype=float,
                    )


                    xi_rel = float(
                        d[
                            "xiE_of_P"
                        ][0]
                    )


                    a_AU = float(
                        d[
                            "a_AU_of_P"
                        ][0]
                    )


                # -------------------------------------------
                # Exact source trajectories from pyLIMA.
                # -------------------------------------------

                diag = trajectory_diagnostics(
                    t=t,
                    u0=float(
                        u0
                    ),
                    qM=float(
                        qM
                    ),
                    xi_rel=xi_rel,
                )


                for key in diagnostic_names:

                    diagnostics[
                        key
                    ][
                        i_u
                    ] = diag[
                        key
                    ]


                xi_rel_arr[
                    i_u
                ] = xi_rel


                a_AU_arr[
                    i_u
                ] = a_AU


                success[
                    i_u
                ] = True


            except Exception:

                success[
                    i_u
                ] = False


    # ========================================================
    # Barycentric orbital scales
    # ========================================================

    xi1 = (
        float(qM)
        / (
            1.0
            + float(qM)
        )
        * xi_rel_arr
    )


    xi2 = (
        1.0
        / (
            1.0
            + float(qM)
        )
        * xi_rel_arr
    )


    xi1_over_u0 = (
        xi1
        / u0_grid
    )


    xi2_over_u0 = (
        xi2
        / u0_grid
    )


    np.savez_compressed(

        outfile,

        u0_grid=u0_grid,

        qM=np.float64(
            qM
        ),

        qf=np.float64(
            QFLUX
        ),

        M1_Msun=np.float64(
            M1
        ),

        M2_Msun=np.float64(
            M2
        ),

        Mtot_Msun=np.float64(
            M1 + M2
        ),

        P_days=np.float64(
            P_DAYS
        ),

        P_over_tE=np.float64(
            P_OVER_TE
        ),

        xi_rel=xi_rel_arr,

        xi1=xi1,

        xi2=xi2,

        xi1_over_u0=(
            xi1_over_u0
        ),

        xi2_over_u0=(
            xi2_over_u0
        ),

        a_AU=a_AU_arr,

        SUCCESS=success,

        BEST_T0U0TE=best,

        D=metrics["D"],
        RMS=metrics["RMS"],
        MAXABS=metrics["MAXABS"],
        TDEV=metrics["TDEV"],

        DT0=metrics["DT0"],
        DU0=metrics["DU0"],
        DTE=metrics["DTE"],

        Q_A=metrics["Q_A"],

        U1MIN=diagnostics["U1MIN"],
        U2MIN=diagnostics["U2MIN"],

        T_U1MIN=diagnostics["T_U1MIN"],
        T_U2MIN=diagnostics["T_U2MIN"],

        DT_U1MIN_OVER_TE=(
            diagnostics[
                "DT_U1MIN_OVER_TE"
            ]
        ),

        DT_U2MIN_OVER_TE=(
            diagnostics[
                "DT_U2MIN_OVER_TE"
            ]
        ),

        A1MAX=diagnostics["A1MAX"],
        A2MAX=diagnostics["A2MAX"],

        U1_AT_T0=(
            diagnostics[
                "U1_AT_T0"
            ]
        ),

        U2_AT_T0=(
            diagnostics[
                "U2_AT_T0"
            ]
        ),

        t0_true=np.float64(
            T0
        ),

        tE_true=np.float64(
            TE
        ),

        phi_true=np.float64(
            PHI
        ),

        i_true=np.float64(
            INCLINATION
        ),

        theta_true=np.float64(
            THETA
        ),

        rEhat_AU=np.float64(
            REHAT_AU
        ),

        n_time=np.int64(
            N_TIME
        ),

        window_tE=np.float64(
            WINDOW_TE
        ),

        fit_objective=np.array(
            FIT_OBJECTIVE
        ),

        code_commit=np.array(
            CODE_COMMIT
        ),

        code_dirty=np.bool_(
            CODE_DIRTY
        ),
    )


    n_failed = int(
        np.count_nonzero(
            ~success
        )
    )


    return (
        "done",
        str(outfile),
        n_failed,
    )


# ============================================================
# Build combined summary
# ============================================================

def build_summary(
    directory,
    u0_grid,
    qM_grid,
):

    directory = Path(
        directory
    )


    shape = (
        len(
            u0_grid
        ),
        len(
            qM_grid
        ),
    )


    scalar_fields = (
        "D",
        "RMS",
        "MAXABS",
        "TDEV",
        "DT0",
        "DU0",
        "DTE",
        "Q_A",
        "xi_rel",
        "xi1",
        "xi2",
        "xi1_over_u0",
        "xi2_over_u0",
        "a_AU",
        "U1MIN",
        "U2MIN",
        "T_U1MIN",
        "T_U2MIN",
        "DT_U1MIN_OVER_TE",
        "DT_U2MIN_OVER_TE",
        "A1MAX",
        "A2MAX",
        "U1_AT_T0",
        "U2_AT_T0",
    )


    arrays = {
        key:
            np.full(
                shape,
                np.nan,
                dtype=float,
            )
        for key
        in scalar_fields
    }


    success = np.zeros(
        shape,
        dtype=bool,
    )


    best = np.full(
        (
            len(
                u0_grid
            ),
            len(
                qM_grid
            ),
            3,
        ),
        np.nan,
        dtype=float,
    )


    M1_grid = np.full(
        len(
            qM_grid
        ),
        np.nan,
        dtype=float,
    )


    M2_grid = np.full_like(
        M1_grid,
        np.nan,
    )


    for i_q, qM in enumerate(
        qM_grid
    ):


        filename = (
            directory
            / f"qM_{i_q:04d}.npz"
        )


        if not valid_qmass_file(
            filename,
            qM,
            u0_grid,
        ):
            continue


        with np.load(
            filename,
            allow_pickle=False,
        ) as d:


            success[
                :,
                i_q,
            ] = np.asarray(
                d[
                    "SUCCESS"
                ],
                dtype=bool,
            )


            best[
                :,
                i_q,
                :,
            ] = np.asarray(
                d[
                    "BEST_T0U0TE"
                ],
                dtype=float,
            )


            for key in scalar_fields:

                arrays[
                    key
                ][
                    :,
                    i_q,
                ] = np.asarray(
                    d[
                        key
                    ],
                    dtype=float,
                )


            M1_grid[
                i_q
            ] = float(
                d[
                    "M1_Msun"
                ].item()
            )


            M2_grid[
                i_q
            ] = float(
                d[
                    "M2_Msun"
                ].item()
            )


    output = (
        directory
        / "summary_u0_qmass_fixed_period.npz"
    )


    np.savez_compressed(

        output,

        u0_grid=u0_grid,
        qM_grid=qM_grid,

        qf=np.float64(
            QFLUX
        ),

        P_days=np.float64(
            P_DAYS
        ),

        P_over_tE=np.float64(
            P_OVER_TE
        ),

        Mtot_Msun=np.float64(
            MTOT_MSUN
        ),

        M1_grid=M1_grid,
        M2_grid=M2_grid,

        t0_true=np.float64(
            T0
        ),

        tE_true=np.float64(
            TE
        ),

        phi_true=np.float64(
            PHI
        ),

        i_true=np.float64(
            INCLINATION
        ),

        theta_true=np.float64(
            THETA
        ),

        rEhat_AU=np.float64(
            REHAT_AU
        ),

        n_time=np.int64(
            N_TIME
        ),

        window_tE=np.float64(
            WINDOW_TE
        ),

        SUCCESS=success,

        BEST_T0U0TE=best,

        fit_objective=np.array(
            FIT_OBJECTIVE
        ),

        code_commit=np.array(
            CODE_COMMIT
        ),

        code_dirty=np.bool_(
            CODE_DIRTY
        ),

        **arrays,
    )


    print()
    print(
        "Summary saved:",
        output,
    )


    return output


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--n-u0",
        type=int,
        default=DEFAULT_N_U0,
    )


    parser.add_argument(
        "--n-qm",
        type=int,
        default=DEFAULT_N_QM,
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=min(
            12,
            os.cpu_count()
            or 1,
        ),
    )


    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )


    parser.add_argument(
        "--allow-dirty",
        action="store_true",
    )


    args = parser.parse_args()


    if (
        CODE_DIRTY
        and not args.allow_dirty
    ):

        raise RuntimeError(
            "Repository is dirty. "
            "Commit/stash changes before the production scan, "
            "or use --allow-dirty for a test run."
        )


    u0_grid = np.logspace(
        np.log10(
            U0_MIN
        ),
        np.log10(
            U0_MAX
        ),
        args.n_u0,
    )


    qM_grid = np.logspace(
        np.log10(
            QM_MIN
        ),
        np.log10(
            QM_MAX
        ),
        args.n_qm,
    )


    if args.output is None:

        suffix = (
            f"{CODE_COMMIT}"
            + (
                "_dirty"
                if CODE_DIRTY
                else ""
            )
        )


        directory = (
            REPO_ROOT
            / "results"
            / (
                "diagnostic_u0_qmass_"
                "PoverTE1_qf0_"
                f"{suffix}"
            )
        )

    else:

        directory = (
            args.output
            .expanduser()
            .resolve()
        )


    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    tasks = [

        (
            i_q,
            float(qM),
            u0_grid,
            str(
                directory
            ),
        )

        for i_q, qM
        in enumerate(
            qM_grid
        )
    ]


    print()
    print("=" * 80)
    print("DIAGNOSTIC u0 x qM SCAN")
    print("=" * 80)

    print(
        "commit       =",
        CODE_COMMIT,
    )

    print(
        "dirty        =",
        CODE_DIRTY,
    )

    print(
        "objective    =",
        FIT_OBJECTIVE,
    )

    print(
        "P/tE         =",
        P_OVER_TE,
    )

    print(
        "P [d]        =",
        P_DAYS,
    )

    print(
        "qf           =",
        QFLUX,
    )

    print(
        "Mtot [Msun]  =",
        MTOT_MSUN,
    )

    print(
        "u0 grid      =",
        len(
            u0_grid
        ),
        u0_grid[0],
        "->",
        u0_grid[-1],
    )

    print(
        "qM grid      =",
        len(
            qM_grid
        ),
        qM_grid[0],
        "->",
        qM_grid[-1],
    )

    print(
        "total fits   =",
        len(
            u0_grid
        )
        * len(
            qM_grid
        ),
    )

    print(
        "workers      =",
        args.workers,
    )

    print(
        "output       =",
        directory,
    )

    print("=" * 80)


    done = 0
    skipped = 0
    failed_jobs = 0
    failed_fits = 0


    with ProcessPoolExecutor(
        max_workers=args.workers,
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


        for number, future in enumerate(
            as_completed(
                futures
            ),
            start=1,
        ):


            try:


                (
                    status,
                    _,
                    n_failed,
                ) = future.result()


                failed_fits += (
                    n_failed
                )


                if status == "done":
                    done += 1

                else:
                    skipped += 1


            except Exception as error:


                failed_jobs += 1

                print(
                    "ERROR:",
                    repr(
                        error
                    ),
                )


            if (
                number % 10 == 0
                or number
                == len(tasks)
            ):

                print(
                    f"{number}/{len(tasks)} | "
                    f"done={done} "
                    f"skip={skipped} "
                    f"failed_jobs={failed_jobs} "
                    f"failed_fits={failed_fits}"
                )


    if failed_jobs:

        raise RuntimeError(
            f"{failed_jobs} qM jobs failed."
        )


    summary = build_summary(
        directory,
        u0_grid,
        qM_grid,
    )


    print()
    print("=" * 80)
    print("SCAN FINISHED")
    print("=" * 80)

    print(
        "failed fits =",
        failed_fits,
    )

    print(
        "summary     =",
        summary,
    )


if __name__ == "__main__":
    main()
