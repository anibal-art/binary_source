#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Remaining final experiments for the intrinsic BSPL--PSPL paper.

Experiments
-----------
1. Mass--luminosity / flux-ratio tracks.
2. Photocenter cancellation in the small-xi_rel limit.

All fits use the intrinsic magnification-space objective defined
in degeneracy_fit.py.
"""

import contextlib
import io
import multiprocessing as mp
import sys
import tempfile

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

from pathlib import Path

import numpy as np


# ============================================================
# Imports
# ============================================================

SCAN_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCAN_DIR.parent

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


from degeneracy_fit import (
    FIT_OBJECTIVE,
    run_grid_and_save_npz_kepler,
)

from final_config import (
    CODE_COMMIT,
    CODE_DIRTY,
    FINAL_N_1D,
    FINAL_N_TIME,
    FINAL_N_PHOTOCENTER_QM,
    FINAL_XI_OVER_U0,
    MAX_WORKERS,
    final_output_dir,
)


# ============================================================
# Common configuration
# ============================================================

T0 = 50.0
U0 = 0.1
TE = 150.0

PHI = 0.0
INCLINATION = np.pi / 2.0
THETA = 0.0

REHAT_AU = 5.0

MSOURCE = 24.0
MTOTAL = 24.0

WINDOW_TE = 3.5


def time_grid():

    return np.linspace(
        T0 - WINDOW_TE * TE,
        T0 + WINDOW_TE * TE,
        FINAL_N_TIME,
    )


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


def valid_core_npz(
    filename,
    expected_P,
):

    filename = Path(
        filename
    )

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

            if str(
                d["code_commit"].item()
            ) != CODE_COMMIT:
                return False

            if bool(
                d["code_dirty"].item()
            ):
                return False

            if str(
                d["fit_objective"].item()
            ) != FIT_OBJECTIVE:
                return False

            P = np.asarray(
                d["P_grid"],
                dtype=float,
            )

            expected_P = np.asarray(
                expected_P,
                dtype=float,
            )

            if P.shape != expected_P.shape:
                return False

            if not np.allclose(
                P,
                expected_P,
                rtol=1e-12,
                atol=0.0,
            ):
                return False

    except Exception:
        return False

    return True


# ============================================================
# ============================================================
# 1. MASS--LUMINOSITY / FLUX-RATIO TRACKS
# ============================================================
# ============================================================

MASSLUM_QM_GRID = np.logspace(
    -4,
    0,
    FINAL_N_1D,
)

MASSLUM_P_OVER_TE = np.array(
    [
        0.3,
        1.0,
        3.0,
    ],
    dtype=float,
)

MASSLUM_P_GRID = (
    MASSLUM_P_OVER_TE
    * TE
)


# ------------------------------------------------------------
# Controlled tracks
#
# dark:
#     only source 1 contributes light.
#
# power4_toy:
#     q_f = q_M^4.
#     This is an illustrative main-sequence-like scaling,
#     NOT a bandpass-specific physical isochrone.
#
# photocenter_cancel:
#     q_f = q_M.
#     The first-order photocenter displacement vanishes.
# ------------------------------------------------------------

MASSLUM_TRACKS = (
    "dark",
    "power4_toy",
    "photocenter_cancel",
)


def masslum_qflux(
    track,
    qM,
):

    qM = float(
        qM
    )

    if track == "dark":
        return 0.0

    if track == "power4_toy":
        return qM**4

    if track == "photocenter_cancel":
        return qM

    raise ValueError(
        track
    )


def worker_mass_luminosity(
    task,
):

    (
        track,
        i_q,
        qM,
        directory,
    ) = task

    directory = Path(
        directory
    )

    outfile = (
        directory
        / (
            f"{track}"
            f"_qM_{i_q:04d}.npz"
        )
    )

    if valid_core_npz(
        outfile,
        MASSLUM_P_GRID,
    ):

        with np.load(
            outfile,
            allow_pickle=False,
        ) as d:

            n_failed = int(
                np.count_nonzero(
                    ~d["SUCCESS"].astype(bool)
                )
            )

        return (
            "skip",
            str(outfile),
            n_failed,
        )


    qf = masslum_qflux(
        track,
        qM,
    )


    # ========================================================
    # Here M1 is fixed to 1 Msun.
    #
    # Therefore qM = M2/M1 = M2, and changing qM changes
    # the total source mass consistently with a physical
    # primary-companion sequence.
    # ========================================================

    M1 = 1.0
    M2 = float(
        qM
    )


    run_grid_and_save_npz_kepler(

        out_npz_path=str(
            outfile
        ),

        t=time_grid(),

        t0_true=T0,
        u0_true=U0,
        tE_true=TE,

        phi_true=PHI,
        i_true=INCLINATION,
        theta_true=THETA,

        qflux_true=qf,

        M1_Msun=M1,
        M2_Msun=M2,

        rEhat_AU=REHAT_AU,

        P_grid=MASSLUM_P_GRID,

        msource_true=MSOURCE,
        mtotal_true=MTOTAL,

        override_xiE=None,

        set_flux_from_truth_photometry=True,
        rms_on_magnification=True,

        store_curves=False,
    )


    with np.load(
        outfile,
        allow_pickle=False,
    ) as d:

        n_failed = int(
            np.count_nonzero(
                ~d["SUCCESS"].astype(bool)
            )
        )


    return (
        "done",
        str(outfile),
        n_failed,
    )


def build_mass_luminosity_summary(
    directory,
):

    directory = Path(
        directory
    )

    n_track = len(
        MASSLUM_TRACKS
    )

    n_q = len(
        MASSLUM_QM_GRID
    )

    n_P = len(
        MASSLUM_P_GRID
    )

    shape = (
        n_track,
        n_q,
        n_P,
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

    metrics = {
        name:
            np.full(
                shape,
                np.nan,
                dtype=float,
            )
        for name
        in metric_names
    }

    success = np.zeros(
        shape,
        dtype=bool,
    )

    qf_values = np.full(
        (
            n_track,
            n_q,
        ),
        np.nan,
        dtype=float,
    )

    xi_rel = np.full(
        (
            n_q,
            n_P,
        ),
        np.nan,
        dtype=float,
    )


    for i_track, track in enumerate(
        MASSLUM_TRACKS
    ):

        for i_q, qM in enumerate(
            MASSLUM_QM_GRID
        ):

            filename = (
                directory
                / (
                    f"{track}"
                    f"_qM_{i_q:04d}.npz"
                )
            )

            if not valid_core_npz(
                filename,
                MASSLUM_P_GRID,
            ):
                continue


            with np.load(
                filename,
                allow_pickle=False,
            ) as d:

                for name in metric_names:

                    metrics[
                        name
                    ][
                        i_track,
                        i_q,
                        :,
                    ] = np.asarray(
                        d[name],
                        dtype=float,
                    )


                success[
                    i_track,
                    i_q,
                    :,
                ] = np.asarray(
                    d["SUCCESS"],
                    dtype=bool,
                )


                xi_rel[
                    i_q,
                    :,
                ] = np.asarray(
                    d["xiE_of_P"],
                    dtype=float,
                )


            qf_values[
                i_track,
                i_q,
            ] = masslum_qflux(
                track,
                qM,
            )


    QM = (
        MASSLUM_QM_GRID[
            None,
            :,
            None,
        ]
    )

    QF = (
        qf_values[
            :,
            :,
            None,
        ]
    )

    XI = (
        xi_rel[
            None,
            :,
            :,
        ]
    )


    # ========================================================
    # First-order photocenter amplitude
    # ========================================================

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


    output = (
        directory
        / "summary_mass_luminosity.npz"
    )


    np.savez_compressed(

        output,

        track_names=np.array(
            MASSLUM_TRACKS
        ),

        qM_grid=MASSLUM_QM_GRID,

        qf_values=qf_values,

        P_grid=MASSLUM_P_GRID,

        P_over_tE_grid=(
            MASSLUM_P_OVER_TE
        ),

        M1_Msun=np.float64(
            1.0
        ),

        M2_grid=MASSLUM_QM_GRID,

        xi_rel=xi_rel,

        xi_phot=xi_phot,

        SUCCESS=success,

        D=metrics["D"],
        RMS=metrics["RMS"],
        MAXABS=metrics["MAXABS"],
        TDEV=metrics["TDEV"],

        DT0=metrics["DT0"],
        DU0=metrics["DU0"],
        DTE=metrics["DTE"],

        Q_A=metrics["Q_A"],

        fit_objective=np.array(
            FIT_OBJECTIVE
        ),

        code_commit=np.array(
            CODE_COMMIT
        ),

        code_dirty=np.bool_(
            CODE_DIRTY
        ),

        n_time=np.int64(
            FINAL_N_TIME
        ),

        window_tE=np.float64(
            WINDOW_TE
        ),
    )


    print(
        f"Summary saved: {output}"
    )


def run_mass_luminosity():

    directory = final_output_dir(
        "mass_luminosity_tE150"
    )


    tasks = [

        (
            track,
            i_q,
            float(qM),
            str(directory),
        )

        for track in MASSLUM_TRACKS

        for i_q, qM
        in enumerate(
            MASSLUM_QM_GRID
        )
    ]


    print()
    print("=" * 80)
    print("FINAL MASS--LUMINOSITY TRACK SCAN")
    print("=" * 80)

    print(
        f"qM points    = "
        f"{len(MASSLUM_QM_GRID)}"
    )

    print(
        f"tracks       = "
        f"{MASSLUM_TRACKS}"
    )

    print(
        f"P/tE         = "
        f"{MASSLUM_P_OVER_TE}"
    )

    print(
        f"jobs         = "
        f"{len(tasks)}"
    )

    print(
        f"total fits   = "
        f"{len(tasks) * len(MASSLUM_P_GRID)}"
    )

    print(
        f"output       = "
        f"{directory}"
    )


    done = 0
    skipped = 0
    failed_jobs = 0
    failed_fits = 0


    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=mp.get_context(
            "spawn"
        ),
    ) as executor:


        futures = [
            executor.submit(
                worker_mass_luminosity,
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
                    "ERROR mass-luminosity:",
                    repr(error),
                )


            if (
                number % 100 == 0
                or number == len(tasks)
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
            f"{failed_jobs} mass-luminosity jobs failed."
        )


    build_mass_luminosity_summary(
        directory
    )


    if failed_fits:

        raise RuntimeError(
            f"{failed_fits} individual fits failed "
            "in mass-luminosity scan."
        )


# ============================================================
# ============================================================
# 2. PHOTOCENTER / SMALL-XI LIMIT
# ============================================================
# ============================================================

PHOTO_QM_GRID = np.logspace(
    -4,
    0,
    FINAL_N_PHOTOCENTER_QM,
)

PHOTO_XI_OVER_U0 = np.asarray(
    FINAL_XI_OVER_U0,
    dtype=float,
)

PHOTO_XI_REL = (
    U0
    * PHOTO_XI_OVER_U0
)

PHOTO_P_OVER_TE = 1.0

PHOTO_P_GRID = np.array(
    [
        PHOTO_P_OVER_TE
        * TE
    ],
    dtype=float,
)

PHOTO_FAMILIES = (
    "dark",
    "photocenter_cancel",
)


def photo_qflux(
    family,
    qM,
):

    if family == "dark":
        return 0.0

    if family == "photocenter_cancel":
        return float(
            qM
        )

    raise ValueError(
        family
    )


def valid_photo_file(
    filename,
    qM,
    family,
):

    filename = Path(
        filename
    )

    if not filename.exists():
        return False

    try:

        with np.load(
            filename,
            allow_pickle=False,
        ) as d:

            required = [
                "xi_over_u0_grid",
                "xi_rel_grid",
                "qM",
                "family",
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


            if str(
                d["code_commit"].item()
            ) != CODE_COMMIT:
                return False


            if bool(
                d["code_dirty"].item()
            ):
                return False


            if str(
                d["fit_objective"].item()
            ) != FIT_OBJECTIVE:
                return False


            if str(
                d["family"].item()
            ) != family:
                return False


            if not np.isclose(
                float(
                    d["qM"].item()
                ),
                float(qM),
                rtol=1e-13,
                atol=0.0,
            ):
                return False


            x = np.asarray(
                d["xi_over_u0_grid"],
                dtype=float,
            )


            if (
                x.shape
                != PHOTO_XI_OVER_U0.shape
            ):
                return False


            if not np.allclose(
                x,
                PHOTO_XI_OVER_U0,
                rtol=1e-13,
                atol=0.0,
            ):
                return False


    except Exception:
        return False


    return True


def worker_photocenter(
    task,
):

    (
        family,
        i_q,
        qM,
        directory,
    ) = task


    directory = Path(
        directory
    )


    outfile = (
        directory
        / (
            f"{family}"
            f"_qM_{i_q:04d}.npz"
        )
    )


    if valid_photo_file(
        outfile,
        qM,
        family,
    ):

        with np.load(
            outfile,
            allow_pickle=False,
        ) as d:

            n_failed = int(
                np.count_nonzero(
                    ~d["SUCCESS"].astype(bool)
                )
            )

        return (
            "skip",
            str(outfile),
            n_failed,
        )


    qf = photo_qflux(
        family,
        qM,
    )


    M1, M2 = masses_fixed_mtot(
        qM,
        Mtot=3.0,
    )


    n_xi = len(
        PHOTO_XI_OVER_U0
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


    metrics = {
        key:
            np.full(
                n_xi,
                np.nan,
                dtype=float,
            )
        for key
        in metric_names
    }


    success = np.zeros(
        n_xi,
        dtype=bool,
    )


    best = np.full(
        (
            n_xi,
            3,
        ),
        np.nan,
        dtype=float,
    )


    # ========================================================
    # Each xi_rel requires a separate call because
    # override_xiE is a scalar parameter.
    #
    # Temporary NPZ files are kept only inside /tmp and removed
    # automatically when this worker finishes.
    # ========================================================

    with tempfile.TemporaryDirectory(
        prefix=f"bspl_photo_{i_q:04d}_"
    ) as tmpdir:


        tmpdir = Path(
            tmpdir
        )


        for i_xi, xi_rel in enumerate(
            PHOTO_XI_REL
        ):


            tmpfile = (
                tmpdir
                / f"xi_{i_xi:04d}.npz"
            )


            # Avoid tens of thousands of "Saved:" messages.
            with contextlib.redirect_stdout(
                io.StringIO()
            ):


                run_grid_and_save_npz_kepler(

                    out_npz_path=str(
                        tmpfile
                    ),

                    t=time_grid(),

                    t0_true=T0,
                    u0_true=U0,
                    tE_true=TE,

                    phi_true=PHI,
                    i_true=INCLINATION,
                    theta_true=THETA,

                    qflux_true=qf,

                    M1_Msun=M1,
                    M2_Msun=M2,

                    rEhat_AU=REHAT_AU,

                    P_grid=PHOTO_P_GRID,

                    msource_true=MSOURCE,
                    mtotal_true=MTOTAL,

                    # ---------------------------------------
                    # Here xi_rel is controlled independently
                    # of Kepler's relation.
                    # ---------------------------------------

                    override_xiE=float(
                        xi_rel
                    ),

                    set_flux_from_truth_photometry=True,

                    rms_on_magnification=True,

                    store_curves=False,
                )


            with np.load(
                tmpfile,
                allow_pickle=False,
            ) as d:


                success[
                    i_xi
                ] = bool(
                    d["SUCCESS"][0]
                )


                for key in metric_names:

                    metrics[
                        key
                    ][
                        i_xi
                    ] = float(
                        d[key][0]
                    )


                best[
                    i_xi,
                    :,
                ] = np.asarray(
                    d["BEST_T0U0TE"][0],
                    dtype=float,
                )


    np.savez_compressed(

        outfile,

        family=np.array(
            family
        ),

        qM=np.float64(
            qM
        ),

        qf=np.float64(
            qf
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

        xi_over_u0_grid=(
            PHOTO_XI_OVER_U0
        ),

        xi_rel_grid=(
            PHOTO_XI_REL
        ),

        P_over_tE=np.float64(
            PHOTO_P_OVER_TE
        ),

        P_days=np.float64(
            PHOTO_P_GRID[0]
        ),

        D=metrics["D"],
        RMS=metrics["RMS"],
        MAXABS=metrics["MAXABS"],
        TDEV=metrics["TDEV"],

        DT0=metrics["DT0"],
        DU0=metrics["DU0"],
        DTE=metrics["DTE"],

        Q_A=metrics["Q_A"],

        BEST_T0U0TE=best,

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

        n_time=np.int64(
            FINAL_N_TIME
        ),

        window_tE=np.float64(
            WINDOW_TE
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


def build_photocenter_summary(
    directory,
):

    directory = Path(
        directory
    )

    n_family = len(
        PHOTO_FAMILIES
    )

    n_q = len(
        PHOTO_QM_GRID
    )

    n_xi = len(
        PHOTO_XI_OVER_U0
    )


    shape = (
        n_family,
        n_q,
        n_xi,
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


    metrics = {
        name:
            np.full(
                shape,
                np.nan,
                dtype=float,
            )
        for name
        in metric_names
    }


    success = np.zeros(
        shape,
        dtype=bool,
    )


    qf_values = np.full(
        (
            n_family,
            n_q,
        ),
        np.nan,
        dtype=float,
    )


    for i_family, family in enumerate(
        PHOTO_FAMILIES
    ):


        for i_q, qM in enumerate(
            PHOTO_QM_GRID
        ):


            filename = (
                directory
                / (
                    f"{family}"
                    f"_qM_{i_q:04d}.npz"
                )
            )


            if not valid_photo_file(
                filename,
                qM,
                family,
            ):
                continue


            with np.load(
                filename,
                allow_pickle=False,
            ) as d:


                for name in metric_names:

                    metrics[
                        name
                    ][
                        i_family,
                        i_q,
                        :,
                    ] = np.asarray(
                        d[name],
                        dtype=float,
                    )


                success[
                    i_family,
                    i_q,
                    :,
                ] = np.asarray(
                    d["SUCCESS"],
                    dtype=bool,
                )


                qf_values[
                    i_family,
                    i_q,
                ] = float(
                    d["qf"].item()
                )


    QM = (
        PHOTO_QM_GRID[
            None,
            :,
            None,
        ]
    )

    QF = (
        qf_values[
            :,
            :,
            None,
        ]
    )

    XI = (
        PHOTO_XI_REL[
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


    output = (
        directory
        / "summary_photocenter_small_xi.npz"
    )


    np.savez_compressed(

        output,

        family_names=np.array(
            PHOTO_FAMILIES
        ),

        qM_grid=PHOTO_QM_GRID,

        qf_values=qf_values,

        xi_over_u0_grid=(
            PHOTO_XI_OVER_U0
        ),

        xi_rel_grid=(
            PHOTO_XI_REL
        ),

        xi_phot=xi_phot,

        P_over_tE=np.float64(
            PHOTO_P_OVER_TE
        ),

        P_days=np.float64(
            PHOTO_P_GRID[0]
        ),

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

        n_time=np.int64(
            FINAL_N_TIME
        ),

        window_tE=np.float64(
            WINDOW_TE
        ),
    )


    print(
        f"Summary saved: {output}"
    )


def run_photocenter():

    directory = final_output_dir(
        "photocenter_small_xi_tE150"
    )


    tasks = [

        (
            family,
            i_q,
            float(qM),
            str(directory),
        )

        for family in PHOTO_FAMILIES

        for i_q, qM
        in enumerate(
            PHOTO_QM_GRID
        )
    ]


    total_fits = (
        len(tasks)
        * len(
            PHOTO_XI_OVER_U0
        )
    )


    print()
    print("=" * 80)
    print("FINAL PHOTOCENTER SMALL-XI SCAN")
    print("=" * 80)

    print(
        f"qM points    = "
        f"{len(PHOTO_QM_GRID)}"
    )

    print(
        f"xi points    = "
        f"{len(PHOTO_XI_OVER_U0)}"
    )

    print(
        f"xi/u0 range  = "
        f"{PHOTO_XI_OVER_U0[0]:.3e} "
        f"-> "
        f"{PHOTO_XI_OVER_U0[-1]:.3e}"
    )

    print(
        f"families     = "
        f"{PHOTO_FAMILIES}"
    )

    print(
        f"P/tE         = "
        f"{PHOTO_P_OVER_TE:g}"
    )

    print(
        f"total fits   = "
        f"{total_fits}"
    )

    print(
        f"output       = "
        f"{directory}"
    )


    done = 0
    skipped = 0
    failed_jobs = 0
    failed_fits = 0


    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=mp.get_context(
            "spawn"
        ),
    ) as executor:


        futures = [
            executor.submit(
                worker_photocenter,
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
                    "ERROR photocenter:",
                    repr(error),
                )


            if (
                number % 20 == 0
                or number == len(tasks)
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
            f"{failed_jobs} photocenter jobs failed."
        )


    build_photocenter_summary(
        directory
    )


    if failed_fits:

        raise RuntimeError(
            f"{failed_fits} individual fits failed "
            "in photocenter scan."
        )


