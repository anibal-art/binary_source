#!/usr/bin/env python3

"""
Orbital-geometry robustness tests for the intrinsic BSPL -> PSPL
degeneracy analysis.

The script addresses two questions.

A. BROAD GEOMETRY ROBUSTNESS

   How much do D_BSPL-PSPL and the absorbed PSPL parameter shifts
   change when the projected orbital geometry (theta, phi, inclination)
   is varied?

B. PHOTOCENTER SCALING ROBUSTNESS

   Does the small-separation scaling remain

       D ~ xi_rel       for q_f = 0

   and

       D ~ xi_rel^2     for q_f = q_M

   across orbital geometries?

This is a robustness/stress test, not a population-weighted
marginalization over binary orientations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Repository root
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Existing project implementation
# ============================================================

from binary_source.analysis import roman_bspl_pspl_asimov as roman

from binary_source.validation.validate_intrinsic_multistart_time_resolution import (
    fit_one_start,
)


# ============================================================
# Fiducial intrinsic setup
# ============================================================

T0 = 50.0
TE = 150.0

U0_FID = 0.1

M_TOTAL = 3.0
REHAT_AU = 5.0

Q_MASS = 0.5

WINDOW_TE = 3.5
N_TIME = 10_000

MAXITER_DEFAULT = 50_000

SEED = 24681357


# ============================================================
# Small-separation scaling experiment
# ============================================================

SCALING_P_DAYS = 150.0
SCALING_U0 = 0.1

# Exactly the asymptotic small-separation regime.
XI_OVER_U0_GRID = np.logspace(
    -4.0,
    -2.0,
    9,
)

D_NUMERICAL_FLOOR = 1.0e-14


# ============================================================
# Geometry definitions
# ============================================================

@dataclass(frozen=True)
class Geometry:
    geometry_id: str
    geometry_kind: str

    theta_rad: float
    phi_rad: float
    inclination_rad: float


@dataclass(frozen=True)
class BroadCase:
    key: str
    description: str

    u0: float
    P_days: float

    q_mass: float
    qflux: float


# ============================================================
# Time grid
# ============================================================

def make_time_grid():
    return np.linspace(
        T0 - WINDOW_TE * TE,
        T0 + WINDOW_TE * TE,
        N_TIME,
    )


# ============================================================
# Geometry grid
# ============================================================

def build_geometry_grid(
    n_random=12,
    seed=SEED,
):
    """
    Deterministic stress-test grid + isotropic random orientations.

    For the random subset:
        cos(i) is uniform in [0,1],
        phi and theta are uniform in [0,2*pi).

    Since i and pi-i are equivalent for the present circular
    projected geometry, i in [0, pi/2] is sufficient here.
    """

    geometries = []

    inclinations_deg = [
        0.0,
        45.0,
        90.0,
    ]

    phases_deg = [
        0.0,
        90.0,
        180.0,
        270.0,
    ]

    theta_deg = [
        0.0,
        45.0,
        90.0,
        135.0,
    ]


    # --------------------------------------------------------
    # Deterministic grid
    # --------------------------------------------------------

    counter = 0

    for inc in inclinations_deg:
        for phi in phases_deg:
            for theta in theta_deg:

                geometries.append(
                    Geometry(
                        geometry_id=(
                            f"det_{counter:03d}"
                        ),
                        geometry_kind=(
                            "deterministic"
                        ),
                        theta_rad=float(
                            np.deg2rad(theta)
                        ),
                        phi_rad=float(
                            np.deg2rad(phi)
                        ),
                        inclination_rad=float(
                            np.deg2rad(inc)
                        ),
                    )
                )

                counter += 1


    # --------------------------------------------------------
    # Random isotropic orientations
    # --------------------------------------------------------

    rng = np.random.default_rng(
        int(seed)
    )

    for j in range(
        int(n_random)
    ):

        cos_i = rng.uniform(
            0.0,
            1.0,
        )

        inclination = np.arccos(
            cos_i
        )

        phi = rng.uniform(
            0.0,
            2.0 * np.pi,
        )

        theta = rng.uniform(
            0.0,
            2.0 * np.pi,
        )

        geometries.append(
            Geometry(
                geometry_id=(
                    f"rnd_{j:03d}"
                ),
                geometry_kind=(
                    "random_isotropic"
                ),
                theta_rad=float(theta),
                phi_rad=float(phi),
                inclination_rad=float(
                    inclination
                ),
            )
        )


    return geometries


# ============================================================
# Broad representative cases
# ============================================================

def build_broad_cases():

    return [

        BroadCase(
            key="one_short",
            description=(
                "One luminous source; short-period Kepler regime"
            ),
            u0=0.1,
            P_days=10.0,
            q_mass=0.5,
            qflux=0.0,
        ),

        BroadCase(
            key="one_intermediate",
            description=(
                "One luminous source; intermediate P~tE regime"
            ),
            u0=0.1,
            P_days=210.0,
            q_mass=0.5,
            qflux=0.0,
        ),

        BroadCase(
            key="one_long",
            description=(
                "One luminous source; long-period absorbed regime"
            ),
            u0=0.1,
            P_days=6000.0,
            q_mass=0.5,
            qflux=0.0,
        ),

        BroadCase(
            key="one_small_u0",
            description=(
                "One luminous source; high-magnification case"
            ),
            u0=0.01,
            P_days=210.0,
            q_mass=0.5,
            qflux=0.0,
        ),

        BroadCase(
            key="two_cancel",
            description=(
                "Two luminous sources on qf=qM"
            ),
            u0=0.1,
            P_days=150.0,
            q_mass=0.5,
            qflux=0.5,
        ),

        BroadCase(
            key="two_off_cancel",
            description=(
                "Two luminous sources away from photocenter cancellation"
            ),
            u0=0.1,
            P_days=150.0,
            q_mass=0.5,
            qflux=0.1,
        ),
    ]


# ============================================================
# Helpers
# ============================================================

def geometry_metadata(g):

    return {
        "geometry_id": (
            g.geometry_id
        ),
        "geometry_kind": (
            g.geometry_kind
        ),

        "theta_rad": float(
            g.theta_rad
        ),
        "phi_rad": float(
            g.phi_rad
        ),
        "inclination_rad": float(
            g.inclination_rad
        ),

        "theta_deg": float(
            np.rad2deg(
                g.theta_rad
            )
        ),
        "phi_deg": float(
            np.rad2deg(
                g.phi_rad
            )
        ),
        "inclination_deg": float(
            np.rad2deg(
                g.inclination_rad
            )
        ),
    }


def is_fiducial_geometry(
    g,
):
    """
    Production geometry:
        theta = 0
        phi   = 0
        i     = pi/2
    """

    return (
        np.isclose(
            g.theta_rad,
            0.0,
        )
        and np.isclose(
            g.phi_rad,
            0.0,
        )
        and np.isclose(
            g.inclination_rad,
            0.5 * np.pi,
        )
    )


# ============================================================
# Intrinsic fit
# ============================================================

def fit_intrinsic_truth(
    t,
    A_truth,
    u0_true,
    maxiter,
):
    """
    Same intrinsic PSPL projection used in the validation work:
    Nelder-Mead in (t0,u0,tE), initialized at the barycentric truth.
    """

    x0 = np.array(
        [
            T0,
            u0_true,
            TE,
        ],
        dtype=float,
    )

    fit = fit_one_start(
        t=t,
        A_truth=A_truth,
        x0=x0,
        maxiter=maxiter,
    )

    return fit


# ============================================================
# Truth generator
# ============================================================

def make_truth(
    t,
    u0,
    P_days,
    q_mass,
    qflux,
    geometry,
    rEhat_AU=REHAT_AU,
):

    result = (
        roman.bspl_truth_magnification(
            t=t,

            t0_true=T0,
            u0_true=u0,
            tE_true=TE,

            P_days=P_days,

            q_mass=q_mass,
            qflux=qflux,

            Mtot_Msun=M_TOTAL,
            rEhat_AU=rEhat_AU,

            theta=geometry.theta_rad,
            phi=geometry.phi_rad,
            inclination=(
                geometry.inclination_rad
            ),
        )
    )

    A_truth = np.asarray(
        result["A_bspl"],
        dtype=float,
    )

    return (
        A_truth,
        result,
    )


# ============================================================
# BROAD TEST
# ============================================================

def run_broad_task(
    payload,
):
    case_dict, geometry_dict, maxiter = payload

    case = BroadCase(
        **case_dict
    )

    geometry = Geometry(
        **geometry_dict
    )

    t = make_time_grid()

    A_truth, truth_info = (
        make_truth(
            t=t,
            u0=case.u0,
            P_days=case.P_days,
            q_mass=case.q_mass,
            qflux=case.qflux,
            geometry=geometry,
        )
    )

    fit = fit_intrinsic_truth(
        t=t,
        A_truth=A_truth,
        u0_true=case.u0,
        maxiter=maxiter,
    )


    dt0_over_tE = (
        (
            fit["best_t0"]
            - T0
        )
        / TE
    )

    du0_over_u0 = (
        (
            fit["best_u0"]
            - case.u0
        )
        / case.u0
    )

    dtE_over_tE = (
        (
            fit["best_tE"]
            - TE
        )
        / TE
    )


    row = {
        "case": case.key,
        "description": (
            case.description
        ),

        "u0": float(
            case.u0
        ),
        "P_days": float(
            case.P_days
        ),
        "P_over_tE": float(
            case.P_days
            / TE
        ),

        "q_mass": float(
            case.q_mass
        ),
        "qflux": float(
            case.qflux
        ),

        **geometry_metadata(
            geometry
        ),

        "fiducial_geometry": bool(
            is_fiducial_geometry(
                geometry
            )
        ),

        "xi_rel": float(
            truth_info["xi_rel"]
        ),
        "a_rel_AU": float(
            truth_info["a_rel_AU"]
        ),

        "success": bool(
            fit["success"]
        ),

        "D": float(
            fit["D"]
        ),
        "J": float(
            fit["J"]
        ),

        "best_t0": float(
            fit["best_t0"]
        ),
        "best_u0": float(
            fit["best_u0"]
        ),
        "best_tE": float(
            fit["best_tE"]
        ),

        "dt0_over_tE": float(
            dt0_over_tE
        ),
        "du0_over_u0": float(
            du0_over_u0
        ),
        "dtE_over_tE": float(
            dtE_over_tE
        ),
    }

    return row


def run_broad(
    geometries,
    cases,
    workers,
    maxiter,
):

    tasks = [
        (
            asdict(case),
            asdict(geometry),
            int(maxiter),
        )
        for case in cases
        for geometry in geometries
    ]


    print()
    print("=" * 100)
    print("BROAD GEOMETRY ROBUSTNESS")
    print("=" * 100)

    print(
        "N geometries =",
        len(geometries),
    )

    print(
        "N cases      =",
        len(cases),
    )

    print(
        "N fits       =",
        len(tasks),
    )

    print(
        "workers      =",
        workers,
    )

    print()


    rows = []

    if workers <= 1:

        for i, task in enumerate(
            tasks,
            start=1,
        ):

            row = run_broad_task(
                task
            )

            rows.append(
                row
            )

            print(
                f"[{i:4d}/{len(tasks):4d}] "
                f"{row['case']:20s} "
                f"{row['geometry_id']:8s} "
                f"D={row['D']:.6e}"
            )

    else:

        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:

            future_map = {
                executor.submit(
                    run_broad_task,
                    task,
                ): task
                for task in tasks
            }

            completed = 0

            for future in as_completed(
                future_map
            ):

                row = future.result()

                rows.append(
                    row
                )

                completed += 1

                print(
                    f"[{completed:4d}/{len(tasks):4d}] "
                    f"{row['case']:20s} "
                    f"{row['geometry_id']:8s} "
                    f"D={row['D']:.6e}"
                )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Broad summaries
# ============================================================

def summarize_broad(
    df,
):

    summaries = []

    for case_name, group in df.groupby(
        "case"
    ):

        D = np.asarray(
            group["D"],
            dtype=float,
        )

        valid = np.isfinite(
            D
        )

        D = D[
            valid
        ]

        fid = group[
            group[
                "fiducial_geometry"
            ]
        ]

        if len(fid) != 1:
            raise RuntimeError(
                f"{case_name}: expected exactly "
                "one fiducial geometry."
            )

        D_fid = float(
            fid.iloc[0]["D"]
        )


        summaries.append(
            {
                "case": case_name,

                "n_geometry": int(
                    len(group)
                ),

                "n_valid": int(
                    np.sum(valid)
                ),

                "D_fiducial": D_fid,

                "D_min": float(
                    np.min(D)
                ),
                "D_p16": float(
                    np.percentile(
                        D,
                        16,
                    )
                ),
                "D_median": float(
                    np.median(D)
                ),
                "D_p84": float(
                    np.percentile(
                        D,
                        84,
                    )
                ),
                "D_max": float(
                    np.max(D)
                ),

                "D_min_over_fid": float(
                    np.min(D)
                    / D_fid
                ),

                "D_max_over_fid": float(
                    np.max(D)
                    / D_fid
                ),
            }
        )


    return pd.DataFrame(
        summaries
    )


# ============================================================
# Pairwise physical checks
# ============================================================

def broad_pairwise_tests(
    df,
):

    rows = []


    # ========================================================
    # 1. Short / intermediate / long ordering
    # ========================================================

    keys = [
        "one_short",
        "one_intermediate",
        "one_long",
    ]

    pieces = []

    for key in keys:

        piece = (
            df[
                df["case"]
                == key
            ][
                [
                    "geometry_id",
                    "D",
                ]
            ]
            .rename(
                columns={
                    "D": f"D_{key}"
                }
            )
        )

        pieces.append(
            piece
        )


    merged = pieces[0]

    for piece in pieces[1:]:

        merged = merged.merge(
            piece,
            on="geometry_id",
            how="inner",
        )


    for _, row in merged.iterrows():

        mid_larger_than_both = (
            row[
                "D_one_intermediate"
            ]
            >
            row[
                "D_one_short"
            ]
            and
            row[
                "D_one_intermediate"
            ]
            >
            row[
                "D_one_long"
            ]
        )

        rows.append(
            {
                "test": (
                    "one_luminous_temporal_order"
                ),

                "geometry_id": (
                    row["geometry_id"]
                ),

                "D_short": float(
                    row["D_one_short"]
                ),
                "D_intermediate": float(
                    row[
                        "D_one_intermediate"
                    ]
                ),
                "D_long": float(
                    row["D_one_long"]
                ),

                "condition": bool(
                    mid_larger_than_both
                ),
            }
        )


    # ========================================================
    # 2. Cancellation vs off-cancellation at finite xi
    # ========================================================

    cancel = (
        df[
            df["case"]
            == "two_cancel"
        ][
            [
                "geometry_id",
                "D",
            ]
        ]
        .rename(
            columns={
                "D": "D_cancel"
            }
        )
    )

    off = (
        df[
            df["case"]
            == "two_off_cancel"
        ][
            [
                "geometry_id",
                "D",
            ]
        ]
        .rename(
            columns={
                "D": "D_off_cancel"
            }
        )
    )

    paired = cancel.merge(
        off,
        on="geometry_id",
        how="inner",
    )


    for _, row in paired.iterrows():

        rows.append(
            {
                "test": (
                    "finite_xi_cancel_vs_off"
                ),

                "geometry_id": (
                    row["geometry_id"]
                ),

                "D_cancel": float(
                    row["D_cancel"]
                ),

                "D_off_cancel": float(
                    row["D_off_cancel"]
                ),

                "ratio_cancel_over_off": float(
                    row["D_cancel"]
                    / row["D_off_cancel"]
                ),

                "condition": bool(
                    row["D_cancel"]
                    <
                    row["D_off_cancel"]
                ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# SMALL-SEPARATION SCALING TEST
# ============================================================

def run_scaling_geometry_family(
    payload,
):

    geometry_dict, family, maxiter = payload

    geometry = Geometry(
        **geometry_dict
    )

    if family == "dark":

        qflux = 0.0
        target_slope = 1.0

    elif family == "cancel":

        qflux = Q_MASS
        target_slope = 2.0

    else:

        raise ValueError(
            family
        )


    t = make_time_grid()

    # Keplerian physical a_rel at fixed P.
    # We vary rEhat only to control xi_rel while retaining
    # the same pyLIMA orbital parametrization.
    a_rel_AU = float(
        roman.a_from_P_kepler_days(
            SCALING_P_DAYS,
            M_TOTAL,
        )
    )


    rows = []

    for xi_over_u0 in (
        XI_OVER_U0_GRID
    ):

        xi_rel_target = (
            SCALING_U0
            * xi_over_u0
        )

        rEhat_required = (
            a_rel_AU
            / xi_rel_target
        )


        A_truth, truth_info = (
            make_truth(
                t=t,

                u0=SCALING_U0,
                P_days=SCALING_P_DAYS,

                q_mass=Q_MASS,
                qflux=qflux,

                geometry=geometry,

                rEhat_AU=(
                    rEhat_required
                ),
            )
        )


        fit = fit_intrinsic_truth(
            t=t,
            A_truth=A_truth,
            u0_true=SCALING_U0,
            maxiter=maxiter,
        )


        rows.append(
            {
                "geometry_id": (
                    geometry.geometry_id
                ),

                "geometry_kind": (
                    geometry.geometry_kind
                ),

                "family": family,
                "target_slope": (
                    target_slope
                ),

                **{
                    k: v
                    for k, v
                    in geometry_metadata(
                        geometry
                    ).items()
                    if k not in (
                        "geometry_id",
                        "geometry_kind",
                    )
                },

                "fiducial_geometry": bool(
                    is_fiducial_geometry(
                        geometry
                    )
                ),

                "xi_over_u0": float(
                    xi_over_u0
                ),

                "xi_rel_target": float(
                    xi_rel_target
                ),

                "xi_rel_actual": float(
                    truth_info[
                        "xi_rel"
                    ]
                ),

                "rEhat_AU": float(
                    rEhat_required
                ),

                "D": float(
                    fit["D"]
                ),

                "best_t0": float(
                    fit["best_t0"]
                ),

                "best_u0": float(
                    fit["best_u0"]
                ),

                "best_tE": float(
                    fit["best_tE"]
                ),
            }
        )


    df = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Fit asymptotic slope
    # --------------------------------------------------------

    valid = (
        np.isfinite(
            df["D"]
        )
        & (
            df["D"]
            > D_NUMERICAL_FLOOR
        )
        & np.isfinite(
            df["xi_rel_actual"]
        )
        & (
            df["xi_rel_actual"]
            > 0.0
        )
    )


    x = np.log10(
        np.asarray(
            df.loc[
                valid,
                "xi_rel_actual",
            ],
            dtype=float,
        )
    )

    y = np.log10(
        np.asarray(
            df.loc[
                valid,
                "D",
            ],
            dtype=float,
        )
    )


    if len(x) >= 4:

        slope, intercept = (
            np.polyfit(
                x,
                y,
                1,
            )
        )

        y_model = (
            slope * x
            + intercept
        )

        ss_res = float(
            np.sum(
                (
                    y
                    - y_model
                ) ** 2
            )
        )

        ss_tot = float(
            np.sum(
                (
                    y
                    - np.mean(y)
                ) ** 2
            )
        )

        if ss_tot > 0.0:

            r2 = (
                1.0
                - ss_res
                / ss_tot
            )

        else:

            r2 = np.nan

    else:

        slope = np.nan
        intercept = np.nan
        r2 = np.nan


    slope_row = {
        "geometry_id": (
            geometry.geometry_id
        ),
        "geometry_kind": (
            geometry.geometry_kind
        ),

        "family": family,
        "target_slope": (
            target_slope
        ),

        **{
            k: v
            for k, v
            in geometry_metadata(
                geometry
            ).items()
            if k not in (
                "geometry_id",
                "geometry_kind",
            )
        },

        "fiducial_geometry": bool(
            is_fiducial_geometry(
                geometry
            )
        ),

        "n_fit_points": int(
            len(x)
        ),

        "slope": float(
            slope
        ),

        "intercept": float(
            intercept
        ),

        "r2": float(
            r2
        ),

        "slope_minus_target": float(
            slope
            - target_slope
        )
        if np.isfinite(slope)
        else np.nan,
    }


    return (
        rows,
        slope_row,
    )


def run_scaling(
    geometries,
    workers,
    maxiter,
):

    tasks = [
        (
            asdict(geometry),
            family,
            int(maxiter),
        )
        for geometry in geometries
        for family in [
            "dark",
            "cancel",
        ]
    ]


    n_fits = (
        len(tasks)
        * len(
            XI_OVER_U0_GRID
        )
    )


    print()
    print("=" * 100)
    print("PHOTOCENTER SCALING GEOMETRY ROBUSTNESS")
    print("=" * 100)

    print(
        "N geometries =",
        len(geometries),
    )

    print(
        "families     = dark, cancel"
    )

    print(
        "xi/u0 nodes  =",
        len(
            XI_OVER_U0_GRID
        ),
    )

    print(
        "total fits   =",
        n_fits,
    )

    print(
        "workers      =",
        workers,
    )

    print()


    all_rows = []
    slope_rows = []


    if workers <= 1:

        for i, task in enumerate(
            tasks,
            start=1,
        ):

            rows, slope = (
                run_scaling_geometry_family(
                    task
                )
            )

            all_rows.extend(
                rows
            )

            slope_rows.append(
                slope
            )

            print(
                f"[{i:3d}/{len(tasks):3d}] "
                f"{slope['geometry_id']:8s} "
                f"{slope['family']:6s} "
                f"slope={slope['slope']:.6f} "
                f"R2={slope['r2']:.6f}"
            )

    else:

        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:

            future_map = {
                executor.submit(
                    run_scaling_geometry_family,
                    task,
                ): task
                for task in tasks
            }

            completed = 0

            for future in as_completed(
                future_map
            ):

                rows, slope = (
                    future.result()
                )

                all_rows.extend(
                    rows
                )

                slope_rows.append(
                    slope
                )

                completed += 1

                print(
                    f"[{completed:3d}/{len(tasks):3d}] "
                    f"{slope['geometry_id']:8s} "
                    f"{slope['family']:6s} "
                    f"slope={slope['slope']:.6f} "
                    f"R2={slope['r2']:.6f}"
                )


    return (
        pd.DataFrame(
            all_rows
        ),
        pd.DataFrame(
            slope_rows
        ),
    )


# ============================================================
# Scaling summary
# ============================================================

def summarize_slopes(
    df_slopes,
):

    rows = []

    for family, group in df_slopes.groupby(
        "family"
    ):

        slopes = np.asarray(
            group["slope"],
            dtype=float,
        )

        target = float(
            group[
                "target_slope"
            ].iloc[0]
        )

        valid = np.isfinite(
            slopes
        )

        slopes_valid = slopes[
            valid
        ]


        rows.append(
            {
                "family": family,

                "target_slope": (
                    target
                ),

                "n_geometry": int(
                    len(group)
                ),

                "n_valid": int(
                    np.sum(valid)
                ),

                "slope_min": float(
                    np.min(
                        slopes_valid
                    )
                ),

                "slope_p16": float(
                    np.percentile(
                        slopes_valid,
                        16,
                    )
                ),

                "slope_median": float(
                    np.median(
                        slopes_valid
                    )
                ),

                "slope_p84": float(
                    np.percentile(
                        slopes_valid,
                        84,
                    )
                ),

                "slope_max": float(
                    np.max(
                        slopes_valid
                    )
                ),

                "median_abs_error": float(
                    np.median(
                        np.abs(
                            slopes_valid
                            - target
                        )
                    )
                ),

                "fraction_within_0p05": float(
                    np.mean(
                        np.abs(
                            slopes_valid
                            - target
                        )
                        <= 0.05
                    )
                ),

                "fraction_within_0p10": float(
                    np.mean(
                        np.abs(
                            slopes_valid
                            - target
                        )
                        <= 0.10
                    )
                ),
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# Git provenance
# ============================================================

def git_info():

    def run_git(
        *args,
    ):
        try:

            result = subprocess.run(
                [
                    "git",
                    *args,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            return (
                result.stdout.strip()
            )

        except Exception:

            return None


    commit = run_git(
        "rev-parse",
        "HEAD",
    )

    short_commit = run_git(
        "rev-parse",
        "--short",
        "HEAD",
    )

    status = run_git(
        "status",
        "--porcelain",
    )

    return {
        "commit": commit,
        "short_commit": (
            short_commit
        ),
        "working_tree_clean": (
            status == ""
        )
        if status is not None
        else None,
        "status_porcelain": (
            status
        ),
    }


# ============================================================
# Print scientific summary
# ============================================================

def print_scientific_summary(
    broad_summary,
    pairwise,
    slope_summary,
    slopes,
):

    print()
    print()
    print("#" * 100)
    print("SCIENTIFIC GEOMETRY-ROBUSTNESS SUMMARY")
    print("#" * 100)


    # --------------------------------------------------------
    # Broad cases
    # --------------------------------------------------------

    if (
        broad_summary
        is not None
        and len(
            broad_summary
        )
        > 0
    ):

        print()
        print("BROAD D DISTRIBUTIONS")
        print("-" * 100)

        for _, row in (
            broad_summary
            .sort_values(
                "case"
            )
            .iterrows()
        ):

            print(
                f"{row['case']:22s} "
                f"Dfid={row['D_fiducial']:.6e} "
                f"median={row['D_median']:.6e} "
                f"[p16,p84]="
                f"[{row['D_p16']:.6e}, "
                f"{row['D_p84']:.6e}] "
                f"range/fid="
                f"[{row['D_min_over_fid']:.3g}, "
                f"{row['D_max_over_fid']:.3g}]"
            )


    # --------------------------------------------------------
    # Pairwise tests
    # --------------------------------------------------------

    if (
        pairwise
        is not None
        and len(
            pairwise
        )
        > 0
    ):

        temporal = pairwise[
            pairwise["test"]
            == "one_luminous_temporal_order"
        ]

        cancel = pairwise[
            pairwise["test"]
            == "finite_xi_cancel_vs_off"
        ]


        if len(
            temporal
        ):

            print()
            print(
                "Fraction of geometries with "
                "D_intermediate > D_short and D_long:"
            )

            print(
                " ",
                float(
                    np.mean(
                        temporal[
                            "condition"
                        ]
                    )
                ),
            )


        if len(
            cancel
        ):

            ratios = np.asarray(
                cancel[
                    "ratio_cancel_over_off"
                ],
                dtype=float,
            )

            print()
            print(
                "Finite-separation qf=qM / off-cancellation D ratio:"
            )

            print(
                "  median =",
                float(
                    np.median(
                        ratios
                    )
                ),
            )

            print(
                "  p16    =",
                float(
                    np.percentile(
                        ratios,
                        16,
                    )
                ),
            )

            print(
                "  p84    =",
                float(
                    np.percentile(
                        ratios,
                        84,
                    )
                ),
            )

            print(
                "  fraction D_cancel < D_off =",
                float(
                    np.mean(
                        cancel[
                            "condition"
                        ]
                    )
                ),
            )


    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    if (
        slope_summary
        is not None
        and len(
            slope_summary
        )
        > 0
    ):

        print()
        print("ASYMPTOTIC SLOPES")
        print("-" * 100)

        for _, row in (
            slope_summary
            .sort_values(
                "family"
            )
            .iterrows()
        ):

            print(
                f"{row['family']:8s} "
                f"target={row['target_slope']:.1f} "
                f"median={row['slope_median']:.6f} "
                f"[p16,p84]="
                f"[{row['slope_p16']:.6f}, "
                f"{row['slope_p84']:.6f}] "
                f"range="
                f"[{row['slope_min']:.6f}, "
                f"{row['slope_max']:.6f}] "
                f"within0.1="
                f"{row['fraction_within_0p10']:.3f}"
            )


        # ----------------------------------------------------
        # Largest slope deviations
        # ----------------------------------------------------

        tmp = slopes.copy()

        tmp[
            "abs_slope_error"
        ] = np.abs(
            tmp["slope"]
            - tmp[
                "target_slope"
            ]
        )

        tmp = tmp.sort_values(
            "abs_slope_error",
            ascending=False,
        )


        print()
        print(
            "Largest slope deviations:"
        )

        for _, row in (
            tmp.head(
                10
            ).iterrows()
        ):

            print(
                f"  "
                f"{row['family']:6s} "
                f"{row['geometry_id']:8s} "
                f"i={row['inclination_deg']:7.2f} "
                f"phi={row['phi_deg']:7.2f} "
                f"theta={row['theta_deg']:7.2f} "
                f"slope={row['slope']:.6f} "
                f"target={row['target_slope']:.1f} "
                f"R2={row['r2']:.6f}"
            )


    print()
    print("#" * 100)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "broad",
            "scaling",
            "all",
        ],
        default="all",
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )


    parser.add_argument(
        "--n-random",
        type=int,
        default=12,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )


    parser.add_argument(
        "--maxiter",
        type=int,
        default=MAXITER_DEFAULT,
    )


    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "validation_geometry_robustness"
        ),
    )


    args = parser.parse_args()


    outdir = (
        ROOT
        / args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    geometries = build_geometry_grid(
        n_random=args.n_random,
        seed=args.seed,
    )

    cases = build_broad_cases()


    # ========================================================
    # Smoke mode:
    #
    # only production/fiducial geometry
    # ========================================================

    if args.mode == "smoke":

        geometries = [
            g
            for g in geometries
            if is_fiducial_geometry(
                g
            )
        ]

        if len(
            geometries
        ) != 1:

            raise RuntimeError(
                "Smoke mode could not identify "
                "exactly one fiducial geometry."
            )


    print("=" * 100)
    print("ORBITAL GEOMETRY ROBUSTNESS")
    print("=" * 100)

    print(
        "mode              =",
        args.mode,
    )

    print(
        "N geometries      =",
        len(geometries),
    )

    print(
        "N deterministic   =",
        sum(
            g.geometry_kind
            == "deterministic"
            for g in geometries
        ),
    )

    print(
        "N random isotropic=",
        sum(
            g.geometry_kind
            == "random_isotropic"
            for g in geometries
        ),
    )

    print(
        "N_time            =",
        N_TIME,
    )

    print(
        "time window       = "
        f"t0 +/- {WINDOW_TE} tE"
    )

    print()


    # ========================================================
    # Save geometry table
    # ========================================================

    geometry_df = pd.DataFrame(
        [
            {
                **geometry_metadata(
                    g
                ),
                "fiducial_geometry": (
                    is_fiducial_geometry(
                        g
                    )
                ),
            }
            for g in geometries
        ]
    )

    geometry_df.to_csv(
        outdir
        / "geometry_grid.csv",
        index=False,
    )


    # ========================================================
    # Containers
    # ========================================================

    df_broad = pd.DataFrame()
    broad_summary = (
        pd.DataFrame()
    )
    pairwise = pd.DataFrame()

    df_scaling = pd.DataFrame()
    df_slopes = pd.DataFrame()
    slope_summary = (
        pd.DataFrame()
    )


    # ========================================================
    # Broad
    # ========================================================

    if args.mode in (
        "smoke",
        "broad",
        "all",
    ):

        df_broad = run_broad(
            geometries=geometries,
            cases=cases,
            workers=args.workers,
            maxiter=args.maxiter,
        )

        df_broad.to_csv(
            outdir
            / "geometry_broad.csv",
            index=False,
        )


        broad_summary = (
            summarize_broad(
                df_broad
            )
        )

        broad_summary.to_csv(
            outdir
            / "geometry_broad_summary.csv",
            index=False,
        )


        pairwise = (
            broad_pairwise_tests(
                df_broad
            )
        )

        pairwise.to_csv(
            outdir
            / "geometry_pairwise_tests.csv",
            index=False,
        )


    # ========================================================
    # Scaling
    # ========================================================

    if args.mode in (
        "smoke",
        "scaling",
        "all",
    ):

        (
            df_scaling,
            df_slopes,
        ) = run_scaling(
            geometries=geometries,
            workers=args.workers,
            maxiter=args.maxiter,
        )


        df_scaling.to_csv(
            outdir
            / "geometry_scaling_points.csv",
            index=False,
        )


        df_slopes.to_csv(
            outdir
            / "geometry_scaling_slopes.csv",
            index=False,
        )


        slope_summary = (
            summarize_slopes(
                df_slopes
            )
        )

        slope_summary.to_csv(
            outdir
            / "geometry_scaling_summary.csv",
            index=False,
        )


    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "git": git_info(),

        "t0": T0,
        "tE": TE,

        "window_tE": WINDOW_TE,
        "n_time": N_TIME,

        "M_total_Msun": (
            M_TOTAL
        ),

        "rEhat_AU_broad": (
            REHAT_AU
        ),

        "q_mass": (
            Q_MASS
        ),

        "seed": int(
            args.seed
        ),

        "n_random": int(
            args.n_random
        ),

        "n_geometries": int(
            len(geometries)
        ),

        "maxiter": int(
            args.maxiter
        ),

        "broad_cases": [
            asdict(case)
            for case in cases
        ],

        "scaling": {
            "P_days": (
                SCALING_P_DAYS
            ),

            "u0": (
                SCALING_U0
            ),

            "xi_over_u0_grid": (
                XI_OVER_U0_GRID.tolist()
            ),

            "numerical_floor": (
                D_NUMERICAL_FLOOR
            ),
        },
    }


    (
        outdir
        / "geometry_validation_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )


    # ========================================================
    # Final summary
    # ========================================================

    print_scientific_summary(
        broad_summary=(
            broad_summary
            if len(
                broad_summary
            )
            else None
        ),

        pairwise=(
            pairwise
            if len(
                pairwise
            )
            else None
        ),

        slope_summary=(
            slope_summary
            if len(
                slope_summary
            )
            else None
        ),

        slopes=(
            df_slopes
            if len(
                df_slopes
            )
            else None
        ),
    )


    print()
    print("Saved in:")
    print(
        " ",
        outdir,
    )


if __name__ == "__main__":
    main()
