#!/usr/bin/env python3

import argparse
import sys
import inspect
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from binary_source.analysis import intrinsic_fit_tools as base
from binary_source.analysis import intrinsic_parallax as par


# ============================================================
# Scientific configuration
# ============================================================

T0_TRUE = 2461849.0
TE_TRUE = 30.0

FIT_WINDOW_TE = 3.5
N_TIME_FULL = 10_000

PIE_BOUND = 40.0

Q_MASS = 0.5
Q_FLUX = 0.0

EPHEMERIS_PATH = Path(
    "/export/storage3/rubin/microlensing/romanrubin/"
    "roman_rubin/"
    "ephemerides/Roman_positions.npy"
)

OUTPUT_DIR = (
    Path("results")
    / (
        "intrinsic_xallarap_vs_pspl_parallax_tE30_smoke"
        if "--smoke" in sys.argv
        else "intrinsic_xallarap_vs_pspl_parallax_tE30"
    )
)


# ============================================================
# Exact draft grids
# ============================================================

def make_full_grids():

    # Original intrinsic u0 grid:
    # 200 logarithmic nodes from 1e-2 to 10.
    # Roman/detection comparison retained u0 <= 1,
    # which gives 133 nodes.
    u0_all = np.logspace(
        -2.0,
        1.0,
        200,
    )

    u0_grid = u0_all[
        u0_all <= 1.0
    ]

    # Original period grid:
    # 60 logarithmic nodes, P = 10 ... 1e5 d.
    P_days = np.logspace(
        1.0,
        5.0,
        60,
    )

    return (
        np.asarray(u0_grid, dtype=float),
        np.asarray(P_days, dtype=float),
    )


# ============================================================
# Basic PSPL magnification
# ============================================================

def pspl_magnification(
    t,
    t0,
    u0,
    tE,
):
    tau = (
        np.asarray(t, dtype=float)
        - float(t0)
    ) / float(tE)

    u = np.sqrt(
        float(u0) ** 2
        + tau ** 2
    )

    # Numerical guard only.
    u = np.maximum(
        u,
        1.0e-14,
    )

    u2 = u * u

    return (
        (u2 + 2.0)
        /
        (
            u
            * np.sqrt(
                u2 + 4.0
            )
        )
    )


# ============================================================
# Resolve the existing validated PSPL+parallax evaluator
#
# We deliberately do NOT reimplement the Roman ephemeris
# machinery here.  We find the evaluator already present in
# intrinsic_parallax.py by its public behavior:
#
#   magnification(t0,u0,tE,piEN,piEE)
#
# ============================================================

def _constructor_kwargs(
    cls,
    t,
    t0_reference,
):
    sig = inspect.signature(
        cls.__init__
    )

    kwargs = {}

    for name, p in sig.parameters.items():

        if name == "self":
            continue

        lname = name.lower()

        if lname in {
            "t",
            "time",
            "times",
        }:
            kwargs[name] = np.asarray(
                t,
                dtype=float,
            )

        elif lname in {
            "t0_reference",
            "t0_ref",
            "t0",
        }:
            kwargs[name] = float(
                t0_reference
            )

        elif (
            "ephemer" in lname
            or lname in {
                "path_ephemerides",
                "ephemerides_path",
            }
        ):
            kwargs[name] = str(
                EPHEMERIS_PATH
            )

        elif p.default is inspect._empty:
            raise TypeError(
                f"Cannot satisfy required constructor "
                f"argument {name!r} for {cls.__name__}"
            )

    return kwargs


def resolve_parallax_evaluator(
    t,
    t0_reference,
):
    required_mag_args = {
        "t0",
        "u0",
        "tE",
        "piEN",
        "piEE",
    }

    candidates = []

    for name, obj in vars(par).items():

        if not inspect.isclass(obj):
            continue

        if not hasattr(
            obj,
            "magnification",
        ):
            continue

        try:
            msig = inspect.signature(
                obj.magnification
            )

        except Exception:
            continue

        params = set(
            msig.parameters.keys()
        )

        if not required_mag_args.issubset(
            params
        ):
            continue

        # Prefer a PSPL/parallax-specific evaluator with the
        # fewest additional magnification arguments.
        score = 0

        low_name = name.lower()

        if "pspl" in low_name:
            score += 100

        if "parallax" in low_name:
            score += 100

        extra = (
            params
            - required_mag_args
            - {"self"}
        )

        score -= len(extra)

        candidates.append(
            (
                score,
                name,
                obj,
            )
        )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    errors = []

    for score, name, cls in candidates:

        try:
            kwargs = _constructor_kwargs(
                cls,
                t=t,
                t0_reference=t0_reference,
            )

            evaluator = cls(
                **kwargs
            )

            # Critical zero-parallax equivalence check.
            test_u0 = 0.1

            A_par0 = np.asarray(
                evaluator.magnification(
                    t0=float(t0_reference),
                    u0=test_u0,
                    tE=TE_TRUE,
                    piEN=0.0,
                    piEE=0.0,
                ),
                dtype=float,
            )

            A_plain = pspl_magnification(
                t=t,
                t0=t0_reference,
                u0=test_u0,
                tE=TE_TRUE,
            )

            if (
                A_par0.shape
                != A_plain.shape
            ):
                raise RuntimeError(
                    "shape mismatch"
                )

            max_abs = float(
                np.max(
                    np.abs(
                        A_par0
                        - A_plain
                    )
                )
            )

            if (
                not np.isfinite(max_abs)
                or max_abs > 1.0e-8
            ):
                raise RuntimeError(
                    "piE=0 does not reproduce PSPL: "
                    f"max_abs={max_abs}"
                )

            return evaluator, name, max_abs

        except Exception as exc:
            errors.append(
                f"{name}: {exc!r}"
            )

    raise RuntimeError(
        "Could not identify the existing validated "
        "PSPL+parallax evaluator.\n"
        + "\n".join(errors)
    )


# ============================================================
# Exact trapezoidal D objective
#
# Uniform grid:
#
# D^2 =
# sum_i w_i (Atruth - Amodel)^2
# --------------------------------
# sum_i w_i (Atruth - 1)^2
#
# with endpoint trapezoidal weights 1/2.
#
# The least_squares residual is normalized so
#
#       sum(residual^2) == D^2
#
# ============================================================

def objective_normalization(
    A_truth,
):
    A_truth = np.asarray(
        A_truth,
        dtype=float,
    )

    weights = np.ones_like(
        A_truth
    )

    weights[0] = 0.5
    weights[-1] = 0.5

    denominator = float(
        np.sum(
            weights
            * (
                A_truth
                - 1.0
            ) ** 2
        )
    )

    if (
        not np.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise RuntimeError(
            "Invalid intrinsic-D denominator."
        )

    scale = (
        np.sqrt(weights)
        / np.sqrt(denominator)
    )

    return scale


def D_from_model(
    A_model,
    A_truth,
    scale,
):
    r = (
        np.asarray(
            A_model,
            dtype=float,
        )
        - np.asarray(
            A_truth,
            dtype=float,
        )
    ) * scale

    return float(
        np.sqrt(
            np.dot(
                r,
                r,
            )
        )
    )


# ============================================================
# Plain PSPL intrinsic fit
# ============================================================

def fit_plain_pspl(
    t,
    A_truth,
    u0_true,
    previous_x=None,
):
    lower, upper = (
        base.shape_bounds()
    )

    scale = objective_normalization(
        A_truth
    )

    def residual(x):

        t0, u0, tE = (
            base.unpack_shape_parameters(
                x,
                T0_TRUE,
                TE_TRUE,
            )
        )

        try:
            A = pspl_magnification(
                t=t,
                t0=t0,
                u0=u0,
                tE=tE,
            )

            if np.any(
                ~np.isfinite(A)
            ):
                raise ValueError

            return (
                A
                - A_truth
            ) * scale

        except Exception:
            return np.full(
                len(t),
                1.0e6,
                dtype=float,
            )

    starts = [
        np.array(
            [
                0.0,
                +float(u0_true),
                0.0,
            ]
        ),
        np.array(
            [
                0.0,
                -float(u0_true),
                0.0,
            ]
        ),
    ]

    if previous_x is not None:
        starts.append(
            np.asarray(
                previous_x,
                dtype=float,
            )
        )

    fits = []

    for i, x0 in enumerate(starts):

        x0 = np.minimum(
            np.maximum(
                x0,
                lower + 1.0e-12,
            ),
            upper - 1.0e-12,
        )

        opt = least_squares(
            residual,
            x0,
            bounds=(
                lower,
                upper,
            ),
            method="trf",
            xtol=1.0e-10,
            ftol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=2000,
        )

        r = residual(
            opt.x
        )

        D = float(
            np.sqrt(
                np.dot(
                    r,
                    r,
                )
            )
        )

        fits.append(
            {
                "D": D,
                "x": np.asarray(
                    opt.x,
                    dtype=float,
                ),
                "success": bool(
                    opt.success
                ),
                "nfev": int(
                    opt.nfev
                ),
                "optimality": float(
                    opt.optimality
                ),
                "start_id": int(i),
            }
        )

    return min(
        fits,
        key=lambda x: x["D"],
    )


# ============================================================
# PSPL + parallax intrinsic fit
# ============================================================

def fit_pspl_parallax(
    evaluator,
    A_truth,
    u0_true,
    plain_fit,
    previous_x=None,
):
    shape_lower, shape_upper = (
        base.shape_bounds()
    )

    lower = np.concatenate(
        [
            shape_lower,
            [
                -PIE_BOUND,
                -PIE_BOUND,
            ],
        ]
    )

    upper = np.concatenate(
        [
            shape_upper,
            [
                +PIE_BOUND,
                +PIE_BOUND,
            ],
        ]
    )

    scale = objective_normalization(
        A_truth
    )

    def residual(x):

        t0, u0, tE = (
            base.unpack_shape_parameters(
                x[:3],
                T0_TRUE,
                TE_TRUE,
            )
        )

        try:
            A = np.asarray(
                evaluator.magnification(
                    t0=t0,
                    u0=u0,
                    tE=tE,
                    piEN=float(x[3]),
                    piEE=float(x[4]),
                ),
                dtype=float,
            )

            if (
                A.shape != A_truth.shape
                or np.any(
                    ~np.isfinite(A)
                )
            ):
                raise ValueError

            return (
                A
                - A_truth
            ) * scale

        except Exception:
            return np.full(
                len(A_truth),
                1.0e6,
                dtype=float,
            )

    plain_x = np.asarray(
        plain_fit["x"],
        dtype=float,
    )

    # --------------------------------------------------------
    # Mandatory nested zero-parallax candidate.
    # This guarantees that PSPL+pi cannot be worse than PSPL.
    # --------------------------------------------------------

    nested_x = np.concatenate(
        [
            plain_x,
            [
                0.0,
                0.0,
            ],
        ]
    )

    nested_r = residual(
        nested_x
    )

    nested_D = float(
        np.sqrt(
            np.dot(
                nested_r,
                nested_r,
            )
        )
    )

    candidates = [
        {
            "D": nested_D,
            "x": nested_x.copy(),
            "success": True,
            "nfev": 0,
            "optimality": np.nan,
            "start_id": "nested_zero_eval",
        }
    ]

    # --------------------------------------------------------
    # Optimization starts:
    #
    # 1. best plain PSPL + piE=0
    # 2. same physical shape with opposite u0 sign
    # 3. continuation from neighbouring period
    #
    # Previous Roman tests showed that the two u0 signs recover
    # the relevant branches for the tE=30 regime; no jerk starts
    # are included in the production scan.
    # --------------------------------------------------------

    mirror_shape = plain_x.copy()
    mirror_shape[1] *= -1.0

    starts = [
        (
            "plain_plus",
            nested_x.copy(),
        ),
        (
            "plain_mirror",
            np.concatenate(
                [
                    mirror_shape,
                    [
                        0.0,
                        0.0,
                    ],
                ]
            ),
        ),
    ]

    if previous_x is not None:
        starts.append(
            (
                "continuation",
                np.asarray(
                    previous_x,
                    dtype=float,
                ),
            )
        )

    # Also include the truth geometric start.
    starts.extend(
        [
            (
                "truth_plus",
                np.array(
                    [
                        0.0,
                        +float(u0_true),
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ),
            (
                "truth_minus",
                np.array(
                    [
                        0.0,
                        -float(u0_true),
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ),
        ]
    )

    # Remove essentially duplicate starts.
    unique_starts = []

    for label, x0 in starts:

        x0 = np.asarray(
            x0,
            dtype=float,
        )

        duplicate = any(
            np.allclose(
                x0,
                old_x,
                rtol=0.0,
                atol=1.0e-12,
            )
            for _, old_x
            in unique_starts
        )

        if not duplicate:
            unique_starts.append(
                (
                    label,
                    x0,
                )
            )

    for label, x0 in unique_starts:

        x0 = np.minimum(
            np.maximum(
                x0,
                lower + 1.0e-12,
            ),
            upper - 1.0e-12,
        )

        try:
            opt = least_squares(
                residual,
                x0,
                bounds=(
                    lower,
                    upper,
                ),
                method="trf",
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
                max_nfev=3000,
            )

            r = residual(
                opt.x
            )

            D = float(
                np.sqrt(
                    np.dot(
                        r,
                        r,
                    )
                )
            )

            candidates.append(
                {
                    "D": D,
                    "x": np.asarray(
                        opt.x,
                        dtype=float,
                    ),
                    "success": bool(
                        opt.success
                    ),
                    "nfev": int(
                        opt.nfev
                    ),
                    "optimality": float(
                        opt.optimality
                    ),
                    "start_id": label,
                }
            )

        except Exception:
            continue

    best = min(
        candidates,
        key=lambda x: x["D"],
    )

    # Numerical nesting check.
    if best["D"] > plain_fit["D"] + 1.0e-8:
        raise RuntimeError(
            "Nested PSPL+parallax fit is worse than plain PSPL: "
            f"Dpi={best['D']} D0={plain_fit['D']}"
        )

    return best


# ============================================================
# Worker state
# ============================================================

_STATE = {}


def init_worker(
    t,
):
    t = np.asarray(
        t,
        dtype=float,
    )

    truth_evaluator = (
        base.PhysicalCircularBSPL(
            t=t,
            t0_reference=T0_TRUE,
        )
    )

    parallax_evaluator, class_name, zero_error = (
        resolve_parallax_evaluator(
            t=t,
            t0_reference=T0_TRUE,
        )
    )

    _STATE.clear()

    _STATE.update(
        {
            "t": t,
            "truth_evaluator":
                truth_evaluator,
            "parallax_evaluator":
                parallax_evaluator,
            "parallax_class":
                class_name,
            "zero_error":
                zero_error,
        }
    )


# ============================================================
# One u0 row
# ============================================================

def run_u0_row(
    task,
):
    (
        iu,
        u0_true,
        P_days_grid,
    ) = task

    t = _STATE["t"]

    truth_evaluator = (
        _STATE[
            "truth_evaluator"
        ]
    )

    parallax_evaluator = (
        _STATE[
            "parallax_evaluator"
        ]
    )

    rows = []

    previous_plain_x = None
    previous_parallax_x = None

    for ip, P_days in enumerate(
        P_days_grid
    ):

        try:
            omega = (
                2.0
                * np.pi
                / float(P_days)
            )

            A_truth = np.asarray(
                truth_evaluator.magnification(
                    t0=T0_TRUE,
                    u0=float(u0_true),
                    tE=TE_TRUE,
                    omega=omega,
                ),
                dtype=float,
            )

            if np.any(
                ~np.isfinite(
                    A_truth
                )
            ):
                raise RuntimeError(
                    "non-finite BSPL truth"
                )

            plain = fit_plain_pspl(
                t=t,
                A_truth=A_truth,
                u0_true=u0_true,
                previous_x=previous_plain_x,
            )

            parfit = fit_pspl_parallax(
                evaluator=parallax_evaluator,
                A_truth=A_truth,
                u0_true=u0_true,
                plain_fit=plain,
                previous_x=previous_parallax_x,
            )

            previous_plain_x = (
                plain["x"].copy()
            )

            previous_parallax_x = (
                parfit["x"].copy()
            )

            (
                t0_plain,
                u0_plain,
                tE_plain,
            ) = (
                base.unpack_shape_parameters(
                    plain["x"],
                    T0_TRUE,
                    TE_TRUE,
                )
            )

            (
                t0_pi,
                u0_pi,
                tE_pi,
            ) = (
                base.unpack_shape_parameters(
                    parfit["x"][:3],
                    T0_TRUE,
                    TE_TRUE,
                )
            )

            piEN = float(
                parfit["x"][3]
            )

            piEE = float(
                parfit["x"][4]
            )

            piE = float(
                np.hypot(
                    piEN,
                    piEE,
                )
            )

            D0 = float(
                plain["D"]
            )

            Dpi = float(
                parfit["D"]
            )

            ratio = (
                Dpi / D0
                if D0 > 0.0
                else np.nan
            )

            rows.append(
                {
                    "status":
                        "ok",

                    "iu":
                        int(iu),

                    "ip":
                        int(ip),

                    "u0_true":
                        float(u0_true),

                    "tE_true":
                        TE_TRUE,

                    "P_days":
                        float(P_days),

                    "P_over_tE":
                        float(
                            P_days
                            / TE_TRUE
                        ),

                    "D0":
                        D0,

                    "Dpi":
                        Dpi,

                    "Dpi_over_D0":
                        float(ratio),

                    "log10_D0":
                        float(
                            np.log10(
                                max(
                                    D0,
                                    1.0e-300,
                                )
                            )
                        ),

                    "log10_Dpi":
                        float(
                            np.log10(
                                max(
                                    Dpi,
                                    1.0e-300,
                                )
                            )
                        ),

                    "plain_t0":
                        float(t0_plain),

                    "plain_u0":
                        float(u0_plain),

                    "plain_tE":
                        float(tE_plain),

                    "par_t0":
                        float(t0_pi),

                    "par_u0":
                        float(u0_pi),

                    "par_tE":
                        float(tE_pi),

                    "piEN_fit":
                        piEN,

                    "piEE_fit":
                        piEE,

                    "piE_fit":
                        piE,

                    "plain_success":
                        bool(
                            plain[
                                "success"
                            ]
                        ),

                    "par_success":
                        bool(
                            parfit[
                                "success"
                            ]
                        ),

                    "plain_nfev":
                        int(
                            plain[
                                "nfev"
                            ]
                        ),

                    "par_nfev":
                        int(
                            parfit[
                                "nfev"
                            ]
                        ),

                    "plain_optimality":
                        float(
                            plain[
                                "optimality"
                            ]
                        ),

                    "par_optimality":
                        float(
                            parfit[
                                "optimality"
                            ]
                        ),

                    "par_start":
                        str(
                            parfit[
                                "start_id"
                            ]
                        ),

                    "near_pi_bound":
                        bool(
                            max(
                                abs(piEN),
                                abs(piEE),
                            )
                            > 0.95
                            * PIE_BOUND
                        ),

                    "parallax_evaluator":
                        _STATE[
                            "parallax_class"
                        ],

                    "piE_zero_max_abs":
                        float(
                            _STATE[
                                "zero_error"
                            ]
                        ),

                    "error":
                        "",
                }
            )

        except Exception:

            rows.append(
                {
                    "status":
                        "failed",

                    "iu":
                        int(iu),

                    "ip":
                        int(ip),

                    "u0_true":
                        float(u0_true),

                    "tE_true":
                        TE_TRUE,

                    "P_days":
                        float(P_days),

                    "P_over_tE":
                        float(
                            P_days
                            / TE_TRUE
                        ),

                    "D0":
                        np.nan,

                    "Dpi":
                        np.nan,

                    "Dpi_over_D0":
                        np.nan,

                    "error":
                        traceback.format_exc(),
                }
            )

            # Do not propagate a failed continuation.
            previous_plain_x = None
            previous_parallax_x = None

    return rows


# ============================================================
# Save helpers
# ============================================================

def save_results(
    rows,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(
        rows
    )

    if len(df) == 0:
        return df

    df = df.sort_values(
        [
            "iu",
            "ip",
        ]
    ).reset_index(
        drop=True
    )

    csv_path = (
        output_dir
        / "intrinsic_parallax_grid.csv"
    )

    parquet_path = (
        output_dir
        / "intrinsic_parallax_grid.parquet"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    df.to_parquet(
        parquet_path,
        index=False,
    )

    return df


def save_npz_and_plot(
    df,
    u0_grid,
    P_days_grid,
    output_dir,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    nu = len(
        u0_grid
    )

    np_ = len(
        P_days_grid
    )

    D0 = np.full(
        (nu, np_),
        np.nan,
    )

    Dpi = np.full(
        (nu, np_),
        np.nan,
    )

    ratio = np.full(
        (nu, np_),
        np.nan,
    )

    piE = np.full(
        (nu, np_),
        np.nan,
    )

    for _, row in df.iterrows():

        if row.get(
            "status"
        ) != "ok":
            continue

        iu = int(
            row["iu"]
        )

        ip = int(
            row["ip"]
        )

        D0[
            iu,
            ip,
        ] = float(
            row["D0"]
        )

        Dpi[
            iu,
            ip,
        ] = float(
            row["Dpi"]
        )

        ratio[
            iu,
            ip,
        ] = float(
            row[
                "Dpi_over_D0"
            ]
        )

        piE[
            iu,
            ip,
        ] = float(
            row[
                "piE_fit"
            ]
        )

    P_over_tE = (
        np.asarray(
            P_days_grid
        )
        / TE_TRUE
    )

    np.savez_compressed(
        output_dir
        / "intrinsic_parallax_grid.npz",

        u0_grid=np.asarray(
            u0_grid
        ),

        P_days_grid=np.asarray(
            P_days_grid
        ),

        P_over_tE=P_over_tE,

        D0=D0,

        Dpi=Dpi,

        Dpi_over_D0=ratio,

        piE_fit=piE,
    )

    # --------------------------------------------------------
    # Diagnostic three-panel figure
    # --------------------------------------------------------

    finite_D = np.concatenate(
        [
            D0[
                np.isfinite(D0)
                & (D0 > 0)
            ],
            Dpi[
                np.isfinite(Dpi)
                & (Dpi > 0)
            ],
        ]
    )

    if finite_D.size == 0:
        return

    vmin = max(
        float(
            np.nanmin(
                finite_D
            )
        ),
        1.0e-8,
    )

    vmax = float(
        np.nanmax(
            finite_D
        )
    )

    X, Y = np.meshgrid(
        P_over_tE,
        u0_grid,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            4.6,
        ),
        constrained_layout=True,
    )

    m0 = axes[0].pcolormesh(
        X,
        Y,
        D0,
        shading="auto",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    axes[0].set_title(
        r"$D_0$: BSPL vs PSPL"
    )

    fig.colorbar(
        m0,
        ax=axes[0],
        label=r"$D_0$",
    )

    m1 = axes[1].pcolormesh(
        X,
        Y,
        Dpi,
        shading="auto",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    axes[1].set_title(
        r"$D_\pi$: BSPL vs PSPL+parallax"
    )

    fig.colorbar(
        m1,
        ax=axes[1],
        label=r"$D_\pi$",
    )

    finite_pi = np.isfinite(
        piE
    )

    if np.any(
        finite_pi
    ):
        levels = np.array(
            [
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
                3.0,
                10.0,
            ]
        )

        zmin = np.nanmin(
            piE[
                finite_pi
            ]
        )

        zmax = np.nanmax(
            piE[
                finite_pi
            ]
        )

        use_levels = levels[
            (levels > zmin)
            & (levels < zmax)
        ]

        if len(
            use_levels
        ) > 0:
            cs = axes[1].contour(
                X,
                Y,
                piE,
                levels=use_levels,
                linewidths=0.8,
            )

            axes[1].clabel(
                cs,
                inline=True,
                fontsize=7,
                fmt="%g",
            )

    log_ratio = np.full_like(
        ratio,
        np.nan,
    )

    mask = (
        np.isfinite(ratio)
        & (ratio > 0)
    )

    log_ratio[
        mask
    ] = np.log10(
        ratio[
            mask
        ]
    )

    m2 = axes[2].pcolormesh(
        X,
        Y,
        log_ratio,
        shading="auto",
    )

    axes[2].set_title(
        r"$\log_{10}(D_\pi/D_0)$"
    )

    fig.colorbar(
        m2,
        ax=axes[2],
        label=r"$\log_{10}(D_\pi/D_0)$",
    )

    for ax in axes:

        ax.set_xscale(
            "log"
        )

        ax.set_yscale(
            "log"
        )

        ax.set_xlabel(
            r"$P/t_E$"
        )

        ax.set_ylabel(
            r"$u_0$"
        )

    fig.savefig(
        output_dir
        / "intrinsic_parallax_grid.png",
        dpi=180,
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use a small 4 x 6 grid "
            "for validation."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    args = parser.parse_args()

    if not EPHEMERIS_PATH.exists():
        raise FileNotFoundError(
            EPHEMERIS_PATH
        )

    u0_grid, P_days_grid = (
        make_full_grids()
    )

    n_time = N_TIME_FULL

    if args.smoke:

        u0_indices = np.array(
            [
                0,
                40,
                80,
                len(u0_grid) - 1,
            ]
        )

        P_indices = np.array(
            [
                0,
                10,
                20,
                30,
                40,
                59,
            ]
        )

        u0_grid = (
            u0_grid[
                u0_indices
            ]
        )

        P_days_grid = (
            P_days_grid[
                P_indices
            ]
        )

        n_time = 3000

    t = np.linspace(
        T0_TRUE
        - FIT_WINDOW_TE
        * TE_TRUE,

        T0_TRUE
        + FIT_WINDOW_TE
        * TE_TRUE,

        int(
            n_time
        ),
    )

    print("=" * 100)
    print(
        "INTRINSIC BSPL/XALLARAP vs PSPL+PARALLAX GRID"
    )
    print("=" * 100)

    print(
        "t0             =",
        T0_TRUE,
    )

    print(
        "tE             =",
        TE_TRUE,
    )

    print(
        "time window    =",
        (
            t.min(),
            t.max(),
        ),
    )

    print(
        "N_time         =",
        len(t),
    )

    print(
        "u0 nodes       =",
        len(
            u0_grid
        ),
    )

    print(
        "P nodes        =",
        len(
            P_days_grid
        ),
    )

    print(
        "P/tE range     =",
        (
            P_days_grid.min()
            / TE_TRUE,

            P_days_grid.max()
            / TE_TRUE,
        ),
    )

    print(
        "piE bounds     =",
        (
            -PIE_BOUND,
            +PIE_BOUND,
        ),
    )

    print(
        "workers        =",
        args.workers,
    )

    print(
        "ephemerides    =",
        EPHEMERIS_PATH,
    )

    print(
        "output         =",
        args.output_dir,
    )

    print("=" * 100)

    # --------------------------------------------------------
    # Before multiprocessing, validate evaluator once.
    # --------------------------------------------------------

    evaluator, class_name, zero_error = (
        resolve_parallax_evaluator(
            t=t,
            t0_reference=T0_TRUE,
        )
    )

    del evaluator

    print(
        "parallax evaluator =",
        class_name,
    )

    print(
        "piE=0 PSPL max abs difference =",
        zero_error,
    )

    tasks = [
        (
            int(iu),
            float(u0),
            np.asarray(
                P_days_grid,
                dtype=float,
            ),
        )
        for iu, u0
        in enumerate(
            u0_grid
        )
    ]

    rows = []

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        context = (
            mp.get_context(
                "fork"
            )
        )

    except ValueError:
        context = None

    executor_kwargs = {
        "max_workers":
            int(
                args.workers
            ),

        "initializer":
            init_worker,

        "initargs":
            (
                t,
            ),
    }

    if context is not None:
        executor_kwargs[
            "mp_context"
        ] = context

    with ProcessPoolExecutor(
        **executor_kwargs
    ) as executor:

        future_to_task = {
            executor.submit(
                run_u0_row,
                task,
            ):
            task
            for task in tasks
        }

        for completed, future in enumerate(
            as_completed(
                future_to_task
            ),
            start=1,
        ):

            task = (
                future_to_task[
                    future
                ]
            )

            iu, u0, _ = task

            try:
                new_rows = (
                    future.result()
                )

            except Exception:

                new_rows = [
                    {
                        "status":
                            "worker_failed",

                        "iu":
                            int(iu),

                        "ip":
                            -1,

                        "u0_true":
                            float(u0),

                        "error":
                            traceback.format_exc(),
                    }
                ]

            rows.extend(
                new_rows
            )

            df = save_results(
                rows=rows,
                output_dir=args.output_dir,
            )

            ok = int(
                np.count_nonzero(
                    df[
                        "status"
                    ]
                    == "ok"
                )
            )

            failed = (
                len(df)
                - ok
            )

            print(
                f"[{completed:3d}/"
                f"{len(tasks):3d}] "
                f"u0={u0:.6g} "
                f"rows={len(df)} "
                f"ok={ok} "
                f"failed={failed}"
            )

    df = save_results(
        rows=rows,
        output_dir=args.output_dir,
    )

    save_npz_and_plot(
        df=df,
        u0_grid=u0_grid,
        P_days_grid=P_days_grid,
        output_dir=args.output_dir,
    )

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(
        df[
            "status"
        ].value_counts(
            dropna=False
        )
    )

    good = df[
        df["status"]
        == "ok"
    ].copy()

    if len(good) > 0:

        print()
        print(
            "max(Dpi - D0) =",
            np.nanmax(
                good["Dpi"]
                - good["D0"]
            ),
        )

        print(
            "median Dpi/D0 =",
            np.nanmedian(
                good[
                    "Dpi_over_D0"
                ]
            ),
        )

        print(
            "min Dpi/D0    =",
            np.nanmin(
                good[
                    "Dpi_over_D0"
                ]
            ),
        )

        print(
            "max |piE|     =",
            np.nanmax(
                good[
                    "piE_fit"
                ]
            ),
        )

        print(
            "piE bound hits=",
            int(
                np.count_nonzero(
                    good[
                        "near_pi_bound"
                    ]
                )
            ),
        )

        print(
            "par fit success fraction =",
            float(
                np.mean(
                    good[
                        "par_success"
                    ]
                )
            ),
        )

    print()
    print(
        "CSV:",
        args.output_dir
        / "intrinsic_parallax_grid.csv",
    )

    print(
        "Parquet:",
        args.output_dir
        / "intrinsic_parallax_grid.parquet",
    )

    print(
        "NPZ:",
        args.output_dir
        / "intrinsic_parallax_grid.npz",
    )

    print(
        "Figure:",
        args.output_dir
        / "intrinsic_parallax_grid.png",
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
