#!/usr/bin/env python3

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base,
)
from binary_source.analysis import (
    roman_bspl_pspl_lrt_parallax as par,
)


# ============================================================
# CHE roman_rubin snapshot used by the validated Pandeia
# backend and Roman ephemerides.
#
# This override is local to this decision test.  It avoids
# changing the production backend while we validate the new
# fitting strategy.
# ============================================================

ROMAN_RUBIN_ROOT = Path(
    "/export/storage3/rubin/microlensing/romanrubin/"
    "roman_rubin_lrt_pandeia_20261016"
).resolve()

if not (
    ROMAN_RUBIN_ROOT
    / "photometry/data/"
    "roman_f146_pandeia_2026p1.csv"
).exists():
    raise FileNotFoundError(
        "Missing Roman F146 Pandeia grid: "
        f"{ROMAN_RUBIN_ROOT}"
    )

base.f146_backend.get_roman_rubin_root = (
    lambda: ROMAN_RUBIN_ROOT
)


# ============================================================
# Science configuration
# ============================================================

TE_TRUE = 30.0
U0_TRUE = 0.10
SOURCE_MAG = 21.0

PERIOD_RATIOS = [
    12.67,
    20.16,
    34.65,
]

PERIODS = [
    r * TE_TRUE
    for r in PERIOD_RATIOS
]

SEEDS = [
    2033101601,
    2033101602,
    2033101603,
    2033101604,
    2033101605,
]

CONTINUATION_WINDOWS = [
    15.0,
    30.0,
    50.0,
    75.0,
    105.0,
]

PIE_BOUND = 40.0

OUTDIR = Path(
    "/tmp/roman_branch_aware_h0"
)


# ============================================================
# Build deterministic BSPL truth with real Roman noise model
# ============================================================

def make_bspl_truth(
    t,
    t0_true,
    P_days,
):
    truth_model = (
        base.roman.bspl_truth_magnification(
            t=t,
            t0_true=t0_true,
            u0_true=U0_TRUE,
            tE_true=TE_TRUE,
            P_days=P_days,
        )
    )

    A = np.asarray(
        truth_model["A_bspl"],
        dtype=float,
    )

    mag_truth = (
        SOURCE_MAG
        - 2.5
        * np.log10(A)
    )

    instrument = (
        base.f146_backend.evaluate_f146_pandeia_ab(
            mag_truth
        )
    )

    err_mag = np.asarray(
        instrument["sigma_mag"],
        dtype=float,
    )

    valid = np.asarray(
        instrument["valid"],
        dtype=bool,
    )

    keep = (
        valid
        & np.isfinite(err_mag)
        & (err_mag > 0.0)
    )

    if np.count_nonzero(keep) < 10:
        raise RuntimeError(
            "Too few retained Roman epochs."
        )

    return {
        "A":
            A,

        "mag_truth":
            mag_truth,

        "err_mag":
            err_mag,

        "keep":
            keep,

        "t":
            np.asarray(t)[keep],

        "mag_truth_used":
            mag_truth[keep],

        "err_mag_used":
            err_mag[keep],
    }


# ============================================================
# Magnitudes -> flux dataset
# ============================================================

def dataset_from_magnitudes(
    t,
    mag,
    err_mag,
):
    ev, tel = (
        base.roman.build_event_from_magnitudes(
            t=t,
            magnitude=mag,
            err_mag=err_mag,
            name="Roman",
        )
    )

    flux = np.asarray(
        base.plain(
            tel.lightcurve["flux"]
        ),
        dtype=float,
    )

    err_flux = np.asarray(
        base.plain(
            tel.lightcurve["err_flux"]
        ),
        dtype=float,
    )

    return {
        "event": ev,
        "telescope": tel,
        "flux": flux,
        "err_flux": err_flux,
    }


def make_asimov_dataset(
    truth,
):
    return dataset_from_magnitudes(
        t=truth["t"],
        mag=truth["mag_truth_used"],
        err_mag=truth["err_mag_used"],
    )


def make_noisy_dataset(
    truth,
    seed,
):
    return base.make_noisy_dataset(
        t=truth["t"],
        mag_truth=truth[
            "mag_truth_used"
        ],
        err_mag=truth[
            "err_mag_used"
        ],
        seed=int(seed),
    )


# ============================================================
# H0 PSPL+parallax fit
#
# Same profiled-flux chi2 objective as the LRT.
# ============================================================

def fit_h0_parallax(
    evaluator,
    flux,
    err_flux,
    t0_ref,
    start,
):
    shape_lower, shape_upper = (
        base.shape_bounds()
    )

    lower = np.concatenate(
        [
            np.asarray(
                shape_lower,
                dtype=float,
            ),
            [
                -PIE_BOUND,
                -PIE_BOUND,
            ],
        ]
    )

    upper = np.concatenate(
        [
            np.asarray(
                shape_upper,
                dtype=float,
            ),
            [
                +PIE_BOUND,
                +PIE_BOUND,
            ],
        ]
    )

    x0 = np.array(
        [
            float(
                start.get(
                    "dt0_scaled",
                    0.0,
                )
            ),

            float(
                start["u0"]
            ),

            float(
                start.get(
                    "log_tE",
                    0.0,
                )
            ),

            float(
                start["piEN"]
            ),

            float(
                start["piEE"]
            ),
        ],
        dtype=float,
    )

    x0 = np.clip(
        x0,
        lower + 1e-12,
        upper - 1e-12,
    )

    def residuals(x):

        t0, u0, tE = (
            base.unpack_shape_parameters(
                x[:3],
                t0_ref,
                TE_TRUE,
            )
        )

        A = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            piEN=x[3],
            piEE=x[4],
        )

        prof = base.profile_flux(
            A,
            flux,
            err_flux,
        )

        return prof["residual"]

    tic = time.perf_counter()

    opt = least_squares(
        residuals,
        x0,
        bounds=(
            lower,
            upper,
        ),
        method="trf",
        x_scale="jac",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=5000,
    )

    elapsed = (
        time.perf_counter()
        - tic
    )

    t0, u0, tE = (
        base.unpack_shape_parameters(
            opt.x[:3],
            t0_ref,
            TE_TRUE,
        )
    )

    A = evaluator.magnification(
        t0=t0,
        u0=u0,
        tE=tE,
        piEN=opt.x[3],
        piEE=opt.x[4],
    )

    prof = base.profile_flux(
        A,
        flux,
        err_flux,
    )

    return {
        "success":
            bool(opt.success),

        "chi2":
            float(
                prof["chi2"]
            ),

        "t0":
            float(t0),

        "u0":
            float(u0),

        "tE":
            float(tE),

        "piEN":
            float(opt.x[3]),

        "piEE":
            float(opt.x[4]),

        "piE":
            float(
                np.hypot(
                    opt.x[3],
                    opt.x[4],
                )
            ),

        "fs":
            float(
                prof["fs"]
            ),

        "ft":
            float(
                prof["ft"]
            ),

        "nfev":
            int(opt.nfev),

        "optimality":
            float(
                opt.optimality
            ),

        "elapsed":
            float(elapsed),

        "x":
            np.asarray(
                opt.x,
                dtype=float,
            ),
    }


def result_to_start(
    result,
    t0_ref,
):
    # Preserve exactly the internal shape parameterization
    # used by shape_bounds/unpack_shape_parameters.
    return {
        "dt0_scaled":
            float(
                result["x"][0]
            ),

        "u0":
            float(
                result["x"][1]
            ),

        "log_tE":
            float(
                result["x"][2]
            ),

        "piEN":
            float(
                result["piEN"]
            ),

        "piEE":
            float(
                result["piEE"]
            ),
    }


# ============================================================
# Local Roman jerk direction
# ============================================================

def jerk_starts(
    evaluator,
    t0_ref,
):
    t = np.asarray(
        evaluator.t,
        dtype=float,
    )

    delta = np.asarray(
        evaluator.telescope.deltas_positions[
            "photometry"
        ],
        dtype=float,
    )

    dt = (
        t
        - float(t0_ref)
    )

    # Wide enough to be numerically stable, but local
    # compared with Roman orbital timescale.
    use = (
        np.abs(dt)
        <= 20.0
    )

    if np.count_nonzero(use) < 20:
        raise RuntimeError(
            "Too few points for local jerk estimate."
        )

    coeffs = []

    for component in range(2):

        c = (
            np.polynomial.polynomial.polyfit(
                dt[use],
                delta[
                    component,
                    use,
                ],
                deg=5,
            )
        )

        coeffs.append(c)

    coeffs = np.asarray(
        coeffs
    )

    alpha_vec = (
        2.0
        * coeffs[:, 2]
    )

    jerk_vec = (
        6.0
        * coeffs[:, 3]
    )

    alpha = float(
        np.linalg.norm(
            alpha_vec
        )
    )

    epar = (
        alpha_vec
        / alpha
    )

    eperp = np.array(
        [
            -epar[1],
            +epar[0],
        ],
        dtype=float,
    )

    pi_j = (
        (4.0 / 3.0)
        * jerk_vec
        / (
            alpha**2
            * TE_TRUE
        )
    )

    pi_j_perp = float(
        np.dot(
            pi_j,
            eperp,
        )
    )

    # For piE_truth = 0, Gould small-u0 branch is
    #
    # pi'_parallel = 0
    # pi'_perp     = -pi_j_perp
    #
    jerk_NE = (
        -pi_j_perp
        * eperp
    )

    print(
        "local geometry:",
        "alpha=",
        alpha,
        "pi_j_perp=",
        pi_j_perp,
        "jerk_NE=",
        jerk_NE,
    )

    starts = []

    # Both signs are included deliberately:
    # one is the formal jerk prediction,
    # the other protects against coordinate/sign convention.
    for branch_name, vec in [
        (
            "jerk",
            jerk_NE,
        ),
        (
            "opposite_jerk",
            -jerk_NE,
        ),
    ]:

        for sign in [
            +1.0,
            -1.0,
        ]:

            starts.append(
                {
                    "name":
                        f"{branch_name}_u0"
                        f"{int(sign):+d}",

                    "dt0_scaled":
                        0.0,

                    "u0":
                        sign
                        * U0_TRUE,

                    "log_tE":
                        0.0,

                    "piEN":
                        float(
                            vec[0]
                        ),

                    "piEE":
                        float(
                            vec[1]
                        ),
                }
            )

    return starts


# ============================================================
# Build evaluators once for each time subset
# ============================================================

def build_window_evaluators(
    t,
    t0_ref,
):
    evaluators = {}

    masks = {}

    for W in CONTINUATION_WINDOWS:

        mask = (
            np.abs(
                np.asarray(t)
                - float(t0_ref)
            )
            <= float(W)
        )

        if np.count_nonzero(mask) < 20:
            continue

        masks[W] = mask

        evaluators[W] = (
            par.PSPLParallaxEvaluator(
                t=np.asarray(t)[mask],
                t0_reference=t0_ref,
            )
        )

    evaluators["full"] = (
        par.PSPLParallaxEvaluator(
            t=np.asarray(t),
            t0_reference=t0_ref,
        )
    )

    masks["full"] = np.ones(
        len(t),
        dtype=bool,
    )

    return evaluators, masks


# ============================================================
# One continuation track
# ============================================================

def run_continuation(
    *,
    start,
    evaluators,
    masks,
    flux,
    err_flux,
    t0_ref,
):
    current_start = dict(
        start
    )

    stages = []

    for stage in (
        CONTINUATION_WINDOWS
        + ["full"]
    ):

        if stage not in evaluators:
            continue

        mask = masks[stage]

        fit = fit_h0_parallax(
            evaluator=evaluators[stage],
            flux=np.asarray(flux)[mask],
            err_flux=np.asarray(err_flux)[mask],
            t0_ref=t0_ref,
            start=current_start,
        )

        fit[
            "stage"
        ] = str(stage)

        stages.append(
            fit
        )

        if not (
            fit["success"]
            and np.isfinite(
                fit["chi2"]
            )
        ):
            break

        current_start = (
            result_to_start(
                fit,
                t0_ref,
            )
        )

    if (
        not stages
        or stages[-1]["stage"]
        != "full"
    ):
        return {
            "success":
                False,

            "chi2":
                np.inf,

            "stages":
                stages,
        }

    result = dict(
        stages[-1]
    )

    result[
        "stages"
    ] = stages

    return result


# ============================================================
# One observational dataset
# ============================================================

def run_dataset(
    *,
    label,
    seed,
    P_days,
    truth,
    data,
    evaluators,
    masks,
    jerk_bank,
    t0_ref,
):
    flux = np.asarray(
        data["flux"],
        dtype=float,
    )

    err_flux = np.asarray(
        data["err_flux"],
        dtype=float,
    )

    full_eval = (
        evaluators["full"]
    )


    # --------------------------------------------------------
    # 1. Baseline single start
    # --------------------------------------------------------

    baseline_start = {
        "name":
            "zero_u0+",

        "dt0_scaled":
            0.0,

        "u0":
            U0_TRUE,

        "log_tE":
            0.0,

        "piEN":
            0.0,

        "piEE":
            0.0,
    }

    baseline = fit_h0_parallax(
        evaluator=full_eval,
        flux=flux,
        err_flux=err_flux,
        t0_ref=t0_ref,
        start=baseline_start,
    )


    # --------------------------------------------------------
    # 2. Direct full-data jerk starts
    # --------------------------------------------------------

    direct_rows = []

    for start in jerk_bank:

        fit = fit_h0_parallax(
            evaluator=full_eval,
            flux=flux,
            err_flux=err_flux,
            t0_ref=t0_ref,
            start=start,
        )

        fit[
            "start_name"
        ] = start[
            "name"
        ]

        direct_rows.append(
            fit
        )

    direct_best = min(
        direct_rows,
        key=lambda r:
            r["chi2"],
    )


    # --------------------------------------------------------
    # 3. Window-continuation jerk starts
    # --------------------------------------------------------

    continuation_rows = []

    for start in jerk_bank:

        fit = run_continuation(
            start=start,
            evaluators=evaluators,
            masks=masks,
            flux=flux,
            err_flux=err_flux,
            t0_ref=t0_ref,
        )

        fit[
            "start_name"
        ] = start[
            "name"
        ]

        continuation_rows.append(
            fit
        )

    continuation_best = min(
        continuation_rows,
        key=lambda r:
            r["chi2"],
    )


    # --------------------------------------------------------
    # Never allow branch-aware strategy to worsen baseline.
    # --------------------------------------------------------

    candidates = [
        (
            "baseline",
            baseline,
        ),
        (
            "direct_jerk",
            direct_best,
        ),
        (
            "continuation",
            continuation_best,
        ),
    ]

    strategy, best = min(
        candidates,
        key=lambda pair:
            pair[1]["chi2"],
    )


    improvement = (
        float(
            baseline["chi2"]
        )
        - float(
            best["chi2"]
        )
    )


    print()
    print(
        f"{label:8s}",
        f"P={P_days:9.3f}",
        f"seed={str(seed):>10s}",
    )

    print(
        " baseline     ",
        f"chi2={baseline['chi2']:.6f}",
        f"pi=({baseline['piEN']:+.6f},"
        f"{baseline['piEE']:+.6f})",
        f"u0={baseline['u0']:+.6f}",
        f"nfev={baseline['nfev']}",
    )

    print(
        " direct best  ",
        f"chi2={direct_best['chi2']:.6f}",
        f"pi=({direct_best['piEN']:+.6f},"
        f"{direct_best['piEE']:+.6f})",
        f"start={direct_best['start_name']}",
    )

    print(
        " cont. best   ",
        f"chi2={continuation_best['chi2']:.6f}",
        f"pi=({continuation_best['piEN']:+.6f},"
        f"{continuation_best['piEE']:+.6f})",
        f"start={continuation_best['start_name']}",
    )

    print(
        " WINNER       ",
        strategy,
        f"delta_chi2={improvement:+.6f}",
    )


    return {
        "dataset":
            label,

        "seed":
            seed,

        "P_days":
            float(P_days),

        "P_over_tE":
            float(
                P_days
                / TE_TRUE
            ),

        "n_epochs":
            int(
                len(
                    truth["t"]
                )
            ),

        "chi2_baseline":
            float(
                baseline["chi2"]
            ),

        "baseline_piEN":
            float(
                baseline["piEN"]
            ),

        "baseline_piEE":
            float(
                baseline["piEE"]
            ),

        "baseline_u0":
            float(
                baseline["u0"]
            ),

        "baseline_tE":
            float(
                baseline["tE"]
            ),

        "baseline_nfev":
            int(
                baseline["nfev"]
            ),

        "baseline_optimality":
            float(
                baseline[
                    "optimality"
                ]
            ),

        "chi2_direct":
            float(
                direct_best[
                    "chi2"
                ]
            ),

        "direct_piEN":
            float(
                direct_best[
                    "piEN"
                ]
            ),

        "direct_piEE":
            float(
                direct_best[
                    "piEE"
                ]
            ),

        "direct_start":
            direct_best[
                "start_name"
            ],

        "chi2_continuation":
            float(
                continuation_best[
                    "chi2"
                ]
            ),

        "continuation_piEN":
            float(
                continuation_best[
                    "piEN"
                ]
            ),

        "continuation_piEE":
            float(
                continuation_best[
                    "piEE"
                ]
            ),

        "continuation_start":
            continuation_best[
                "start_name"
            ],

        "best_strategy":
            strategy,

        "chi2_best":
            float(
                best[
                    "chi2"
                ]
            ),

        "best_piEN":
            float(
                best[
                    "piEN"
                ]
            ),

        "best_piEE":
            float(
                best[
                    "piEE"
                ]
            ),

        "best_u0":
            float(
                best[
                    "u0"
                ]
            ),

        "best_tE":
            float(
                best[
                    "tE"
                ]
            ),

        "delta_chi2_improvement":
            float(
                improvement
            ),
    }


# ============================================================
# Main
# ============================================================

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


t_all, t0_true = (
    base.roman.build_roman_times(
        tE_true=TE_TRUE
    )
)

t_all = np.asarray(
    t_all,
    dtype=float,
)


print()
print("=" * 120)
print("ROMAN OBSERVATIONAL TEST OF BRANCH-AWARE PSPL+PARALLAX FITTING")
print("=" * 120)

print(
    "t0_true    =",
    t0_true,
)

print(
    "tE_true    =",
    TE_TRUE,
)

print(
    "u0_true    =",
    U0_TRUE,
)

print(
    "source_mag =",
    SOURCE_MAG,
)

print(
    "periods    =",
    PERIODS,
)


rows = []


for P_days in PERIODS:

    print()
    print("#" * 120)
    print(
        "BSPL TRUTH:",
        f"P={P_days:.6f} d",
        f"P/tE={P_days/TE_TRUE:.6f}",
    )
    print("#" * 120)


    truth = make_bspl_truth(
        t=t_all,
        t0_true=t0_true,
        P_days=P_days,
    )


    print(
        "Roman epochs:",
        len(t_all),
        "-> retained:",
        len(
            truth["t"]
        ),
    )


    evaluators, masks = (
        build_window_evaluators(
            truth["t"],
            t0_true,
        )
    )


    jerk_bank = jerk_starts(
        evaluators["full"],
        t0_true,
    )


    print(
        "branch starts:"
    )

    for s in jerk_bank:
        print(
            " ",
            s["name"],
            f"u0={s['u0']:+.3f}",
            f"pi=({s['piEN']:+.6f},"
            f"{s['piEE']:+.6f})",
        )


    # --------------------------------------------------------
    # Asimov
    # --------------------------------------------------------

    asimov = (
        make_asimov_dataset(
            truth
        )
    )

    rows.append(
        run_dataset(
            label="asimov",
            seed=None,
            P_days=P_days,
            truth=truth,
            data=asimov,
            evaluators=evaluators,
            masks=masks,
            jerk_bank=jerk_bank,
            t0_ref=t0_true,
        )
    )


    # --------------------------------------------------------
    # Noisy Roman realizations
    # --------------------------------------------------------

    for seed in SEEDS:

        data = (
            make_noisy_dataset(
                truth,
                seed,
            )
        )

        rows.append(
            run_dataset(
                label="noise",
                seed=seed,
                P_days=P_days,
                truth=truth,
                data=data,
                evaluators=evaluators,
                masks=masks,
                jerk_bank=jerk_bank,
                t0_ref=t0_true,
            )
        )


df = pd.DataFrame(
    rows
)


out = (
    OUTDIR
    / "roman_branch_aware_h0.csv"
)

df.to_csv(
    out,
    index=False,
)


print()
print("=" * 120)
print("ALL RESULTS")
print("=" * 120)

print(
    df[
        [
            "dataset",
            "seed",
            "P_over_tE",
            "chi2_baseline",
            "chi2_direct",
            "chi2_continuation",
            "chi2_best",
            "delta_chi2_improvement",
            "best_strategy",
            "baseline_piEN",
            "baseline_piEE",
            "best_piEN",
            "best_piEE",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 120)
print("NOISY-ROMAN SUMMARY")
print("=" * 120)


noise = (
    df[
        df["dataset"]
        == "noise"
    ]
    .copy()
)


for ratio, g in noise.groupby(
    "P_over_tE"
):

    d = (
        g[
            "delta_chi2_improvement"
        ]
        .to_numpy(
            dtype=float
        )
    )

    print()
    print(
        f"P/tE={ratio:.6f}"
    )

    print(
        "  N                    =",
        len(g),
    )

    print(
        "  median delta chi2    =",
        np.median(d),
    )

    print(
        "  max delta chi2       =",
        np.max(d),
    )

    print(
        "  improved > 0.1       =",
        int(
            np.count_nonzero(
                d > 0.1
            )
        ),
        "/",
        len(d),
    )

    print(
        "  improved > 1         =",
        int(
            np.count_nonzero(
                d > 1.0
            )
        ),
        "/",
        len(d),
    )

    print(
        "  improved > 9         =",
        int(
            np.count_nonzero(
                d > 9.0
            )
        ),
        "/",
        len(d),
    )

    print(
        "  winners:"
    )

    print(
        g[
            "best_strategy"
        ].value_counts()
        .to_string()
    )


print()
print("=" * 120)
print("GLOBAL SUMMARY")
print("=" * 120)

d = (
    noise[
        "delta_chi2_improvement"
    ]
    .to_numpy(
        dtype=float
    )
)

print(
    "N noisy               =",
    len(noise),
)

print(
    "median improvement    =",
    np.median(d),
)

print(
    "q90 improvement       =",
    np.quantile(
        d,
        0.90,
    ),
)

print(
    "max improvement       =",
    np.max(d),
)

print(
    "fraction > 0.1        =",
    np.mean(
        d > 0.1
    ),
)

print(
    "fraction > 1          =",
    np.mean(
        d > 1.0
    ),
)

print(
    "fraction > 9          =",
    np.mean(
        d > 9.0
    ),
)

print()
print(
    "strategy winners:"
)

print(
    noise[
        "best_strategy"
    ]
    .value_counts()
    .to_string()
)

print()
print(
    "saved:",
    out
)
