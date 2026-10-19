#!/usr/bin/env python3

import argparse
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
# Validated CHE Roman/Pandeia snapshot
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
        f"Missing Roman/Pandeia snapshot: "
        f"{ROMAN_RUBIN_ROOT}"
    )

base.f146_backend.get_roman_rubin_root = (
    lambda: ROMAN_RUBIN_ROOT
)


# ============================================================
# Validation grid
# ============================================================

U0_GRID = [
    0.03,
    0.10,
    0.30,
]

TE_GRID = [
    10.0,
    30.0,
    100.0,
]

MAG_GRID = [
    19.0,
    21.0,
    23.0,
]

PERIOD_RATIO_GRID = [
    1.0,
    3.0,
    12.67,
    20.16,
    34.65,
]

DEFAULT_SEEDS = [
    2033101601,
    2033101602,
    2033101603,
]

PIE_BOUND = 40.0

OUTROOT = Path(
    "/tmp/roman_two_sign_policy_validation"
)


# ============================================================
# Oracle subset
#
# Covers:
# - small/large u0
# - short/long tE
# - bright/faint source
# - short/intermediate/long P/tE
# - known fiducial branch cases
# ============================================================

ORACLE_CONFIGS = [
    (0.03, 10.0, 19.0, 1.0),
    (0.30, 10.0, 23.0, 34.65),

    (0.03, 30.0, 23.0, 12.67),
    (0.10, 30.0, 21.0, 12.67),
    (0.10, 30.0, 21.0, 20.16),
    (0.10, 30.0, 21.0, 34.65),
    (0.30, 30.0, 19.0, 3.0),

    (0.03, 100.0, 23.0, 12.67),
    (0.10, 100.0, 21.0, 20.16),
    (0.30, 100.0, 19.0, 34.65),
]

ORACLE_SEEDS = [
    2033101602,
    2033101603,
]


# ============================================================
# Utilities
# ============================================================

def make_bspl_truth(
    *,
    t,
    t0,
    u0,
    tE,
    source_mag,
    P_days,
):
    truth_model = (
        base.roman.bspl_truth_magnification(
            t=t,
            t0_true=t0,
            u0_true=u0,
            tE_true=tE,
            P_days=P_days,
        )
    )

    A = np.asarray(
        truth_model["A_bspl"],
        dtype=float,
    )

    mag_truth = (
        float(source_mag)
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

    n_used = int(
        np.count_nonzero(keep)
    )

    if n_used < 10:
        raise RuntimeError(
            f"Too few Roman epochs: "
            f"{n_used}/{len(t)}"
        )

    return {
        "A":
            A,

        "t":
            np.asarray(
                t,
                dtype=float,
            )[keep],

        "mag_truth":
            mag_truth[keep],

        "err_mag":
            err_mag[keep],

        "keep":
            keep,

        "n_total":
            int(len(t)),

        "n_used":
            n_used,

        "fraction_rejected":
            float(
                1.0
                - n_used / len(t)
            ),

        "xi_rel":
            float(
                truth_model["xi_rel"]
            ),

        "omega":
            float(
                truth_model["omega"]
            ),
    }


def asimov_dataset(
    truth,
):
    ev, tel = (
        base.roman.build_event_from_magnitudes(
            t=truth["t"],
            magnitude=truth[
                "mag_truth"
            ],
            err_mag=truth[
                "err_mag"
            ],
            name="Roman",
        )
    )

    return {
        "event":
            ev,

        "telescope":
            tel,

        "flux":
            np.asarray(
                base.plain(
                    tel.lightcurve[
                        "flux"
                    ]
                ),
                dtype=float,
            ),

        "err_flux":
            np.asarray(
                base.plain(
                    tel.lightcurve[
                        "err_flux"
                    ]
                ),
                dtype=float,
            ),
    }


def noisy_dataset(
    truth,
    seed,
):
    return (
        base.make_noisy_dataset(
            t=truth["t"],
            mag_truth=truth[
                "mag_truth"
            ],
            err_mag=truth[
                "err_mag"
            ],
            seed=int(seed),
        )
    )


# ============================================================
# H0 PSPL + parallax fit
# ============================================================

def fit_h0_parallax(
    *,
    evaluator,
    flux,
    err_flux,
    t0_ref,
    tE_ref,
    start_u0,
    start_piEN=0.0,
    start_piEE=0.0,
    start_x=None,
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

    if start_x is None:

        x0 = np.array(
            [
                0.0,
                float(start_u0),
                0.0,
                float(start_piEN),
                float(start_piEE),
            ],
            dtype=float,
        )

    else:

        x0 = np.asarray(
            start_x,
            dtype=float,
        ).copy()

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
                tE_ref,
            )
        )

        A = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            piEN=x[3],
            piEE=x[4],
        )

        return base.profile_flux(
            A,
            flux,
            err_flux,
        )["residual"]

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
            tE_ref,
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

    pie_absmax = float(
        max(
            abs(opt.x[3]),
            abs(opt.x[4]),
        )
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
            float(
                opt.x[3]
            ),

        "piEE":
            float(
                opt.x[4]
            ),

        "piE":
            float(
                np.hypot(
                    opt.x[3],
                    opt.x[4],
                )
            ),

        "near_pie_bound":
            bool(
                pie_absmax
                > 0.95 * PIE_BOUND
            ),

        "nfev":
            int(
                opt.nfev
            ),

        "optimality":
            float(
                opt.optimality
            ),

        "elapsed":
            float(
                elapsed
            ),

        "x":
            np.asarray(
                opt.x,
                dtype=float,
            ),
    }


def trustworthy(
    fit,
):
    return bool(
        fit["success"]
        and np.isfinite(
            fit["chi2"]
        )
        and np.isfinite(
            fit["optimality"]
        )
        and not fit[
            "near_pie_bound"
        ]
    )


# ============================================================
# Generic two-sign policy
# ============================================================

def run_two_sign(
    *,
    evaluator,
    data,
    t0_ref,
    tE_ref,
    u0_guess,
):
    plus = fit_h0_parallax(
        evaluator=evaluator,
        flux=data["flux"],
        err_flux=data[
            "err_flux"
        ],
        t0_ref=t0_ref,
        tE_ref=tE_ref,
        start_u0=+abs(
            u0_guess
        ),
    )

    minus = fit_h0_parallax(
        evaluator=evaluator,
        flux=data["flux"],
        err_flux=data[
            "err_flux"
        ],
        t0_ref=t0_ref,
        tE_ref=tE_ref,
        start_u0=-abs(
            u0_guess
        ),
    )

    candidates = [
        ("plus", plus),
        ("minus", minus),
    ]

    valid = [
        pair
        for pair in candidates
        if trustworthy(
            pair[1]
        )
    ]

    if not valid:
        raise RuntimeError(
            "Both generic H0 fits failed."
        )

    name, best = min(
        valid,
        key=lambda pair:
            pair[1][
                "chi2"
            ],
    )

    gain = (
        float(
            plus["chi2"]
        )
        - float(
            best["chi2"]
        )
    )

    return {
        "plus":
            plus,

        "minus":
            minus,

        "best_name":
            name,

        "best":
            best,

        "gain":
            float(gain),
    }


# ============================================================
# Jerk starts used only as validation oracle
# ============================================================

def jerk_vectors(
    *,
    evaluator,
    t0_ref,
    tE_ref,
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

    local_window = min(
        20.0,
        max(
            5.0,
            0.75
            * float(tE_ref),
        ),
    )

    use = (
        np.abs(dt)
        <= local_window
    )

    if np.count_nonzero(
        use
    ) < 20:
        raise RuntimeError(
            "Too few epochs for jerk estimate."
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
            * float(tE_ref)
        )
    )

    pi_j_perp = float(
        np.dot(
            pi_j,
            eperp,
        )
    )

    jerk_NE = (
        -pi_j_perp
        * eperp
    )

    return {
        "alpha":
            alpha,

        "pi_j_perp":
            pi_j_perp,

        "jerk_NE":
            np.asarray(
                jerk_NE,
                dtype=float,
            ),
    }


def make_jerk_starts(
    *,
    evaluator,
    t0_ref,
    tE_ref,
    u0_guess,
):
    j = jerk_vectors(
        evaluator=evaluator,
        t0_ref=t0_ref,
        tE_ref=tE_ref,
    )

    starts = []

    for branch_name, vec in [
        (
            "jerk",
            j["jerk_NE"],
        ),
        (
            "opposite_jerk",
            -j["jerk_NE"],
        ),
    ]:

        for sign in [
            +1.0,
            -1.0,
        ]:

            starts.append(
                {
                    "name":
                        f"{branch_name}_"
                        f"u0{int(sign):+d}",

                    "u0":
                        sign
                        * abs(
                            u0_guess
                        ),

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

    return starts, j


# ============================================================
# Continuation oracle
# ============================================================

def continuation_windows(
    tE,
):
    ratios = [
        0.5,
        1.0,
        5.0 / 3.0,
        2.5,
        3.5,
    ]

    return [
        float(r)
        * float(tE)
        for r in ratios
    ]


def build_window_evaluators(
    *,
    t,
    t0_ref,
    tE_ref,
):
    evaluators = {}
    masks = {}

    for W in continuation_windows(
        tE_ref
    ):

        mask = (
            np.abs(
                np.asarray(t)
                - float(t0_ref)
            )
            <= float(W)
        )

        if np.count_nonzero(
            mask
        ) < 20:
            continue

        key = float(W)

        masks[key] = mask

        evaluators[key] = (
            par.PSPLParallaxEvaluator(
                t=np.asarray(t)[mask],
                t0_reference=t0_ref,
            )
        )

    masks["full"] = np.ones(
        len(t),
        dtype=bool,
    )

    evaluators["full"] = (
        par.PSPLParallaxEvaluator(
            t=np.asarray(t),
            t0_reference=t0_ref,
        )
    )

    return evaluators, masks


def run_continuation(
    *,
    start,
    evaluators,
    masks,
    data,
    t0_ref,
    tE_ref,
):
    current_x = None
    final = None

    stages = (
        [
            x
            for x in evaluators.keys()
            if x != "full"
        ]
        + ["full"]
    )

    stages_numeric = sorted(
        [
            x
            for x in stages
            if x != "full"
        ]
    )

    stages = (
        stages_numeric
        + ["full"]
    )

    for i, stage in enumerate(
        stages
    ):

        mask = masks[
            stage
        ]

        evaluator = evaluators[
            stage
        ]

        if i == 0:

            fit = fit_h0_parallax(
                evaluator=evaluator,
                flux=np.asarray(
                    data["flux"]
                )[mask],
                err_flux=np.asarray(
                    data["err_flux"]
                )[mask],
                t0_ref=t0_ref,
                tE_ref=tE_ref,
                start_u0=start[
                    "u0"
                ],
                start_piEN=start[
                    "piEN"
                ],
                start_piEE=start[
                    "piEE"
                ],
            )

        else:

            fit = fit_h0_parallax(
                evaluator=evaluator,
                flux=np.asarray(
                    data["flux"]
                )[mask],
                err_flux=np.asarray(
                    data["err_flux"]
                )[mask],
                t0_ref=t0_ref,
                tE_ref=tE_ref,
                start_u0=start[
                    "u0"
                ],
                start_x=current_x,
            )

        if not trustworthy(
            fit
        ):
            return {
                "success":
                    False,

                "chi2":
                    np.inf,
            }

        current_x = fit["x"]
        final = fit

    return final


def run_oracle(
    *,
    full_evaluator,
    window_evaluators,
    window_masks,
    data,
    t0_ref,
    tE_ref,
    u0_guess,
    two_sign,
):
    starts, jerk_info = (
        make_jerk_starts(
            evaluator=full_evaluator,
            t0_ref=t0_ref,
            tE_ref=tE_ref,
            u0_guess=u0_guess,
        )
    )

    direct = []

    continuation = []

    for start in starts:

        d = fit_h0_parallax(
            evaluator=full_evaluator,
            flux=data["flux"],
            err_flux=data[
                "err_flux"
            ],
            t0_ref=t0_ref,
            tE_ref=tE_ref,
            start_u0=start[
                "u0"
            ],
            start_piEN=start[
                "piEN"
            ],
            start_piEE=start[
                "piEE"
            ],
        )

        d["start_name"] = (
            start["name"]
        )

        direct.append(d)

        c = run_continuation(
            start=start,
            evaluators=window_evaluators,
            masks=window_masks,
            data=data,
            t0_ref=t0_ref,
            tE_ref=tE_ref,
        )

        c["start_name"] = (
            start["name"]
        )

        continuation.append(c)

    direct_valid = [
        x
        for x in direct
        if trustworthy(x)
    ]

    continuation_valid = [
        x
        for x in continuation
        if trustworthy(x)
    ]

    candidates = [
        (
            "two_sign",
            two_sign["best"],
        )
    ]

    if direct_valid:

        direct_best = min(
            direct_valid,
            key=lambda x:
                x["chi2"],
        )

        candidates.append(
            (
                "direct_jerk",
                direct_best,
            )
        )

    else:
        direct_best = None

    if continuation_valid:

        continuation_best = min(
            continuation_valid,
            key=lambda x:
                x["chi2"],
        )

        candidates.append(
            (
                "continuation",
                continuation_best,
            )
        )

    else:
        continuation_best = None

    best_name, best = min(
        candidates,
        key=lambda pair:
            pair[1]["chi2"],
    )

    jerk_extra = (
        float(
            two_sign["best"][
                "chi2"
            ]
        )
        - float(
            best["chi2"]
        )
    )

    return {
        "best_name":
            best_name,

        "best":
            best,

        "direct_best":
            direct_best,

        "continuation_best":
            continuation_best,

        "jerk_extra":
            float(
                jerk_extra
            ),

        "pi_j_perp":
            float(
                jerk_info[
                    "pi_j_perp"
                ]
            ),
    }


# ============================================================
# One configuration
# ============================================================

def prepare_configuration(
    *,
    u0,
    tE,
    source_mag,
    period_ratio,
    need_windows=False,
):
    P_days = (
        float(period_ratio)
        * float(tE)
    )

    t, t0 = (
        base.roman.build_roman_times(
            tE_true=float(tE)
        )
    )

    t = np.asarray(
        t,
        dtype=float,
    )

    truth = make_bspl_truth(
        t=t,
        t0=t0,
        u0=u0,
        tE=tE,
        source_mag=source_mag,
        P_days=P_days,
    )

    full_evaluator = (
        par.PSPLParallaxEvaluator(
            t=truth["t"],
            t0_reference=t0,
        )
    )

    if need_windows:

        window_evaluators, window_masks = (
            build_window_evaluators(
                t=truth["t"],
                t0_ref=t0,
                tE_ref=tE,
            )
        )

    else:

        window_evaluators = None
        window_masks = None

    return {
        "P_days":
            P_days,

        "t0":
            float(t0),

        "truth":
            truth,

        "full_evaluator":
            full_evaluator,

        "window_evaluators":
            window_evaluators,

        "window_masks":
            window_masks,
    }


def datasets_for_truth(
    truth,
    seeds,
):
    yield (
        "asimov",
        None,
        asimov_dataset(
            truth
        ),
    )

    for seed in seeds:

        yield (
            "noise",
            int(seed),
            noisy_dataset(
                truth,
                seed,
            ),
        )


# ============================================================
# Policy grid
# ============================================================

def run_policy_grid(
    seeds,
    max_configs=None,
):
    rows = []

    configs = [
        (
            u0,
            tE,
            mag,
            ratio,
        )
        for u0 in U0_GRID
        for tE in TE_GRID
        for mag in MAG_GRID
        for ratio in PERIOD_RATIO_GRID
    ]

    if max_configs is not None:
        configs = configs[
            : int(max_configs)
        ]

    n_configs = len(
        configs
    )

    tic_all = time.perf_counter()

    for i, (
        u0,
        tE,
        mag,
        ratio,
    ) in enumerate(
        configs,
        start=1,
    ):

        print()
        print(
            f"[POLICY {i}/{n_configs}] "
            f"u0={u0:g} "
            f"tE={tE:g} "
            f"mag={mag:g} "
            f"P/tE={ratio:g}",
            flush=True,
        )

        prep = prepare_configuration(
            u0=u0,
            tE=tE,
            source_mag=mag,
            period_ratio=ratio,
            need_windows=False,
        )

        truth = prep[
            "truth"
        ]

        for (
            dataset_type,
            seed,
            data,
        ) in datasets_for_truth(
            truth,
            seeds,
        ):

            result = run_two_sign(
                evaluator=prep[
                    "full_evaluator"
                ],
                data=data,
                t0_ref=prep[
                    "t0"
                ],
                tE_ref=tE,
                u0_guess=u0,
            )

            plus = result[
                "plus"
            ]

            minus = result[
                "minus"
            ]

            best = result[
                "best"
            ]

            row = {
                "dataset":
                    dataset_type,

                "seed":
                    seed,

                "u0_true":
                    u0,

                "tE_true":
                    tE,

                "source_mag":
                    mag,

                "P_over_tE":
                    ratio,

                "P_days":
                    prep[
                        "P_days"
                    ],

                "n_epochs_total":
                    truth[
                        "n_total"
                    ],

                "n_epochs_used":
                    truth[
                        "n_used"
                    ],

                "fraction_rejected":
                    truth[
                        "fraction_rejected"
                    ],

                "chi2_plus":
                    plus[
                        "chi2"
                    ],

                "chi2_minus":
                    minus[
                        "chi2"
                    ],

                "best_sign":
                    result[
                        "best_name"
                    ],

                "chi2_two_sign":
                    best[
                        "chi2"
                    ],

                "two_sign_gain":
                    result[
                        "gain"
                    ],

                "best_u0":
                    best[
                        "u0"
                    ],

                "best_tE":
                    best[
                        "tE"
                    ],

                "best_piEN":
                    best[
                        "piEN"
                    ],

                "best_piEE":
                    best[
                        "piEE"
                    ],

                "best_piE":
                    best[
                        "piE"
                    ],

                "plus_success":
                    plus[
                        "success"
                    ],

                "minus_success":
                    minus[
                        "success"
                    ],

                "plus_near_pie_bound":
                    plus[
                        "near_pie_bound"
                    ],

                "minus_near_pie_bound":
                    minus[
                        "near_pie_bound"
                    ],

                "plus_optimality":
                    plus[
                        "optimality"
                    ],

                "minus_optimality":
                    minus[
                        "optimality"
                    ],

                "fit_seconds":
                    (
                        plus[
                            "elapsed"
                        ]
                        + minus[
                            "elapsed"
                        ]
                    ),
            }

            rows.append(
                row
            )

            print(
                f"  {dataset_type:6s} "
                f"seed={str(seed):>10s} "
                f"best={result['best_name']:5s} "
                f"gain={result['gain']:.6g} "
                f"pi=({best['piEN']:+.4f},"
                f"{best['piEE']:+.4f})",
                flush=True,
            )

    df = pd.DataFrame(
        rows
    )

    outdir = (
        OUTROOT
        / "policy_grid"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        outdir
        / "policy_grid.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    elapsed = (
        time.perf_counter()
        - tic_all
    )

    print()
    print("=" * 100)
    print("POLICY GRID SUMMARY")
    print("=" * 100)

    print(
        "configs =",
        n_configs,
    )

    print(
        "datasets =",
        len(df),
    )

    print(
        "runtime_s =",
        elapsed,
    )

    print(
        "fit failures =",
        int(
            np.count_nonzero(
                ~df[
                    "plus_success"
                ]
                | ~df[
                    "minus_success"
                ]
            )
        ),
    )

    print(
        "piE bound warnings =",
        int(
            np.count_nonzero(
                df[
                    "plus_near_pie_bound"
                ]
                | df[
                    "minus_near_pie_bound"
                ]
            )
        ),
    )

    noisy = df[
        df["dataset"]
        == "noise"
    ]

    gain = noisy[
        "two_sign_gain"
    ].to_numpy(
        dtype=float
    )

    print()
    print(
        "NOISY DATASETS"
    )

    print(
        "N =",
        len(noisy),
    )

    print(
        "negative sign wins =",
        int(
            np.count_nonzero(
                noisy[
                    "best_sign"
                ]
                == "minus"
            )
        ),
        "/",
        len(noisy),
    )

    print(
        "fraction negative wins =",
        float(
            np.mean(
                noisy[
                    "best_sign"
                ]
                == "minus"
            )
        ),
    )

    print(
        "median two-sign gain =",
        float(
            np.median(
                gain
            )
        ),
    )

    print(
        "q90 two-sign gain =",
        float(
            np.quantile(
                gain,
                0.90,
            )
        ),
    )

    print(
        "q99 two-sign gain =",
        float(
            np.quantile(
                gain,
                0.99,
            )
        ),
    )

    print(
        "max two-sign gain =",
        float(
            np.max(
                gain
            )
        ),
    )

    print()
    print(
        "TOP 15 TWO-SIGN GAINS"
    )

    cols = [
        "u0_true",
        "tE_true",
        "source_mag",
        "P_over_tE",
        "seed",
        "two_sign_gain",
        "best_sign",
        "best_piEN",
        "best_piEE",
        "best_u0",
    ]

    print(
        noisy.sort_values(
            "two_sign_gain",
            ascending=False,
        )[
            cols
        ].head(
            15
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "saved:",
        path
    )

    return df


# ============================================================
# Oracle subset
# ============================================================

def run_oracle_subset(
    max_configs=None,
):
    rows = []

    configs = list(
        ORACLE_CONFIGS
    )

    if max_configs is not None:
        configs = configs[
            : int(max_configs)
        ]

    tic_all = time.perf_counter()

    for i, (
        u0,
        tE,
        mag,
        ratio,
    ) in enumerate(
        configs,
        start=1,
    ):

        print()
        print(
            "#" * 100
        )

        print(
            f"[ORACLE {i}/{len(configs)}] "
            f"u0={u0:g} "
            f"tE={tE:g} "
            f"mag={mag:g} "
            f"P/tE={ratio:g}"
        )

        print(
            "#" * 100
        )

        prep = prepare_configuration(
            u0=u0,
            tE=tE,
            source_mag=mag,
            period_ratio=ratio,
            need_windows=True,
        )

        truth = prep[
            "truth"
        ]

        for (
            dataset_type,
            seed,
            data,
        ) in datasets_for_truth(
            truth,
            ORACLE_SEEDS,
        ):

            two_sign = run_two_sign(
                evaluator=prep[
                    "full_evaluator"
                ],
                data=data,
                t0_ref=prep[
                    "t0"
                ],
                tE_ref=tE,
                u0_guess=u0,
            )

            oracle = run_oracle(
                full_evaluator=prep[
                    "full_evaluator"
                ],
                window_evaluators=prep[
                    "window_evaluators"
                ],
                window_masks=prep[
                    "window_masks"
                ],
                data=data,
                t0_ref=prep[
                    "t0"
                ],
                tE_ref=tE,
                u0_guess=u0,
                two_sign=two_sign,
            )

            direct = oracle[
                "direct_best"
            ]

            cont = oracle[
                "continuation_best"
            ]

            row = {
                "dataset":
                    dataset_type,

                "seed":
                    seed,

                "u0_true":
                    u0,

                "tE_true":
                    tE,

                "source_mag":
                    mag,

                "P_over_tE":
                    ratio,

                "P_days":
                    prep[
                        "P_days"
                    ],

                "n_epochs_used":
                    truth[
                        "n_used"
                    ],

                "two_sign_start":
                    two_sign[
                        "best_name"
                    ],

                "chi2_two_sign":
                    two_sign[
                        "best"
                    ][
                        "chi2"
                    ],

                "two_sign_piEN":
                    two_sign[
                        "best"
                    ][
                        "piEN"
                    ],

                "two_sign_piEE":
                    two_sign[
                        "best"
                    ][
                        "piEE"
                    ],

                "two_sign_gain":
                    two_sign[
                        "gain"
                    ],

                "oracle_strategy":
                    oracle[
                        "best_name"
                    ],

                "chi2_oracle":
                    oracle[
                        "best"
                    ][
                        "chi2"
                    ],

                "oracle_piEN":
                    oracle[
                        "best"
                    ][
                        "piEN"
                    ],

                "oracle_piEE":
                    oracle[
                        "best"
                    ][
                        "piEE"
                    ],

                "jerk_extra":
                    oracle[
                        "jerk_extra"
                    ],

                "pi_j_perp":
                    oracle[
                        "pi_j_perp"
                    ],

                "chi2_direct_jerk":
                    (
                        direct[
                            "chi2"
                        ]
                        if direct
                        is not None
                        else np.nan
                    ),

                "chi2_continuation":
                    (
                        cont[
                            "chi2"
                        ]
                        if cont
                        is not None
                        else np.nan
                    ),
            }

            rows.append(
                row
            )

            print(
                f"  {dataset_type:6s} "
                f"seed={str(seed):>10s} "
                f"two={two_sign['best']['chi2']:.6f} "
                f"oracle={oracle['best']['chi2']:.6f} "
                f"extra={oracle['jerk_extra']:.9g} "
                f"winner={oracle['best_name']}",
                flush=True,
            )

    df = pd.DataFrame(
        rows
    )

    outdir = (
        OUTROOT
        / "oracle_subset"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        outdir
        / "oracle_subset.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    elapsed = (
        time.perf_counter()
        - tic_all
    )

    extra = df[
        "jerk_extra"
    ].to_numpy(
        dtype=float
    )

    print()
    print("=" * 100)
    print("ORACLE SUMMARY")
    print("=" * 100)

    print(
        "configs =",
        len(configs),
    )

    print(
        "datasets =",
        len(df),
    )

    print(
        "runtime_s =",
        elapsed,
    )

    print(
        "median jerk extra =",
        float(
            np.median(
                extra
            )
        ),
    )

    print(
        "q90 jerk extra =",
        float(
            np.quantile(
                extra,
                0.90,
            )
        ),
    )

    print(
        "max jerk extra =",
        float(
            np.max(
                extra
            )
        ),
    )

    print(
        "extra > 1e-4 =",
        int(
            np.count_nonzero(
                extra > 1e-4
            )
        ),
        "/",
        len(extra),
    )

    print(
        "extra > 1e-3 =",
        int(
            np.count_nonzero(
                extra > 1e-3
            )
        ),
        "/",
        len(extra),
    )

    print(
        "extra > 0.01 =",
        int(
            np.count_nonzero(
                extra > 0.01
            )
        ),
        "/",
        len(extra),
    )

    print()
    print(
        "TOP ORACLE GAINS"
    )

    cols = [
        "dataset",
        "seed",
        "u0_true",
        "tE_true",
        "source_mag",
        "P_over_tE",
        "two_sign_start",
        "jerk_extra",
        "oracle_strategy",
        "two_sign_piEN",
        "two_sign_piEE",
        "oracle_piEN",
        "oracle_piEE",
    ]

    print(
        df.sort_values(
            "jerk_extra",
            ascending=False,
        )[
            cols
        ].head(
            15
        ).to_string(
            index=False
        )
    )

    pilot_pass = bool(
        np.all(
            np.isfinite(
                extra
            )
        )
        and np.max(
            extra
        )
        < 1e-3
    )

    print()
    print(
        "TWO-SIGN ORACLE PILOT PASS =",
        pilot_pass,
    )

    print(
        "criterion: max jerk extra < 1e-3"
    )

    print()
    print(
        "saved:",
        path
    )

    return df


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "policy",
            "oracle",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
    )

    args = parser.parse_args()

    OUTROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Roman/Pandeia root =",
        ROMAN_RUBIN_ROOT,
    )

    print(
        "mode =",
        args.mode,
    )

    if args.mode in {
        "policy",
        "all",
    }:

        run_policy_grid(
            seeds=args.seeds,
            max_configs=args.max_configs,
        )

    if args.mode in {
        "oracle",
        "all",
    }:

        run_oracle_subset(
            max_configs=args.max_configs,
        )


if __name__ == "__main__":
    main()
