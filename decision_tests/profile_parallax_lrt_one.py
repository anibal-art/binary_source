#!/usr/bin/env python3

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base,
)
from binary_source.analysis import (
    roman_bspl_pspl_lrt_levels as levels,
)
from binary_source.analysis import (
    roman_bspl_pspl_lrt_parallax as par,
)


# ======================================================================
# Timing registry
# ======================================================================

TIMING = defaultdict(
    lambda: {
        "calls": 0,
        "seconds": 0.0,
    }
)


def timed_wrapper(
    original,
    key,
):

    def wrapped(
        *args,
        **kwargs,
    ):

        t0 = time.perf_counter()

        try:
            return original(
                *args,
                **kwargs,
            )

        finally:

            dt = (
                time.perf_counter()
                - t0
            )

            TIMING[key][
                "calls"
            ] += 1

            TIMING[key][
                "seconds"
            ] += dt

    wrapped.__name__ = getattr(
        original,
        "__name__",
        key,
    )

    wrapped.__doc__ = getattr(
        original,
        "__doc__",
        None,
    )

    return wrapped


def install_timers():

    # ----------------------------------------------------------
    # Core numerical calls
    # ----------------------------------------------------------

    base.profile_flux = timed_wrapper(
        base.profile_flux,
        "profile_flux",
    )

    par.least_squares = timed_wrapper(
        par.least_squares,
        "scipy_least_squares",
    )

    # ----------------------------------------------------------
    # Magnification evaluators
    # ----------------------------------------------------------

    par.PSPLParallaxEvaluator.magnification = (
        timed_wrapper(
            par.PSPLParallaxEvaluator.magnification,
            "pspl_parallax_magnification",
        )
    )

    par.PhysicalCircularBSPLParallax.magnification = (
        timed_wrapper(
            par.PhysicalCircularBSPLParallax.magnification,
            "bspl_parallax_magnification",
        )
    )

    # ----------------------------------------------------------
    # Fit calls
    # ----------------------------------------------------------

    par.fit_h0_pspl_parallax = timed_wrapper(
        par.fit_h0_pspl_parallax,
        "fit_h0_pspl_parallax",
    )

    par._fit_h1_fixed_omega_parallax = (
        timed_wrapper(
            par._fit_h1_fixed_omega_parallax,
            "fit_h1_fixed_omega_parallax",
        )
    )

    par.fit_h1_joint_parallax = (
        timed_wrapper(
            par.fit_h1_joint_parallax,
            "fit_h1_joint_parallax",
        )
    )


def timed_stage(
    timings,
    name,
    func,
):

    t0 = time.perf_counter()

    result = func()

    timings[name] = (
        time.perf_counter()
        - t0
    )

    print(
        f"[stage] {name:<30s} "
        f"{timings[name]:10.3f} s",
        flush=True,
    )

    return result


# ======================================================================
# Main profiling run
# ======================================================================

def run(args):

    install_timers()

    stage = {}

    total_start = (
        time.perf_counter()
    )

    # ----------------------------------------------------------
    # Roman timestamps
    # ----------------------------------------------------------

    (
        t,
        t0_true,
    ) = timed_stage(
        stage,
        "roman_times",
        lambda:
            base.roman.build_roman_times(
                tE_true=args.tE,
            ),
    )

    # ----------------------------------------------------------
    # Pandeia truth/mask
    # ----------------------------------------------------------

    prepared = timed_stage(
        stage,
        "prepare_h0_configuration",
        lambda:
            base.prepare_h0_configuration(
                t=t,
                t0_true=t0_true,
                u0_true=args.u0,
                tE_true=args.tE,
                source_mag=args.source_mag,
                evaluator=None,
                blend_ratio=args.blend_ratio,
            ),
    )

    t_used = np.asarray(
        prepared["t"],
        dtype=float,
    )

    # ----------------------------------------------------------
    # Noise realization
    # ----------------------------------------------------------

    data = timed_stage(
        stage,
        "make_noisy_dataset",
        lambda:
            base.make_noisy_dataset(
                t=t_used,
                mag_truth=prepared[
                    "mag_truth"
                ],
                err_mag=prepared[
                    "err_mag"
                ],
                seed=args.seed,
            ),
    )

    # ----------------------------------------------------------
    # Evaluators / Roman ephemeris
    # ----------------------------------------------------------

    h0_evaluator = timed_stage(
        stage,
        "build_H0_parallax_evaluator",
        lambda:
            par.PSPLParallaxEvaluator(
                t=t_used,
                t0_reference=t0_true,
                ephemerides_path=(
                    args.ephemerides
                ),
            ),
    )

    h1_evaluator = timed_stage(
        stage,
        "build_H1_parallax_evaluator",
        lambda:
            par.PhysicalCircularBSPLParallax(
                t=t_used,
                t0_reference=t0_true,
                ephemerides_path=(
                    args.ephemerides
                ),
            ),
    )

    # ----------------------------------------------------------
    # H0
    # ----------------------------------------------------------

    h0 = timed_stage(
        stage,
        "fit_H0_PSPL_parallax",
        lambda:
            par.fit_h0_pspl_parallax(
                evaluator=h0_evaluator,
                flux=data["flux"],
                err_flux=data[
                    "err_flux"
                ],
                t0_ref=t0_true,
                u0_guess=args.u0,
                tE_ref=args.tE,
                piEN_start=0.0,
                piEE_start=0.0,
            ),
    )

    # ----------------------------------------------------------
    # Period grid
    # ----------------------------------------------------------

    P_grid = np.logspace(
        np.log10(
            args.p_min
        ),
        np.log10(
            args.p_max
        ),
        args.n_omega,
    )

    omega_grid = (
        2.0
        * np.pi
        / P_grid
    )

    profiles = {}

    # ----------------------------------------------------------
    # H1 level 1
    # ----------------------------------------------------------

    profiles[
        levels.LEVEL_1
    ] = timed_stage(
        stage,
        "H1_level1_omega",
        lambda:
            par.fit_level_1_parallax(
                evaluator=h1_evaluator,
                flux=data["flux"],
                err_flux=data[
                    "err_flux"
                ],
                omega_grid=omega_grid,
                h0_fit=h0,
                t0_ref=t0_true,
                tE_ref=args.tE,
            ),
    )

    # ----------------------------------------------------------
    # H1 level 2
    # ----------------------------------------------------------

    if (
        levels.level_number(
            args.max_level
        )
        >= 2
    ):

        profiles[
            levels.LEVEL_2
        ] = timed_stage(
            stage,
            "H1_level2_qmass",
            lambda:
                par.fit_level_2_parallax(
                    evaluator=h1_evaluator,
                    flux=data["flux"],
                    err_flux=data[
                        "err_flux"
                    ],
                    omega_grid=omega_grid,
                    t0_ref=t0_true,
                    tE_ref=args.tE,
                    previous_profile=profiles[
                        levels.LEVEL_1
                    ],
                ),
        )

    # ----------------------------------------------------------
    # H1 level 3
    # ----------------------------------------------------------

    if (
        levels.level_number(
            args.max_level
        )
        >= 3
    ):

        profiles[
            levels.LEVEL_3
        ] = timed_stage(
            stage,
            "H1_level3_qflux",
            lambda:
                par.fit_level_3_parallax(
                    evaluator=h1_evaluator,
                    flux=data["flux"],
                    err_flux=data[
                        "err_flux"
                    ],
                    omega_grid=omega_grid,
                    t0_ref=t0_true,
                    tE_ref=args.tE,
                    previous_profile=profiles[
                        levels.LEVEL_2
                    ],
                ),
        )

    total = (
        time.perf_counter()
        - total_start
    )

    stage[
        "TOTAL"
    ] = total

    # ==================================================================
    # Scientific result summary
    # ==================================================================

    print()
    print("=" * 120)
    print("SCIENTIFIC RESULT")
    print("=" * 120)

    print(
        "ephemerides =",
        h0_evaluator.ephemerides_path,
    )

    print(
        "epochs      =",
        len(t_used),
    )

    print(
        "H0 chi2     =",
        h0["chi2"],
    )

    print(
        "H0 piEN     =",
        h0["piEN"],
    )

    print(
        "H0 piEE     =",
        h0["piEE"],
    )

    for level in levels.LEVELS:

        if level not in profiles:
            continue

        best = (
            profiles[level][0]
        )

        T = max(
            0.0,
            float(
                h0["chi2"]
                - best["chi2"]
            ),
        )

        print(
            f"{level:24s} "
            f"T={T:10.6f} "
            f"chi2={best['chi2']:14.6f} "
            f"P={best['P_days']:12.4f} "
            f"piEN={best['piEN']:+.6f} "
            f"piEE={best['piEE']:+.6f}"
        )

    # ==================================================================
    # Stage profile
    # ==================================================================

    print()
    print("=" * 120)
    print("STAGE PROFILE")
    print("=" * 120)

    for key, value in stage.items():

        frac = (
            100.0
            * value
            / total
            if total > 0.0
            else np.nan
        )

        print(
            f"{key:<32s} "
            f"{value:10.3f} s "
            f"{frac:8.2f} %"
        )

    # ==================================================================
    # Inclusive call profile
    # ==================================================================

    print()
    print("=" * 120)
    print("INCLUSIVE CALL PROFILE")
    print(
        "(nested times overlap, so percentages "
        "need not sum to 100%)"
    )
    print("=" * 120)

    ordered = sorted(
        TIMING.items(),
        key=lambda item:
            item[1][
                "seconds"
            ],
        reverse=True,
    )

    for key, info in ordered:

        calls = int(
            info["calls"]
        )

        seconds = float(
            info["seconds"]
        )

        mean_ms = (
            1000.0
            * seconds
            / calls
            if calls > 0
            else np.nan
        )

        frac = (
            100.0
            * seconds
            / total
            if total > 0.0
            else np.nan
        )

        print(
            f"{key:<34s} "
            f"calls={calls:8d} "
            f"total={seconds:10.3f} s "
            f"mean={mean_ms:10.3f} ms "
            f"frac={frac:8.2f} %"
        )

    # ==================================================================
    # Machine-readable summary
    # ==================================================================

    summary = {
        "seed":
            int(
                args.seed
            ),

        "source_mag":
            float(
                args.source_mag
            ),

        "blend_ratio":
            float(
                args.blend_ratio
            ),

        "u0":
            float(
                args.u0
            ),

        "tE":
            float(
                args.tE
            ),

        "n_omega":
            int(
                args.n_omega
            ),

        "max_level":
            str(
                args.max_level
            ),

        "n_epochs":
            int(
                len(t_used)
            ),

        "ephemerides_path":
            str(
                h0_evaluator.ephemerides_path
            ),

        "stage_seconds":
            {
                key:
                    float(value)
                for key, value
                in stage.items()
            },

        "call_profile":
            {
                key: {
                    "calls":
                        int(
                            info["calls"]
                        ),

                    "seconds":
                        float(
                            info["seconds"]
                        ),
                }
                for key, info
                in TIMING.items()
            },

        "h0": {
            "chi2":
                float(
                    h0["chi2"]
                ),

            "piEN":
                float(
                    h0["piEN"]
                ),

            "piEE":
                float(
                    h0["piEE"]
                ),

            "nfev":
                int(
                    h0["nfev"]
                ),
        },
    }

    out_json = Path(
        args.json_out
    )

    out_json.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "JSON profile saved:",
        out_json,
    )


# ======================================================================
# CLI
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-mag",
        type=float,
        default=21.0,
    )

    parser.add_argument(
        "--blend-ratio",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--u0",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--tE",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2033101601,
    )

    parser.add_argument(
        "--p-min",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--p-max",
        type=float,
        default=100000.0,
    )

    parser.add_argument(
        "--n-omega",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--max-level",
        choices=[
            levels.LEVEL_1,
            levels.LEVEL_2,
            levels.LEVEL_3,
        ],
        default=levels.LEVEL_3,
    )

    parser.add_argument(
        "--ephemerides",
        default=None,
    )

    parser.add_argument(
        "--prof-out",
        default=(
            "/tmp/"
            "bspl_lrt_parallax_profile.prof"
        ),
    )

    parser.add_argument(
        "--json-out",
        default=(
            "/tmp/"
            "bspl_lrt_parallax_profile.json"
        ),
    )

    args = parser.parse_args()

    profiler = cProfile.Profile()

    profiler.enable()

    try:

        run(
            args
        )

    finally:

        profiler.disable()

        profiler.dump_stats(
            args.prof_out
        )

    print()
    print("=" * 120)
    print("CPROFILE — TOP 40 BY CUMULATIVE TIME")
    print("=" * 120)

    stats = pstats.Stats(
        profiler
    )

    stats.strip_dirs()
    stats.sort_stats(
        "cumulative"
    )
    stats.print_stats(
        40
    )

    print()
    print("=" * 120)
    print("CPROFILE — TOP 40 BY INTERNAL TIME")
    print("=" * 120)

    stats.sort_stats(
        "tottime"
    )
    stats.print_stats(
        40
    )

    print()
    print(
        "cProfile saved:",
        args.prof_out,
    )


if __name__ == "__main__":
    main()
