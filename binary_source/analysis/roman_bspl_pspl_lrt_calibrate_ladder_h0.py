#!/usr/bin/env python3

"""
Parallel H0 calibration of the Roman BSPL-vs-PSPL LRT ladder.

For each noisy PSPL realization, fit the same dataset with

    H1(1): omega
    H1(2): omega + q_mass
    H1(3): omega + q_mass + q_flux
    H1(4): omega + q_mass + q_flux + geometry

up to --max-level.

The same noisy realization, timestamps, Pandeia saturation mask,
H0 fit and omega search range are shared by all H1 levels.

This script intentionally leaves the original validated level-1
calibration driver untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import traceback

from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

from pathlib import Path

import numpy as np

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base,
)

from binary_source.analysis import (
    roman_bspl_pspl_lrt_levels as ladder,
)


# ============================================================
# Worker state
# ============================================================

_STATE = {}


def _init_worker(
    source_mag,
    u0_true,
    tE_true,
    p_min,
    p_max,
    n_omega,
    max_level,
):
    """
    Build deterministic quantities once per worker.

    In particular:
      - Roman timestamps,
      - truth-derived Pandeia saturation mask,
      - retained timestamps,
      - BSPL evaluator,
      - omega grid.
    """

    source_mag = float(source_mag)
    u0_true = float(u0_true)
    tE_true = float(tE_true)

    t, t0_true = (
        base.roman.build_roman_times(
            tE_true=tE_true,
        )
    )

    # This is used only to construct the validated deterministic
    # H0 truth / saturation mask.
    legacy_evaluator = (
        base.PhysicalCircularBSPL(
            t=t,
            t0_reference=t0_true,
        )
    )

    prepared = (
        base.prepare_h0_configuration(
            t=t,
            t0_true=t0_true,
            u0_true=u0_true,
            tE_true=tE_true,
            source_mag=source_mag,
            evaluator=legacy_evaluator,
        )
    )

    t_used = np.asarray(
        prepared["t"],
        dtype=float,
    )

    evaluator = (
        ladder.PhysicalCircularBSPLLadder(
            t=t_used,
            t0_reference=t0_true,
        )
    )

    P_grid = np.logspace(
        np.log10(float(p_min)),
        np.log10(float(p_max)),
        int(n_omega),
    )

    omega_grid = (
        2.0
        * np.pi
        / P_grid
    )

    _STATE.clear()

    _STATE.update(
        {
            "source_mag":
                source_mag,

            "u0_true":
                u0_true,

            "tE_true":
                tE_true,

            "t0_true":
                float(t0_true),

            "prepared":
                prepared,

            "evaluator":
                evaluator,

            "omega_grid":
                omega_grid,

            "p_min":
                float(p_min),

            "p_max":
                float(p_max),

            "max_level":
                str(max_level),
        }
    )


# ============================================================
# One realization
# ============================================================

def _run_one(
    seed,
):

    try:

        seed = int(seed)

        prepared = (
            _STATE[
                "prepared"
            ]
        )

        t_used = np.asarray(
            prepared["t"],
            dtype=float,
        )

        t0_true = float(
            _STATE[
                "t0_true"
            ]
        )

        u0_true = float(
            _STATE[
                "u0_true"
            ]
        )

        tE_true = float(
            _STATE[
                "tE_true"
            ]
        )

        source_mag = float(
            _STATE[
                "source_mag"
            ]
        )

        data = (
            base.make_noisy_dataset(
                t=t_used,
                mag_truth=prepared[
                    "mag_truth"
                ],
                err_mag=prepared[
                    "err_mag"
                ],
                seed=seed,
            )
        )

        h0 = (
            base.fit_h0_pspl(
                t=t_used,
                flux=data[
                    "flux"
                ],
                err_flux=data[
                    "err_flux"
                ],
                t0_ref=t0_true,
                u0_guess=u0_true,
                tE_ref=tE_true,
            )
        )

        profiles = (
            ladder.fit_h1_ladder(
                evaluator=_STATE[
                    "evaluator"
                ],
                flux=data[
                    "flux"
                ],
                err_flux=data[
                    "err_flux"
                ],
                omega_grid=_STATE[
                    "omega_grid"
                ],
                h0_fit=h0,
                t0_ref=t0_true,
                tE_ref=tE_true,
                max_level=_STATE[
                    "max_level"
                ],
            )
        )

        truth = (
            prepared[
                "truth"
            ]
        )

        rows = []

        for level in ladder.LEVELS:

            if level not in profiles:
                continue

            best = (
                profiles[
                    level
                ][0]
            )

            raw_delta = (
                float(
                    h0[
                        "chi2"
                    ]
                )
                - float(
                    best[
                        "chi2"
                    ]
                )
            )

            T = max(
                0.0,
                raw_delta,
            )

            rows.append(
                {
                    "seed":
                        seed,

                    "source_mag":
                        source_mag,

                    "u0_true":
                        u0_true,

                    "tE_true":
                        tE_true,

                    "h1_level":
                        level,

                    "level_number":
                        int(
                            ladder.level_number(
                                level
                            )
                        ),

                    "chi2_h0":
                        float(
                            h0[
                                "chi2"
                            ]
                        ),

                    "chi2_h1":
                        float(
                            best[
                                "chi2"
                            ]
                        ),

                    "delta_chi2_raw":
                        float(
                            raw_delta
                        ),

                    "T":
                        float(
                            T
                        ),

                    "best_P_days":
                        float(
                            best[
                                "P_days"
                            ]
                        ),

                    "best_omega":
                        float(
                            best[
                                "omega"
                            ]
                        ),

                    "best_q_mass":
                        float(
                            best[
                                "q_mass"
                            ]
                        ),

                    "best_q_flux":
                        float(
                            best[
                                "q_flux"
                            ]
                        ),

                    "best_theta":
                        float(
                            best[
                                "theta"
                            ]
                        ),

                    "best_phi":
                        float(
                            best[
                                "phi"
                            ]
                        ),

                    "best_inclination":
                        float(
                            best[
                                "inclination"
                            ]
                        ),

                    "h0_t0":
                        float(
                            h0[
                                "t0"
                            ]
                        ),

                    "h0_u0":
                        float(
                            h0[
                                "u0"
                            ]
                        ),

                    "h0_tE":
                        float(
                            h0[
                                "tE"
                            ]
                        ),

                    "h1_t0":
                        float(
                            best[
                                "t0"
                            ]
                        ),

                    "h1_u0":
                        float(
                            best[
                                "u0"
                            ]
                        ),

                    "h1_tE":
                        float(
                            best[
                                "tE"
                            ]
                        ),

                    "n_epochs_total":
                        int(
                            truth[
                                "n_epochs_total"
                            ]
                        ),

                    "n_epochs_used":
                        int(
                            truth[
                                "n_epochs_used"
                            ]
                        ),

                    "n_epochs_rejected":
                        int(
                            truth[
                                "n_epochs_rejected"
                            ]
                        ),

                    "n_full_saturated":
                        int(
                            truth[
                                "n_full_saturated"
                            ]
                        ),

                    "n_partial_saturated":
                        int(
                            truth[
                                "n_partial_saturated"
                            ]
                        ),

                    "fraction_rejected":
                        float(
                            truth[
                                "fraction_rejected"
                            ]
                        ),
                }
            )

        # ----------------------------------------------------
        # Mandatory nesting checks
        # ----------------------------------------------------

        for i in range(
            1,
            len(rows),
        ):

            previous = (
                rows[
                    i - 1
                ]
            )

            current = (
                rows[
                    i
                ]
            )

            if (
                current[
                    "chi2_h1"
                ]
                >
                previous[
                    "chi2_h1"
                ]
                + 1.0e-8
            ):

                raise RuntimeError(
                    "Nestedness violation: "
                    f"seed={seed}, "
                    f"{previous['h1_level']} -> "
                    f"{current['h1_level']}, "
                    f"chi2 previous="
                    f"{previous['chi2_h1']}, "
                    f"chi2 current="
                    f"{current['chi2_h1']}"
                )

            if (
                current[
                    "T"
                ]
                <
                previous[
                    "T"
                ]
                - 1.0e-8
            ):

                raise RuntimeError(
                    "T monotonicity violation: "
                    f"seed={seed}, "
                    f"{previous['h1_level']} -> "
                    f"{current['h1_level']}"
                )

        return {
            "ok":
                True,

            "seed":
                seed,

            "rows":
                rows,
        }

    except Exception as exc:

        return {
            "ok":
                False,

            "seed":
                int(seed),

            "error":
                repr(
                    exc
                ),

            "traceback":
                traceback.format_exc(),
        }


# ============================================================
# I/O
# ============================================================

def write_csv(
    path,
    rows,
):

    if len(rows) == 0:
        raise RuntimeError(
            "No successful rows to write."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = (
            csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def summarize(
    rows,
    failed,
    p_max,
):

    summary = {
        "n_failed_realizations":
            int(
                len(
                    failed
                )
            ),

        "levels":
            {},
    }

    for level in ladder.LEVELS:

        subset = [
            row
            for row
            in rows
            if (
                row[
                    "h1_level"
                ]
                == level
            )
        ]

        if len(
            subset
        ) == 0:
            continue

        T = np.asarray(
            [
                row[
                    "T"
                ]
                for row
                in subset
            ],
            dtype=float,
        )

        P = np.asarray(
            [
                row[
                    "best_P_days"
                ]
                for row
                in subset
            ],
            dtype=float,
        )

        qM = np.asarray(
            [
                row[
                    "best_q_mass"
                ]
                for row
                in subset
            ],
            dtype=float,
        )

        qf = np.asarray(
            [
                row[
                    "best_q_flux"
                ]
                for row
                in subset
            ],
            dtype=float,
        )

        summary[
            "levels"
        ][level] = {
            "n_success":
                int(
                    len(
                        subset
                    )
                ),

            "fraction_T_zero":
                float(
                    np.mean(
                        T == 0.0
                    )
                ),

            "T_median":
                float(
                    np.median(
                        T
                    )
                ),

            "T_q90":
                float(
                    np.quantile(
                        T,
                        0.90,
                    )
                ),

            "T_q95":
                float(
                    np.quantile(
                        T,
                        0.95,
                    )
                ),

            "T_q99":
                float(
                    np.quantile(
                        T,
                        0.99,
                    )
                ),

            "T_max":
                float(
                    np.max(
                        T
                    )
                ),

            "fraction_P_at_upper":
                float(
                    np.mean(
                        P
                        >= (
                            0.999999
                            * float(
                                p_max
                            )
                        )
                    )
                ),

            "q_mass_min":
                float(
                    np.min(
                        qM
                    )
                ),

            "q_mass_median":
                float(
                    np.median(
                        qM
                    )
                ),

            "q_mass_max":
                float(
                    np.max(
                        qM
                    )
                ),

            "q_flux_min":
                float(
                    np.min(
                        qf
                    )
                ),

            "q_flux_median":
                float(
                    np.median(
                        qf
                    )
                ),

            "q_flux_max":
                float(
                    np.max(
                        qf
                    )
                ),
        }

    return summary


def git_metadata():

    root = Path.cwd()

    try:

        commit = (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=root,
                text=True,
            )
            .strip()
        )

    except Exception:

        commit = None

    try:

        status = (
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--short",
                ],
                cwd=root,
                text=True,
            )
            .strip()
        )

    except Exception:

        status = None

    return (
        commit,
        status,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--source-mag",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--u0",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--tE",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--n-null",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed0",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--p-min",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--p-max",
        type=float,
        default=1.0e5,
    )

    parser.add_argument(
        "--n-omega",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--max-level",
        choices=ladder.LEVELS,
        default=ladder.LEVEL_3,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = (
        parser.parse_args()
    )

    outdir = Path(
        args.output_dir
    )

    if (
        outdir.exists()
        and any(
            outdir.iterdir()
        )
    ):
        raise RuntimeError(
            f"Output directory is not empty: "
            f"{outdir}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = [
        int(
            args.seed0
            + i
        )
        for i
        in range(
            int(
                args.n_null
            )
        )
    ]

    print(
        "=" * 90
    )

    print(
        "ROMAN BSPL-vs-PSPL H0 LRT LADDER CALIBRATION"
    )

    print(
        "=" * 90
    )

    print(
        "F146       =",
        args.source_mag,
    )

    print(
        "u0         =",
        args.u0,
    )

    print(
        "tE         =",
        args.tE,
    )

    print(
        "N null     =",
        args.n_null,
    )

    print(
        "seed0      =",
        args.seed0,
    )

    print(
        "workers    =",
        args.workers,
    )

    print(
        "max level  =",
        args.max_level,
    )

    print(
        "output     =",
        outdir,
    )

    print(
        "=" * 90
    )

    rows = []

    failed = []

    with ProcessPoolExecutor(
        max_workers=int(
            args.workers
        ),
        initializer=_init_worker,
        initargs=(
            args.source_mag,
            args.u0,
            args.tE,
            args.p_min,
            args.p_max,
            args.n_omega,
            args.max_level,
        ),
    ) as executor:

        futures = {
            executor.submit(
                _run_one,
                seed,
            ):
                seed

            for seed
            in seeds
        }

        done = 0

        for future in as_completed(
            futures
        ):

            result = (
                future.result()
            )

            done += 1

            if result[
                "ok"
            ]:

                rows.extend(
                    result[
                        "rows"
                    ]
                )

            else:

                failed.append(
                    result
                )

            if (
                done <= 10
                or done % 25 == 0
                or done == len(
                    seeds
                )
            ):

                print(
                    f"[{done:5d}/"
                    f"{len(seeds):5d}] "
                    f"failed="
                    f"{len(failed)}",
                    flush=True,
                )

    rows = sorted(
        rows,
        key=lambda row:
            (
                int(
                    row[
                        "seed"
                    ]
                ),
                int(
                    row[
                        "level_number"
                    ]
                ),
            ),
    )

    realizations_path = (
        outdir
        / "h0_ladder_realizations.csv"
    )

    summary_path = (
        outdir
        / "h0_ladder_summary.json"
    )

    metadata_path = (
        outdir
        / "metadata.json"
    )

    failures_path = (
        outdir
        / "failures.json"
    )

    write_csv(
        realizations_path,
        rows,
    )

    summary = (
        summarize(
            rows=rows,
            failed=failed,
            p_max=args.p_max,
        )
    )

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    with open(
        failures_path,
        "w",
    ) as f:

        json.dump(
            failed,
            f,
            indent=2,
            sort_keys=True,
        )

    commit, status = (
        git_metadata()
    )

    metadata = {
        "source_mag":
            float(
                args.source_mag
            ),

        "u0_true":
            float(
                args.u0
            ),

        "tE_true":
            float(
                args.tE
            ),

        "n_null_requested":
            int(
                args.n_null
            ),

        "seed0":
            int(
                args.seed0
            ),

        "workers":
            int(
                args.workers
            ),

        "p_min_days":
            float(
                args.p_min
            ),

        "p_max_days":
            float(
                args.p_max
            ),

        "n_omega":
            int(
                args.n_omega
            ),

        "max_level":
            str(
                args.max_level
            ),

        "levels":
            list(
                ladder.LEVELS[
                    : ladder.level_number(
                        args.max_level
                    )
                ]
            ),

        "qmass_bounds":
            [
                float(
                    ladder.QMASS_MIN
                ),
                float(
                    ladder.QMASS_MAX
                ),
            ],

        "qflux_bounds":
            [
                float(
                    ladder.QFLUX_MIN
                ),
                float(
                    ladder.QFLUX_MAX
                ),
            ],

        "qmass_starts":
            [
                float(x)
                for x
                in ladder.QMASS_STARTS
            ],

        "qflux_starts":
            [
                float(x)
                for x
                in ladder.QFLUX_STARTS
            ],

        "statistic":
            "T=max(0, chi2_H0-chi2_H1)",

        "noise_generation":
            (
                "Gaussian perturbation in magnitude using "
                "truth-derived Pandeia sigma_mag; fit objective "
                "is profiled chi2 in flux. This is the current "
                "validated pilot mechanism."
            ),

        "saturation_policy":
            (
                "truth-derived Pandeia full-saturation mask; "
                "same retained epochs for H0 and all H1 levels"
            ),

        "git_commit":
            commit,

        "git_status_short":
            status,
    }

    with open(
        metadata_path,
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            sort_keys=True,
        )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "realizations =",
        realizations_path,
    )

    print(
        "summary      =",
        summary_path,
    )

    print(
        "metadata     =",
        metadata_path,
    )

    print(
        "failures     =",
        failures_path,
    )


if __name__ == "__main__":
    main()
