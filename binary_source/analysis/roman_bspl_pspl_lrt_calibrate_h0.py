#!/usr/bin/env python3

"""
Parallel H0 calibration for the Roman BSPL-vs-PSPL LRT.

Each worker builds its own pyLIMA BSPL evaluator and then runs
independent parametric-bootstrap PSPL realizations.

The core statistic is implemented in roman_bspl_pspl_lrt.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path

import numpy as np

from binary_source.analysis import (
    roman_bspl_pspl_asimov as roman
)

from binary_source.analysis import (
    roman_bspl_pspl_lrt as lrt
)


_STATE = {}


def _init_worker(
    tE,
    P_min,
    P_max,
    n_omega,
):
    # Do not let numerical libraries create threads inside
    # each multiprocessing worker.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    roman.set_roman_noise_backend(
        "pandeia"
    )

    t, t0 = roman.build_roman_times(
        tE_true=float(tE),
        anchor_season_index=2,
        include_off_seasons=True,
    )

    P_grid = np.logspace(
        np.log10(float(P_min)),
        np.log10(float(P_max)),
        int(n_omega),
    )

    omega_grid = (
        2.0
        * np.pi
        / P_grid
    )

    evaluator = lrt.PhysicalCircularBSPL(
        t=t,
        t0_reference=t0,
    )

    _STATE.clear()

    _STATE.update(
        {
            "t": t,
            "t0": float(t0),
            "tE": float(tE),
            "omega_grid": omega_grid,
            "evaluator": evaluator,
            "prepared_cache": {},
        }
    )


def _run_one(task):
    config_id, source_mag, u0, seed = task

    try:

        cache_key = (
            float(source_mag),
            float(u0),
        )

        prepared = (
            _STATE[
                "prepared_cache"
            ].get(
                cache_key
            )
        )

        if prepared is None:

            prepared = (
                lrt.prepare_h0_configuration(
                    t=_STATE["t"],
                    t0_true=_STATE["t0"],
                    u0_true=float(u0),
                    tE_true=_STATE["tE"],
                    source_mag=float(
                        source_mag
                    ),
                    evaluator=_STATE[
                        "evaluator"
                    ],
                )
            )

            _STATE[
                "prepared_cache"
            ][
                cache_key
            ] = prepared

        result = (
            lrt.run_h0_realization(
                t=_STATE["t"],
                t0_true=_STATE["t0"],
                u0_true=float(u0),
                tE_true=_STATE["tE"],
                source_mag=float(
                    source_mag
                ),
                seed=int(seed),
                evaluator=prepared[
                    "evaluator"
                ],
                omega_grid=_STATE[
                    "omega_grid"
                ],
                prepared=prepared,
                validate_pylima=False,
            )
        )

        result[
            "config_id"
        ] = int(
            config_id
        )

        result[
            "status"
        ] = "ok"

        return result

    except Exception as exc:

        return {
            "config_id":
                int(
                    config_id
                ),

            "source_mag":
                float(
                    source_mag
                ),

            "u0_true":
                float(
                    u0
                ),

            "tE_true":
                float(
                    _STATE["tE"]
                ),

            "seed":
                int(
                    seed
                ),

            "status":
                "failed",

            "error":
                repr(
                    exc
                ),
        }



def write_csv(
    rows,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with path.open(
        "w",
        newline="",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def summarize(
    rows,
    source_mags,
    u0_values,
    alphas,
    P_max,
):
    summary = []

    config_id = 0

    for mag in source_mags:

        for u0 in u0_values:

            subset = [
                row
                for row in rows
                if (
                    row.get(
                        "status"
                    )
                    == "ok"
                    and int(
                        row[
                            "config_id"
                        ]
                    )
                    == config_id
                )
            ]

            failed = [
                row
                for row in rows
                if (
                    int(
                        row[
                            "config_id"
                        ]
                    )
                    == config_id
                    and row.get(
                        "status"
                    )
                    != "ok"
                )
            ]

            T = np.asarray(
                [
                    row["T"]
                    for row in subset
                ],
                dtype=float,
            )

            Pbest = np.asarray(
                [
                    row["best_P_days"]
                    for row in subset
                ],
                dtype=float,
            )

            item = {
                "config_id": config_id,
                "source_mag": float(
                    mag
                ),
                "u0": float(
                    u0
                ),
                "n_success": int(
                    len(subset)
                ),
                "n_failed": int(
                    len(failed)
                ),
            }

            if len(T) > 0:

                item.update(
                    {
                        "T_min":
                            float(
                                np.min(T)
                            ),

                        "T_median":
                            float(
                                np.median(T)
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
                                np.max(T)
                            ),

                        "fraction_T_zero":
                            float(
                                np.mean(
                                    T == 0.0
                                )
                            ),

                        "fraction_P_at_upper":
                            float(
                                np.mean(
                                    np.isclose(
                                        Pbest,
                                        float(
                                            P_max
                                        ),
                                        rtol=1e-6,
                                        atol=0.0,
                                    )
                                )
                            ),
                    }
                )

                for alpha in alphas:

                    threshold = (
                        lrt.empirical_threshold(
                            T,
                            float(alpha),
                        )
                    )

                    item[
                        f"c_alpha_{alpha:g}"
                    ] = float(
                        threshold
                    )

                    item[
                        f"strict_tail_{alpha:g}"
                    ] = float(
                        np.mean(
                            T
                            > threshold
                        )
                    )

            summary.append(
                item
            )

            config_id += 1

    return summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-mags",
        nargs="+",
        type=float,
        default=[
            19.0,
            21.0,
            23.0,
        ],
    )

    parser.add_argument(
        "--u0s",
        nargs="+",
        type=float,
        default=[
            0.01,
            0.03,
            0.10,
            0.30,
            0.9771241535346497,
        ],
    )

    parser.add_argument(
        "--tE",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--n-null-per-config",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--P-min",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--P-max",
        type=float,
        default=100000.0,
    )

    parser.add_argument(
        "--n-omega",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed0",
        type=int,
        default=2026101600,
    )

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[
            0.05,
            0.01,
        ],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/roman_lrt/"
            "h0_calibration_pilot"
        ),
    )

    args = parser.parse_args()

    tasks = []

    config_id = 0

    for mag in args.source_mags:

        for u0 in args.u0s:

            for j in range(
                args.n_null_per_config
            ):

                seed = (
                    args.seed0
                    + config_id
                    * 1_000_000
                    + j
                )

                tasks.append(
                    (
                        config_id,
                        float(mag),
                        float(u0),
                        int(seed),
                    )
                )

            config_id += 1

    print()
    print("=" * 90)
    print("ROMAN BSPL-vs-PSPL H0 CALIBRATION")
    print("=" * 90)
    print(
        "source mags        =",
        args.source_mags,
    )
    print(
        "u0 values          =",
        args.u0s,
    )
    print(
        "tE [d]             =",
        args.tE,
    )
    print(
        "null/config        =",
        args.n_null_per_config,
    )
    print(
        "configurations     =",
        config_id,
    )
    print(
        "total realizations =",
        len(tasks),
    )
    print(
        "omega multistarts  =",
        args.n_omega,
    )
    print(
        "workers            =",
        args.workers,
    )
    print("=" * 90)

    rows = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(
            args.tE,
            args.P_min,
            args.P_max,
            args.n_omega,
        ),
    ) as executor:

        futures = [
            executor.submit(
                _run_one,
                task,
            )
            for task in tasks
        ]

        total = len(
            futures
        )

        for i, future in enumerate(
            as_completed(
                futures
            ),
            1,
        ):

            result = (
                future.result()
            )

            rows.append(
                result
            )

            if (
                i == 1
                or i % 50 == 0
                or i == total
            ):

                n_failed = sum(
                    row.get(
                        "status"
                    )
                    != "ok"
                    for row in rows
                )

                print(
                    f"[{i:5d}/{total:5d}] "
                    f"failed={n_failed}"
                )

    rows = sorted(
        rows,
        key=lambda row: (
            int(
                row["config_id"]
            ),
            int(
                row["seed"]
            ),
        ),
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        args.output_dir
        / "h0_realizations.csv"
    )

    write_csv(
        rows,
        csv_path,
    )

    summary = summarize(
        rows=rows,
        source_mags=args.source_mags,
        u0_values=args.u0s,
        alphas=args.alphas,
        P_max=args.P_max,
    )

    summary_path = (
        args.output_dir
        / "h0_summary.json"
    )

    metadata_path = (
        args.output_dir
        / "metadata.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    metadata = {
        "noise_backend":
            "roman_rubin_pandeia",

        "saturation_policy":
            (
                "mask fixed from truth before noise; "
                "fully saturated epochs rejected; "
                "valid partially saturated epochs retained; "
                "identical retained timestamps used for H0 and H1"
            ),

        "hypothesis_h0":
            "PSPL",

        "hypothesis_h1":
            (
                "PSPL union physical circular BSPL; "
                "qM/qf/geometry fixed; "
                "t0,u0,tE,omega profiled; "
                "omega grid used only as multistart"
            ),

        "source_mags":
            args.source_mags,

        "u0_values":
            args.u0s,

        "tE_days":
            args.tE,

        "n_null_per_config":
            args.n_null_per_config,

        "P_min_days":
            args.P_min,

        "P_max_days":
            args.P_max,

        "n_omega_multistarts":
            args.n_omega,

        "seed0":
            args.seed0,

        "alphas":
            args.alphas,

        "workers":
            args.workers,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for row in summary:

        print(
            f"F146={row['source_mag']:4.1f}  "
            f"u0={row['u0']:.5f}  "
            f"N={row['n_success']:4d}  "
            f"fail={row['n_failed']:3d}  "
            f"f0={row.get('fraction_T_zero', np.nan):.3f}  "
            f"q95={row.get('T_q95', np.nan):.4g}  "
            f"q99={row.get('T_q99', np.nan):.4g}  "
            f"c05={row.get('c_alpha_0.05', np.nan):.4g}  "
            f"c01={row.get('c_alpha_0.01', np.nan):.4g}"
        )

    print()
    print(
        "realizations =",
        csv_path,
    )
    print(
        "summary      =",
        summary_path,
    )
    print(
        "metadata     =",
        metadata_path,
    )


if __name__ == "__main__":
    main()
