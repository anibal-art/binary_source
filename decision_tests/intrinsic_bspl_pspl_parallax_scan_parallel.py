#!/usr/bin/env python3

from __future__ import annotations

import argparse
from functools import partial
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import intrinsic_bspl_pspl_parallax_scan as seq

from binary_source.analysis import (
    intrinsic_fit_tools as base,
)
from binary_source.analysis import (
    intrinsic_parallax as par,
)


# ============================================================
# Worker-global state
# ============================================================

_WORKER = {}


def init_worker(
    *,
    t0_true,
    u0_true,
    tE_true,
    n_time,
    pie_bound,
    start_scales,
):
    """
    Initialize expensive/static pyLIMA objects once per worker.

    Every task handled by this worker reuses:
      - the same dense theoretical time grid,
      - the same BSPL evaluator,
      - the same PSPL+parallax evaluator.
    """

    # Explicitly prevent nested BLAS/OpenMP parallelism.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    t0_true = float(t0_true)
    u0_true = float(u0_true)
    tE_true = float(tE_true)

    t = np.linspace(
        t0_true
        - seq.WINDOW_TE * tE_true,
        t0_true
        + seq.WINDOW_TE * tE_true,
        int(n_time),
    )

    bspl = base.PhysicalCircularBSPL(
        t=t,
        t0_reference=t0_true,
    )

    pspl_pi = par.PSPLParallaxEvaluator(
        t=t,
        t0_reference=t0_true,
    )

    _WORKER.clear()

    _WORKER.update(
        {
            "t":
                t,

            "t0_true":
                t0_true,

            "u0_true":
                u0_true,

            "tE_true":
                tE_true,

            "pie_bound":
                float(pie_bound),

            "start_scales":
                tuple(
                    float(x)
                    for x in start_scales
                ),

            "bspl":
                bspl,

            "pspl_pi":
                pspl_pi,
        }
    )


def run_one_period(task):
    """
    One independent intrinsic-degeneracy calculation.

    task = (grid_index, P_days)
    """

    grid_index, P_days = task

    t = _WORKER["t"]
    t0_true = _WORKER["t0_true"]
    u0_true = _WORKER["u0_true"]
    tE_true = _WORKER["tE_true"]

    bspl = _WORKER["bspl"]
    pspl_pi = _WORKER["pspl_pi"]

    pie_bound = _WORKER["pie_bound"]
    start_scales = _WORKER["start_scales"]

    P_days = float(P_days)

    omega = (
        2.0
        * np.pi
        / P_days
    )

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # BSPL truth, no parallax
    # --------------------------------------------------------

    A_true = bspl.magnification(
        t0=t0_true,
        u0=u0_true,
        tE=tE_true,
        omega=omega,
    )

    # --------------------------------------------------------
    # Intrinsic PSPL best fit
    # --------------------------------------------------------

    fit0 = (
        seq.fit_pspl_no_parallax(
            t=t,
            A_true=A_true,
            t0_true=t0_true,
            u0_true=u0_true,
            tE_true=tE_true,
        )
    )

    # --------------------------------------------------------
    # Intrinsic PSPL + parallax best fit
    # --------------------------------------------------------

    fitpi, all_starts = (
        seq.fit_pspl_parallax(
            evaluator=pspl_pi,
            t=t,
            A_true=A_true,
            no_parallax_fit=fit0,
            t0_true=t0_true,
            tE_true=tE_true,
            pie_bound=pie_bound,
            start_scales=start_scales,
        )
    )

    bound_hit = (
        abs(
            fitpi["piEN"]
        )
        > 0.995 * pie_bound
        or
        abs(
            fitpi["piEE"]
        )
        > 0.995 * pie_bound
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "grid_index":
            int(grid_index),

        "P_days":
            P_days,

        "P_over_tE":
            float(
                P_days
                / tE_true
            ),

        "tE_true":
            float(tE_true),

        "u0_true":
            float(u0_true),

        "D_no_parallax":
            float(
                fit0["D"]
            ),

        "RMS_no_parallax":
            float(
                fit0["rms"]
            ),

        "D_parallax":
            float(
                fitpi["D"]
            ),

        "RMS_parallax":
            float(
                fitpi["rms"]
            ),

        "absorption_ratio":
            float(
                fitpi["D"]
                / fit0["D"]
            ),

        "piEN":
            float(
                fitpi["piEN"]
            ),

        "piEE":
            float(
                fitpi["piEE"]
            ),

        "piE":
            float(
                fitpi["piE"]
            ),

        "piE_angle":
            float(
                fitpi[
                    "piE_angle"
                ]
            ),

        "fit_t0":
            float(
                fitpi["t0"]
            ),

        "fit_u0":
            float(
                fitpi["u0"]
            ),

        "fit_tE":
            float(
                fitpi["tE"]
            ),

        "nfev":
            int(
                fitpi["nfev"]
            ),

        "winning_start_piEN":
            float(
                fitpi[
                    "start_piEN"
                ]
            ),

        "winning_start_piEE":
            float(
                fitpi[
                    "start_piEE"
                ]
            ),

        "pie_bound_hit":
            bool(
                bound_hit
            ),

        "elapsed_s":
            float(
                elapsed
            ),

        "n_parallax_starts":
            int(
                len(
                    all_starts
                )
            ),
    }


def main():
    parser = argparse.ArgumentParser()

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
        "--n-periods",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--n-time",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--pie-bound",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--start-scales",
        type=float,
        nargs="*",
        default=[
            1.0,
            5.0,
        ],
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/tmp/"
            "intrinsic_bspl_pspl_parallax_parallel"
        ),
    )

    args = parser.parse_args()

    outdir = Path(
        args.output_dir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Reference epoch
    # --------------------------------------------------------

    _, t0_true = (
        base.roman.build_roman_times(
            tE_true=args.tE,
        )
    )

    # --------------------------------------------------------
    # Validate theoretical dense interval against ephemeris.
    # --------------------------------------------------------

    t_check = np.linspace(
        t0_true
        - seq.WINDOW_TE
        * args.tE,

        t0_true
        + seq.WINDOW_TE
        * args.tE,

        args.n_time,
    )

    eph_path, eph = (
        par.load_roman_ephemerides()
    )

    if (
        t_check[0]
        < eph[0, 0]
        or
        t_check[-1]
        > eph[-1, 0]
    ):
        raise RuntimeError(
            "Dense theoretical interval "
            "outside Roman ephemeris support.\n"
            f"time = [{t_check[0]}, "
            f"{t_check[-1]}]\n"
            f"eph  = [{eph[0,0]}, "
            f"{eph[-1,0]}]"
        )

    P_grid = np.logspace(
        np.log10(
            args.p_min
        ),
        np.log10(
            args.p_max
        ),
        args.n_periods,
    )

    tasks = list(
        enumerate(
            P_grid
        )
    )

    print("=" * 120)

    print(
        "PARALLEL INTRINSIC "
        "BSPL -> PSPL(+PARALLAX) SCAN"
    )

    print("=" * 120)

    print(
        "Roman ephemeris =",
        eph_path,
    )

    print(
        "dense interval  =",
        t_check[0],
        t_check[-1],
    )

    print(
        "N time          =",
        args.n_time,
    )

    print(
        "N periods       =",
        args.n_periods,
    )

    print(
        "workers         =",
        args.workers,
    )

    print(
        "piE bounds      =",
        -args.pie_bound,
        args.pie_bound,
    )

    print(
        "start scales    =",
        args.start_scales,
    )

    print("=" * 120)

    rows = []

    wall_start = (
        time.perf_counter()
    )

    # spawn is slower to initialize than fork, but safer for
    # pyLIMA / astropy numerical state on a shared cluster node.
    ctx = mp.get_context(
        "spawn"
    )

    worker_initializer = partial(
        init_worker,
        t0_true=t0_true,
        u0_true=args.u0,
        tE_true=args.tE,
        n_time=args.n_time,
        pie_bound=args.pie_bound,
        start_scales=args.start_scales,
    )

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=worker_initializer,
    ) as executor:

        futures = {
            executor.submit(
                run_one_period,
                task,
            ):
            task
            for task in tasks
        }

        completed = 0

        for future in as_completed(
            futures
        ):
            task = futures[
                future
            ]

            try:
                row = future.result()

            except Exception as exc:
                grid_index, P_days = task

                print(
                    f"[FAIL] "
                    f"i={grid_index} "
                    f"P={P_days:.8g} d "
                    f"{exc!r}",
                    flush=True,
                )

                raise

            rows.append(
                row
            )

            completed += 1

            print(
                f"[{completed:4d}/"
                f"{args.n_periods:4d}] "
                f"P/tE="
                f"{row['P_over_tE']:11.4g} "
                f"D0="
                f"{row['D_no_parallax']:11.4e} "
                f"Dpi="
                f"{row['D_parallax']:11.4e} "
                f"ratio="
                f"{row['absorption_ratio']:8.4f} "
                f"piE="
                f"({row['piEN']:+7.3f},"
                f"{row['piEE']:+7.3f}) "
                f"|piE|="
                f"{row['piE']:7.3f} "
                f"t="
                f"{row['elapsed_s']:6.2f}s",
                flush=True,
            )

    wall_time = (
        time.perf_counter()
        - wall_start
    )

    # as_completed returns arbitrary order.
    df = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "grid_index"
        )
        .reset_index(
            drop=True
        )
    )

    csv_path = (
        outdir
        / "intrinsic_parallax_scan.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    figure_path = (
        outdir
        / "intrinsic_parallax_scan.png"
    )

    seq.make_figure(
        df=df,
        output=figure_path,
        pie_bound=args.pie_bound,
    )

    print()

    print("=" * 120)

    print("SUMMARY")

    print("=" * 120)

    print(
        "wall time      =",
        wall_time,
        "s",
    )

    print(
        "sum task time  =",
        float(
            df[
                "elapsed_s"
            ].sum()
        ),
        "s",
    )

    print(
        "speedup proxy  =",
        float(
            df[
                "elapsed_s"
            ].sum()
        )
        / wall_time,
    )

    print(
        "bound hits     =",
        int(
            df[
                "pie_bound_hit"
            ].sum()
        ),
        "/",
        len(df),
    )

    print(
        "|piE| median   =",
        float(
            np.median(
                df["piE"]
            )
        ),
    )

    print(
        "|piE| q90      =",
        float(
            np.quantile(
                df["piE"],
                0.90,
            )
        ),
    )

    print(
        "|piE| max      =",
        float(
            np.max(
                df["piE"]
            )
        ),
    )

    print(
        "ratio min      =",
        float(
            np.min(
                df[
                    "absorption_ratio"
                ]
            )
        ),
    )

    print(
        "ratio median   =",
        float(
            np.median(
                df[
                    "absorption_ratio"
                ]
            )
        ),
    )

    print()

    print(
        "CSV            =",
        csv_path,
    )

    print(
        "figure         =",
        figure_path,
    )


if __name__ == "__main__":
    main()
