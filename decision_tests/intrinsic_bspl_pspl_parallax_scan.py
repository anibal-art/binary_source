#!/usr/bin/env python3

"""
Intrinsic BSPL -> PSPL(+parallax) degeneracy scan.

NO photometric uncertainties.
NO noise.
NO survey cadence.
NO flux fitting.

For every physical BSPL period P:

    1. Generate A_BSPL(t) on a dense uniform theoretical grid.
    2. Fit a standard PSPL model.
    3. Fit a PSPL + Roman/L2 parallax model.
    4. Minimize the unweighted magnification residual.
    5. Record RMS, D, and best-fit piEN/piEE.

Truth has piEN = piEE = 0.

The time window is t0 +/- 3.5 tE, matching the intrinsic
magnification-space experiment used in the manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import least_squares

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base,
)
from binary_source.analysis import (
    roman_bspl_pspl_lrt_parallax as par,
)


WINDOW_TE = 3.5


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

    return (
        (u ** 2 + 2.0)
        /
        (
            u
            * np.sqrt(
                u ** 2 + 4.0
            )
        )
    )


def trapezoid_weights(n):
    """
    Residual weights such that least_squares minimizes
    the trapezoidal integral on a uniform grid.
    """

    w = np.ones(
        int(n),
        dtype=float,
    )

    w[0] = np.sqrt(0.5)
    w[-1] = np.sqrt(0.5)

    return w


def rms_trapezoid(
    residual,
    t,
):
    residual = np.asarray(
        residual,
        dtype=float,
    )

    t = np.asarray(
        t,
        dtype=float,
    )

    integral = np.trapz(
        residual ** 2,
        t,
    )

    duration = (
        t[-1]
        - t[0]
    )

    return float(
        np.sqrt(
            integral
            / duration
        )
    )


def metrics(
    A_true,
    A_fit,
    t,
):
    residual = (
        np.asarray(A_true)
        - np.asarray(A_fit)
    )

    rms = rms_trapezoid(
        residual,
        t,
    )

    event_rms = rms_trapezoid(
        np.asarray(A_true) - 1.0,
        t,
    )

    D = (
        rms / event_rms
        if event_rms > 0.0
        else np.nan
    )

    return {
        "rms": float(rms),
        "event_rms": float(
            event_rms
        ),
        "D": float(D),
        "maxabs": float(
            np.max(
                np.abs(residual)
            )
        ),
    }


def decode_shape(
    x,
    t0_ref,
    tE_ref,
):
    t0 = (
        float(t0_ref)
        + float(x[0])
        * float(tE_ref)
    )

    u0 = float(
        x[1]
    )

    tE = (
        float(tE_ref)
        * np.exp(
            float(x[2])
        )
    )

    return t0, u0, tE


def fit_pspl_no_parallax(
    t,
    A_true,
    t0_true,
    u0_true,
    tE_true,
):
    """
    Intrinsic PSPL fit in magnification space.
    """

    w = trapezoid_weights(
        len(t)
    )

    x0 = np.array(
        [
            0.0,
            u0_true,
            0.0,
        ],
        dtype=float,
    )

    lower = np.array(
        [
            -WINDOW_TE,
            -5.0,
            np.log(0.05),
        ]
    )

    upper = np.array(
        [
            +WINDOW_TE,
            +5.0,
            np.log(20.0),
        ]
    )

    def residuals(x):
        t0, u0, tE = decode_shape(
            x,
            t0_true,
            tE_true,
        )

        A = pspl_magnification(
            t,
            t0,
            u0,
            tE,
        )

        return (
            w
            * (
                A
                - A_true
            )
        )

    opt = least_squares(
        residuals,
        x0,
        bounds=(
            lower,
            upper,
        ),
        method="trf",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=3000,
    )

    t0, u0, tE = decode_shape(
        opt.x,
        t0_true,
        tE_true,
    )

    A_fit = pspl_magnification(
        t,
        t0,
        u0,
        tE,
    )

    out = metrics(
        A_true,
        A_fit,
        t,
    )

    out.update(
        {
            "success":
                bool(opt.success),

            "t0":
                float(t0),

            "u0":
                float(u0),

            "tE":
                float(tE),

            "nfev":
                int(opt.nfev),

            "A_fit":
                A_fit,

            "x":
                np.asarray(
                    opt.x,
                    dtype=float,
                ),
        }
    )

    return out


def parallax_start_grid(
    scales,
):
    """
    Absolute, bound-independent piE starts.

    This is a diagnostic/offline calculation, not the production
    multistart policy.
    """

    starts = [
        (0.0, 0.0),
    ]

    for s in scales:
        s = float(s)

        starts.extend(
            [
                (+s, 0.0),
                (-s, 0.0),
                (0.0, +s),
                (0.0, -s),
            ]
        )

    return starts


def fit_pspl_parallax(
    evaluator,
    t,
    A_true,
    no_parallax_fit,
    t0_true,
    tE_true,
    pie_bound,
    start_scales,
):
    """
    Fit PSPL + Roman parallax directly in magnification space.

    Free parameters:
        t0, u0, tE, piEN, piEE.
    """

    w = trapezoid_weights(
        len(t)
    )

    lower = np.array(
        [
            -WINDOW_TE,
            -5.0,
            np.log(0.05),
            -pie_bound,
            -pie_bound,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            +WINDOW_TE,
            +5.0,
            np.log(20.0),
            +pie_bound,
            +pie_bound,
        ],
        dtype=float,
    )

    # Start shape from the intrinsic no-parallax optimum.
    base_shape = np.asarray(
        no_parallax_fit["x"],
        dtype=float,
    )

    starts = parallax_start_grid(
        start_scales
    )

    fits = []

    def residuals(x):
        t0, u0, tE = decode_shape(
            x[:3],
            t0_true,
            tE_true,
        )

        piEN = float(
            x[3]
        )

        piEE = float(
            x[4]
        )

        A = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            piEN=piEN,
            piEE=piEE,
        )

        return (
            w
            * (
                A
                - A_true
            )
        )

    for piEN0, piEE0 in starts:

        x0 = np.concatenate(
            [
                base_shape,
                [
                    piEN0,
                    piEE0,
                ],
            ]
        )

        x0 = np.maximum(
            x0,
            lower + 1e-10,
        )

        x0 = np.minimum(
            x0,
            upper - 1e-10,
        )

        try:
            opt = least_squares(
                residuals,
                x0,
                bounds=(
                    lower,
                    upper,
                ),
                method="trf",
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
                max_nfev=5000,
            )

            t0, u0, tE = (
                decode_shape(
                    opt.x[:3],
                    t0_true,
                    tE_true,
                )
            )

            piEN = float(
                opt.x[3]
            )

            piEE = float(
                opt.x[4]
            )

            A_fit = (
                evaluator.magnification(
                    t0=t0,
                    u0=u0,
                    tE=tE,
                    piEN=piEN,
                    piEE=piEE,
                )
            )

            m = metrics(
                A_true,
                A_fit,
                t,
            )

            m.update(
                {
                    "success":
                        bool(opt.success),

                    "t0":
                        float(t0),

                    "u0":
                        float(u0),

                    "tE":
                        float(tE),

                    "piEN":
                        piEN,

                    "piEE":
                        piEE,

                    "piE":
                        float(
                            np.hypot(
                                piEN,
                                piEE,
                            )
                        ),

                    "piE_angle":
                        float(
                            np.arctan2(
                                piEE,
                                piEN,
                            )
                        ),

                    "nfev":
                        int(
                            opt.nfev
                        ),

                    "start_piEN":
                        float(
                            piEN0
                        ),

                    "start_piEE":
                        float(
                            piEE0
                        ),

                    "A_fit":
                        A_fit,
                }
            )

            fits.append(
                m
            )

        except Exception as exc:
            fits.append(
                {
                    "success": False,
                    "D": np.inf,
                    "rms": np.inf,
                    "error": repr(exc),
                    "start_piEN":
                        float(piEN0),
                    "start_piEE":
                        float(piEE0),
                }
            )

    valid = [
        row
        for row in fits
        if (
            row.get(
                "success",
                False,
            )
            and np.isfinite(
                row["rms"]
            )
        )
    ]

    if not valid:
        raise RuntimeError(
            "All parallax starts failed."
        )

    best = min(
        valid,
        key=lambda row:
            row["rms"],
    )

    return best, fits


def make_figure(
    df,
    output,
    pie_bound,
):
    x = (
        df["P_days"]
        / df["tE_true"]
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
    )

    # ----------------------------------------------------------
    # A: intrinsic mismatch
    # ----------------------------------------------------------

    ax = axes[0, 0]

    ax.plot(
        x,
        df["D_no_parallax"],
        marker="o",
        markersize=3,
        label="PSPL",
    )

    ax.plot(
        x,
        df["D_parallax"],
        marker="o",
        markersize=3,
        label=r"PSPL + $\pi_E$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"$P/t_E$"
    )

    ax.set_ylabel(
        r"$D$"
    )

    ax.set_title(
        "Intrinsic model mismatch"
    )

    ax.legend()

    # ----------------------------------------------------------
    # B: fraction absorbed by parallax
    # ----------------------------------------------------------

    ax = axes[0, 1]

    ratio = (
        df["D_parallax"]
        /
        df["D_no_parallax"]
    )

    ax.plot(
        x,
        ratio,
        marker="o",
        markersize=3,
    )

    ax.axhline(
        1.0,
        linestyle="--",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"$P/t_E$"
    )

    ax.set_ylabel(
        r"$D_{\rm PSPL+\pi_E}/D_{\rm PSPL}$"
    )

    ax.set_title(
        "Residual remaining after allowing parallax"
    )

    # ----------------------------------------------------------
    # C: best parallax as function of period
    # ----------------------------------------------------------

    ax = axes[1, 0]

    ax.plot(
        x,
        df["piEN"],
        marker="o",
        markersize=3,
        label=r"$\pi_{E,N}$",
    )

    ax.plot(
        x,
        df["piEE"],
        marker="o",
        markersize=3,
        label=r"$\pi_{E,E}$",
    )

    ax.plot(
        x,
        df["piE"],
        marker="o",
        markersize=3,
        label=r"$|\pi_E|$",
    )

    ax.axhline(
        +pie_bound,
        linestyle=":",
    )

    ax.axhline(
        -pie_bound,
        linestyle=":",
    )

    ax.set_xscale("log")

    ax.set_xlabel(
        r"$P/t_E$"
    )

    ax.set_ylabel(
        "Best-fit parallax"
    )

    ax.set_title(
        "Parallax used to absorb the BSPL signal"
    )

    ax.legend()

    # ----------------------------------------------------------
    # D: piE plane
    # ----------------------------------------------------------

    ax = axes[1, 1]

    sc = ax.scatter(
        df["piEN"],
        df["piEE"],
        c=np.log10(x),
        s=28,
    )

    ax.axvline(
        0.0,
        linewidth=0.8,
    )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.set_xlim(
        -pie_bound,
        pie_bound,
    )

    ax.set_ylim(
        -pie_bound,
        pie_bound,
    )

    ax.set_xlabel(
        r"$\pi_{E,N}$"
    )

    ax.set_ylabel(
        r"$\pi_{E,E}$"
    )

    ax.set_title(
        "Location of intrinsic best fits"
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
    )

    cbar.set_label(
        r"$\log_{10}(P/t_E)$"
    )

    for ax in axes.flat:
        ax.grid(
            True,
            alpha=0.2,
        )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=180,
    )

    plt.close(
        fig
    )


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
        default=60,
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
        "--output-dir",
        default=(
            "/tmp/"
            "intrinsic_bspl_pspl_parallax"
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

    # ----------------------------------------------------------
    # Absolute epoch.
    #
    # Only used to define the Roman/L2 parallax geometry.
    # We do NOT use the Roman observational timestamps.
    # ----------------------------------------------------------

    _, t0_true = (
        base.roman.build_roman_times(
            tE_true=args.tE,
        )
    )

    # Dense, uniform theoretical sampling.
    t = np.linspace(
        t0_true
        - WINDOW_TE
        * args.tE,

        t0_true
        + WINDOW_TE
        * args.tE,

        args.n_time,
    )

    # Check that the dense theoretical interval is actually
    # covered by the Roman ephemeris.
    eph_path, eph = (
        par.load_roman_ephemerides()
    )

    if (
        t[0] < eph[0, 0]
        or t[-1] > eph[-1, 0]
    ):
        raise RuntimeError(
            "Dense theoretical time interval extends "
            "outside Roman ephemeris support.\n"
            f"time = [{t[0]}, {t[-1]}]\n"
            f"eph  = [{eph[0,0]}, {eph[-1,0]}]"
        )

    print(
        "Roman ephemeris =",
        eph_path,
    )

    print(
        "dense interval  =",
        t[0],
        t[-1],
    )

    print(
        "N time          =",
        len(t),
    )

    # Truth evaluator: BSPL, NO parallax.
    bspl = base.PhysicalCircularBSPL(
        t=t,
        t0_reference=t0_true,
    )

    # Fit evaluator: PSPL + Roman Full parallax.
    pspl_pi = par.PSPLParallaxEvaluator(
        t=t,
        t0_reference=t0_true,
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

    rows = []

    print()
    print("=" * 130)
    print(
        "INTRINSIC BSPL -> PSPL(+PARALLAX) SCAN"
    )
    print("=" * 130)

    for i, P_days in enumerate(
        P_grid
    ):
        omega = (
            2.0
            * np.pi
            / P_days
        )

        A_true = bspl.magnification(
            t0=t0_true,
            u0=args.u0,
            tE=args.tE,
            omega=omega,
        )

        # Standard intrinsic PSPL fit.
        fit0 = fit_pspl_no_parallax(
            t=t,
            A_true=A_true,
            t0_true=t0_true,
            u0_true=args.u0,
            tE_true=args.tE,
        )

        # Same intrinsic problem, now with Roman parallax freedom.
        fitpi, all_starts = (
            fit_pspl_parallax(
                evaluator=pspl_pi,
                t=t,
                A_true=A_true,
                no_parallax_fit=fit0,
                t0_true=t0_true,
                tE_true=args.tE,
                pie_bound=args.pie_bound,
                start_scales=(
                    args.start_scales
                ),
            )
        )

        bound_hit = (
            abs(fitpi["piEN"])
            > 0.995
            * args.pie_bound
            or
            abs(fitpi["piEE"])
            > 0.995
            * args.pie_bound
        )

        row = {
            "P_days":
                float(P_days),

            "P_over_tE":
                float(
                    P_days
                    / args.tE
                ),

            "tE_true":
                float(args.tE),

            "u0_true":
                float(args.u0),

            "D_no_parallax":
                float(fit0["D"]),

            "RMS_no_parallax":
                float(fit0["rms"]),

            "D_parallax":
                float(fitpi["D"]),

            "RMS_parallax":
                float(fitpi["rms"]),

            "absorption_ratio":
                float(
                    fitpi["D"]
                    / fit0["D"]
                ),

            "piEN":
                float(fitpi["piEN"]),

            "piEE":
                float(fitpi["piEE"]),

            "piE":
                float(fitpi["piE"]),

            "piE_angle":
                float(
                    fitpi[
                        "piE_angle"
                    ]
                ),

            "fit_t0":
                float(fitpi["t0"]),

            "fit_u0":
                float(fitpi["u0"]),

            "fit_tE":
                float(fitpi["tE"]),

            "nfev":
                int(fitpi["nfev"]),

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
                bool(bound_hit),
        }

        rows.append(
            row
        )

        print(
            f"[{i+1:3d}/{len(P_grid):3d}] "
            f"P/tE={row['P_over_tE']:11.4g} "
            f"D0={row['D_no_parallax']:11.4e} "
            f"Dpi={row['D_parallax']:11.4e} "
            f"ratio={row['absorption_ratio']:9.4f} "
            f"piE=({row['piEN']:+8.3f},"
            f"{row['piEE']:+8.3f}) "
            f"|piE|={row['piE']:8.3f} "
            f"bound={bound_hit}",
            flush=True,
        )

    df = pd.DataFrame(
        rows
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

    make_figure(
        df=df,
        output=figure_path,
        pie_bound=args.pie_bound,
    )

    print()
    print("=" * 130)
    print("SUMMARY")
    print("=" * 130)

    print(
        "bound hits =",
        int(
            df[
                "pie_bound_hit"
            ].sum()
        ),
        "/",
        len(df),
    )

    print(
        "|piE| median =",
        np.median(
            df["piE"]
        ),
    )

    print(
        "|piE| q90    =",
        np.quantile(
            df["piE"],
            0.90,
        ),
    )

    print(
        "|piE| max    =",
        np.max(
            df["piE"]
        ),
    )

    print()
    print(
        "CSV    =",
        csv_path,
    )

    print(
        "figure =",
        figure_path,
    )


if __name__ == "__main__":
    main()
