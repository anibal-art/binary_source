#!/usr/bin/env python3

"""
Figure: Roman tE-bound validation.

Uses:
    results/validation_roman_te_bound/
        roman_te_bound_relaxation.csv

Produces:
    figures/appendix/
        roman_te_bound_validation.pdf
        roman_te_bound_validation.png

The figure focuses on the final F146/tE=30 production configuration
with the largest stored best-fit tE relative to the nominal upper
bound.  This provides the most conservative available test of the
sensitivity to relaxing the upper tE bound.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "results"
    / "validation_roman_te_bound"
    / "roman_te_bound_relaxation.csv"
)

OUTDIR = (
    ROOT
    / "figures"
    / "appendix"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Style
# ============================================================

def apply_style():

    try:

        from binary_source.plot_codes.paper_style import (
            apply_paper_style,
        )

        apply_paper_style()

    except Exception:

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": [
                    "STIXGeneral",
                    "DejaVu Serif",
                ],
                "mathtext.fontset": "stix",
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 12,
                "legend.fontsize": 9,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "axes.linewidth": 0.8,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.top": True,
                "ytick.right": True,
            }
        )


# ============================================================
# Main
# ============================================================

def main():

    apply_style()


    if not INPUT.exists():

        raise FileNotFoundError(
            f"Missing input file:\n{INPUT}"
        )


    df = pd.read_csv(
        INPUT
    )


    # --------------------------------------------------------
    # Select the most conservative physical cell automatically.
    #
    # Among the configurations included in the current validation,
    # choose the cell whose stored production fit has the largest
    # best-fit tE at the nominal upper-factor = 20 setup.
    # --------------------------------------------------------

    nominal = df[
        np.isclose(
            df["upper_factor"],
            20.0,
        )
    ].copy()


    if nominal.empty:

        raise RuntimeError(
            "No rows found for upper_factor=20."
        )


    cell_scores = (
        nominal.groupby(
            [
                "i",
                "j",
            ]
        )[
            "stored_best_tE"
        ]
        .max()
    )


    i_plot, j_plot = (
        cell_scores.idxmax()
    )


    i_plot = int(
        i_plot
    )

    j_plot = int(
        j_plot
    )


    sub = df[
        (df["i"] == i_plot)
        & (df["j"] == j_plot)
    ].copy()


    if sub.empty:

        raise RuntimeError(
            "Automatically selected Roman tE-bound cell "
            f"({i_plot}, {j_plot}) was not found."
        )


    sub = sub.sort_values(
        [
            "F146",
            "upper_factor",
        ]
    )


    # Infer the true event timescale from the stored physical
    # coordinates rather than hard-coding it.
    te_values = (
        sub["P_days"]
        / sub["P_over_tE"]
    ).to_numpy(
        dtype=float
    )


    te_values = te_values[
        np.isfinite(
            te_values
        )
    ]


    if len(te_values) == 0:

        raise RuntimeError(
            "Could not infer tE_true from P_days/P_over_tE."
        )


    tE_true = float(
        np.median(
            te_values
        )
    )


    if not np.allclose(
        te_values,
        tE_true,
        rtol=1.0e-8,
        atol=1.0e-8,
    ):

        raise RuntimeError(
            "Inconsistent tE_true values inferred from "
            "P_days/P_over_tE."
        )


    print(
        "Selected conservative Roman tE-bound cell:"
    )

    print(
        f"  (i, j)   = ({i_plot}, {j_plot})"
    )

    print(
        f"  u0       = {sub['u0_true'].iloc[0]:.10g}"
    )

    print(
        f"  P/tE     = {sub['P_over_tE'].iloc[0]:.10g}"
    )

    print(
        f"  tE_true  = {tE_true:.10g} d"
    )

    print(
        "  max stored best_tE = "
        f"{cell_scores.loc[(i_plot, j_plot)]:.10g} d"
    )


    # --------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------

    sub[
        "best_tE_over_true"
    ] = (
        sub["best_tE"]
        / tE_true
    )


    sub[
        "teff_fit"
    ] = (
        np.abs(
            sub["best_u0"]
        )
        * sub["best_tE"]
    )


    sub[
        "dchi2_frac_percent"
    ] = np.nan


    for mag in sorted(
        sub["F146"].unique()
    ):

        m = (
            sub["F146"]
            == mag
        )

        s = (
            sub.loc[m]
            .sort_values(
                "upper_factor"
            )
        )


        ref = float(
            s.loc[
                np.isclose(
                    s[
                        "upper_factor"
                    ],
                    20.0,
                ),
                "delta_chi2",
            ].iloc[0]
        )


        sub.loc[
            s.index,
            "dchi2_frac_percent",
        ] = (
            100.0
            * (
                s[
                    "delta_chi2"
                ].to_numpy()
                / ref
                - 1.0
            )
        )


    # --------------------------------------------------------
    # Metadata for title/caption context
    # --------------------------------------------------------

    u0_true = float(
        sub[
            "u0_true"
        ].iloc[0]
    )

    P_over_tE = float(
        sub[
            "P_over_tE"
        ].iloc[0]
    )


    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            11.5,
            3.7,
        ),
    )


    markers = [
        "o",
        "s",
        "^",
    ]


    magnitudes = sorted(
        sub[
            "F146"
        ].unique()
    )


    # ========================================================
    # Panel A
    # ========================================================

    ax = axes[0]


    x_reference = np.array(
        [
            20.0,
            50.0,
            100.0,
            200.0,
        ]
    )


    ax.plot(
        x_reference,
        x_reference,
        linestyle="--",
        linewidth=1.2,
        label=r"$t_{E,\rm fit}=t_{E,\max}$",
    )


    for marker, mag in zip(
        markers,
        magnitudes,
    ):

        d = (
            sub[
                sub[
                    "F146"
                ]
                == mag
            ]
            .sort_values(
                "upper_factor"
            )
        )


        ax.plot(
            d[
                "upper_factor"
            ],
            d[
                "best_tE_over_true"
            ],
            marker=marker,
            linewidth=1.5,
            markersize=5,
            label=(
                rf"F146={mag:.0f}"
            ),
        )


    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )


    ax.set_xticks(
        x_reference
    )

    ax.set_xticklabels(
        [
            "20",
            "50",
            "100",
            "200",
        ]
    )


    ax.set_yticks(
        x_reference
    )

    ax.set_yticklabels(
        [
            "20",
            "50",
            "100",
            "200",
        ]
    )


    ax.set_xlabel(
        r"Upper bound $t_{E,\max}/t_{E,\rm true}$"
    )

    ax.set_ylabel(
        r"$t_{E,\rm fit}/t_{E,\rm true}$"
    )


    ax.legend(
        frameon=False,
        loc="upper left",
    )


    ax.text(
        0.04,
        0.95,
        "(a)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


    # ========================================================
    # Panel B
    # ========================================================

    ax = axes[1]


    for marker, mag in zip(
        markers,
        magnitudes,
    ):

        d = (
            sub[
                sub[
                    "F146"
                ]
                == mag
            ]
            .sort_values(
                "upper_factor"
            )
        )


        ax.plot(
            d[
                "upper_factor"
            ],
            d[
                "teff_fit"
            ],
            marker=marker,
            linewidth=1.5,
            markersize=5,
            label=(
                rf"F146={mag:.0f}"
            ),
        )


    ax.set_xscale(
        "log"
    )


    ax.set_xticks(
        x_reference
    )

    ax.set_xticklabels(
        [
            "20",
            "50",
            "100",
            "200",
        ]
    )


    ax.set_xlabel(
        r"Upper bound $t_{E,\max}/t_{E,\rm true}$"
    )

    ax.set_ylabel(
        r"$|u_{0,\rm fit}|\,t_{E,\rm fit}$ [d]"
    )


    ax.text(
        0.04,
        0.95,
        "(b)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


    # ========================================================
    # Panel C
    # ========================================================

    ax = axes[2]


    for marker, mag in zip(
        markers,
        magnitudes,
    ):

        d = (
            sub[
                sub[
                    "F146"
                ]
                == mag
            ]
            .sort_values(
                "upper_factor"
            )
        )


        ax.plot(
            d[
                "upper_factor"
            ],
            d[
                "dchi2_frac_percent"
            ],
            marker=marker,
            linewidth=1.5,
            markersize=5,
            label=(
                rf"F146={mag:.0f}"
            ),
        )


    ax.axhline(
        0.0,
        linestyle=":",
        linewidth=1.0,
    )


    ax.set_xscale(
        "log"
    )


    ax.set_xticks(
        x_reference
    )

    ax.set_xticklabels(
        [
            "20",
            "50",
            "100",
            "200",
        ]
    )


    ax.set_xlabel(
        r"Upper bound $t_{E,\max}/t_{E,\rm true}$"
    )

    ax.set_ylabel(
        r"$100[\Delta\chi^2/\Delta\chi^2_{20}-1]$ [\%]"
    )


    ax.text(
        0.04,
        0.95,
        "(c)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


    # ========================================================
    # Common formatting
    # ========================================================

    for ax in axes:

        ax.grid(
            alpha=0.25,
        )


    fig.suptitle(
        (
            r"Roman $t_E$-bound validation: "
            rf"$u_0={u0_true:.3f}$, "
            rf"$P/t_E={P_over_tE:.3f}$"
        ),
        y=1.02,
        fontsize=12,
    )


    fig.tight_layout()


    pdf_path = (
        OUTDIR
        / "roman_te_bound_validation.pdf"
    )

    png_path = (
        OUTDIR
        / "roman_te_bound_validation.png"
    )


    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    # ========================================================
    # Console summary
    # ========================================================

    print(
        "Saved:",
        pdf_path
    )

    print(
        "Saved:",
        png_path
    )


    print()
    print(
        "Selected physical configuration:"
    )

    print(
        f"  u0_true  = {u0_true:.12g}"
    )

    print(
        f"  P/tE     = {P_over_tE:.12g}"
    )


    print()
    print(
        "Change from upper factor 20 -> 200:"
    )


    for mag in magnitudes:

        d = (
            sub[
                sub[
                    "F146"
                ]
                == mag
            ]
            .sort_values(
                "upper_factor"
            )
        )


        first = (
            d.iloc[0]
        )

        last = (
            d.iloc[-1]
        )


        dchi2_percent = (
            100.0
            * (
                last[
                    "delta_chi2"
                ]
                / first[
                    "delta_chi2"
                ]
                - 1.0
            )
        )


        Droman_percent = (
            100.0
            * (
                last[
                    "D_roman_w"
                ]
                / first[
                    "D_roman_w"
                ]
                - 1.0
            )
        )


        print(
            f"  F146={mag:.0f}: "
            f"Delta chi2 {dchi2_percent:+.5f}% ; "
            f"D_Roman,w {Droman_percent:+.5f}% ; "
            f"teff(final)={last['teff_fit']:.5f} d"
        )


if __name__ == "__main__":

    main()
