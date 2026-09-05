#!/usr/bin/env python3

"""
Two inexpensive final validation tests.

1. WINDOW SENSITIVITY
   Recompute representative intrinsic BSPL -> PSPL projections using

       t0 +/- 2.5 tE
       t0 +/- 3.5 tE
       t0 +/- 5.0 tE

   while keeping approximately the same temporal sampling density
   as the production calculation.

2. ROMAN TRUE-BLENDING TEST
   Repeat four representative Roman W149=21 Asimov configurations
   for several positive true blend ratios

       beta = Fb / Fs

   keeping the source magnitude fixed and recomputing the W149
   uncertainty from the total observed flux.

The script writes CSV summaries only. It does not overwrite any
production results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUTDIR_DEFAULT = (
    ROOT
    / "results"
    / "validation_final_window_blending"
)


# ============================================================
# Existing project code
# ============================================================

from binary_source.analysis import roman_bspl_pspl_asimov as roman

from binary_source.validation.validate_intrinsic_multistart_time_resolution import (
    fit_one_start,
)


# ============================================================
# Constants
# ============================================================

T0_INTRINSIC = 50.0
TE = 150.0

Q_MASS = 0.5

M_TOTAL = 3.0
REHAT_AU = 5.0

THETA = 0.0
PHI = 0.0
INCLINATION = np.pi / 2.0

N_TIME_PRODUCTION = 10_000
WINDOW_PRODUCTION = 3.5

WINDOWS_TE = [
    2.5,
    3.5,
    5.0,
]

# Fixed source magnitude for the Roman blending experiment.
SOURCE_MAG = 21.0

# True blend ratios:
#
#     beta = Fb / Fs
#
BLEND_RATIOS = [
    0.0,
    0.1,
    0.3,
    1.0,
    3.0,
]

W149_ZP = 27.615


# ============================================================
# Cases
# ============================================================

@dataclass(frozen=True)
class IntrinsicCase:
    key: str
    description: str

    u0: float
    P_days: float

    q_mass: float = 0.5
    qflux: float = 0.0


@dataclass(frozen=True)
class RomanCase:
    key: str
    description: str

    u0: float
    P_days: float

    q_mass: float = 0.5
    qflux: float = 0.0


def build_intrinsic_cases():

    return [

        IntrinsicCase(
            key="one_short",
            description=(
                "one luminous source; short period"
            ),
            u0=0.1,
            P_days=10.0,
        ),

        IntrinsicCase(
            key="one_intermediate",
            description=(
                "one luminous source; intermediate period"
            ),
            u0=0.1,
            P_days=210.0,
        ),

        IntrinsicCase(
            key="one_long",
            description=(
                "one luminous source; long period"
            ),
            u0=0.1,
            P_days=6000.0,
        ),

        IntrinsicCase(
            key="one_hidden_long",
            description=(
                "extreme strongly degenerate long-period case"
            ),
            u0=0.01,
            P_days=100_000.0,
        ),

        IntrinsicCase(
            key="one_small_u0",
            description=(
                "small-u0 broad ordinary case"
            ),
            u0=0.01,
            P_days=142.08308,
        ),

        IntrinsicCase(
            key="two_cancel",
            description=(
                "two luminous sources on qf=qM"
            ),
            u0=0.1,
            P_days=150.0,
            q_mass=0.5,
            qflux=0.5,
        ),

        IntrinsicCase(
            key="two_off_cancel",
            description=(
                "two luminous sources away from qf=qM"
            ),
            u0=0.1,
            P_days=150.0,
            q_mass=0.5,
            qflux=0.1,
        ),
    ]


def build_roman_cases():
    """
    Same W149=21 representative cases used in the Roman appendix.
    """

    return [

        RomanCase(
            key="A_hidden",
            description=(
                "extreme hidden configuration"
            ),
            u0=0.01,
            P_days=100_000.0,
        ),

        RomanCase(
            key="B_near_100",
            description=(
                "intrinsically degenerate case near Delta chi2=100"
            ),
            u0=0.0476861,
            P_days=13_141.47,
        ),

        RomanCase(
            key="C_near_500",
            description=(
                "intrinsically degenerate case near Delta chi2=500"
            ),
            u0=0.0511143,
            P_days=7_038.136,
        ),

        RomanCase(
            key="D_clear",
            description=(
                "clear short-period mismatch"
            ),
            u0=0.0493705,
            P_days=10.0,
        ),
    ]


# ============================================================
# Small helpers
# ============================================================

def git_info():

    def run_git(*args):

        try:

            out = subprocess.run(
                [
                    "git",
                    *args,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            return out.stdout.strip()

        except Exception:

            return None


    status = run_git(
        "status",
        "--porcelain",
    )

    return {
        "commit": run_git(
            "rev-parse",
            "HEAD",
        ),

        "short_commit": run_git(
            "rev-parse",
            "--short",
            "HEAD",
        ),

        "working_tree_clean": (
            status == ""
            if status is not None
            else None
        ),

        "status_porcelain": status,
    }


def as_float_array(x):

    if hasattr(
        x,
        "value",
    ):

        x = x.value

    return np.asarray(
        x,
        dtype=float,
    )


def mag_to_flux(
    mag,
    zp=W149_ZP,
):

    return (
        10.0
        ** (
            (
                float(zp)
                - np.asarray(
                    mag,
                    dtype=float,
                )
            )
            / 2.5
        )
    )


def flux_to_mag(
    flux,
    zp=W149_ZP,
):

    flux = np.asarray(
        flux,
        dtype=float,
    )

    return (
        float(zp)
        - 2.5
        * np.log10(
            flux
        )
    )


def magerr_to_fluxerr(
    flux,
    sigma_mag,
):

    return (
        np.asarray(
            flux,
            dtype=float,
        )
        * np.log(10.0)
        / 2.5
        * np.asarray(
            sigma_mag,
            dtype=float,
        )
    )


def assign_lightcurve_column(
    lightcurve,
    name,
    values,
):
    """
    Replace an Astropy/QTable column while preserving its unit
    when one is already attached.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if name in lightcurve.colnames:

        unit = getattr(
            lightcurve[name],
            "unit",
            None,
        )

        if unit is not None:

            try:

                lightcurve[name] = (
                    values
                    * unit
                )

                return

            except Exception:

                pass

    lightcurve[name] = values


# ============================================================
# W149 uncertainty wrapper
# ============================================================

def sigma_w149_vector(
    magnitude,
):
    """
    Robust wrapper around the project's sigma_w149_safe().
    """

    magnitude = np.asarray(
        magnitude,
        dtype=float,
    )

    try:

        out = (
            roman.sigma_w149_safe(
                magnitude
            )
        )

        # Some helper implementations can return
        # (sigma, floor_mask/metadata).
        if isinstance(
            out,
            tuple,
        ):

            out = out[0]

        sigma = np.asarray(
            out,
            dtype=float,
        )

        if sigma.shape == magnitude.shape:

            return sigma

    except Exception:

        pass


    # Scalar fallback
    sigma = np.array(
        [
            float(
                roman.sigma_w149_safe(
                    float(m)
                )
            )
            for m in magnitude
        ],
        dtype=float,
    )

    return sigma


# ============================================================
# Roman time setup
# ============================================================

def resolve_roman_times():
    """
    Resolve both the Roman epoch array and the t0 used by the existing
    build_roman_times() implementation without assuming one exact
    return container.
    """

    out = (
        roman.build_roman_times(
            tE_true=TE,
            anchor_season_index=2,
            fit_window_te=3.5,
            include_off_seasons=True,
        )
    )


    t = None
    t0 = None


    # --------------------------------------------------------
    # Dictionary return
    # --------------------------------------------------------

    if isinstance(
        out,
        dict,
    ):

        for key in [
            "t",
            "time",
            "times",
            "roman_times",
        ]:

            if key in out:

                arr = np.asarray(
                    out[key],
                    dtype=float,
                )

                if (
                    arr.ndim == 1
                    and len(arr) > 10
                ):

                    t = arr

                    break


        for key in [
            "t0",
            "t0_true",
            "anchor_t0",
            "event_t0",
        ]:

            if key in out:

                try:

                    t0 = float(
                        out[key]
                    )

                    break

                except Exception:

                    pass


    # --------------------------------------------------------
    # Tuple/list return
    # --------------------------------------------------------

    elif isinstance(
        out,
        (
            tuple,
            list,
        ),
    ):

        arrays = []

        scalars = []

        for item in out:

            try:

                arr = np.asarray(
                    item,
                    dtype=float,
                )

            except Exception:

                continue


            if (
                arr.ndim == 1
                and arr.size > 10
            ):

                arrays.append(
                    arr
                )

            elif arr.size == 1:

                scalars.append(
                    float(
                        arr.reshape(-1)[0]
                    )
                )


        if arrays:

            t = max(
                arrays,
                key=len,
            )


        if (
            t is not None
            and scalars
        ):

            # A JD-like t0 should lie reasonably close to
            # the returned Roman epoch range.
            plausible = [
                x
                for x in scalars
                if (
                    np.min(t)
                    - 10.0 * TE
                    <= x
                    <= np.max(t)
                    + 10.0 * TE
                )
            ]

            if plausible:

                t0 = min(
                    plausible,
                    key=lambda x: abs(
                        x
                        - np.median(t)
                    ),
                )


    # --------------------------------------------------------
    # Array-only return
    # --------------------------------------------------------

    else:

        try:

            arr = np.asarray(
                out,
                dtype=float,
            )

            if (
                arr.ndim == 1
                and len(arr) > 10
            ):

                t = arr

        except Exception:

            pass


    # --------------------------------------------------------
    # Search module globals for t0 if necessary
    # --------------------------------------------------------

    if t0 is None:

        for name in [
            "T0_TRUE",
            "t0_true",
            "T0_ROMAN",
            "t0_roman",
        ]:

            if hasattr(
                roman,
                name,
            ):

                try:

                    candidate = float(
                        getattr(
                            roman,
                            name,
                        )
                    )

                except Exception:

                    continue


                if (
                    t is None
                    or (
                        np.min(t)
                        - 10.0 * TE
                        <= candidate
                        <= np.max(t)
                        + 10.0 * TE
                    )
                ):

                    t0 = candidate

                    break


    if t is None:

        raise RuntimeError(
            "Could not resolve the Roman time array from "
            "build_roman_times()."
        )


    if t0 is None:

        raise RuntimeError(
            "Could not resolve t0 from build_roman_times(). "
            "Run:\n\n"
            "  python - <<'PY'\n"
            "  from binary_source.analysis import "
            "roman_bspl_pspl_asimov as r\n"
            "  print(r.build_roman_times(150.0))\n"
            "  PY\n"
        )


    t = np.asarray(
        t,
        dtype=float,
    )


    print()
    print(
        "Resolved Roman setup:"
    )

    print(
        "  N epochs =",
        len(t),
    )

    print(
        "  t0       =",
        f"{t0:.10f}",
    )

    print(
        "  t min    =",
        f"{np.min(t):.10f}",
    )

    print(
        "  t max    =",
        f"{np.max(t):.10f}",
    )


    return (
        t,
        float(t0),
    )


# ============================================================
# WINDOW TEST
# ============================================================

def n_time_for_window(
    window_te,
):
    """
    Preserve approximately the production temporal density.

    Production:
        10000 points across 7 tE.
    """

    dt_production = (
        2.0
        * WINDOW_PRODUCTION
        * TE
        / (
            N_TIME_PRODUCTION
            - 1
        )
    )

    width_days = (
        2.0
        * float(window_te)
        * TE
    )

    n_time = int(
        np.round(
            width_days
            / dt_production
        )
    ) + 1

    return max(
        n_time,
        100,
    )


def run_window_test(
    outdir,
    maxiter,
):
    cases = (
        build_intrinsic_cases()
    )


    print()
    print("=" * 100)
    print("INTRINSIC WINDOW-SENSITIVITY TEST")
    print("=" * 100)


    rows = []


    for case in cases:

        print()
        print("-" * 100)

        print(
            case.key,
            ":",
            case.description,
        )

        print("-" * 100)


        for window_te in (
            WINDOWS_TE
        ):

            n_time = (
                n_time_for_window(
                    window_te
                )
            )

            t = np.linspace(
                T0_INTRINSIC
                - window_te
                * TE,

                T0_INTRINSIC
                + window_te
                * TE,

                n_time,
            )


            truth = (
                roman.bspl_truth_magnification(
                    t=t,

                    t0_true=T0_INTRINSIC,
                    u0_true=case.u0,
                    tE_true=TE,

                    P_days=case.P_days,

                    q_mass=case.q_mass,
                    qflux=case.qflux,

                    Mtot_Msun=M_TOTAL,
                    rEhat_AU=REHAT_AU,

                    theta=THETA,
                    phi=PHI,
                    inclination=INCLINATION,
                )
            )


            A_truth = np.asarray(
                truth[
                    "A_bspl"
                ],
                dtype=float,
            )


            fit = fit_one_start(
                t=t,
                A_truth=A_truth,

                x0=np.array(
                    [
                        T0_INTRINSIC,
                        case.u0,
                        TE,
                    ],
                    dtype=float,
                ),

                maxiter=maxiter,
            )


            dt0_over_te = (
                fit[
                    "best_t0"
                ]
                - T0_INTRINSIC
            ) / TE


            du0_over_u0 = (
                (
                    fit[
                        "best_u0"
                    ]
                    - case.u0
                )
                / case.u0
            )


            dtE_over_tE = (
                fit[
                    "best_tE"
                ]
                - TE
            ) / TE


            row = {
                "case": (
                    case.key
                ),

                "description": (
                    case.description
                ),

                "window_tE": float(
                    window_te
                ),

                "n_time": int(
                    n_time
                ),

                "dt_days": float(
                    np.median(
                        np.diff(t)
                    )
                ),

                "u0_true": float(
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

                "xi_rel": float(
                    truth[
                        "xi_rel"
                    ]
                ),

                "D": float(
                    fit[
                        "D"
                    ]
                ),

                "J": float(
                    fit[
                        "J"
                    ]
                ),

                "best_t0": float(
                    fit[
                        "best_t0"
                    ]
                ),

                "best_u0": float(
                    fit[
                        "best_u0"
                    ]
                ),

                "best_tE": float(
                    fit[
                        "best_tE"
                    ]
                ),

                "dt0_over_tE": float(
                    dt0_over_te
                ),

                "du0_over_u0": float(
                    du0_over_u0
                ),

                "dtE_over_tE": float(
                    dtE_over_tE
                ),

                "success": bool(
                    fit[
                        "success"
                    ]
                ),
            }


            rows.append(
                row
            )


            print(
                f"window=+/-{window_te:3.1f} tE "
                f"N={n_time:6d} "
                f"D={fit['D']:.12e} "
                f"best="
                f"({fit['best_t0']:.8g}, "
                f"{fit['best_u0']:.8g}, "
                f"{fit['best_tE']:.8g})"
            )


    df = pd.DataFrame(
        rows
    )


    # ========================================================
    # Normalize each case to the production 3.5 tE value
    # ========================================================

    df[
        "D_reference_3p5"
    ] = np.nan

    df[
        "D_over_3p5"
    ] = np.nan

    df[
        "fractional_D_change"
    ] = np.nan


    for case_name in (
        df[
            "case"
        ].unique()
    ):

        mask = (
            df[
                "case"
            ]
            == case_name
        )

        ref = df[
            mask
            & np.isclose(
                df[
                    "window_tE"
                ],
                3.5,
            )
        ]


        if len(ref) != 1:

            raise RuntimeError(
                f"{case_name}: expected one "
                "3.5-tE reference row."
            )


        D_ref = float(
            ref.iloc[0][
                "D"
            ]
        )


        df.loc[
            mask,
            "D_reference_3p5",
        ] = D_ref


        df.loc[
            mask,
            "D_over_3p5",
        ] = (
            df.loc[
                mask,
                "D",
            ]
            / D_ref
        )


        df.loc[
            mask,
            "fractional_D_change",
        ] = (
            (
                df.loc[
                    mask,
                    "D",
                ]
                - D_ref
            )
            / D_ref
        )


    csv_path = (
        outdir
        / "window_sensitivity.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )


    # ========================================================
    # Physical ordering check
    # ========================================================

    ordering_rows = []


    for window_te in (
        WINDOWS_TE
    ):

        this = df[
            np.isclose(
                df[
                    "window_tE"
                ],
                window_te,
            )
        ].set_index(
            "case"
        )


        D_short = float(
            this.loc[
                "one_short",
                "D",
            ]
        )

        D_mid = float(
            this.loc[
                "one_intermediate",
                "D",
            ]
        )

        D_long = float(
            this.loc[
                "one_long",
                "D",
            ]
        )


        ordering_rows.append(
            {
                "window_tE": float(
                    window_te
                ),

                "D_short": D_short,
                "D_intermediate": D_mid,
                "D_long": D_long,

                "intermediate_largest": bool(
                    D_mid
                    > D_short
                    and D_mid
                    > D_long
                ),

                "D_long_over_D_intermediate": float(
                    D_long
                    / D_mid
                ),
            }
        )


    ordering_df = pd.DataFrame(
        ordering_rows
    )


    ordering_path = (
        outdir
        / "window_ordering_summary.csv"
    )

    ordering_df.to_csv(
        ordering_path,
        index=False,
    )


    # ========================================================
    # Print summary
    # ========================================================

    print()
    print("=" * 100)
    print("WINDOW-SENSITIVITY SUMMARY")
    print("=" * 100)


    for case_name in (
        df[
            "case"
        ].unique()
    ):

        this = (
            df[
                df[
                    "case"
                ]
                == case_name
            ]
            .sort_values(
                "window_tE"
            )
        )


        print()
        print(
            case_name
        )


        for _, row in (
            this.iterrows()
        ):

            print(
                f"  +/-{row['window_tE']:.1f} tE "
                f"D={row['D']:.8e} "
                f"D/D3.5={row['D_over_3p5']:.6f} "
                f"dD/D={row['fractional_D_change']:+.3e}"
            )


    print()
    print(
        "Period ordering:"
    )


    for _, row in (
        ordering_df.iterrows()
    ):

        print(
            f"  +/-{row['window_tE']:.1f} tE : "
            f"Dshort={row['D_short']:.4e}, "
            f"Dmid={row['D_intermediate']:.4e}, "
            f"Dlong={row['D_long']:.4e}, "
            f"mid_largest="
            f"{bool(row['intermediate_largest'])}"
        )


    print()
    print(
        "Saved:",
        csv_path,
    )

    print(
        "Saved:",
        ordering_path,
    )


    return (
        df,
        ordering_df,
    )


# ============================================================
# TRUE-BLENDING ROMAN TEST
# ============================================================

def make_blended_asimov_event(
    t,
    A_bspl,
    source_mag,
    fb_over_fs,
):
    """
    Start from the project's ordinary Roman Asimov event as a
    correctly configured pyLIMA event/telescope, then replace its
    photometry with a positively blended BSPL truth.

        Fs = source flux
        Fb = beta Fs
        beta = Fb/Fs

        F_truth(t) = Fs A_BSPL(t) + Fb

    The W149 uncertainty is recomputed from the TOTAL observed
    magnitude at each epoch.
    """

    base = (
        roman.make_roman_asimov_event(
            t=t,
            A_bspl=A_bspl,
            source_mag=source_mag,
        )
    )


    telescope = base[
        "telescope"
    ]

    lightcurve = (
        telescope.lightcurve
    )


    Fs = float(
        mag_to_flux(
            source_mag
        )
    )

    beta = float(
        fb_over_fs
    )

    Fb = (
        beta
        * Fs
    )

    Fbase = (
        Fs
        + Fb
    )


    A_bspl = np.asarray(
        A_bspl,
        dtype=float,
    )


    F_truth = (
        Fs
        * A_bspl
        + Fb
    )


    if np.any(
        F_truth <= 0.0
    ):

        raise RuntimeError(
            "Blended truth contains non-positive flux."
        )


    mag_truth = flux_to_mag(
        F_truth
    )


    sigma_mag = (
        sigma_w149_vector(
            mag_truth
        )
    )


    sigma_F = (
        magerr_to_fluxerr(
            F_truth,
            sigma_mag,
        )
    )


    # --------------------------------------------------------
    # Replace all representations consistently
    # --------------------------------------------------------

    assign_lightcurve_column(
        lightcurve,
        "flux",
        F_truth,
    )

    assign_lightcurve_column(
        lightcurve,
        "err_flux",
        sigma_F,
    )


    if (
        "inv_err_flux"
        in lightcurve.colnames
    ):

        assign_lightcurve_column(
            lightcurve,
            "inv_err_flux",
            1.0
            / sigma_F,
        )


    if (
        "mag"
        in lightcurve.colnames
    ):

        assign_lightcurve_column(
            lightcurve,
            "mag",
            mag_truth,
        )


    if (
        "err_mag"
        in lightcurve.colnames
    ):

        assign_lightcurve_column(
            lightcurve,
            "err_mag",
            sigma_mag,
        )


    return {
        "event": base[
            "event"
        ],

        "telescope": telescope,

        "Fs_true": Fs,
        "Fb_true": Fb,
        "Fbase_true": Fbase,

        "F_truth": F_truth,

        "mag_truth": mag_truth,

        "sigma_mag": sigma_mag,
        "sigma_F": sigma_F,

        "baseline_mag": float(
            flux_to_mag(
                Fbase
            )
        ),

        "n_bright_floor": int(
            np.sum(
                sigma_mag
                <= (
                    1.0e-3
                    * (
                        1.0
                        + 1.0e-12
                    )
                )
            )
        ),
    }


def run_roman_blended_case(
    case,
    t,
    t0_true,
    fb_over_fs,
):
    truth = (
        roman.bspl_truth_magnification(
            t=t,

            t0_true=t0_true,
            u0_true=case.u0,
            tE_true=TE,

            P_days=case.P_days,

            q_mass=case.q_mass,
            qflux=case.qflux,

            Mtot_Msun=M_TOTAL,
            rEhat_AU=REHAT_AU,

            theta=THETA,
            phi=PHI,
            inclination=INCLINATION,
        )
    )


    A_bspl = np.asarray(
        truth[
            "A_bspl"
        ],
        dtype=float,
    )


    asimov = (
        make_blended_asimov_event(
            t=t,
            A_bspl=A_bspl,
            source_mag=SOURCE_MAG,
            fb_over_fs=fb_over_fs,
        )
    )


    fit = (
        roman.fit_pspl_roman(
            ev=asimov[
                "event"
            ],

            t0_guess=t0_true,
            u0_guess=case.u0,
            tE_guess=TE,
        )
    )


    delta_chi2 = float(
        fit[
            "chi2"
        ]
    )


    # ========================================================
    # Blended event S/N
    #
    # Definition relative to the TRUE blended baseline:
    #
    # F_truth - Fbase
    #     = Fs (A_BSPL - 1)
    # ========================================================

    sigma_F = np.asarray(
        asimov[
            "sigma_F"
        ],
        dtype=float,
    )


    signal = (
        np.asarray(
            asimov[
                "F_truth"
            ],
            dtype=float,
        )
        - float(
            asimov[
                "Fbase_true"
            ]
        )
    )


    snr_event = float(
        np.sqrt(
            np.sum(
                (
                    signal
                    / sigma_F
                ) ** 2
            )
        )
    )


    if (
        np.isfinite(
            snr_event
        )
        and snr_event > 0.0
    ):

        D_roman_w = float(
            np.sqrt(
                max(
                    delta_chi2,
                    0.0,
                )
            )
            / snr_event
        )

    else:

        D_roman_w = np.nan


    best_model = np.asarray(
        fit[
            "best_model"
        ],
        dtype=float,
    ).reshape(-1)


    # Store the last two fitted photometric parameters without
    # assuming more than the single-telescope ftotal ordering.
    photometric_1 = (
        float(
            best_model[-2]
        )
        if len(best_model) >= 5
        else np.nan
    )

    photometric_2 = (
        float(
            best_model[-1]
        )
        if len(best_model) >= 5
        else np.nan
    )


    return {
        "case": (
            case.key
        ),

        "description": (
            case.description
        ),

        "source_mag": float(
            SOURCE_MAG
        ),

        "fb_over_fs_true": float(
            fb_over_fs
        ),

        "u0_true": float(
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

        "xi_rel": float(
            truth[
                "xi_rel"
            ]
        ),

        "Fs_true": float(
            asimov[
                "Fs_true"
            ]
        ),

        "Fb_true": float(
            asimov[
                "Fb_true"
            ]
        ),

        "Fbase_true": float(
            asimov[
                "Fbase_true"
            ]
        ),

        "baseline_mag": float(
            asimov[
                "baseline_mag"
            ]
        ),

        "delta_chi2": float(
            delta_chi2
        ),

        "snr_event": float(
            snr_event
        ),

        "D_roman_w": float(
            D_roman_w
        ),

        "best_t0": float(
            fit[
                "best_t0"
            ]
        ),

        "best_u0": float(
            fit[
                "best_u0"
            ]
        ),

        "best_tE": float(
            fit[
                "best_tE"
            ]
        ),

        "dt0_over_tE": float(
            (
                fit[
                    "best_t0"
                ]
                - t0_true
            )
            / TE
        ),

        "du0_over_u0": float(
            (
                fit[
                    "best_u0"
                ]
                - case.u0
            )
            / case.u0
        ),

        "dtE_over_tE": float(
            (
                fit[
                    "best_tE"
                ]
                - TE
            )
            / TE
        ),

        "fit_photometric_parameter_1": (
            photometric_1
        ),

        "fit_photometric_parameter_2": (
            photometric_2
        ),

        "chi2_reported": float(
            fit[
                "chi2_reported"
            ]
        ),

        "chi2_recomputed": float(
            fit[
                "chi2_recomputed"
            ]
        ),

        "n_obs": int(
            len(t)
        ),

        "n_bright_floor": int(
            asimov[
                "n_bright_floor"
            ]
        ),
    }


def run_blending_test(
    outdir,
):
    cases = (
        build_roman_cases()
    )


    t, t0_true = (
        resolve_roman_times()
    )


    print()
    print("=" * 100)
    print("ROMAN TRUE-BLENDING TEST")
    print("=" * 100)

    print(
        "source magnitude =",
        SOURCE_MAG,
    )

    print(
        "blend ratios Fb/Fs =",
        BLEND_RATIOS,
    )


    rows = []


    for case in cases:

        print()
        print("-" * 100)
        print(
            case.key,
            ":",
            case.description,
        )
        print("-" * 100)


        for beta in (
            BLEND_RATIOS
        ):

            row = (
                run_roman_blended_case(
                    case=case,
                    t=t,
                    t0_true=t0_true,
                    fb_over_fs=beta,
                )
            )


            rows.append(
                row
            )


            print(
                f"Fb/Fs={beta:4.1f} "
                f"mbase={row['baseline_mag']:.4f} "
                f"dchi2={row['delta_chi2']:.8e} "
                f"SNR={row['snr_event']:.6e} "
                f"Droman={row['D_roman_w']:.8e}"
            )


    df = pd.DataFrame(
        rows
    )


    # ========================================================
    # Normalize to the unblended truth for each case
    # ========================================================

    df[
        "delta_chi2_reference_unblended"
    ] = np.nan

    df[
        "delta_chi2_over_unblended"
    ] = np.nan

    df[
        "Droman_reference_unblended"
    ] = np.nan

    df[
        "Droman_over_unblended"
    ] = np.nan


    for case_name in (
        df[
            "case"
        ].unique()
    ):

        mask = (
            df[
                "case"
            ]
            == case_name
        )


        ref = df[
            mask
            & np.isclose(
                df[
                    "fb_over_fs_true"
                ],
                0.0,
            )
        ]


        if len(ref) != 1:

            raise RuntimeError(
                f"{case_name}: expected one "
                "unblended reference."
            )


        dchi_ref = float(
            ref.iloc[0][
                "delta_chi2"
            ]
        )


        D_ref = float(
            ref.iloc[0][
                "D_roman_w"
            ]
        )


        df.loc[
            mask,
            "delta_chi2_reference_unblended",
        ] = dchi_ref


        df.loc[
            mask,
            "delta_chi2_over_unblended",
        ] = (
            df.loc[
                mask,
                "delta_chi2",
            ]
            / dchi_ref
        )


        df.loc[
            mask,
            "Droman_reference_unblended",
        ] = D_ref


        df.loc[
            mask,
            "Droman_over_unblended",
        ] = (
            df.loc[
                mask,
                "D_roman_w",
            ]
            / D_ref
        )


    csv_path = (
        outdir
        / "roman_true_blending.csv"
    )


    df.to_csv(
        csv_path,
        index=False,
    )


    # ========================================================
    # Compact summary
    # ========================================================

    print()
    print("=" * 100)
    print("ROMAN TRUE-BLENDING SUMMARY")
    print("=" * 100)


    for case_name in (
        df[
            "case"
        ].unique()
    ):

        this = (
            df[
                df[
                    "case"
                ]
                == case_name
            ]
            .sort_values(
                "fb_over_fs_true"
            )
        )


        print()
        print(
            case_name
        )


        for _, row in (
            this.iterrows()
        ):

            print(
                f"  Fb/Fs={row['fb_over_fs_true']:4.1f} "
                f"dchi2={row['delta_chi2']:.6e} "
                f"dchi2/dchi2_0="
                f"{row['delta_chi2_over_unblended']:.6f} "
                f"Drom/Drom_0="
                f"{row['Droman_over_unblended']:.6f}"
            )


    print()
    print(
        "Saved:",
        csv_path,
    )


    return df


# ============================================================
# Metadata
# ============================================================

def save_metadata(
    outdir,
    args,
):
    metadata = {
        "git": git_info(),

        "tE_days": TE,

        "intrinsic_t0": (
            T0_INTRINSIC
        ),

        "intrinsic_production_window_tE": (
            WINDOW_PRODUCTION
        ),

        "intrinsic_production_n_time": (
            N_TIME_PRODUCTION
        ),

        "window_test_values_tE": (
            WINDOWS_TE
        ),

        "source_mag_blending_test": (
            SOURCE_MAG
        ),

        "blend_ratios_fb_over_fs": (
            BLEND_RATIOS
        ),

        "W149_zero_point": (
            W149_ZP
        ),

        "intrinsic_cases": [
            asdict(
                x
            )
            for x in (
                build_intrinsic_cases()
            )
        ],

        "roman_cases": [
            asdict(
                x
            )
            for x in (
                build_roman_cases()
            )
        ],

        "maxiter": int(
            args.maxiter
        ),
    }


    path = (
        outdir
        / "validation_metadata.json"
    )


    path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(
        "--mode",
        choices=[
            "window",
            "blending",
            "all",
        ],
        default="all",
    )


    parser.add_argument(
        "--maxiter",
        type=int,
        default=50_000,
    )


    parser.add_argument(
        "--output-dir",
        default=str(
            OUTDIR_DEFAULT
        ),
    )


    args = parser.parse_args()


    outdir = Path(
        args.output_dir
    )

    if not outdir.is_absolute():

        outdir = (
            ROOT
            / outdir
        )


    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("=" * 100)
    print("FINAL CHEAP VALIDATION TESTS")
    print("=" * 100)

    print(
        "mode   =",
        args.mode,
    )

    print(
        "outdir =",
        outdir,
    )


    if args.mode in (
        "window",
        "all",
    ):

        run_window_test(
            outdir=outdir,
            maxiter=args.maxiter,
        )


    if args.mode in (
        "blending",
        "all",
    ):

        run_blending_test(
            outdir=outdir,
        )


    save_metadata(
        outdir=outdir,
        args=args,
    )


    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        "Results:",
        outdir,
    )


if __name__ == "__main__":
    main()
