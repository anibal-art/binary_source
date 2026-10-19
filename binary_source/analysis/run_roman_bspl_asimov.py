#!/usr/bin/env python3

"""
Project-level Roman BSPL/xallarap runner.

Science orchestration lives here in binary_source.

Roman cadence, ephemerides, F146/Pandeia uncertainties,
Asimov/noisy realization and pyLIMA TRF fitting are provided
exclusively by the canonical roman_rubin repository.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from binary_source.analysis import (
    roman_bspl_roman_rubin as bs,
)


T0_DEFAULT = 2461849.0


def plain(x):
    if hasattr(x, "value"):
        x = x.value

    return np.asarray(
        x,
        dtype=float,
    )


def first_mapping_value(mapping, prefixes):
    for prefix in prefixes:
        for key, value in mapping.items():
            if key.startswith(prefix):
                return float(value)

    raise KeyError(
        f"Could not find any of prefixes {prefixes}; "
        f"keys={list(mapping)}"
    )


def event_snr(simulation):
    """
    Event S/N used only as the historical Roman-weighted diagnostic:

        sqrt(sum[(F_truth - F_baseline)^2 / sigma_F^2])

    This is not the intrinsic D metric.
    """
    model = simulation["model"]
    params = simulation["py_params"]

    if len(model.event.telescopes) != 1:
        raise RuntimeError(
            "Expected exactly one Roman telescope."
        )

    tel = model.event.telescopes[0]

    flux = plain(
        tel.lightcurve["flux"]
    )

    err_flux = plain(
        tel.lightcurve["err_flux"]
    )

    baseline_flux = first_mapping_value(
        params,
        (
            "ftotal_",
            "fsource_",
        ),
    )

    valid = (
        np.isfinite(flux)
        & np.isfinite(err_flux)
        & (err_flux > 0)
    )

    return float(
        np.sqrt(
            np.sum(
                (
                    (
                        flux[valid]
                        - baseline_flux
                    )
                    / err_flux[valid]
                ) ** 2
            )
        )
    )


def verify_asimov(simulation):
    model = simulation["model"]
    params = simulation["py_params"]

    if len(model.event.telescopes) != 1:
        raise RuntimeError(
            "Expected exactly one Roman telescope."
        )

    tel = model.event.telescopes[0]

    observed = plain(
        tel.lightcurve["flux"]
    )

    err_flux = plain(
        tel.lightcurve["err_flux"]
    )

    truth = plain(
        model.compute_the_microlensing_model(
            tel,
            params,
        )["photometry"]
    )

    max_abs = float(
        np.max(
            np.abs(
                observed
                - truth
            )
        )
    )

    if not np.all(
        np.isfinite(err_flux)
    ):
        raise RuntimeError(
            "Non-finite Roman uncertainties."
        )

    if not np.all(
        err_flux > 0
    ):
        raise RuntimeError(
            "Non-positive Roman uncertainties."
        )

    if not np.allclose(
        observed,
        truth,
        rtol=1e-13,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Asimov data differ from exact truth: "
            f"max_abs={max_abs}"
        )

    return max_abs


def run_case(
    *,
    event_index,
    P_days,
    u0,
    tE,
    source_mag_ab,
    fit_window_tE,
):
    simulation = (
        bs.simulate_roman_bspl(
            event_index=int(event_index),
            P_days=float(P_days),
            u0=float(u0),
            tE=float(tE),
            t0=T0_DEFAULT,
            source_mag_ab=float(
                source_mag_ab
            ),
            asimov=True,
            truth_parallax=False,
            q_flux=0.0,
            time_window_te=float(
                fit_window_tE
            ),
        )
    )

    max_asimov_diff = (
        verify_asimov(
            simulation
        )
    )

    with tempfile.TemporaryDirectory(
        prefix="bspl_rr_fit_"
    ) as tmp:
        fit = bs.fit_pspl_comparison(
            simulation,
            output_dir=tmp,
            seed=int(event_index),
        )

    chi2_pspl = float(
        fit["chi2_h0"]
    )

    chi2_parallax = float(
        fit["chi2_h1"]
    )

    if (
        chi2_parallax
        >
        chi2_pspl
        + 1e-6
        * (
            1.0
            + abs(chi2_pspl)
        )
    ):
        raise RuntimeError(
            "Nested-model violation: "
            f"chi2_PSPL={chi2_pspl}, "
            f"chi2_PSPL+piE={chi2_parallax}"
        )

    best0 = np.asarray(
        fit["best_h0"],
        dtype=float,
    )

    best1 = np.asarray(
        fit["best_h1"],
        dtype=float,
    )

    snr = event_snr(
        simulation
    )

    D_roman_eff = (
        np.sqrt(
            max(
                chi2_pspl,
                0.0,
            )
        )
        / snr
        if snr > 0
        else np.nan
    )

    return {
        "simulation":
            simulation,

        "source_mag_ab":
            float(
                source_mag_ab
            ),

        "u0":
            float(u0),

        "P_days":
            float(P_days),

        "P_over_tE":
            float(
                P_days
                / tE
            ),

        "xi_rel":
            float(
                simulation[
                    "xallarap"
                ][
                    "xi_rel"
                ]
            ),

        # Historical Figure-7 statistic:
        # Asimov BSPL truth => chi2_truth = 0.
        "delta_chi2_pspl":
            chi2_pspl,

        # New robustness statistic after allowing annual parallax.
        "delta_chi2_pspl_parallax":
            chi2_parallax,

        "parallax_absorbed_chi2":
            (
                chi2_pspl
                - chi2_parallax
            ),

        "snr_event":
            snr,

        "D_roman_eff":
            float(
                D_roman_eff
            ),

        "best_t0":
            float(best0[0]),

        "best_u0":
            float(best0[1]),

        "best_tE":
            float(best0[2]),

        "best_parallax_t0":
            float(best1[0]),

        "best_parallax_u0":
            float(best1[1]),

        "best_parallax_tE":
            float(best1[2]),

        "best_piEN":
            float(best1[3]),

        "best_piEE":
            float(best1[4]),

        "best_piE":
            float(
                np.hypot(
                    best1[3],
                    best1[4],
                )
            ),

        "max_asimov_flux_difference":
            max_asimov_diff,
    }


def find_coordinate(
    data,
    candidates,
):
    for key in candidates:
        if key in data.files:
            arr = np.asarray(
                data[key],
                dtype=float,
            )

            arr = np.squeeze(arr)

            if arr.ndim == 1:
                return key, arr

    return None, None


def load_intrinsic_grid(
    source,
    tE,
    u0_max,
):
    """
    Load the intrinsic BSPL-vs-PSPL grid used to define the
    Roman observational nodes.

    Supported inputs
    ----------------
    1. Consolidated .parquet/.csv table, e.g. the current
       intrinsic xallarap-vs-parallax grid.  Its D0 column is
       the intrinsic BSPL-vs-PSPL metric.

    2. Historical directory containing
       scan_kepler_u0_*.npz files.

    Returns
    -------
    u0_grid, P_grid, D_intrinsic, source_path
    """
    source = Path(source)

    if not source.exists():
        raise FileNotFoundError(source)

    tE = float(tE)
    u0_max = float(u0_max)

    # ============================================================
    # A. Consolidated table
    # ============================================================

    if source.is_file() and source.suffix.lower() in {
        ".parquet",
        ".pq",
        ".csv",
    }:
        if source.suffix.lower() in {
            ".parquet",
            ".pq",
        }:
            df = pd.read_parquet(source)
        else:
            df = pd.read_csv(source)

        print(
            "Intrinsic consolidated grid:",
            source,
        )
        print(
            "columns =",
            list(df.columns),
        )

        def pick(candidates):
            for candidate in candidates:
                if candidate in df.columns:
                    return candidate
            return None

        u_col = pick(
            [
                "u0",
                "u0_true",
                "u0_input",
            ]
        )

        P_col = pick(
            [
                "P_days",
                "P",
                "period_days",
            ]
        )

        ratio_col = pick(
            [
                "P_over_tE",
                "P_over_TE",
            ]
        )

        D_col = pick(
            [
                "D0",
                "D_intrinsic",
                "D",
                "D_grid",
            ]
        )

        if u_col is None:
            raise RuntimeError(
                "Could not identify u0 column. "
                f"columns={list(df.columns)}"
            )

        work = df.copy()

        if P_col is None:
            if ratio_col is None:
                raise RuntimeError(
                    "Could not identify P_days or P/tE column. "
                    f"columns={list(df.columns)}"
                )

            P_col = "__P_days__"

            work[P_col] = (
                np.asarray(
                    work[ratio_col],
                    dtype=float,
                )
                * tE
            )

        if "status" in work.columns:
            work = work[
                work["status"].astype(str)
                == "ok"
            ].copy()

        if "success" in work.columns:
            success = (
                work["success"]
                .astype(bool)
                .to_numpy()
            )

            work = work.loc[
                success
            ].copy()

        work = work[
            np.isfinite(
                np.asarray(
                    work[u_col],
                    dtype=float,
                )
            )
            &
            np.isfinite(
                np.asarray(
                    work[P_col],
                    dtype=float,
                )
            )
        ].copy()

        work = work[
            np.abs(
                np.asarray(
                    work[u_col],
                    dtype=float,
                )
            )
            <= u0_max
        ].copy()

        u0_grid = np.sort(
            np.unique(
                np.asarray(
                    work[u_col],
                    dtype=float,
                )
            )
        )

        P_grid = np.sort(
            np.unique(
                np.asarray(
                    work[P_col],
                    dtype=float,
                )
            )
        )

        pairs = (
            work[
                [
                    u_col,
                    P_col,
                ]
            ]
            .drop_duplicates()
        )

        expected = (
            len(u0_grid)
            * len(P_grid)
        )

        if len(pairs) != expected:
            raise RuntimeError(
                "Intrinsic table is not a complete rectangular grid: "
                f"pairs={len(pairs)}, expected={expected}"
            )

        D_intrinsic = None

        if D_col is not None:
            pivot = (
                work.pivot(
                    index=u_col,
                    columns=P_col,
                    values=D_col,
                )
                .reindex(
                    index=u0_grid,
                    columns=P_grid,
                )
            )

            D_intrinsic = (
                pivot.to_numpy(
                    dtype=float,
                )
            )

        print(
            "u0 column =",
            u_col,
        )
        print(
            "P column =",
            P_col,
        )
        print(
            "D column =",
            D_col,
        )
        print(
            "grid shape =",
            (
                len(u0_grid),
                len(P_grid),
            ),
        )

        return (
            u0_grid,
            P_grid,
            D_intrinsic,
            source,
        )


    # ============================================================
    # B. Historical per-u0 NPZ directory
    # ============================================================

    if not source.is_dir():
        raise RuntimeError(
            f"Unsupported intrinsic grid source: {source}"
        )

    files = sorted(
        source.glob(
            "scan_kepler_u0_*.npz"
        )
    )

    if not files:
        raise RuntimeError(
            "No scan_kepler_u0_*.npz files under "
            f"{source}"
        )

    P_reference = None
    rows = []

    for path in files:
        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            required = {
                "truth",
                "P_grid",
                "D",
            }

            missing = (
                required
                - set(z.files)
            )

            if missing:
                raise RuntimeError(
                    f"{path}: missing {sorted(missing)}"
                )

            truth = np.asarray(
                z["truth"],
                dtype=float,
            )

            u0 = float(
                truth[1]
            )

            if abs(u0) > u0_max:
                continue

            P_this = np.asarray(
                z["P_grid"],
                dtype=float,
            )

            D_this = np.asarray(
                z["D"],
                dtype=float,
            ).copy()

            if P_reference is None:
                P_reference = (
                    P_this.copy()
                )

            elif not np.allclose(
                P_this,
                P_reference,
                rtol=0.0,
                atol=0.0,
            ):
                raise RuntimeError(
                    f"Inconsistent P_grid in {path}"
                )

            if "SUCCESS" in z.files:
                success = np.asarray(
                    z["SUCCESS"],
                    dtype=bool,
                )

                D_this[
                    ~success
                ] = np.nan

            rows.append(
                (
                    u0,
                    D_this,
                )
            )

    if not rows:
        raise RuntimeError(
            "No u0 rows survive requested u0_max."
        )

    rows.sort(
        key=lambda item: item[0]
    )

    u0_grid = np.asarray(
        [
            item[0]
            for item in rows
        ],
        dtype=float,
    )

    D_intrinsic = np.vstack(
        [
            item[1]
            for item in rows
        ]
    )

    P_grid = np.asarray(
        P_reference,
        dtype=float,
    )

    print(
        "Historical intrinsic directory:",
        source,
    )
    print(
        "grid shape =",
        D_intrinsic.shape,
    )

    return (
        u0_grid,
        P_grid,
        D_intrinsic,
        source,
    )



def make_default_grid(
    *,
    tE,
    n_u0,
    n_P,
    u0_min,
    u0_max,
    P_min,
    P_max,
):
    u = np.geomspace(
        float(u0_min),
        float(u0_max),
        int(n_u0),
    )

    P = np.geomspace(
        float(P_min),
        float(P_max),
        int(n_P),
    )

    return (
        u,
        P,
        None,
        None,
    )


def run_smoke(args):
    result = run_case(
        event_index=args.seed,
        P_days=args.smoke_period,
        u0=args.smoke_u0,
        tE=args.tE,
        source_mag_ab=args.source_mag,
        fit_window_tE=args.fit_window_tE,
    )

    summary = {
        key: value
        for key, value
        in result.items()
        if key != "simulation"
    }

    model = (
        result["simulation"][
            "model"
        ]
    )

    tel = (
        model.event.telescopes[0]
    )

    summary[
        "n_roman_epochs"
    ] = int(
        len(
            tel.lightcurve
        )
    )

    summary[
        "median_cadence_minutes"
    ] = float(
        np.median(
            np.diff(
                plain(
                    tel.lightcurve[
                        "time"
                    ]
                )
            )
        )
        * 24.0
        * 60.0
    )

    print()
    print("=" * 80)
    print("ROMAN BSPL ASIMOV SMOKE")
    print("=" * 80)
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


def run_grid(args):
    if args.intrinsic_grid_dir:
        (
            u0_grid,
            P_grid,
            D_intrinsic,
            intrinsic_source,
        ) = load_intrinsic_grid(
            args.intrinsic_grid_dir,
            tE=args.tE,
            u0_max=args.intrinsic_u0_max,
        )
    else:
        (
            u0_grid,
            P_grid,
            D_intrinsic,
            intrinsic_source,
        ) = make_default_grid(
            tE=args.tE,
            n_u0=args.n_u0,
            n_P=args.n_P,
            u0_min=args.u0_min,
            u0_max=args.intrinsic_u0_max,
            P_min=args.P_min,
            P_max=args.P_max,
        )

    mags = np.asarray(
        args.magnitudes,
        dtype=float,
    )

    shape = (
        len(mags),
        len(u0_grid),
        len(P_grid),
    )

    def blank():
        return np.full(
            shape,
            np.nan,
            dtype=float,
        )

    DELTA_CHI2 = blank()
    DELTA_CHI2_PARALLAX = blank()
    PARALLAX_ABSORBED_CHI2 = blank()

    SNR_EVENT = blank()
    D_ROMAN_EFF = blank()
    XI_REL = blank()

    BEST_T0 = blank()
    BEST_U0 = blank()
    BEST_TE = blank()

    BEST_PARALLAX_T0 = blank()
    BEST_PARALLAX_U0 = blank()
    BEST_PARALLAX_TE = blank()

    BEST_PIEN = blank()
    BEST_PIEE = blank()
    BEST_PIE = blank()

    SUCCESS = np.zeros(
        shape,
        dtype=bool,
    )

    first_times = None

    total = int(
        np.prod(shape)
    )

    counter = 0

    for im, mag in enumerate(mags):
        for iu, u0 in enumerate(u0_grid):
            for ip, P_days in enumerate(P_grid):

                counter += 1

                print(
                    f"[{counter}/{total}] "
                    f"F146={mag:g} "
                    f"u0={u0:.6g} "
                    f"P={P_days:.6g} d"
                )

                event_index = (
                    int(args.seed)
                    + counter
                )

                result = run_case(
                    event_index=event_index,
                    P_days=P_days,
                    u0=u0,
                    tE=args.tE,
                    source_mag_ab=mag,
                    fit_window_tE=(
                        args.fit_window_tE
                    ),
                )

                sim = result[
                    "simulation"
                ]

                if first_times is None:
                    tel = (
                        sim["model"]
                        .event
                        .telescopes[0]
                    )

                    first_times = plain(
                        tel.lightcurve[
                            "time"
                        ]
                    )

                idx = (
                    im,
                    iu,
                    ip,
                )

                DELTA_CHI2[idx] = (
                    result[
                        "delta_chi2_pspl"
                    ]
                )

                DELTA_CHI2_PARALLAX[idx] = (
                    result[
                        "delta_chi2_pspl_parallax"
                    ]
                )

                PARALLAX_ABSORBED_CHI2[idx] = (
                    result[
                        "parallax_absorbed_chi2"
                    ]
                )

                SNR_EVENT[idx] = (
                    result[
                        "snr_event"
                    ]
                )

                D_ROMAN_EFF[idx] = (
                    result[
                        "D_roman_eff"
                    ]
                )

                XI_REL[idx] = (
                    result[
                        "xi_rel"
                    ]
                )

                BEST_T0[idx] = (
                    result["best_t0"]
                )

                BEST_U0[idx] = (
                    result["best_u0"]
                )

                BEST_TE[idx] = (
                    result["best_tE"]
                )

                BEST_PARALLAX_T0[idx] = (
                    result[
                        "best_parallax_t0"
                    ]
                )

                BEST_PARALLAX_U0[idx] = (
                    result[
                        "best_parallax_u0"
                    ]
                )

                BEST_PARALLAX_TE[idx] = (
                    result[
                        "best_parallax_tE"
                    ]
                )

                BEST_PIEN[idx] = (
                    result["best_piEN"]
                )

                BEST_PIEE[idx] = (
                    result["best_piEE"]
                )

                BEST_PIE[idx] = (
                    result["best_piE"]
                )

                SUCCESS[idx] = True

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "roman_times":
            np.asarray(
                first_times,
                dtype=float,
            ),

        "f146_magnitudes":
            mags,

        # Compatibility alias for historical readers.
        "w149_magnitudes":
            mags,

        "u0_grid":
            np.asarray(
                u0_grid,
                dtype=float,
            ),

        "P_grid":
            np.asarray(
                P_grid,
                dtype=float,
            ),

        "P_over_tE":
            np.asarray(
                P_grid,
                dtype=float,
            )
            / float(args.tE),

        "DELTA_CHI2":
            DELTA_CHI2,

        "SQRT_DELTA_CHI2":
            np.sqrt(
                np.maximum(
                    DELTA_CHI2,
                    0.0,
                )
            ),

        "DELTA_CHI2_PARALLAX":
            DELTA_CHI2_PARALLAX,

        "PARALLAX_ABSORBED_CHI2":
            PARALLAX_ABSORBED_CHI2,

        "SNR_EVENT":
            SNR_EVENT,

        "D_ROMAN_EFF":
            D_ROMAN_EFF,

        "XI_REL":
            XI_REL,

        "BEST_T0":
            BEST_T0,

        "BEST_U0":
            BEST_U0,

        "BEST_TE":
            BEST_TE,

        "BEST_PARALLAX_T0":
            BEST_PARALLAX_T0,

        "BEST_PARALLAX_U0":
            BEST_PARALLAX_U0,

        "BEST_PARALLAX_TE":
            BEST_PARALLAX_TE,

        "BEST_PIEN":
            BEST_PIEN,

        "BEST_PIEE":
            BEST_PIEE,

        "BEST_PIE":
            BEST_PIE,

        "SUCCESS":
            SUCCESS,

        "t0_true":
            np.array(
                T0_DEFAULT
            ),

        "tE_true":
            np.array(
                float(args.tE)
            ),

        "q_mass":
            np.array(
                bs.Q_MASS
            ),

        "qflux":
            np.array(
                bs.Q_FLUX
            ),

        "rEhat_AU":
            np.array(
                bs.REHAT_AU
            ),

        "fit_window_tE":
            np.array(
                float(
                    args.fit_window_tE
                )
            ),

        "asimov":
            np.array(True),

        "truth_parallax":
            np.array(False),

        "simulation_backend":
            np.array(
                "roman_rubin"
            ),

        "fit_backend":
            np.array(
                "pyLIMA.TRFfit via roman_rubin"
            ),

        "intrinsic_grid_source":
            np.array(
                ""
                if intrinsic_source is None
                else str(
                    intrinsic_source
                )
            ),
    }

    if D_intrinsic is not None:
        payload[
            "D_INTRINSIC"
        ] = np.asarray(
            D_intrinsic,
            dtype=float,
        )

    np.savez_compressed(
        output,
        **payload,
    )

    print()
    print("=" * 80)
    print("ROMAN ASIMOV GRID COMPLETE")
    print("=" * 80)
    print("output =", output)
    print("shape =", shape)
    print(
        "success =",
        int(
            np.count_nonzero(
                SUCCESS
            )
        ),
        "/",
        SUCCESS.size,
    )


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "grid",
        ),
        default="smoke",
    )

    parser.add_argument(
        "--tE",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--source-mag",
        type=float,
        default=21.0,
    )

    parser.add_argument(
        "--magnitudes",
        nargs="+",
        type=float,
        default=[
            19.0,
            21.0,
            23.0,
        ],
    )

    parser.add_argument(
        "--fit-window-tE",
        type=float,
        default=3.5,
    )

    parser.add_argument(
        "--intrinsic-grid",
        "--intrinsic-grid-dir",
        default=None,
    )

    parser.add_argument(
        "--intrinsic-u0-max",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--n-u0",
        type=int,
        default=133,
    )

    parser.add_argument(
        "--n-P",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--u0-min",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--P-min",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--P-max",
        type=float,
        default=1e5,
    )

    parser.add_argument(
        "--smoke-u0",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--smoke-period",
        type=float,
        default=365.25,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261019,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/roman_asimov_f146_tE30/"
            "roman_intrinsic_grid_f146_tE30.npz"
        ),
    )

    # Retained only so old reproduction commands fail gracefully
    # rather than silently changing semantics.
    parser.add_argument(
        "--anchor-season",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--no-off-seasons",
        action="store_true",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.anchor_season != 2:
        parser.error(
            "The canonical roman_rubin backend fixes the "
            "validated Roman cadence/t0 configuration; "
            "--anchor-season values other than 2 are no "
            "longer supported here."
        )

    if args.no_off_seasons:
        parser.error(
            "--no-off-seasons belonged to the retired "
            "binary_source Roman simulator. Roman sampling "
            "is now defined by roman_rubin."
        )

    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
