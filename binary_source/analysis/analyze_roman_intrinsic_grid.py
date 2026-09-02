#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Input files
# ============================================================

FILES = {
    19.0: Path(
        "results/roman_asimov/"
        "roman_intrinsic_grid_W149_19.npz"
    ),
    21.0: Path(
        "results/roman_asimov/"
        "roman_intrinsic_grid_W149_21.npz"
    ),
    23.0: Path(
        "results/roman_asimov/"
        "roman_intrinsic_grid_W149_23.npz"
    ),
}


OUTDIR = Path(
    "results/roman_asimov/combined_intrinsic_grid"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUT_NPZ = (
    OUTDIR
    / "roman_intrinsic_grid_W149_19_21_23.npz"
)

OUT_LONG_CSV = (
    OUTDIR
    / "roman_intrinsic_grid_long.csv"
)

OUT_SUMMARY_CSV = (
    OUTDIR
    / "roman_intrinsic_grid_summary.csv"
)


# ============================================================
# Load
# ============================================================

loaded = {}

for mag, path in FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    loaded[mag] = np.load(
        path,
        allow_pickle=False,
    )


# ============================================================
# Reference grid
# ============================================================

ref_mag = 21.0
ref = loaded[ref_mag]

u0 = np.asarray(
    ref["u0_grid"],
    dtype=float,
)

P_grid = np.asarray(
    ref["P_grid"],
    dtype=float,
)

P_over_tE = np.asarray(
    ref["P_over_tE"],
    dtype=float,
)

D_intrinsic = np.asarray(
    ref["D_INTRINSIC"],
    dtype=float,
)


Nu0 = len(u0)
NP = len(P_grid)

expected_shape = (
    Nu0,
    NP,
)


print("=" * 90)
print("ROMAN + INTRINSIC GRID COMBINATION")
print("=" * 90)

print(
    f"Reference grid: W149={ref_mag:g}"
)

print(
    f"Nu0 = {Nu0}"
)

print(
    f"NP  = {NP}"
)

print(
    f"N cells = {Nu0 * NP}"
)

print(
    f"u0 range = "
    f"{u0.min():.8g} -- {u0.max():.8g}"
)

print(
    f"P/tE range = "
    f"{P_over_tE.min():.8g} -- "
    f"{P_over_tE.max():.8g}"
)

print(
    f"D intrinsic range = "
    f"{D_intrinsic.min():.8e} -- "
    f"{D_intrinsic.max():.8e}"
)


# ============================================================
# Validate exact common grid
# ============================================================

print()
print("=" * 90)
print("GRID CONSISTENCY")
print("=" * 90)

for mag, d in loaded.items():

    this_u0 = np.asarray(
        d["u0_grid"],
        dtype=float,
    )

    this_P = np.asarray(
        d["P_grid"],
        dtype=float,
    )

    this_ratio = np.asarray(
        d["P_over_tE"],
        dtype=float,
    )

    this_D = np.asarray(
        d["D_INTRINSIC"],
        dtype=float,
    )

    checks = {
        "u0": np.allclose(
            this_u0,
            u0,
            rtol=0.0,
            atol=1e-14,
        ),
        "P": np.allclose(
            this_P,
            P_grid,
            rtol=0.0,
            atol=1e-12,
        ),
        "P/tE": np.allclose(
            this_ratio,
            P_over_tE,
            rtol=0.0,
            atol=1e-14,
        ),
        "D": np.allclose(
            this_D,
            D_intrinsic,
            rtol=1e-12,
            atol=1e-14,
        ),
    }

    print(
        f"W149={mag:g}: {checks}"
    )

    if not all(checks.values()):
        raise RuntimeError(
            f"Grid mismatch for W149={mag:g}"
        )


# ============================================================
# Allocate merged arrays
# ============================================================

magnitudes = np.array(
    sorted(loaded.keys()),
    dtype=float,
)

Nm = len(magnitudes)

shape3 = (
    Nm,
    Nu0,
    NP,
)


DELTA_CHI2 = np.full(
    shape3,
    np.nan,
)

D_ROMAN_EFF = np.full(
    shape3,
    np.nan,
)

SNR_EVENT = np.full(
    shape3,
    np.nan,
)

SUCCESS = np.zeros(
    shape3,
    dtype=bool,
)

DT0_OVER_TE = np.full(
    shape3,
    np.nan,
)

DU0_OVER_U0 = np.full(
    shape3,
    np.nan,
)

DTE_OVER_TE = np.full(
    shape3,
    np.nan,
)


# ============================================================
# Fill
# ============================================================

for im, mag in enumerate(magnitudes):

    d = loaded[float(mag)]

    DELTA_CHI2[im] = np.asarray(
        d["DELTA_CHI2"][0],
        dtype=float,
    )

    D_ROMAN_EFF[im] = np.asarray(
        d["D_ROMAN_EFF"][0],
        dtype=float,
    )

    SNR_EVENT[im] = np.asarray(
        d["SNR_EVENT"][0],
        dtype=float,
    )

    SUCCESS[im] = np.asarray(
        d["SUCCESS"][0],
        dtype=bool,
    )

    DT0_OVER_TE[im] = np.asarray(
        d["DT0_OVER_TE"][0],
        dtype=float,
    )

    DU0_OVER_U0[im] = np.asarray(
        d["DU0_OVER_U0"][0],
        dtype=float,
    )

    DTE_OVER_TE[im] = np.asarray(
        d["DTE_OVER_TE"][0],
        dtype=float,
    )


# ============================================================
# Basic validation
# ============================================================

print()
print("=" * 90)
print("FIT VALIDATION")
print("=" * 90)

for im, mag in enumerate(magnitudes):

    valid = (
        SUCCESS[im]
        & np.isfinite(
            DELTA_CHI2[im]
        )
        & np.isfinite(
            D_ROMAN_EFF[im]
        )
        & np.isfinite(
            SNR_EVENT[im]
        )
    )

    print(
        f"W149={mag:g}: "
        f"{valid.sum()} / {valid.size} "
        f"valid "
        f"({100*valid.mean():.3f}%)"
    )

    if not np.all(valid):
        raise RuntimeError(
            f"Invalid Roman cells for W149={mag:g}"
        )


# ============================================================
# Magnitude monotonicity
# ============================================================

print()
print("=" * 90)
print("MAGNITUDE MONOTONICITY")
print("=" * 90)

for im in range(Nm - 1):

    bright = DELTA_CHI2[im]
    faint = DELTA_CHI2[im + 1]

    monotonic = (
        bright
        >= faint
    )

    print(
        f"W149={magnitudes[im]:g} >= "
        f"W149={magnitudes[im+1]:g}: "
        f"{monotonic.sum()} / "
        f"{monotonic.size} "
        f"({100*monotonic.mean():.3f}%)"
    )


# ============================================================
# Summary tables
# ============================================================

D_THRESHOLDS = [
    1e-1,
    1e-2,
    1e-3,
]

CHI_THRESHOLDS = [
    1,
    10,
    25,
    100,
    500,
]


summary_rows = []


print()
print("=" * 90)
print("SUMMARY BY MAGNITUDE")
print("=" * 90)

for im, mag in enumerate(magnitudes):

    chi = DELTA_CHI2[im]
    deff = D_ROMAN_EFF[im]
    snr = SNR_EVENT[im]

    valid = (
        SUCCESS[im]
        & np.isfinite(chi)
        & np.isfinite(deff)
        & np.isfinite(snr)
        & np.isfinite(D_intrinsic)
    )

    # --------------------------------------------------------
    # D intrinsic vs Roman weighted mismatch
    # --------------------------------------------------------

    comp = (
        valid
        & (D_intrinsic > 0)
        & (deff > 0)
    )

    corr = np.corrcoef(
        np.log10(
            D_intrinsic[comp]
        ),
        np.log10(
            deff[comp]
        ),
    )[0, 1]

    ratio = (
        deff[comp]
        / D_intrinsic[comp]
    )

    print()
    print(
        f"W149 = {mag:g}"
    )
    print("-" * 70)

    print(
        f"DeltaChi2: "
        f"min={chi[valid].min():.6e}  "
        f"median={np.median(chi[valid]):.6e}  "
        f"max={chi[valid].max():.6e}"
    )

    print(
        f"D_Roman_eff: "
        f"min={deff[valid].min():.6e}  "
        f"median={np.median(deff[valid]):.6e}  "
        f"max={deff[valid].max():.6e}"
    )

    print(
        f"SNR: "
        f"min={snr[valid].min():.6e}  "
        f"median={np.median(snr[valid]):.6e}  "
        f"max={snr[valid].max():.6e}"
    )

    print(
        f"corr(log D, log D_eff) = "
        f"{corr:.6f}"
    )

    print(
        "D_eff / D: "
        f"median={np.median(ratio):.6f}, "
        f"P16={np.percentile(ratio,16):.6f}, "
        f"P84={np.percentile(ratio,84):.6f}"
    )

    base_row = {
        "W149": mag,
        "n_cells": int(
            valid.sum()
        ),
        "delta_chi2_min": float(
            chi[valid].min()
        ),
        "delta_chi2_median": float(
            np.median(
                chi[valid]
            )
        ),
        "delta_chi2_max": float(
            chi[valid].max()
        ),
        "D_roman_eff_min": float(
            deff[valid].min()
        ),
        "D_roman_eff_median": float(
            np.median(
                deff[valid]
            )
        ),
        "D_roman_eff_max": float(
            deff[valid].max()
        ),
        "snr_min": float(
            snr[valid].min()
        ),
        "snr_median": float(
            np.median(
                snr[valid]
            )
        ),
        "snr_max": float(
            snr[valid].max()
        ),
        "corr_logD_logDeff": float(
            corr
        ),
        "Deff_over_D_median": float(
            np.median(
                ratio
            )
        ),
        "Deff_over_D_p16": float(
            np.percentile(
                ratio,
                16,
            )
        ),
        "Deff_over_D_p84": float(
            np.percentile(
                ratio,
                84,
            )
        ),
    }

    # --------------------------------------------------------
    # Overall chi2 fractions
    # --------------------------------------------------------

    print()
    print("All cells:")

    for chithr in CHI_THRESHOLDS:

        mask = (
            valid
            & (chi >= chithr)
        )

        frac = (
            mask.sum()
            / valid.sum()
        )

        print(
            f"  DeltaChi2 >= {chithr:3d}: "
            f"{mask.sum():5d} / "
            f"{valid.sum():5d} "
            f"({100*frac:6.2f}%)"
        )

        base_row[
            f"frac_chi2_ge_{chithr}"
        ] = float(
            frac
        )

    # --------------------------------------------------------
    # Conditional on intrinsic D
    # --------------------------------------------------------

    print()
    print("Conditioned on intrinsic degeneracy:")

    for Dthr in D_THRESHOLDS:

        lowD = (
            valid
            & (D_intrinsic <= Dthr)
        )

        print()
        print(
            f"  D <= {Dthr:g}: "
            f"{lowD.sum()} cells"
        )

        base_row[
            f"n_D_le_{Dthr:g}"
        ] = int(
            lowD.sum()
        )

        for chithr in [
            25,
            100,
            500,
        ]:

            joint = (
                lowD
                & (
                    chi
                    >= chithr
                )
            )

            frac = (
                joint.sum()
                / lowD.sum()
            )

            print(
                f"    DeltaChi2 >= {chithr:3d}: "
                f"{joint.sum():5d} / "
                f"{lowD.sum():5d} "
                f"({100*frac:6.2f}%)"
            )

            key = (
                f"frac_D_le_{Dthr:g}"
                f"_chi2_ge_{chithr}"
            )

            base_row[key] = float(
                frac
            )

    summary_rows.append(
        base_row
    )


# ============================================================
# Long-form table
# ============================================================

rows = []

for im, mag in enumerate(magnitudes):

    for iu, u0_value in enumerate(u0):

        for ip, p_value in enumerate(
            P_over_tE
        ):

            rows.append(
                {
                    "W149": mag,
                    "iu0": iu,
                    "iP": ip,
                    "u0": u0_value,
                    "P_days": (
                        P_grid[ip]
                    ),
                    "P_over_tE": (
                        p_value
                    ),
                    "D_intrinsic": (
                        D_intrinsic[
                            iu,
                            ip,
                        ]
                    ),
                    "delta_chi2": (
                        DELTA_CHI2[
                            im,
                            iu,
                            ip,
                        ]
                    ),
                    "D_roman_eff": (
                        D_ROMAN_EFF[
                            im,
                            iu,
                            ip,
                        ]
                    ),
                    "snr_event": (
                        SNR_EVENT[
                            im,
                            iu,
                            ip,
                        ]
                    ),
                    "dt0_over_tE": (
                        DT0_OVER_TE[
                            im,
                            iu,
                            ip,
                        ]
                    ),
                    "du0_over_u0": (
                        DU0_OVER_U0[
                            im,
                            iu,
                            ip,
                        ]
                    ),
                    "dtE_over_tE": (
                        DTE_OVER_TE[
                            im,
                            iu,
                            ip,
                        ]
                    ),
                    "success": (
                        SUCCESS[
                            im,
                            iu,
                            ip,
                        ]
                    ),
                }
            )


df_long = pd.DataFrame(
    rows
)

df_summary = pd.DataFrame(
    summary_rows
)


df_long.to_csv(
    OUT_LONG_CSV,
    index=False,
)

df_summary.to_csv(
    OUT_SUMMARY_CSV,
    index=False,
)


# ============================================================
# Save merged NPZ
# ============================================================

np.savez_compressed(
    OUT_NPZ,

    w149_magnitudes=(
        magnitudes
    ),

    u0_grid=(
        u0
    ),

    P_grid=(
        P_grid
    ),

    P_over_tE=(
        P_over_tE
    ),

    D_INTRINSIC=(
        D_intrinsic
    ),

    DELTA_CHI2=(
        DELTA_CHI2
    ),

    D_ROMAN_EFF=(
        D_ROMAN_EFF
    ),

    SNR_EVENT=(
        SNR_EVENT
    ),

    DT0_OVER_TE=(
        DT0_OVER_TE
    ),

    DU0_OVER_U0=(
        DU0_OVER_U0
    ),

    DTE_OVER_TE=(
        DTE_OVER_TE
    ),

    SUCCESS=(
        SUCCESS
    ),

    source_files=np.array(
        [
            str(
                FILES[
                    float(mag)
                ]
            )
            for mag in magnitudes
        ]
    ),

    interpretation=np.array(
        "D_INTRINSIC is the continuous unweighted "
        "BSPL-to-PSPL shape mismatch. "
        "DELTA_CHI2 is the Roman Asimov PSPL rejection "
        "statistic. D_ROMAN_EFF = sqrt(DELTA_CHI2)/SNR_EVENT."
    ),
)


# ============================================================
# Most intrinsically degenerate examples
# ============================================================

print()
print("=" * 90)
print("MOST INTRINSICALLY DEGENERATE CELLS")
print("=" * 90)

indices = np.argsort(
    D_intrinsic,
    axis=None,
)[:10]

for flat in indices:

    iu, ip = np.unravel_index(
        flat,
        D_intrinsic.shape,
    )

    print()
    print(
        f"u0={u0[iu]:.6g}  "
        f"P/tE={P_over_tE[ip]:.6g}  "
        f"D={D_intrinsic[iu,ip]:.3e}"
    )

    for im, mag in enumerate(
        magnitudes
    ):

        print(
            f"    W149={mag:g}: "
            f"DeltaChi2="
            f"{DELTA_CHI2[im,iu,ip]:.3e}  "
            f"D_eff="
            f"{D_ROMAN_EFF[im,iu,ip]:.3e}  "
            f"SNR="
            f"{SNR_EVENT[im,iu,ip]:.3e}"
        )


# ============================================================
# Final
# ============================================================

print()
print("=" * 90)
print("OUTPUTS")
print("=" * 90)

print(
    OUT_NPZ
)

print(
    OUT_LONG_CSV
)

print(
    OUT_SUMMARY_CSV
)

print()
print(
    f"Long table rows = "
    f"{len(df_long)}"
)

print("=" * 90)
