#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# H0 catalog schema smoke test
#
# This is NOT the production H0 catalog.
# It contains one row only, to freeze and audit the schema.
# ============================================================

OUTDIR = Path("catalogs/roman_bspl_lrt_v1")
OUTPATH = OUTDIR / "h0_schema_smoke.parquet"

SCHEMA_VERSION = "roman_bspl_lrt_h0_v1"

EPHEMERIS_NAME = (
    "Roman_positions_2027_2032_periodic_v1.npy"
)

EPHEMERIS_SHA256 = (
    "eeb7bcf5300d2f3d4be3595898170975b9da20b5d47a26791af0c5258f307d58"
)

RA_DEG = 267.92497055
DEC_DEG = -29.15223251

F146_AB_MINUS_VEGA = 1.0364436822930514


# ============================================================
# One smoke-test row
# ============================================================

t0 = 2461849.0
u0 = 0.1
tE = 30.0

source_mag_F146_AB = 21.0

piEN = 0.0
piEE = 0.0

piE = float(
    np.hypot(
        piEN,
        piEE,
    )
)

piE_angle_rad = (
    np.nan
    if piE == 0.0
    else float(
        np.arctan2(
            piEE,
            piEN,
        )
    )
)


row = {
    # --------------------------------------------------------
    # Catalog identity
    # --------------------------------------------------------
    "catalog_schema_version":
        SCHEMA_VERSION,

    "catalog_event_index":
        0,

    "truth_config_id":
        "smoke_h0_tE30_u0p1_m21_piE0",

    "realization_id":
        0,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    "simulation_seed":
        2026102200,

    # --------------------------------------------------------
    # Sky / observing geometry
    # --------------------------------------------------------
    "ra_deg":
        RA_DEG,

    "dec_deg":
        DEC_DEG,

    # --------------------------------------------------------
    # Common PSPL truth parameters
    # --------------------------------------------------------
    "t0_truth_jd":
        t0,

    "u0_truth":
        u0,

    "tE_truth_days":
        tE,

    # --------------------------------------------------------
    # Roman source photometry
    # Scientific catalog convention: F146 AB.
    # Vega value is stored explicitly for backend provenance.
    # --------------------------------------------------------
    "source_mag_F146_AB":
        source_mag_F146_AB,

    "source_mag_F146_Vega":
        (
            source_mag_F146_AB
            - F146_AB_MINUS_VEGA
        ),

    "blend_ratio_F146":
        0.0,

    # --------------------------------------------------------
    # Parallax truth
    #
    # Same columns exist for every H0 row.
    #
    # Test A:
    #   uses the piEN=piEE=0 subset.
    #
    # Test B:
    #   uses the full parallax grid.
    # --------------------------------------------------------
    "piEN_truth":
        piEN,

    "piEE_truth":
        piEE,

    "piE_truth":
        piE,

    "piE_angle_rad":
        piE_angle_rad,

    # Convenient exact selector for Test A.
    "is_zero_parallax":
        bool(
            piEN == 0.0
            and piEE == 0.0
        ),

    # --------------------------------------------------------
    # Frozen Roman ephemeris provenance
    # --------------------------------------------------------
    "ephemeris_name":
        EPHEMERIS_NAME,

    "ephemeris_sha256":
        EPHEMERIS_SHA256,
}


df = pd.DataFrame(
    [row]
)


# ============================================================
# Minimal schema checks before writing
# ============================================================

required_columns = [
    "catalog_schema_version",
    "catalog_event_index",
    "truth_config_id",
    "realization_id",
    "simulation_seed",
    "ra_deg",
    "dec_deg",
    "t0_truth_jd",
    "u0_truth",
    "tE_truth_days",
    "source_mag_F146_AB",
    "source_mag_F146_Vega",
    "blend_ratio_F146",
    "piEN_truth",
    "piEE_truth",
    "piE_truth",
    "piE_angle_rad",
    "is_zero_parallax",
    "ephemeris_name",
    "ephemeris_sha256",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required H0 columns: {missing}"
    )


if not np.allclose(
    df["piE_truth"],
    np.hypot(
        df["piEN_truth"],
        df["piEE_truth"],
    ),
):
    raise RuntimeError(
        "piE_truth is inconsistent with piEN/piEE."
    )


# ============================================================
# Write and read back
# ============================================================

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_parquet(
    OUTPATH,
    index=False,
)

check = pd.read_parquet(
    OUTPATH
)


print("=" * 90)
print("H0 SCHEMA SMOKE TEST")
print("=" * 90)

print()
print("path:")
print(OUTPATH.resolve())

print()
print("shape:")
print(check.shape)

print()
print("columns:")
for i, col in enumerate(
    check.columns
):
    print(
        f"{i:2d}  "
        f"{col:28s}  "
        f"{check[col].dtype}"
    )

print()
print("row:")
print(
    check.T.to_string(
        header=False
    )
)

print()
print("Test A selector:")
print(
    check[
        check["is_zero_parallax"]
    ][
        [
            "catalog_event_index",
            "piEN_truth",
            "piEE_truth",
        ]
    ]
)

print()
print("PASS")
