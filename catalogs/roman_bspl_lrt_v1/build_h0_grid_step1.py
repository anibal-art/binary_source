#!/usr/bin/env python3

import numpy as np


# ============================================================
# H0 v12 -- STEP 1
#
# Define and audit the scientific grid only.
#
# NO parquet.
# NO seeds.
# NO noise realizations.
# NO piEN/piEE yet.
# ============================================================


# ------------------------------------------------------------
# t0
# Primary calibration is conditional on the center of S3.
# ------------------------------------------------------------

T0_PRIMARY_JD = 2461849.0


# ------------------------------------------------------------
# u0
# Main domain: [0.005, 2], positive branch.
# ------------------------------------------------------------

U0_GRID = np.array(
    [
        0.005,
        0.0136,
        0.037,
        0.10,
        0.27,
        0.74,
        2.0,
    ],
    dtype=float,
)


# ------------------------------------------------------------
# tE / magnitude strata
#
# Four strata only:
#
#   tE ~ 30 d:
#       F146 AB = 19, 21, 23
#
#   tE ~ 150 d:
#       F146 AB = 21 only
# ------------------------------------------------------------

STRATA = [
    {
        "stratum_id": "te30_m19",
        "source_mag_F146_AB": 19.0,
        "tE_grid_days": [20.0, 30.0, 45.0],
    },
    {
        "stratum_id": "te30_m21",
        "source_mag_F146_AB": 21.0,
        "tE_grid_days": [20.0, 30.0, 45.0],
    },
    {
        "stratum_id": "te30_m23",
        "source_mag_F146_AB": 23.0,
        "tE_grid_days": [20.0, 30.0, 45.0],
    },
    {
        "stratum_id": "te150_m21",
        "source_mag_F146_AB": 21.0,
        "tE_grid_days": [100.0, 150.0, 225.0],
    },
]


# ------------------------------------------------------------
# Parallax pilot radii
#
# 0.01, 0.02, 0.03 are diagnostic nodes for the small-piE
# regime. They are NOT assumed to survive into the final grid.
#
# The main declared domain reaches |piE| = 2.
# ------------------------------------------------------------

PIE_RADII_PILOT = np.array(
    [
        0.0,
        0.01,
        0.02,
        0.03,
        0.07,
        0.17,
        0.41,
        1.0,
        2.0,
    ],
    dtype=float,
)


# ------------------------------------------------------------
# Direction pilot
#
# Eight non-zero directions, every 45 degrees.
#
# IMPORTANT:
# these angles will be defined RELATIVE TO the projected
# acceleration axis at t0.
#
# We do not calculate piEN/piEE in this step.
# ------------------------------------------------------------

PIE_DIRECTION_OFFSETS_DEG = np.arange(
    0.0,
    360.0,
    45.0,
)


# ============================================================
# Audits
# ============================================================

assert len(U0_GRID) == 7

assert np.isclose(
    U0_GRID.min(),
    0.005,
)

assert np.isclose(
    U0_GRID.max(),
    2.0,
)

assert len(STRATA) == 4

assert np.array_equal(
    PIE_RADII_PILOT[:4],
    np.array(
        [0.0, 0.01, 0.02, 0.03]
    ),
)

assert np.isclose(
    PIE_RADII_PILOT.max(),
    2.0,
)

assert len(
    PIE_DIRECTION_OFFSETS_DEG
) == 8


# One piE=0 configuration has NO direction.
#
# Every non-zero radius has 8 directions.

N_PARALLAX_VECTORS = (
    1
    +
    (
        len(PIE_RADII_PILOT) - 1
    )
    * len(PIE_DIRECTION_OFFSETS_DEG)
)


# Every stratum currently has:
#
#   7 u0
#   3 tE
#   N_PARALLAX_VECTORS

N_CONFIG_PER_STRATUM = (
    len(U0_GRID)
    * 3
    * N_PARALLAX_VECTORS
)

N_TOTAL_CONFIG = (
    len(STRATA)
    * N_CONFIG_PER_STRATUM
)


# Test A uses piE=0 only:
#
#   7 u0 x 3 tE
#   per stratum

N_TEST_A_PER_STRATUM = (
    len(U0_GRID)
    * 3
)

N_TEST_A_TOTAL = (
    len(STRATA)
    * N_TEST_A_PER_STRATUM
)


# ============================================================
# Report
# ============================================================

print("=" * 80)
print("H0 v12 -- SCIENTIFIC GRID -- STEP 1")
print("=" * 80)

print()
print("t0 primary:")
print(T0_PRIMARY_JD)

print()
print("u0 grid:")
for i, value in enumerate(U0_GRID):
    print(
        f"{i:2d}  {value:.8g}"
    )

print()
print("strata:")
for row in STRATA:
    print(
        row["stratum_id"],
        "  F146_AB =",
        row["source_mag_F146_AB"],
        "  tE =",
        row["tE_grid_days"],
    )

print()
print("piE radii pilot:")
print(PIE_RADII_PILOT)

print()
print(
    "direction offsets [deg] =",
    PIE_DIRECTION_OFFSETS_DEG,
)

print()
print(
    "N parallax vectors =",
    N_PARALLAX_VECTORS,
)

print(
    "N configs / stratum =",
    N_CONFIG_PER_STRATUM,
)

print(
    "N H0 configs total =",
    N_TOTAL_CONFIG,
)

print()
print(
    "N piE=0 configs / stratum =",
    N_TEST_A_PER_STRATUM,
)

print(
    "N Test-A H0 configs total =",
    N_TEST_A_TOTAL,
)


# Expected counts for this pilot grid.

if N_PARALLAX_VECTORS != 65:
    raise RuntimeError(
        "Expected 65 parallax vectors."
    )

if N_CONFIG_PER_STRATUM != 1365:
    raise RuntimeError(
        "Expected 1365 configurations per stratum."
    )

if N_TOTAL_CONFIG != 5460:
    raise RuntimeError(
        "Expected 5460 H0 configurations."
    )

if N_TEST_A_TOTAL != 84:
    raise RuntimeError(
        "Expected 84 piE=0 configurations for Test A."
    )


print()
print("PASS")
