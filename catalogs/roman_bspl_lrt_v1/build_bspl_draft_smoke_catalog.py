#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u


OUT = Path(
    "catalogs/roman_bspl_lrt_v1/"
    "bspl_draft_smoke.parquet"
)


# ============================================================================
# Draft-based physical point
# ============================================================================

# Figure 7/8 fixed source-binary parameters.
DL_KPC = 4.0
DS_KPC = 8.0
MU_REL_MAS_YR = 4.0

MTOT_SOURCE_MSUN = 2.0
Q_MASS = 0.5
Q_FLUX = 0.0

U0 = 0.3

DRAFT_THETA_RAD = 0.0
DRAFT_PHI0_RAD = np.pi
DRAFT_LAMBDA_XI_RAD = 0.0


# Fiducial projected Einstein radius used in the draft maps.
REHAT_AU = 5.0


# ============================================================================
# Smoke-test choices
# ============================================================================

# Translate draft t0=0 to the real Roman S3 center.
T0_JD = 2461849.0

# One period inside the draft P=10--1000 d scan.
P_DAYS = 365.25

# Current Roman benchmark magnitude.
F146_AB = 21.0

# Model-flux convention used by the current Roman backend.
AB_MINUS_VEGA = 1.0364436822930514

F146_VEGA = (
    F146_AB
    - AB_MINUS_VEGA
)


# ============================================================================
# Derived binary-source quantities
# ============================================================================

M1_MSUN = (
    MTOT_SOURCE_MSUN
    / (1.0 + Q_MASS)
)

M2_MSUN = (
    Q_MASS
    * M1_MSUN
)


# Kepler:
#
#   P_yr^2 = a_rel_AU^3 / Mtot_Msun
#
P_YR = P_DAYS / 365.25

A_REL_AU = (
    MTOT_SOURCE_MSUN
    * P_YR**2
) ** (1.0 / 3.0)


XI_REL = (
    A_REL_AU
    / REHAT_AU
)


# pyLIMA uses the RELATIVE binary orbit.
#
# Draft xallarap amplitude of the primary:
#
#   xi_E_primary
#       = xi_rel q/(1+q)
#
XI_E_PRIMARY = (
    XI_REL
    * Q_MASS
    / (1.0 + Q_MASS)
)


OMEGA_PER_DAY = (
    2.0
    * np.pi
    / P_DAYS
)


# ============================================================================
# Draft -> pyLIMA geometry translation
# ============================================================================

# theta = 0:
#
#   xi_para = xi_rel
#   xi_perp = 0
#
XI_PARA = XI_REL
XI_PERP = 0.0


# Draft phi0 = pi.
XI_PHASE = np.pi


# The draft lambda_xi=0 corresponds to the usual face-on convention.
#
# pyLIMA uses the complementary inclination convention:
# face-on projected circle -> +/- pi/2.
#
XI_INCLINATION_PYLIMA = (
    np.pi / 2.0
)


# ============================================================================
# Einstein timescale implied by the fiducial draft geometry
# ============================================================================

# At DS [kpc],
#
#   1 AU / DS_kpc = 1/DS_kpc mas
#
THETA_E_MAS = (
    REHAT_AU
    / DS_KPC
)

TE_YR = (
    THETA_E_MAS
    / MU_REL_MAS_YR
)

TE_DAYS = (
    TE_YR
    * 365.25
)


# ============================================================================
# Canonical Roman field
# ============================================================================

RA_DEG = 267.92497054815516
DEC_DEG = -29.152232510353283

coord = SkyCoord(
    ra=RA_DEG * u.deg,
    dec=DEC_DEG * u.deg,
    frame="icrs",
)

L_DEG = float(
    coord.galactic.l.deg
)

B_DEG = float(
    coord.galactic.b.deg
)


# ============================================================================
# One-row materialized catalog
# ============================================================================

row = {
    # ------------------------------------------------------------------------
    # Catalog identity
    # ------------------------------------------------------------------------
    "catalog_schema_version":
        "roman_bspl_lrt_v1_smoke",

    # Physical row 0 == catalog_event_index 0.
    "catalog_event_index":
        0,

    "truth_config_id":
        "bspl_draft_fiducial_P365p25",

    "realization_id":
        0,

    # load_event_from_catalog will use catalog_event_index as its canonical
    # simulation identity. The runner can override the actual noise seed.
    "simulation_seed":
        0,

    "field_id":
        "roman_bspl_draft_smoke",

    "genulens_event_id":
        -1,

    "star_id":
        -1,

    "system_type":
        "BSPL",

    # ------------------------------------------------------------------------
    # Sky position
    # ------------------------------------------------------------------------
    "ra":
        RA_DEG,

    "dec":
        DEC_DEG,

    "l_deg":
        L_DEG,

    "b_deg":
        B_DEG,

    "maf_ra":
        RA_DEG,

    "maf_dec":
        DEC_DEG,

    # ------------------------------------------------------------------------
    # Microlensing truth
    # ------------------------------------------------------------------------
    "t0":
        T0_JD,

    "u0":
        U0,

    "tE":
        TE_DAYS,

    # H1 truth has NO observer parallax.
    "piEN":
        0.0,

    "piEE":
        0.0,

    # ------------------------------------------------------------------------
    # Native pyLIMA circular binary-source truth
    # ------------------------------------------------------------------------
    "xi_para":
        XI_PARA,

    "xi_perp":
        XI_PERP,

    "xi_angular_velocity":
        OMEGA_PER_DAY,

    "xi_phase":
        XI_PHASE,

    "xi_inclination":
        XI_INCLINATION_PYLIMA,

    "xi_mass_ratio":
        Q_MASS,

    # simulation API converts this into q_flux_W149 internally
    "q_flux":
        Q_FLUX,

    # ------------------------------------------------------------------------
    # Photometry
    # ------------------------------------------------------------------------
    "source_mag_F146_AB":
        F146_AB,

    "source_mag_F146_Vega":
        F146_VEGA,

    "W149":
        F146_VEGA,

    # Present only for compatibility with generic catalog readers.
    # Rubin is NOT used in this experiment.
    "u":
        F146_VEGA,

    "g":
        F146_VEGA,

    "r":
        F146_VEGA,

    "i":
        F146_VEGA,

    "z":
        F146_VEGA,

    "Y":
        F146_VEGA,

    "blend_ratio_F146":
        0.0,

    # ------------------------------------------------------------------------
    # Physical provenance
    # ------------------------------------------------------------------------
    "DL_kpc":
        DL_KPC,

    "DS_kpc":
        DS_KPC,

    "mu_rel_mas_yr":
        MU_REL_MAS_YR,

    "Mtot_source_Msun":
        MTOT_SOURCE_MSUN,

    "M1_source_Msun":
        M1_MSUN,

    "M2_source_Msun":
        M2_MSUN,

    "q_mass":
        Q_MASS,

    "q_flux":
        Q_FLUX,

    "P_days":
        P_DAYS,

    "a_rel_AU":
        A_REL_AU,

    "rEhat_AU":
        REHAT_AU,

    "xi_rel":
        XI_REL,

    "xi_E_primary":
        XI_E_PRIMARY,

    "draft_theta_rad":
        DRAFT_THETA_RAD,

    "draft_phi0_rad":
        DRAFT_PHI0_RAD,

    "draft_lambda_xi_rad":
        DRAFT_LAMBDA_XI_RAD,

    "truth_parallax":
        False,

    "truth_double_source":
        True,
}


df = pd.DataFrame(
    [row]
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_parquet(
    OUT,
    index=False,
)


print("=" * 90)
print("BSPL DRAFT SMOKE CATALOG")
print("=" * 90)

print(
    df.T.to_string(
        header=False
    )
)

print()
print("DERIVED CHECKS")
print("-" * 90)

print(
    "M1 [Msun]          =",
    M1_MSUN,
)

print(
    "M2 [Msun]          =",
    M2_MSUN,
)

print(
    "M1 + M2 [Msun]     =",
    M1_MSUN + M2_MSUN,
)

print(
    "a_rel [AU]         =",
    A_REL_AU,
)

print(
    "rEhat [AU]         =",
    REHAT_AU,
)

print(
    "xi_rel             =",
    XI_REL,
)

print(
    "xi_E primary       =",
    XI_E_PRIMARY,
)

print(
    "omega [1/d]        =",
    OMEGA_PER_DAY,
)

print(
    "tE [d]             =",
    TE_DAYS,
)

print(
    "F146 AB            =",
    F146_AB,
)

print(
    "W149/F146 Vega     =",
    F146_VEGA,
)

print()
print(
    "output =",
    OUT.resolve(),
)

print()
print("PASS")
