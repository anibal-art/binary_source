#!/usr/bin/env python3
"""
Build the H1 (BSPL truth) catalogue for the Roman LRT.

Each row is ONE noisy Roman realization of ONE physical BSPL configuration.
The catalogue only defines truths and seeds; it never touches fitting.

Families
--------
one_luminous_u0_P           (u0, P), qM=0.5, qf=0, 3 magnitudes.
                            Geometry MARGINALIZED: the n realizations of each cell
                            use n isotropic orbital geometries (stratified, shared
                            by all cells via geometry_id), one noise seed each.
                            Power per cell = power averaged over isotropic orbits.
one_luminous_u0_P_fiducial  same (u0, P) grid, F146=21, FIXED fiducial geometry of
                            the draft figures; n noise realizations per cell.
                            Bridge to the intrinsic maps (Figure 7).
one_luminous_qM_P           (qM, P), u0=0.1, qf=0, fiducial geometry, F146=21
two_luminous_qM_qf          (qM, qf) at three periods, u0=0.1, fiducial geometry, F146=21
one_luminous_u0_P_tE150     optional (--include-te150), tE=150 d, F146=21,
                            marginalized geometry (needs its own H0 calibration)

Conventions (verified against pyLIMA, see check_t0_u0.py)
----------------------------------------------------------
* (t0, u0, tE) define the reference trajectory ((t-t0)/tE, u0).
  Source 1 lies at (0, u0) at t = t0.  The barycenter moves on the parallel
  line u_ref(t) - xi_E1 R(-alpha) d(t0); its t0/u0 are stored as diagnostics.
* xi_inclination uses the pyLIMA convention: +/- pi/2 = face-on, 0 = edge-on.
  Isotropic orbits:  sin(xi_inclination) ~ U(-1, 1).
* Orientation alpha = atan2(xi_perp, xi_para); pyLIMA rotates with R(-alpha).
* xi_para, xi_perp carry the RELATIVE-orbit amplitude xi_rel = a_rel / rEhat.
  pyLIMA applies q/(1+q) to source 1 internally.
* q_mass = M2/M1 <= 1 (source 1 is the heavier star); q_flux = F2/F1.
* source_mag_F146_AB is the TOTAL unresolved source-system magnitude.
  (This must be confirmed in the smoke test with q_flux = 1.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from binary_source.analysis import lrt_catalog_io as io


# =============================================================================
# SCIENTIFIC SPECIFICATION
# =============================================================================


# Event / Roman setup ---------------------------------------------------------
T0 = 2461849.0                      # centre of Roman season 3
TE_MAIN_DAYS = 30.0
TE_ROBUST_DAYS = 150.0

F146_AB_MAIN = np.array([19.0, 21.0, 23.0])
F146_AB_SECONDARY = 21.0
F146_AB_MINUS_VEGA = 1.0364436822930514

# Source binary / lens ----------------------------------------------------------
REHAT_AU = 5.0
MTOT_SOURCE_MSUN = 3.0
Q_MASS_MAIN = 0.5
PIEN = 0.0
PIEE = 0.0
BLEND_RATIO = 0.0

# Fiducial geometry of the draft figures (pyLIMA convention).
FIDUCIAL_ALPHA = 0.0
FIDUCIAL_PHASE = 0.0
FIDUCIAL_INCLINATION = np.pi / 2.0      # face-on in pyLIMA convention

# Period range allowed by the BSPL fit bounds (omega in [2pi/1e5, 2pi/10]).
P_FIT_MIN_DAYS = 10.0
P_FIT_MAX_DAYS = 1.0e5

# Grids -------------------------------------------------------------------------
U0_GRID = np.array([
    0.010000, 0.0144225, 0.0208008, 0.030000, 0.0448140, 0.0669433,
    0.100000, 0.144225, 0.208008, 0.300000, 0.444697, 0.659184, 0.977124,
])

N_PERIODS = 60
P_ANCHOR_DAYS = 365.25

N_QM_LRT = 13
N_QF_POSITIVE_LRT = 13

# Two-luminous cuts.  The draft uses P/tE = 0.3, 1, 3, but P = 9 d is below
# the BSPL fit bound.  Default replaces it by P = 10 d (P/tE = 1/3).
TWO_LUMINOUS_PERIODS = {
    "p10": np.array([10.0, 30.0, 90.0]),
    "p9":  np.array([9.0, 30.0, 90.0]),     # requires widening the omega bound
}

# Monte Carlo design --------------------------------------------------------------
DEFAULT_N_REALIZATIONS = 50
DEFAULT_GEOMETRY_SEED = 20261023

XI_FIT_BOUND = 100.0
Q_FLUX_FIT_MAX = 2.0


# =============================================================================
# HELPERS
# =============================================================================

def anchored_logspace(low, high, n, anchors=()):
    values = np.geomspace(low, high, n)
    for anchor in anchors:
        if not (low <= anchor <= high):
            raise ValueError(f"anchor {anchor} outside [{low}, {high}]")
        j = int(np.argmin(np.abs(np.log(values) - np.log(anchor))))
        values[j] = float(anchor)
    values = np.sort(values)
    if len(np.unique(values)) != n:
        raise RuntimeError("Anchoring produced duplicate grid nodes.")
    return values


def kepler_a_rel_au(period_days, mtot_msun):
    """a^3 = M_tot P^2 in AU, yr, Msun."""
    return float((mtot_msun * (period_days / 365.25) ** 2) ** (1.0 / 3.0))


def binary_component_masses(q_mass):
    if not (0.0 < q_mass <= 1.0):
        raise ValueError(f"q_mass must lie in (0,1], got {q_mass}")
    m1 = MTOT_SOURCE_MSUN / (1.0 + q_mass)
    return float(m1), float(q_mass * m1)


def source_component_magnitudes(total_mag_ab, q_flux):
    """F146_AB is the TOTAL magnitude; q_flux = F2/F1."""
    m1 = total_mag_ab + 2.5 * np.log10(1.0 + q_flux)
    m2 = np.inf if q_flux == 0.0 else m1 - 2.5 * np.log10(q_flux)
    return float(m1), float(m2)


def barycenter_line(t0, u0, te, xi_e1, alpha, phase, inclination):
    """
    pyLIMA: u_B(t) = u_ref(t) - xi_E1 R(-alpha) d(t0),
            d(t0) = (cos phase, sin(i) sin phase).
    Returns (t0_B, u0_B) of the barycentric straight line.
    """
    d0x = np.cos(phase)
    d0y = np.sin(inclination) * np.sin(phase)
    # R(-alpha) d0
    off_tau = np.cos(alpha) * d0x + np.sin(alpha) * d0y
    off_beta = -np.sin(alpha) * d0x + np.cos(alpha) * d0y
    t0_b = t0 + te * xi_e1 * off_tau
    u0_b = u0 - xi_e1 * off_beta
    return float(t0_b), float(u0_b)


def latin_hypercube(n, dim, rng):
    """Stratified (Latin hypercube) sample in [0,1)^dim."""
    out = np.empty((n, dim))
    for k in range(dim):
        out[:, k] = (rng.permutation(n) + rng.uniform(size=n)) / n
    return out


def fiducial_geometry():
    return {
        "geometry_id": -1,
        "geometry_kind": "fiducial",
        "orbit_alpha_rad": FIDUCIAL_ALPHA,
        "xi_phase": FIDUCIAL_PHASE,
        "xi_inclination": FIDUCIAL_INCLINATION,
    }


def marginalized_geometries(n, geometry_seed):
    """
    n isotropic circular-orbit geometries, generated ONCE and shared by all
    cells (geometry_id = 0..n-1).  Latin-hypercube stratification in
        alpha ~ U(0, 2pi), phase ~ U(0, 2pi), sin(xi_inclination) ~ U(-1, 1)
    (pyLIMA convention: isotropic orbits <=> sin(i_pyLIMA) uniform).
    """
    rng = np.random.default_rng(int(geometry_seed))
    lhs = latin_hypercube(n, 3, rng)
    rows = []
    for k in range(n):
        rows.append({"geometry_id": k, "geometry_kind": "isotropic_lhs",
                     "orbit_alpha_rad": float(2 * np.pi * lhs[k, 0]),
                     "xi_phase": float(2 * np.pi * lhs[k, 1]),
                     "xi_inclination": float(np.arcsin(2 * lhs[k, 2] - 1.0))})
    return rows


# =============================================================================
# ROW CONSTRUCTION
# =============================================================================

def make_row(*, scan_family, power_cell_id, realization_id, u0, period_days,
             te_days, q_mass, q_flux, f146_ab, geometry,
             u0_grid_index=-1, period_grid_index=-1, qM_grid_index=-1,
             qf_grid_index=-1, ptE_grid_index=-1, config_name=""):

    m1, m2 = binary_component_masses(q_mass)
    a_rel = kepler_a_rel_au(period_days, MTOT_SOURCE_MSUN)
    xi_rel = a_rel / REHAT_AU
    xi_e1 = q_mass / (1.0 + q_mass) * xi_rel

    alpha = float(geometry["orbit_alpha_rad"])
    phase = float(geometry["xi_phase"])
    incl = float(geometry["xi_inclination"])

    t0_b, u0_b = barycenter_line(T0, u0, te_days, xi_e1, alpha, phase, incl)
    m1_ab, m2_ab = source_component_magnitudes(f146_ab, q_flux)

    # canonical columns shared with H0 (identity, sky, trajectory, photometry,
    # blending, ephemeris provenance)
    row = io.common_fields(
        catalog_kind="h1", config_id=power_cell_id, realization_id=realization_id,
        scan_family=scan_family, system_type="BSPL", u0=u0, tE=te_days,
        f146_ab=f146_ab, piEN=PIEN, piEE=PIEE, t0=T0, truth_double_source=True,
        q_flux=q_flux)

    row.update({
        "config_name": config_name,
        "geometry_id": int(geometry["geometry_id"]),
        "geometry_kind": geometry["geometry_kind"],
        "field_id": "roman_bspl_lrt_h1",
        "genulens_event_id": -1,
        "star_id": -1,

        # native pyLIMA circular double source
        "xi_para": float(xi_rel * np.cos(alpha)),
        "xi_perp": float(xi_rel * np.sin(alpha)),
        "xi_angular_velocity": float(2.0 * np.pi / period_days),
        "xi_phase": phase,
        "xi_inclination": incl,
        "xi_mass_ratio": float(q_mass),
        "q_flux": float(q_flux),
        "q_flux_W149": float(q_flux),

        # geometry semantics
        "orbit_alpha_rad": alpha,
        "sin_xi_inclination": float(np.sin(incl)),
        "inclination_convention": "pyLIMA: +-pi/2 face-on, 0 edge-on",
        "rotation_convention": "pyLIMA R(-alpha), alpha=atan2(xi_perp,xi_para)",

        # physical binary-source metadata
        "Mtot_source_Msun": MTOT_SOURCE_MSUN,
        "M1_source_Msun": m1,
        "M2_source_Msun": m2,
        "q_mass": float(q_mass),
        "P_days": float(period_days),
        "P_over_tE": float(period_days / te_days),
        "a_rel_AU": a_rel,
        "rEhat_AU": REHAT_AU,
        "xi_rel": float(xi_rel),
        "xi_E_primary": float(xi_e1),
        "xi_E_over_u0": float(xi_e1 / u0),

        # barycentric diagnostics (derived, not inputs)
        "t0_barycenter": t0_b,
        "u0_barycenter_line": u0_b,

        # component magnitudes (diagnostic; F146_AB is the system total)
        "source1_mag_F146_AB": m1_ab,
        "source2_mag_F146_AB": m2_ab,

        # grid coordinates
        "u0_grid_index": int(u0_grid_index),
        "period_grid_index": int(period_grid_index),
        "qM_grid_index": int(qM_grid_index),
        "qf_grid_index": int(qf_grid_index),
        "ptE_grid_index": int(ptE_grid_index),
    })
    return row


# =============================================================================
# CATALOGUE
# =============================================================================

def build_catalog(*, n_real, geometry_seed, smoke, include_te150,
                  two_luminous_cut):

    if smoke:
        u0_grid = np.array([0.01, 0.1, 0.977124])
        period_grid = np.array([10.0, 365.25, 1.0e5])
        qM_grid = np.array([1.0e-4, 0.5, 1.0])
        qf_grid = np.array([0.0, 0.5, 1.0])
        mags_main = np.array([21.0])
        n_real = min(n_real, 3)
    else:
        u0_grid = U0_GRID.copy()
        period_grid = anchored_logspace(P_FIT_MIN_DAYS, P_FIT_MAX_DAYS,
                                        N_PERIODS, anchors=(P_ANCHOR_DAYS,))
        qM_grid = anchored_logspace(1.0e-4, 1.0, N_QM_LRT, anchors=(0.5,))
        qf_grid = np.concatenate([[0.0],
                                  anchored_logspace(1.0e-4, 1.0,
                                                    N_QF_POSITIVE_LRT,
                                                    anchors=(0.5,))])
        mags_main = F146_AB_MAIN

    two_lum_periods = TWO_LUMINOUS_PERIODS[two_luminous_cut]
    fid = fiducial_geometry()
    geoms = marginalized_geometries(n_real, geometry_seed)
    rows = []

    # A. one luminous source, (u0, P), all magnitudes, MARGINALIZED geometry --
    #    realization r uses geometry_id = r (same set in every cell) and its
    #    own noise seed.
    for iu, u0 in enumerate(u0_grid):
        for ip, P in enumerate(period_grid):
            for mag in mags_main:
                cell = f"one_luminous_u0_P:u{iu:03d}:p{ip:03d}:m{int(mag):02d}"
                for r in range(n_real):
                    rows.append(make_row(
                        scan_family="one_luminous_u0_P", power_cell_id=cell,
                        realization_id=r, u0=u0, period_days=P,
                        te_days=TE_MAIN_DAYS, q_mass=Q_MASS_MAIN, q_flux=0.0,
                        f146_ab=mag, geometry=geoms[r],
                        u0_grid_index=iu, period_grid_index=ip))

    # A'. same (u0, P) grid, F146 = 21, FIXED fiducial geometry (bridge to Fig. 7)
    for iu, u0 in enumerate(u0_grid):
        for ip, P in enumerate(period_grid):
            cell = f"one_luminous_u0_P_fiducial:u{iu:03d}:p{ip:03d}:m21"
            for r in range(n_real):
                rows.append(make_row(
                    scan_family="one_luminous_u0_P_fiducial",
                    power_cell_id=cell, realization_id=r, u0=u0,
                    period_days=P, te_days=TE_MAIN_DAYS,
                    q_mass=Q_MASS_MAIN, q_flux=0.0,
                    f146_ab=F146_AB_SECONDARY, geometry=fid,
                    u0_grid_index=iu, period_grid_index=ip))

    # B. one luminous source, (qM, P), F146 = 21 ------------------------------
    for iq, qm in enumerate(qM_grid):
        for ip, P in enumerate(period_grid):
            cell = f"one_luminous_qM_P:q{iq:03d}:p{ip:03d}:m21"
            for r in range(n_real):
                rows.append(make_row(
                    scan_family="one_luminous_qM_P", power_cell_id=cell,
                    realization_id=r, u0=0.1, period_days=P,
                    te_days=TE_MAIN_DAYS, q_mass=qm, q_flux=0.0,
                    f146_ab=F146_AB_SECONDARY, geometry=fid,
                    qM_grid_index=iq, period_grid_index=ip))

    # C. two luminous sources, (qM, qf) at three periods, F146 = 21 ----------
    for iq, qm in enumerate(qM_grid):
        for jf, qf in enumerate(qf_grid):
            for kp, P in enumerate(two_lum_periods):
                cell = (f"two_luminous_qM_qf:q{iq:03d}:f{jf:03d}:"
                        f"r{kp:02d}:m21")
                for r in range(n_real):
                    rows.append(make_row(
                        scan_family="two_luminous_qM_qf", power_cell_id=cell,
                        realization_id=r, u0=0.1, period_days=P,
                        te_days=TE_MAIN_DAYS, q_mass=qm, q_flux=qf,
                        f146_ab=F146_AB_SECONDARY, geometry=fid,
                        qM_grid_index=iq, qf_grid_index=jf, ptE_grid_index=kp))

    # E. optional tE = 150 d table, F146 = 21, MARGINALIZED geometry ----------
    if include_te150:
        for iu, u0 in enumerate(u0_grid):
            for ip, P in enumerate(period_grid):
                cell = f"one_luminous_u0_P_tE150:u{iu:03d}:p{ip:03d}:m21"
                for r in range(n_real):
                    rows.append(make_row(
                        scan_family="one_luminous_u0_P_tE150",
                        power_cell_id=cell, realization_id=r, u0=u0,
                        period_days=P, te_days=TE_ROBUST_DAYS,
                        q_mass=Q_MASS_MAIN, q_flux=0.0,
                        f146_ab=F146_AB_SECONDARY, geometry=geoms[r],
                        u0_grid_index=iu, period_grid_index=ip))

    # index, seeds (H1 block), truth_config_id + canonical validation
    df = io.finalize_catalog(rows, "h1")

    # ------------------------------------------------------------------ checks
    def require(cond, msg):
        if not bool(cond):
            raise RuntimeError(msg)

    require(np.allclose(np.hypot(df["xi_para"], df["xi_perp"]), df["xi_rel"],
                        rtol=1e-12, atol=1e-14), "|xi| != xi_rel")
    require(np.allclose(df["M1_source_Msun"] + df["M2_source_Msun"],
                        MTOT_SOURCE_MSUN, rtol=0, atol=1e-12), "M1+M2 != Mtot")
    require(np.allclose(df["M2_source_Msun"] / df["M1_source_Msun"],
                        df["q_mass"], rtol=1e-12), "q_mass inconsistent")
    require((df["q_mass"] > 0).all() and (df["q_mass"] <= 1).all(),
            "q_mass outside (0,1]")
    require((df["q_flux"] >= 0).all() and (df["q_flux"] <= Q_FLUX_FIT_MAX).all(),
            "q_flux outside fit bound")
    require((df[["xi_para", "xi_perp"]].abs() <= XI_FIT_BOUND).all().all(),
            "xi outside fit bound")
    require(np.allclose(np.sin(df["xi_inclination"]), df["sin_xi_inclination"]),
            "inclination bookkeeping")
    for fam in ("one_luminous_u0_P", "one_luminous_u0_P_tE150"):
        sub = df[df["scan_family"] == fam]
        if len(sub) == 0:
            continue
        per_cell = sub.groupby("power_cell_id")["geometry_id"].apply(
            lambda g: tuple(sorted(g)))
        require((per_cell == tuple(range(n_real))).all(),
                f"{fam}: each cell must use geometry_id 0..{n_real-1} exactly once")
        require((sub["geometry_kind"] == "isotropic_lhs").all(),
                f"{fam}: non-isotropic geometry found")
    for fam in ("one_luminous_u0_P_fiducial", "one_luminous_qM_P",
                "two_luminous_qM_qf"):
        sub = df[df["scan_family"] == fam]
        require((sub["geometry_kind"] == "fiducial").all(),
                f"{fam}: must use the fiducial geometry")
    if two_luminous_cut == "p10":
        require((df["P_days"] >= P_FIT_MIN_DAYS).all(),
                "period below BSPL fit bound")

    grids = {
        "u0_grid": [float(x) for x in u0_grid],
        "period_grid_days": [float(x) for x in period_grid],
        "qM_grid": [float(x) for x in qM_grid],
        "qf_grid": [float(x) for x in qf_grid],
        "two_luminous_periods_days": [float(x) for x in two_lum_periods],
        "marginalized_geometries": geoms,
    }
    return df, grids


# =============================================================================
# IO
# =============================================================================

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("catalogs/roman_bspl_lrt_v1/h1_bspl_candidate_v3.parquet"))
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--n-realizations", type=int, default=DEFAULT_N_REALIZATIONS)
    ap.add_argument("--geometry-seed", type=int, default=DEFAULT_GEOMETRY_SEED)
    ap.add_argument("--two-luminous-short-cut", choices=["p10", "p9"], default="p10",
                    help="p10: P=10 d (inside fit bounds). p9: draft P/tE=0.3 "
                         "(requires widening the omega fit bound).")
    ap.add_argument("--include-te150", action="store_true",
                    help="add the tE=150 d table (needs its own H0 calibration)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.n_realizations <= 0:
        raise ValueError("number of realizations must be > 0")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest.expanduser().resolve() if args.manifest
                     else output.with_suffix(".manifest.json"))

    df, grids = build_catalog(
        n_real=args.n_realizations,
        geometry_seed=args.geometry_seed,
        smoke=args.smoke,
        include_te150=args.include_te150,
        two_luminous_cut=args.two_luminous_short_cut,
    )

    df.to_parquet(output, index=False, compression="zstd")
    digest = sha256_file(output)
    family_counts = {k: int(v) for k, v in df.groupby("scan_family").size().items()}

    manifest = {
        "catalog_schema_version": io.SCHEMA_VERSION,
        "catalog_kind": "h1",
        "status": "SMOKE" if args.smoke else "CANDIDATE",
        "production_frozen": False,
        "output": str(output),
        "sha256": digest,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "family_counts": family_counts,
        "n_realizations": args.n_realizations,
        "geometry_design": {
            "marginalized_families": ["one_luminous_u0_P", "one_luminous_u0_P_tE150"],
            "marginalized": "n isotropic geometries (LHS in alpha, phase, sin i_pyLIMA), "
                            "shared across cells via geometry_id; one noise seed per row",
            "fiducial_families": ["one_luminous_u0_P_fiducial", "one_luminous_qM_P",
                                  "two_luminous_qM_qf"],
            "uncertainty": "report binomial error per cell AND bootstrap over geometry_id",
        },
        "geometry_seed": args.geometry_seed,
        "noise_seed_base": io.SEED_BASE["h1"],
        "noise_seed_range": [int(df["simulation_seed"].min()),
                             int(df["simulation_seed"].max())],
        "two_luminous_short_cut": args.two_luminous_short_cut,
        "include_te150": bool(args.include_te150),
        "t0": T0,
        "tE_main_days": TE_MAIN_DAYS,
        "Mtot_source_Msun": MTOT_SOURCE_MSUN,
        "rEhat_AU": REHAT_AU,
        "blend_ratio": BLEND_RATIO,
        "sky_position": {"ra": io.RA_FIELD, "dec": io.DEC_FIELD,
                         "note": "fixed Roman field used by roman_rubin"},
        "conventions": {
            "trajectory": "(t0,u0) = source 1 at (0,u0) at t0; barycenter line stored as diagnostic",
            "inclination": "pyLIMA: +-pi/2 face-on, 0 edge-on; isotropic = sin(i) ~ U(-1,1)",
            "rotation": "pyLIMA R(-alpha), alpha = atan2(xi_perp, xi_para)",
            "xi": "xi_para, xi_perp carry xi_rel = a_rel/rEhat (relative orbit)",
            "masses": "q_mass = M2/M1 <= 1",
            "flux": "q_flux = F2/F1",
            "magnitude": "source_mag_F146_AB = total unresolved source system",
        },
        "fiducial_geometry": {"alpha": FIDUCIAL_ALPHA, "phase": FIDUCIAL_PHASE,
                              "xi_inclination": FIDUCIAL_INCLINATION},
        "pending_verification": [
            "W149 = source-1 magnitude; check_catalog_simulation must give baseline = total",
            "effective blending recorded as zero in simulated outputs",
            "noise-free simulated curve equals degeneracy_fit.py magnification",
            "u0 grid resolution audit against Figure 7",
        ],
        "grids": grids,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print("=" * 80)
    print("H1 BSPL CATALOGUE  status =", manifest["status"])
    print("=" * 80)
    print("rows     =", len(df))
    print("output   =", output)
    print("sha256   =", digest)
    print("manifest =", manifest_path)
    print()
    print(df.groupby("scan_family").size().to_string())
    print()
    for key in ["u0", "P_days", "q_mass", "q_flux", "source_mag_F146_AB",
                "xi_rel", "xi_E_over_u0", "xi_inclination",
                "t0_barycenter", "u0_barycenter_line"]:
        print(f"{key:22s} [{df[key].min():.6g}, {df[key].max():.6g}]")
    print()
    print("PENDING VERIFICATION (smoke):")
    for item in manifest["pending_verification"]:
        print("  -", item)
    print()
    print("CATALOG BUILD: PASS")


if __name__ == "__main__":
    main()
