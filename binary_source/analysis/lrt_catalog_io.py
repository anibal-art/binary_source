#!/usr/bin/env python3
"""
Single source of truth for the Roman BSPL LRT catalogues (H0 and H1).

* Defines the canonical column schema shared by H0 (PSPL / PSPL+parallax
  truths) and H1 (BSPL truths).  Column names are the ones read by
  roman_rubin (t0, u0, tE, piEN, piEE, W149, blend_ratio_W149, ra, dec ...).
* Validates whole catalogues (used by the builders before writing).
* Loads ONE row for the runner WITHOUT modifying it.  In particular the
  catalogue simulation_seed is preserved (roman_rubin's
  catalog.event_reader.load_event_from_catalog overwrites it with
  catalog_event_index, which would give H0 row k and H1 row k the same noise).
* Checks that the requested --truth-kind is compatible with the row.

CLI:
    python -m binary_source.analysis.lrt_catalog_io validate H0.parquet H1.parquet ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# Frozen conventions shared by every catalogue
# =============================================================================

SCHEMA_VERSION = "roman_bspl_lrt_catalog_v4"

T0_SEASON3 = 2461849.0
RA_FIELD = 267.92497054815516          # fit_lc.RA_FIELD in roman_rubin
DEC_FIELD = -29.152232510353276        # fit_lc.DEC_FIELD in roman_rubin
F146_AB_MINUS_VEGA = 1.0364436822930514

EPHEMERIS_NAME = "Roman_positions_2027_2032_periodic_v1.npy"
EPHEMERIS_SHA256 = "eeb7bcf5300d2f3d4be3595898170975b9da20b5d47a26791af0c5258f307d58"

ROMAN_RUBIN_REPO_DEFAULT = "/share/storage3/rubin/microlensing/romanrubin/roman_rubin"

# Disjoint noise-seed blocks (all < 2**32).
SEED_BASE = {"h1": 2_000_000_000, "h0": 3_000_000_000}
SEED_BLOCK = 1_000_000_000

COMMON_COLUMNS = [
    # identity
    "catalog_schema_version", "catalog_kind", "catalog_event_index",
    "simulation_seed", "event_seed",
    "config_id", "power_cell_id", "truth_config_id", "realization_id",
    "scan_family", "system_type",
    # sky (fixed Roman field actually used by roman_rubin)
    "ra", "dec", "maf_ra", "maf_dec", "l_deg", "b_deg",
    # reference trajectory
    "t0", "u0", "tE", "trajectory_reference",
    # parallax truth
    "piEN", "piEE", "piE", "piE_angle_rad", "is_zero_parallax",
    # photometry (F146 AB = total unresolved source; W149 = Vega for backend)
    "source_mag_F146_AB", "source_mag_F146_Vega", "W149",
    "blend_ratio_W149", "blend_ratio_F146",
    # materialized W149 fluxes required by roman_rubin sim_event (v4)
    "flux_zero_point_W149", "fsource_W149", "ftotal_W149",
    # provenance
    "ephemeris_name", "ephemeris_sha256",
    # truth switch fixed by the catalogue kind
    "truth_double_source",
]

BSPL_COLUMNS = [
    "xi_para", "xi_perp", "xi_angular_velocity", "xi_phase",
    "xi_inclination", "xi_mass_ratio", "q_flux", "q_flux_W149",
]

BACKEND_UNUSED_BANDS = ("u", "g", "r", "i", "z", "Y")


# =============================================================================
# Row helpers for the builders
# =============================================================================

def roman_rubin_repo():
    """The roman_rubin checkout used by every LRT script (env ROMAN_RUBIN_REPO)."""
    repo = Path(os.environ.get("ROMAN_RUBIN_REPO", ROMAN_RUBIN_REPO_DEFAULT)).resolve()
    if not repo.exists():
        raise RuntimeError(f"roman_rubin repository not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def roman_rubin_commit(repo=None):
    """HEAD commit of the roman_rubin checkout ('unknown' if git is unavailable)."""
    repo = roman_rubin_repo() if repo is None else Path(repo)
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def w149_zero_point():
    """roman_rubin simulation zero point of W149 (fluxes are materialized with it)."""
    roman_rubin_repo()
    from photometry.constants import SIMULATION_BAND_ZERO_POINTS
    return float(SIMULATION_BAND_ZERO_POINTS["W149"])


_ZP_W149 = None


def l_b_of_field():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = SkyCoord(ra=RA_FIELD * u.deg, dec=DEC_FIELD * u.deg, frame="icrs").galactic
    return float(g.l.deg), float(g.b.deg)


_LB = None


def common_fields(*, catalog_kind, config_id, realization_id, scan_family,
                  system_type, u0, tE, f146_ab, piEN=0.0, piEE=0.0,
                  t0=T0_SEASON3, truth_double_source=False, q_flux=0.0):
    """
    Canonical columns common to H0 and H1 (identity/seed added later).

    f146_ab is the TOTAL unresolved source magnitude (AB).  roman_rubin builds
    the baseline as F(W149) * (1 + q_flux) (verified with a q_flux = 1 smoke:
    baseline 0.753 mag brighter than W149), i.e. W149 is the magnitude of
    source 1.  Hence W149 = total Vega magnitude + 2.5 log10(1 + q_flux).
    """
    global _LB, _ZP_W149
    if _LB is None:
        _LB = l_b_of_field()
    if _ZP_W149 is None:
        _ZP_W149 = w149_zero_point()
    piE = float(np.hypot(piEN, piEE))
    vega = float(f146_ab) - F146_AB_MINUS_VEGA                  # total, Vega
    w149 = vega + 2.5 * np.log10(1.0 + float(q_flux))           # source 1, Vega
    row = {
        "catalog_schema_version": SCHEMA_VERSION,
        "catalog_kind": catalog_kind,
        "config_id": config_id,
        "power_cell_id": config_id,
        "realization_id": int(realization_id),
        "scan_family": scan_family,
        "system_type": system_type,
        "ra": RA_FIELD, "dec": DEC_FIELD,
        "maf_ra": RA_FIELD, "maf_dec": DEC_FIELD,
        "l_deg": _LB[0], "b_deg": _LB[1],
        "t0": float(t0), "u0": float(u0), "tE": float(tE),
        "trajectory_reference": "source1_at_(0,u0)_at_t0",
        "piEN": float(piEN), "piEE": float(piEE), "piE": piE,
        "piE_angle_rad": (np.nan if piE == 0.0 else float(np.arctan2(piEE, piEN))),
        "is_zero_parallax": bool(piE == 0.0),
        "source_mag_F146_AB": float(f146_ab),
        "source_mag_F146_Vega": vega,
        "W149": w149,
        "W149_semantics": "source-1 Vega magnitude (backend total = W149 flux x (1+q_flux))",
        "blend_ratio_W149": 0.0,
        "blend_ratio_F146": 0.0,
        # roman_rubin convention: fsource = source-1 flux, ftotal = fsource (1 + g);
        # pyLIMA baseline = ftotal + q_flux fsource (= total source flux for g = 0).
        "flux_zero_point_W149": _ZP_W149,
        "fsource_W149": 10.0 ** ((_ZP_W149 - w149) / 2.5),
        "ftotal_W149": 10.0 ** ((_ZP_W149 - w149) / 2.5),
        "ephemeris_name": EPHEMERIS_NAME,
        "ephemeris_sha256": EPHEMERIS_SHA256,
        "truth_double_source": bool(truth_double_source),
    }
    for band in BACKEND_UNUSED_BANDS:        # loader compatibility only
        row[band] = w149
    return row


def finalize_catalog(rows, catalog_kind):
    """Assign index, seeds and per-row truth_config_id; validate."""
    df = pd.DataFrame(rows).reset_index(drop=True)
    n = len(df)
    if n >= SEED_BLOCK:
        raise RuntimeError("catalogue larger than its seed block")
    df.insert(0, "catalog_event_index", np.arange(n, dtype=np.int64))
    seeds = np.int64(SEED_BASE[catalog_kind]) + df["catalog_event_index"].to_numpy(np.int64)
    df.insert(1, "simulation_seed", seeds)
    df["event_seed"] = df["simulation_seed"]
    df["truth_config_id"] = df["config_id"] + ":r" + df["realization_id"].map("{:04d}".format)
    validate_catalog(df, catalog_kind)
    return df


# =============================================================================
# Validation
# =============================================================================

class CatalogError(RuntimeError):
    pass


def _require(cond, msg):
    if not bool(cond):
        raise CatalogError(msg)


def validate_catalog(df: pd.DataFrame, catalog_kind: str) -> dict:
    """Raise CatalogError on any violation; return a short summary."""
    _require(catalog_kind in ("h0", "h1"), f"unknown catalog_kind {catalog_kind!r}")
    missing = [c for c in COMMON_COLUMNS if c not in df.columns]
    _require(not missing, f"missing canonical columns: {missing}")
    _require((df["catalog_schema_version"] == SCHEMA_VERSION).all(),
             "catalog_schema_version mismatch")
    _require((df["catalog_kind"] == catalog_kind).all(), "catalog_kind mismatch")
    _require(np.array_equal(df["catalog_event_index"].to_numpy(), np.arange(len(df))),
             "catalog_event_index must equal the physical row number")

    seeds = df["simulation_seed"].to_numpy(np.int64)
    lo = SEED_BASE[catalog_kind]
    _require(df["simulation_seed"].is_unique, "simulation_seed not unique")
    _require(((seeds >= lo) & (seeds < lo + SEED_BLOCK)).all(),
             f"simulation_seed outside the {catalog_kind} block [{lo}, {lo + SEED_BLOCK})")
    _require((seeds < 2**32).all(), "simulation_seed >= 2**32")
    _require((df["event_seed"] == df["simulation_seed"]).all(), "event_seed != simulation_seed")
    _require(df["truth_config_id"].is_unique, "truth_config_id not unique")
    _require((df["config_id"] == df["power_cell_id"]).all(), "config_id != power_cell_id")

    _require(np.allclose(df["ra"], RA_FIELD, rtol=0, atol=1e-12)
             and np.allclose(df["dec"], DEC_FIELD, rtol=0, atol=1e-12),
             "ra/dec differ from the fixed Roman field")
    qf = df["q_flux"].to_numpy(float) if "q_flux" in df.columns else np.zeros(len(df))
    _require(np.allclose(df["source_mag_F146_Vega"],
                         df["source_mag_F146_AB"] - F146_AB_MINUS_VEGA),
             "source_mag_F146_Vega != F146_AB - (AB-Vega)")
    _require(np.allclose(df["W149"], df["source_mag_F146_Vega"] + 2.5 * np.log10(1.0 + qf)),
             "W149 != total Vega + 2.5 log10(1+q_flux) (source-1 magnitude)")
    _require((df["blend_ratio_W149"] == 0).all() and (df["blend_ratio_F146"] == 0).all(),
             "non-zero blending")
    _require(np.allclose(df["fsource_W149"],
                         10.0 ** ((df["flux_zero_point_W149"] - df["W149"]) / 2.5),
                         rtol=1e-12, atol=0),
             "fsource_W149 != 10**((ZP - W149)/2.5)")
    _require(np.allclose(df["ftotal_W149"],
                         df["fsource_W149"] * (1.0 + df["blend_ratio_W149"]),
                         rtol=1e-12, atol=0),
             "ftotal_W149 != fsource_W149 (1 + blend_ratio_W149)")
    _require((df["ephemeris_name"] == EPHEMERIS_NAME).all()
             and (df["ephemeris_sha256"] == EPHEMERIS_SHA256).all(),
             "ephemeris provenance mismatch")
    _require((df["u0"] > 0).all() and (df["tE"] > 0).all(), "u0 and tE must be > 0")
    _require(np.allclose(df["piE"], np.hypot(df["piEN"], df["piEE"])), "piE inconsistent")
    _require((df["is_zero_parallax"] == (df["piE"] == 0)).all(),
             "is_zero_parallax inconsistent")

    # realizations of one configuration differ only by the seed (and, in H1,
    # by geometry for marginalized families)
    phys = ["t0", "u0", "tE", "W149", "piEN", "piEE"]
    per_cfg = df.groupby("config_id")[phys].nunique()
    _require((per_cfg <= 1).all().all(),
             "rows sharing config_id differ in (t0,u0,tE,W149,piEN,piEE)")

    if catalog_kind == "h0":
        _require((df["system_type"] == "PSPL").all(), "H0 system_type must be PSPL")
        _require(~df["truth_double_source"].astype(bool).any(),
                 "H0 rows with truth_double_source=True")
        present = [c for c in BSPL_COLUMNS if c in df.columns]
        _require(not present, f"H0 must not carry binary-source columns: {present}")
        zero = df[df["is_zero_parallax"]].groupby(
            ["t0", "u0", "tE", "W149"])["config_id"].nunique()
        _require((zero <= 1).all(),
                 "pi_E = 0 must be ONE configuration per (t0,u0,tE,mag)")
    else:
        _require((df["system_type"] == "BSPL").all(), "H1 system_type must be BSPL")
        _require(df["truth_double_source"].astype(bool).all(),
                 "H1 rows must have truth_double_source=True")
        _require(df["is_zero_parallax"].all(), "H1 truth must have pi_E = 0")
        missing = [c for c in BSPL_COLUMNS if c not in df.columns]
        _require(not missing, f"H1 missing binary-source columns: {missing}")
        _require((df["q_flux"] == df["q_flux_W149"]).all(), "q_flux != q_flux_W149")
        _require(((df["xi_mass_ratio"] > 0) & (df["xi_mass_ratio"] <= 1)).all(),
                 "xi_mass_ratio outside (0, 1]")

    return {
        "rows": int(len(df)),
        "configurations": int(df["config_id"].nunique()),
        "seed_range": [int(seeds.min()), int(seeds.max())],
        "families": df.groupby("scan_family").size().to_dict(),
    }


# =============================================================================
# Runner-side loading
# =============================================================================

def read_parquet_row(path, row_number):
    """Read one physical row without loading the whole file."""
    pf = pq.ParquetFile(path)
    row_number = int(row_number)
    if not 0 <= row_number < pf.metadata.num_rows:
        raise IndexError((str(path), row_number))
    remaining = row_number
    for group in range(pf.num_row_groups):
        n = pf.metadata.row_group(group).num_rows
        if remaining < n:
            return pf.read_row_group(group).slice(remaining, 1).to_pylist()[0]
        remaining -= n
    raise IndexError((str(path), row_number))


def load_event_from_catalog(catalog_path, catalog_event_index, expected_kind=None):
    """
    Drop-in replacement for roman_rubin's catalog.event_reader.load_event_from_catalog
    for the LRT catalogues.  Returns the row as a dict, unchanged (the catalogue
    simulation_seed is preserved), after validating it.
    """
    catalog_path = Path(catalog_path).expanduser().resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing catalogue: {catalog_path}")
    event = read_parquet_row(catalog_path, catalog_event_index)

    missing = [c for c in COMMON_COLUMNS if c not in event]
    if missing:
        raise CatalogError(f"{catalog_path.name}: missing canonical columns {missing}")
    if event["catalog_schema_version"] != SCHEMA_VERSION:
        raise CatalogError(f"schema {event['catalog_schema_version']!r} != {SCHEMA_VERSION!r}")
    if int(event["catalog_event_index"]) != int(catalog_event_index):
        raise CatalogError("physical-row / catalog_event_index invariant failed")
    if expected_kind is not None and event["catalog_kind"] != expected_kind:
        raise CatalogError(f"expected {expected_kind} row, got {event['catalog_kind']}")
    for key in ("ra", "dec", "t0", "u0", "tE", "W149", "piEN", "piEE"):
        if not np.isfinite(float(event[key])):
            raise CatalogError(f"non-finite {key}")
    if float(event["blend_ratio_W149"]) != 0.0:
        raise CatalogError("blend_ratio_W149 must be 0")
    if event["catalog_kind"] == "h0":
        present = [c for c in BSPL_COLUMNS if c in event]
        if present:
            raise CatalogError(f"H0 row carries binary-source columns {present}")
    else:
        missing = [c for c in BSPL_COLUMNS if c not in event]
        if missing:
            raise CatalogError(f"H1 row lacks binary-source columns {missing}")
    return event


def check_truth_kind(event: dict, truth_kind: str) -> None:
    """
    h0a : PSPL truth without parallax  -> H0 row with pi_E = 0
    h0b : PSPL+parallax truth          -> any H0 row
    h1  : BSPL truth without parallax  -> H1 row
    """
    kind = event["catalog_kind"]
    if truth_kind == "h1":
        ok = kind == "h1"
    elif truth_kind == "h0a":
        ok = kind == "h0" and bool(event["is_zero_parallax"])
    elif truth_kind == "h0b":
        ok = kind == "h0"
    else:
        raise ValueError(f"unknown truth_kind {truth_kind!r}")
    if not ok:
        raise CatalogError(
            f"--truth-kind {truth_kind} is incompatible with this row "
            f"(catalog_kind={kind}, is_zero_parallax={event['is_zero_parallax']})")


def row_identity(event: dict) -> dict:
    """Small identity block to store in the runner output."""
    keys = ["catalog_schema_version", "catalog_kind", "catalog_event_index",
            "config_id", "truth_config_id", "realization_id", "scan_family",
            "simulation_seed", "t0", "u0", "tE", "piEN", "piEE",
            "source_mag_F146_AB", "W149"]
    out = {}
    for k in keys + [c for c in ("geometry_id", "P_days", "q_mass", "q_flux")
                     if c in event]:
        v = event[k]
        out[k] = v.item() if hasattr(v, "item") else v
    return out


# =============================================================================
# CLI: validate catalogues and cross-check seed blocks
# =============================================================================

def _main(argv):
    if len(argv) < 2 or argv[0] != "validate":
        print("usage: python -m binary_source.analysis.lrt_catalog_io validate CAT.parquet ...")
        return 2
    status = 0
    seeds = {}
    for path in map(Path, argv[1:]):
        df = pd.read_parquet(path)
        kind = str(df["catalog_kind"].iloc[0]) if "catalog_kind" in df else "?"
        try:
            summary = validate_catalog(df, kind)
            print(f"PASS  {path.name}  kind={kind}  {summary}")
            seeds[path] = df["simulation_seed"].to_numpy(np.int64)
        except Exception as exc:                       # noqa: BLE001
            print(f"FAIL  {path.name}  kind={kind}  {exc}")
            status = 1
    paths = list(seeds)
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a = pd.read_parquet(paths[i], columns=["catalog_kind"])["catalog_kind"].iloc[0]
            b = pd.read_parquet(paths[j], columns=["catalog_kind"])["catalog_kind"].iloc[0]
            if a == b:
                continue                     # smoke vs candidate of the same kind
            inter = np.intersect1d(seeds[paths[i]], seeds[paths[j]])
            tag = "PASS" if len(inter) == 0 else "FAIL"
            status |= int(len(inter) > 0)
            print(f"{tag}  seeds disjoint: {paths[i].name} vs {paths[j].name}"
                  f" ({len(inter)} shared)")
    print("CATALOG VALIDATION:", "PASS" if status == 0 else "FAIL")
    return status


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
