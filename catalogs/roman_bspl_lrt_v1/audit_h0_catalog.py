#!/usr/bin/env python3
"""
Audit of the H0 (PSPL / PSPL+parallax truth) catalogue before it is used by
the LRT runner.  Read-only: it never modifies the catalogue.

Usage:
    python audit_h0_catalog.py H0.parquet [--h1 H1.parquet ...]

Every check prints PASS / FAIL / WARN.  Exit status is 1 if any FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T0_EXPECTED = 2461849.0
RA_FIELD = 267.92497054815516
DEC_FIELD = -29.152232510353276
AB_MINUS_VEGA = 1.0364436822930514

# Columns the catalogue runner / roman_rubin need for a PSPL(+parallax) truth.
REQUIRED = [
    "catalog_event_index", "simulation_seed",
    "t0", "u0", "tE", "piEN", "piEE",
    "ra", "dec", "W149",
]
RECOMMENDED = [
    "truth_kind", "power_cell_id", "truth_config_id", "realization_id",
    "source_mag_F146_AB", "source_mag_F146_Vega",
    "blend_ratio_W149", "truth_parallax", "truth_double_source",
    "catalog_schema_version",
]
BSPL_ONLY = [
    "xi_para", "xi_perp", "xi_angular_velocity", "xi_phase",
    "xi_inclination", "xi_mass_ratio", "q_flux", "q_flux_W149",
]

FAILS: list[str] = []
WARNS: list[str] = []


def ok(msg):
    print(f"  PASS  {msg}")


def fail(msg):
    print(f"  FAIL  {msg}")
    FAILS.append(msg)


def warn(msg):
    print(f"  WARN  {msg}")
    WARNS.append(msg)


def check(cond, msg_ok, msg_fail, level="fail"):
    if bool(cond):
        ok(msg_ok)
    else:
        (fail if level == "fail" else warn)(msg_fail)


def section(title):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def describe(df, col, fmt="{:.6g}"):
    if col not in df:
        return
    v = df[col]
    if np.issubdtype(v.dtype, np.number):
        u = np.unique(v.to_numpy())
        print(f"  {col:24s} n_unique={len(u):6d}  "
              f"range=[{fmt.format(u.min())}, {fmt.format(u.max())}]")
        if len(u) <= 16:
            print(f"  {'':24s} values=" + ", ".join(fmt.format(x) for x in u))
    else:
        u = v.astype(str).unique()
        print(f"  {col:24s} n_unique={len(u)}  e.g. {list(u[:5])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h0", type=Path)
    ap.add_argument("--h1", type=Path, nargs="*", default=[],
                    help="H1 catalogue(s) to check seed disjointness and "
                         "column compatibility")
    args = ap.parse_args()

    df = pd.read_parquet(args.h0)
    section(f"H0 CATALOGUE  {args.h0}")
    print(f"  rows = {len(df)}   columns = {len(df.columns)}")
    print("  columns:", ", ".join(df.columns))

    # ------------------------------------------------------------------
    section("1. Schema")
    missing = [c for c in REQUIRED if c not in df]
    check(not missing, "all required columns present",
          f"missing required columns: {missing}")
    missing_rec = [c for c in RECOMMENDED if c not in df]
    check(not missing_rec, "all recommended columns present",
          f"missing recommended columns: {missing_rec}", level="warn")
    if missing:
        finish()

    # ------------------------------------------------------------------
    section("2. Identity and seeds")
    idx = df["catalog_event_index"].to_numpy()
    check(np.array_equal(idx, np.arange(len(df))),
          "catalog_event_index == row number 0..N-1",
          "catalog_event_index is not 0..N-1 in row order")
    seeds = df["simulation_seed"].to_numpy(dtype=np.int64)
    check(df["simulation_seed"].is_unique, "simulation_seed unique",
          f"simulation_seed has {len(seeds) - len(np.unique(seeds))} duplicates")
    check((seeds >= 0).all() and (seeds < 2**32).all(),
          "simulation_seed in [0, 2^32)", "simulation_seed outside [0, 2^32)")
    print(f"  seed range = [{seeds.min()}, {seeds.max()}]")
    if "truth_config_id" in df:
        check(df["truth_config_id"].is_unique, "truth_config_id unique",
              "truth_config_id not unique")

    for h1_path in args.h1:
        h1 = pd.read_parquet(h1_path, columns=None)
        inter = np.intersect1d(seeds, h1["simulation_seed"].to_numpy(np.int64))
        check(len(inter) == 0, f"seeds disjoint from {h1_path.name}",
              f"{len(inter)} seeds shared with {h1_path.name}")
        miss_cols = [c for c in REQUIRED if c not in h1]
        check(not miss_cols, f"{h1_path.name} has the same required columns",
              f"{h1_path.name} lacks {miss_cols}")
        for c in ("t0", "ra", "dec"):
            if c in h1:
                same = np.allclose(np.unique(h1[c]), np.unique(df[c])) \
                    if len(np.unique(h1[c])) == len(np.unique(df[c])) else False
                check(same, f"{c} identical in H0 and {h1_path.name}",
                      f"{c} differs between H0 and {h1_path.name}: "
                      f"H0={np.unique(df[c])[:3]} H1={np.unique(h1[c])[:3]}",
                      level="warn")

    # ------------------------------------------------------------------
    section("3. Truth model: PSPL / PSPL+parallax only")
    if "truth_double_source" in df:
        check(~df["truth_double_source"].astype(bool).any(),
              "truth_double_source is False everywhere",
              "some H0 rows have truth_double_source=True")
    present_bspl = [c for c in BSPL_ONLY if c in df]
    if present_bspl:
        active = []
        for c in present_bspl:
            v = pd.to_numeric(df[c], errors="coerce")
            if (v.notna() & (v != 0)).any():
                active.append(c)
        check(not active, "no active binary-source parameters",
              f"H0 rows carry non-zero binary-source parameters: {active}")
    else:
        ok("no binary-source columns present")
    if "truth_kind" in df:
        print("  truth_kind counts:", df["truth_kind"].value_counts().to_dict())

    # ------------------------------------------------------------------
    section("4. Fixed conditions")
    check(np.allclose(df["t0"], T0_EXPECTED), f"t0 == {T0_EXPECTED} (season 3)",
          f"t0 values {np.unique(df['t0'])[:5]} (expected {T0_EXPECTED})",
          level="warn")
    check(np.allclose(df["ra"], RA_FIELD) and np.allclose(df["dec"], DEC_FIELD),
          "ra/dec == fixed Roman field used by roman_rubin",
          f"ra/dec differ from field: ra={np.unique(df['ra'])[:3]} "
          f"dec={np.unique(df['dec'])[:3]}", level="warn")
    for c in ("blend_ratio_W149", "blend_ratio_F146"):
        if c in df:
            check(np.allclose(df[c], 0.0), f"{c} == 0", f"{c} non-zero")
    if "blend_ratio_W149" not in df:
        fail("blend_ratio_W149 missing: roman_rubin would draw blending at random")
    if {"source_mag_F146_AB", "W149"} <= set(df.columns):
        d = df["source_mag_F146_AB"] - df["W149"]
        check(np.allclose(d, AB_MINUS_VEGA),
              "W149 (Vega) = F146_AB - 1.0364", f"AB-Vega offsets: {np.unique(d)[:5]}")
    if {"source_mag_F146_AB", "source_mag_F146_Vega"} <= set(df.columns):
        check(np.allclose(df["source_mag_F146_Vega"], df["W149"]),
              "source_mag_F146_Vega == W149", "source_mag_F146_Vega != W149")

    # ------------------------------------------------------------------
    section("5. Physical grid")
    pie = np.hypot(df["piEN"], df["piEE"])
    df = df.assign(_piE=pie,
                   _theta_deg=np.degrees(np.arctan2(df["piEE"], df["piEN"])) % 360)
    for c in ("u0", "tE", "source_mag_F146_AB", "W149", "_piE", "_theta_deg"):
        describe(df, c)
    check((df["u0"] > 0).all(), "u0 > 0", "u0 <= 0 present", level="warn")
    check((df["tE"] > 0).all(), "tE > 0", "tE <= 0 present")

    # Configurations = rows that differ only by the noise seed.
    cfg_cols = [c for c in ("t0", "u0", "tE", "W149", "piEN", "piEE") if c in df]
    cfg = df.groupby(cfg_cols, dropna=False).size().rename("n_real").reset_index()
    print(f"\n  physical configurations = {len(cfg)}")
    print("  realizations per configuration:",
          cfg["n_real"].value_counts().sort_index().to_dict())

    # pi_E = 0 must exist once per (t0,u0,tE,mag), not once per direction.
    zero = cfg[np.hypot(cfg["piEN"], cfg["piEE"]) == 0]
    base_cols = [c for c in ("t0", "u0", "tE", "W149") if c in cfg]
    if len(zero):
        dup = zero.groupby(base_cols).size()
        check((dup == 1).all(), "pi_E = 0 materialized once per (t0,u0,tE,mag)",
              f"pi_E = 0 duplicated in {int((dup > 1).sum())} base configurations")
        nonzero_base = cfg[np.hypot(cfg["piEN"], cfg["piEE"]) > 0][base_cols] \
            .drop_duplicates()
        print(f"  base (t0,u0,tE,mag) with pi_E=0: {len(zero)} ; "
              f"with pi_E>0: {len(nonzero_base)}")
        if "n_real" in zero:
            print("  realizations at pi_E=0:",
                  zero["n_real"].value_counts().to_dict())
    else:
        warn("no pi_E = 0 configurations: Test A has nothing to calibrate on")

    # Test-A subset vs Test-B subset
    if "truth_kind" in df:
        a = df[df["_piE"] == 0]["truth_kind"].value_counts().to_dict()
        b = df[df["_piE"] > 0]["truth_kind"].value_counts().to_dict()
        print(f"  truth_kind at pi_E=0: {a}")
        print(f"  truth_kind at pi_E>0: {b}")

    # Directions per |pi_E|
    nz = cfg[np.hypot(cfg["piEN"], cfg["piEE"]) > 0].copy()
    if len(nz):
        nz["piE"] = np.round(np.hypot(nz["piEN"], nz["piEE"]), 10)
        nz["theta"] = np.round(np.degrees(np.arctan2(nz["piEE"], nz["piEN"])) % 360, 6)
        per = nz.groupby("piE")["theta"].nunique()
        print("  directions per |pi_E|:", per.to_dict())
        print("  direction set (deg):", [float(x) for x in sorted(nz["theta"].unique())][:16])
        check((np.hypot(df["piEN"], df["piEE"]) <= 1.0).all(),
              "|pi_E| <= 1 (inside current fit bounds [-1,1])",
              "|pi_E| > 1 present: fit bounds must be widened", level="warn")
        comp = np.maximum(np.abs(df["piEN"]), np.abs(df["piEE"]))
        check((comp <= 0.8).all(), "max |pi_E,N|,|pi_E,E| <= 0.8",
              f"component up to {comp.max():.3g}: close to the [-1,1] fit bound",
              level="warn")

    finish()


def finish():
    section("SUMMARY")
    print(f"  FAIL = {len(FAILS)}   WARN = {len(WARNS)}")
    for m in FAILS:
        print("   FAIL:", m)
    for m in WARNS:
        print("   WARN:", m)
    print()
    print("H0 CATALOG AUDIT:", "FAIL" if FAILS else "PASS")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
