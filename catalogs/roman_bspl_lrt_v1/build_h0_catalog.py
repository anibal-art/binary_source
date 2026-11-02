#!/usr/bin/env python3
"""
Build the H0 (PSPL / PSPL+parallax truth) catalogue for the Roman LRT.

One catalogue serves both null hypotheses:
  * Test A (PSPL):            rows with is_zero_parallax == True, run with --truth-kind h0a
  * Test B (PSPL+parallax):   all rows,                            run with --truth-kind h0b

Each physical configuration (t0, u0, tE, magnitude, pi_E vector) has
n realizations that differ only by the noise seed.  pi_E = 0 is materialized
ONCE per (t0, u0, tE, magnitude), not once per direction.

Default grid = stage-1 (localization) design discussed for the protocol:
  u0   : 7 log nodes in [0.005, 2]
  tE   : stratum 30 d -> {20, 30, 45};  stratum 150 d -> {100, 150, 225}
  mag  : F146 AB {19, 21, 23} for the 30 d stratum, {21} for the 150 d stratum
  pi_E : 0  +  |pi_E| in {0.03, 0.07, 0.17, 0.41, 1.0} x 8 directions
Directions are measured from North towards East (theta = atan2(piEE, piEN)),
offset by --direction-offset-deg.  Aligning them with the observer
acceleration axis at t0 is still pending (pass the offset when computed).

NOTE: |pi_E| = 1 along an axis sits ON the current PSPL+parallax fit bound
[-1, 1]; widen the bounds in the fit policy before production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from binary_source.analysis import lrt_catalog_io as io

U0_NODES = np.geomspace(0.005, 2.0, 7)
TE_STRATA = {"30": [20.0, 30.0, 45.0], "150": [100.0, 150.0, 225.0]}
MAGS_STRATA = {"30": [19.0, 21.0, 23.0], "150": [21.0]}
PIE_RADII = [0.03, 0.07, 0.17, 0.41, 1.0]
N_DIRECTIONS = 8
DEFAULT_N_REALIZATIONS = 100


def build(n_real, strata, direction_offset_deg, smoke):
    if smoke:
        u0_nodes = np.array([0.01, 0.1, 1.0])
        te_strata = {"30": [30.0]}
        mags = {"30": [21.0]}
        radii = [0.1, 0.4]
        n_dir = 4
        n_real = min(n_real, 2)
        strata = ["30"]
    else:
        u0_nodes, te_strata, mags = U0_NODES, TE_STRATA, MAGS_STRATA
        radii, n_dir = PIE_RADII, N_DIRECTIONS

    offset = np.radians(direction_offset_deg)
    rows = []
    for stratum in strata:
        for tE in te_strata[stratum]:
            for mag in mags[stratum]:
                for iu, u0 in enumerate(u0_nodes):
                    # parallax vectors: zero once, then radius x direction
                    vectors = [(0.0, 0.0, -1, -1)]
                    for ir, r in enumerate(radii):
                        for k in range(n_dir):
                            th = offset + 2 * np.pi * k / n_dir
                            vectors.append((r * np.cos(th), r * np.sin(th), ir, k))
                    for piEN, piEE, ir, k in vectors:
                        cfg = (f"h0:tE{stratum}:te{tE:g}:m{int(mag):02d}:u{iu:02d}:"
                               + ("pi0" if ir < 0 else f"r{ir:02d}d{k:02d}"))
                        for r in range(n_real):
                            row = io.common_fields(
                                catalog_kind="h0", config_id=cfg, realization_id=r,
                                scan_family=f"h0_tE{stratum}", system_type="PSPL",
                                u0=u0, tE=tE, f146_ab=mag, piEN=piEN, piEE=piEE,
                                truth_double_source=False)
                            row.update({
                                "tE_stratum_days": float(stratum),
                                "u0_grid_index": iu,
                                "piE_radius_index": ir,
                                "piE_radius_nominal": (0.0 if ir < 0 else float(radii[ir])),
                                "piE_direction_index": k,
                                "piE_direction_offset_deg": float(direction_offset_deg),
                            })
                            rows.append(row)
    return io.finalize_catalog(rows, "h0")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("catalogs/roman_bspl_lrt_v1/h0_candidate_v3.parquet"))
    ap.add_argument("--n-realizations", type=int, default=DEFAULT_N_REALIZATIONS)
    ap.add_argument("--strata", nargs="+", choices=["30", "150"], default=["30"])
    ap.add_argument("--direction-offset-deg", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    df = build(args.n_realizations, args.strata, args.direction_offset_deg, args.smoke)
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")
    summary = io.validate_catalog(df, "h0")
    manifest = {
        "catalog_schema_version": io.SCHEMA_VERSION,
        "catalog_kind": "h0",
        "status": "SMOKE" if args.smoke else "CANDIDATE",
        "production_frozen": False,
        "output": str(out),
        "sha256": sha256_file(out),
        "summary": summary,
        "n_realizations": int(df["realization_id"].max() + 1),
        "strata": args.strata if not args.smoke else ["30"],
        "piE_direction_offset_deg": args.direction_offset_deg,
        "notes": [
            "Test A: rows with is_zero_parallax, --truth-kind h0a",
            "Test B: all rows, --truth-kind h0b",
            "directions not yet aligned with observer acceleration axis",
            "|pi_E|=1 on an axis reaches the [-1,1] fit bound: widen bounds",
        ],
    }
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")

    print("=" * 80)
    print("H0 CATALOGUE  status =", manifest["status"])
    print("=" * 80)
    print("output   =", out)
    print("sha256   =", manifest["sha256"])
    print("summary  =", summary)
    zero = df[df["is_zero_parallax"]]
    print(f"pi_E = 0 rows (Test A) = {len(zero)}  configs = {zero['config_id'].nunique()}")
    print(f"all rows (Test B)      = {len(df)}  configs = {df['config_id'].nunique()}")
    for c in ("u0", "tE", "source_mag_F146_AB", "piE"):
        print(f"{c:20s} {sorted(np.round(df[c].unique(), 6))}")
    print("CATALOG BUILD: PASS")


if __name__ == "__main__":
    main()
