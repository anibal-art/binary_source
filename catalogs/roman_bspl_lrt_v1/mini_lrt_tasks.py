#!/usr/bin/env python3
"""
Select the rows for a MINI end-to-end LRT (pipeline test, low statistics).

H0-A : one PSPL configuration, pi_E = 0          (u0=0.1, tE=30, F146=21)  -> h0a AND h0b
       (runner v2 fits PSPL+BSPL for h0a and PSPL_PI+BSPL for h0b; at pi_E = 0
        both kinds simulate the same data, so the pair gives T_A and T_B of
        the pi_E = 0 null configuration of Test B)
H0-B : one PSPL+parallax configuration           (same, |pi_E|=0.17, dir 0) -> h0b
H1   : three BSPL cells of the main (marginalized-geometry) family,
       u0=0.1, F146=21, P closest to 30 d, 365.25 d and 1e4 d               -> h1

Writes a task file with one line per runner call:
    <catalog> <catalog_event_index> <truth_kind> <output_json>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h0", required=True)
    ap.add_argument("--h1", required=True)
    ap.add_argument("--n0", type=int, default=40, help="realizations per H0 config")
    ap.add_argument("--n1", type=int, default=10, help="realizations per H1 cell")
    ap.add_argument("--outdir", default="/tmp/mini_lrt")
    ap.add_argument("--tasks", default="/tmp/mini_lrt/tasks.txt")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "json").mkdir(parents=True, exist_ok=True)

    h0 = pd.read_parquet(args.h0, columns=[
        "catalog_event_index", "u0", "tE", "source_mag_F146_AB", "is_zero_parallax",
        "piE_radius_nominal", "piE_direction_index", "config_id", "realization_id"])
    h1 = pd.read_parquet(args.h1, columns=[
        "catalog_event_index", "scan_family", "u0", "source_mag_F146_AB",
        "P_days", "config_id", "realization_id"])

    base = (np.isclose(h0["u0"], 0.1) & np.isclose(h0["tE"], 30.0)
            & np.isclose(h0["source_mag_F146_AB"], 21.0))
    sel_a = h0[base & h0["is_zero_parallax"]]
    sel_b = h0[base & np.isclose(h0["piE_radius_nominal"], 0.17)
               & (h0["piE_direction_index"] == 0)]
    for name, sel in (("H0-A", sel_a), ("H0-B", sel_b)):
        if sel["config_id"].nunique() != 1:
            raise SystemExit(f"{name}: expected one configuration, got "
                             f"{sel['config_id'].unique()}")
        if len(sel) < args.n0:
            raise SystemExit(f"{name}: only {len(sel)} realizations available")
    sel_a = sel_a.sort_values("realization_id").head(args.n0)
    sel_b = sel_b.sort_values("realization_id").head(args.n0)

    main_fam = h1[(h1["scan_family"] == "one_luminous_u0_P")
                  & np.isclose(h1["u0"], 0.1) & np.isclose(h1["source_mag_F146_AB"], 21.0)]
    periods = np.unique(main_fam["P_days"])
    cells = []
    for target in (30.0, 365.25, 1.0e4):
        p = periods[np.argmin(np.abs(np.log(periods / target)))]
        cell = main_fam[main_fam["P_days"] == p].sort_values("realization_id").head(args.n1)
        cells.append(cell)
        print(f"H1 cell P={p:.4g} d  config={cell['config_id'].iloc[0]}  n={len(cell)}")

    lines = []
    for sel, kind in ((sel_a, "h0a"), (sel_a, "h0b"), (sel_b, "h0b")):
        for idx in sel["catalog_event_index"]:
            lines.append(f"{args.h0} {int(idx)} {kind} {outdir}/json/{kind}_{int(idx)}.json")
    for cell in cells:
        for idx in cell["catalog_event_index"]:
            lines.append(f"{args.h1} {int(idx)} h1 {outdir}/json/h1_{int(idx)}.json")

    Path(args.tasks).write_text("\n".join(lines) + "\n")
    print(f"H0-A config {sel_a['config_id'].iloc[0]}  n={len(sel_a)}")
    print(f"H0-B config {sel_b['config_id'].iloc[0]}  n={len(sel_b)}")
    print(f"tasks = {len(lines)} -> {args.tasks}")


if __name__ == "__main__":
    main()
