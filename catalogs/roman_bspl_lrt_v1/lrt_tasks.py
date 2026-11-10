#!/usr/bin/env python3
"""
Build the task file of a Roman LRT run from a JSON run configuration.

One line per runner call:
    <catalog> <catalog_event_index> <truth_kind> <output_json>

Each run is a config in configs/lrt_tasks/ (mini pipeline test, Test A pilot,
Test A calibration, ...). A config is a list of selections; each selection
picks configurations from the H0 or H1 catalog and the first n realizations of
each:

    {"catalog": "h0" | "h1",
     "kinds": ["h0a", ...],               truth kinds run on every row
     "where": {column: value | [values]}, exact match for bool/int/str,
                                          np.isclose for floats
     "P_days_closest": [30.0, ...],       optional (H1): per target, the
                                          catalog period nearest in log P
     "n_per_config": 20,
     "expect_n_configs": 3}               optional sanity check

Catalog paths are given on the command line because they live on the cluster.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def match(df, where):
    mask = np.ones(len(df), dtype=bool)
    for col, val in where.items():
        vals = val if isinstance(val, list) else [val]
        col_mask = np.zeros(len(df), dtype=bool)
        for v in vals:
            if isinstance(v, (bool, np.bool_)) or isinstance(v, str):
                col_mask |= (df[col] == v).to_numpy()
            elif isinstance(v, (int, np.integer)) and df[col].dtype.kind in "iu":
                col_mask |= (df[col] == v).to_numpy()
            else:
                col_mask |= np.isclose(df[col].to_numpy(dtype=float), float(v))
        mask &= col_mask
    return df[mask]


def select(df, sel):
    sub = match(df, sel.get("where", {}))
    if "P_days_closest" in sel:
        periods = np.unique(sub["P_days"].to_numpy(dtype=float))
        keep = [periods[np.argmin(np.abs(np.log(periods / float(p))))]
                for p in sel["P_days_closest"]]
        sub = sub[np.isin(sub["P_days"].to_numpy(dtype=float), keep)]

    n = int(sel["n_per_config"])
    picked = []
    for cfg, rows in sub.groupby("config_id", sort=True):
        rows = rows.sort_values("realization_id")
        if len(rows) < n:
            raise SystemExit(f"{cfg}: only {len(rows)} realizations (< {n})")
        picked.append(rows.head(n))
    if not picked:
        raise SystemExit(f"selection matched no rows: {sel}")
    out = pd.concat(picked)

    n_cfg = out["config_id"].nunique()
    if "expect_n_configs" in sel and n_cfg != int(sel["expect_n_configs"]):
        raise SystemExit(f"expected {sel['expect_n_configs']} configurations, "
                         f"got {n_cfg}: {sorted(out['config_id'].unique())}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--h0", type=Path, help="H0 catalog parquet")
    ap.add_argument("--h1", type=Path, help="H1 catalog parquet")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    outdir = Path(cfg["outdir"])
    (outdir / "json").mkdir(parents=True, exist_ok=True)

    catalogs = {"h0": args.h0, "h1": args.h1}
    frames = {}
    lines, summary = [], []
    for sel in cfg["selections"]:
        name = sel["catalog"]
        if catalogs[name] is None:
            raise SystemExit(f"selection needs --{name}")
        if name not in frames:
            frames[name] = pd.read_parquet(catalogs[name])
        rows = select(frames[name], sel)
        for kind in sel["kinds"]:
            for idx in rows["catalog_event_index"].astype(int):
                lines.append(f"{catalogs[name]} {idx} {kind} "
                             f"{outdir}/json/{kind}_{name}_{idx}.json")
        summary.append({"catalog": name, "kinds": sel["kinds"],
                        "configs": sorted(rows["config_id"].unique().tolist()),
                        "n_rows": int(len(rows))})

    if len(set(lines)) != len(lines):
        raise SystemExit("duplicated tasks across selections")

    tasks = outdir / "tasks.txt"
    tasks.write_text("\n".join(lines) + "\n")
    (outdir / "run_config.json").write_text(json.dumps(
        dict(cfg, config_path=str(args.config),
             h0=str(args.h0), h1=str(args.h1), selected=summary), indent=2) + "\n")

    for s in summary:
        print(f"{s['catalog']} {s['kinds']}: {len(s['configs'])} configs, "
              f"{s['n_rows']} rows")
        for c in s["configs"]:
            print("   ", c)
    print(f"tasks = {len(lines)} -> {tasks}")


if __name__ == "__main__":
    main()
