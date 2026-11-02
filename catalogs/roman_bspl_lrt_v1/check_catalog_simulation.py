#!/usr/bin/env python3
"""
Simulation-only checks of the H0/H1 catalogues through the real backend
(roman_rubin), WITHOUT fitting.  Run from the binary_source repo root with
ROMAN_RUBIN_REPO and PYTHONPATH set as for the runner.

  1. pi_E = 0 H0 row simulated with --truth-kind h0a and h0b gives the SAME
     data (same seed)  -> one simulation can serve Test A and Test B.
  2. The catalogue seed is used: two rows give different data, and the same
     row twice gives identical data.
  3. Photometric convention with two luminous sources: the noise-free
     baseline of a q_flux = 1 H1 row equals the declared TOTAL magnitude
     (source_mag_F146_Vega).  The catalogue stores W149 = source-1 magnitude
     because roman_rubin builds the baseline as F(W149) x (1 + q_flux).
  4. Same baseline check for a q_flux = 0 row (control) and an H0 row.
  5. Simulated flux parameters are printed so blending can be inspected.

Usage:
  python -u catalogs/roman_bspl_lrt_v1/check_catalog_simulation.py \
      --h0 catalogs/roman_bspl_lrt_v1/h0_smoke_v3.parquet \
      --h1 catalogs/roman_bspl_lrt_v1/h1_bspl_smoke_v3.parquet
"""

from __future__ import annotations

import argparse
import copy
import sys

import numpy as np
import pandas as pd

from binary_source.analysis import roman_bspl_lrt_catalog_runner as runner
from binary_source.analysis import lrt_catalog_io as io

FAILS = []


def report(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def simulate(catalog, index, truth_kind, noise=True):
    event = io.load_event_from_catalog(catalog, index)
    io.check_truth_kind(event, truth_kind)
    tp, tds = runner.truth_switches(truth_kind)
    seed = int(event["simulation_seed"])
    np.random.seed(seed)
    model, params, decision = runner.frr.simulate_event_for_fit(
        i=seed,
        event_params=copy.deepcopy(event),
        path_ephemerides=str(runner.EPHEMERIDES),
        model="PSPL",
        time_window=None,
        use_roman=True,
        use_rubin=False,
        truth_parallax=tp,
        truth_double_source=tds,
        add_photometric_noise=noise,
        apply_detection_criteria=False,
        apply_photometric_filter=True,
    )
    if model is None:
        raise RuntimeError(f"no model for row {index} ({truth_kind})")
    data = runner.extract_fit_data(model)["W149"]
    return event, params, data


def flux_items(params):
    try:
        items = dict(params).items()
    except Exception:                                  # noqa: BLE001
        try:
            items = vars(params).items()
        except Exception:                              # noqa: BLE001
            return {}
    return {k: float(v) for k, v in items
            if any(s in str(k) for s in ("fsource", "ftotal", "fblend", "gblend", "q_flux"))
            and np.ndim(v) == 0}


def baseline_mag(event, data):
    t, mag = data[:, 0], data[:, 1]
    far = np.abs(t - float(event["t0"])) > 10.0 * float(event["tE"])
    if "P_days" in event:                 # stay away from long-period wiggles
        far &= np.abs(t - float(event["t0"])) > 10.0 * float(event["tE"])
    return float(np.median(mag[far])), int(far.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h0", required=True)
    ap.add_argument("--h1", required=True)
    args = ap.parse_args()

    h0 = pd.read_parquet(args.h0)
    h1 = pd.read_parquet(args.h1)

    print("=" * 80)
    print("1. pi_E = 0: h0a and h0b give identical data")
    print("=" * 80)
    i0 = int(h0.index[h0["is_zero_parallax"]][0])
    _, _, da = simulate(args.h0, i0, "h0a")
    _, _, db = simulate(args.h0, i0, "h0b")
    report(np.array_equal(da, db),
           f"row {i0}: h0a hash {runner.sha256_array(da)[:12]} "
           f"vs h0b hash {runner.sha256_array(db)[:12]}")

    print()
    print("=" * 80)
    print("2. seeds: same row reproduces, different rows differ")
    print("=" * 80)
    _, _, da2 = simulate(args.h0, i0, "h0a")
    report(np.array_equal(da, da2), f"row {i0} twice: identical")
    same_cfg = h0.index[(h0["config_id"] == h0.loc[i0, "config_id"]) & (h0.index != i0)]
    if len(same_cfg):
        j = int(same_cfg[0])
        _, _, dj = simulate(args.h0, j, "h0a")
        report(not np.array_equal(da, dj),
               f"rows {i0} and {j} (same configuration, different seed): data differ")
    nz = int(h0.index[~h0["is_zero_parallax"]][0])
    _, _, dnz = simulate(args.h0, nz, "h0b")
    print(f"  info  row {nz} (|pi_E|={h0.loc[nz, 'piE']:.3g}) simulated with h0b, "
          f"N={len(dnz)}")

    print()
    print("=" * 80)
    print("3-4. noise-free baseline magnitude vs declared total magnitude")
    print("=" * 80)
    cases = [("H0 row", args.h0, i0, "h0a")]
    k0 = h1.index[(h1["q_flux"] == 0)]
    if len(k0):
        cases.append(("H1 q_flux=0", args.h1, int(k0[0]), "h1"))
    k1 = h1.index[np.isclose(h1["q_flux"], 1.0)]
    if len(k1):
        cases.append(("H1 q_flux=1", args.h1, int(k1[0]), "h1"))
    for label, cat, idx, kind in cases:
        ev, params, data = simulate(cat, idx, kind, noise=False)
        base, n = baseline_mag(ev, data)
        expected = float(ev["source_mag_F146_Vega"])     # declared TOTAL (Vega)
        diff = base - expected
        verdict = abs(diff) < 0.005
        extra = ""
        if not verdict and "q_flux" in ev and float(ev["q_flux"]) > 0:
            q = float(ev["q_flux"])
            extra = (f"  (expected -2.5log10(1+q)={-2.5 * np.log10(1 + q):+.3f} "
                     f"if W149 were source-1 magnitude)")
        report(verdict, f"{label} (row {idx}): baseline={base:.4f} declared total="
                        f"{expected:.4f}  diff={diff:+.4f}  [{n} far points]{extra}")
        print(f"        flux parameters: {flux_items(params)}")

    print()
    print("=" * 80)
    print("SUMMARY:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    for m in FAILS:
        print("  FAIL:", m)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
