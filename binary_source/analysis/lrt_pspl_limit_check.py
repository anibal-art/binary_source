#!/usr/bin/env python3
"""
Does the BSPL fit reproduce the PSPL minimum in the limit xi -> 0?

In the repaired dark pilot, 9 nodes whose best start was the PSPL start
(xi_para = xi_perp = 1e-4) ended above chi2_PSPL (q down to -1.6). For nested
models this should not happen; a systematic offset of O(1) would bias every q
and the calibrated threshold. For each event and each period this script
compares, with the runner's fitter (run_one_fit, same policy and optimizer):

  chi2_PSPL          stored best PSPL of the runner JSON
  chi2_PSPL_refit    PSPL refitted from its stored best
  lin_PSPL           pyLIMA chi2 of the PSPL at the stored best, fluxes by linear LS
  lin_BSPL_xi0       same for the BSPL at the PSPL start (pspl_start_guess)
  fit_BSPL_xi_fixed  BSPL fit from the PSPL start with xi_para, xi_perp and omega
                     fixed (bounds x (1 +/- 1e-9)): only t0, u0, tE, phase,
                     inclination and fluxes move
  fit_BSPL_free      BSPL fit from the PSPL start at fixed omega, xi free
                     (what --repair does); q = chi2_PSPL - fit_BSPL_free

Reading: lin_BSPL_xi0 - lin_PSPL tests the model at xi -> 0; lin_PSPL - chi2_PSPL
tests the flux treatment; fit_BSPL_xi_fixed - chi2_PSPL and chi2_PSPL_refit -
chi2_PSPL test the optimizer. All should be << 1.

Usage:
    python binary_source/analysis/lrt_pspl_limit_check.py \
        --json results/roman_lrt/gv_pilot_v1_dark/json/h0a_h0_118923.json ... \
        --fit-policy catalogs/roman_bspl_lrt_v1/fit_policy_test_a_dark_v1.json \
        --periods 35.9 118.6 1000 --out results/roman_lrt/gv_pilot_v1_dark/pspl_limit_check
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from binary_source.analysis import roman_bspl_lrt_catalog_runner as runner
from binary_source.analysis.lrt_omega_profile import XI_PSPL_START, code_commits, pspl_start_guess


def event_check(json_path, policy, periods):
    d = json.loads(Path(json_path).read_text())
    a = np.load(Path(json_path).with_suffix(".npz"))
    lc = np.column_stack([a["time"], a["mag"], a["err_mag"]])
    ra, dec, ref = d["event_ra_deg"], d["event_dec_deg"], d["fit_reference"]
    pspl = d["fits"]["PSPL"]["best"]
    chi2_pspl = float(pspl["chi2"])
    spec_b = dict(policy["BSPL"])
    bounds, rule = spec_b["bounds"], spec_b["start_rule"]
    pspl_model = runner.build_fit_model(lc, ra, dec, ref, False)
    bspl_model = runner.build_fit_model(lc, ra, dec, ref, True)
    names = runner.model_parameter_names(pspl_model)
    best = dict(zip(names, pspl["best_model"]))
    sid = int(d["simulation_seed"])

    def fit(key, spec, guess, label):
        t = time.perf_counter()
        r = runner.run_one_fit(source_id=sid, data={"W149": lc}, ra=ra, dec=dec, fit_key=key,
                               spec=spec, start={"label": label, "initial_guess": guess},
                               fit_reference=ref, optimizer_options=policy.get("optimizer_options"))
        return float(r["chi2"]), r.get("optimizer_nfev"), time.perf_counter() - t

    g_pspl = {n: float(best[n]) for n in ("t0", "u0", "tE")}
    refit, nfev_refit, _ = fit("PSPL", dict(policy["PSPL"]), g_pspl, "pspl_refit")
    out = {"event_json": str(json_path), "simulation_seed": sid, "chi2_PSPL": chi2_pspl,
           "chi2_PSPL_refit": refit, "nfev_PSPL_refit": nfev_refit,
           "lin_PSPL": runner.chi2_linear_fluxes(pspl_model, g_pspl), "periods": []}
    for P in periods:
        om = 2 * np.pi / P
        g = pspl_start_guess(pspl["best_model"], names, bounds, rule, om)
        fixed = dict(bounds, xi_angular_velocity=[om * (1 - 1e-9), om * (1 + 1e-9)],
                     xi_para=[XI_PSPL_START * (1 - 1e-9), XI_PSPL_START * (1 + 1e-9)],
                     xi_perp=[XI_PSPL_START * (1 - 1e-9), XI_PSPL_START * (1 + 1e-9)])
        free = dict(bounds, xi_angular_velocity=[om * (1 - 1e-9), om * (1 + 1e-9)])
        c_fix, n_fix, s_fix = fit("BSPL", dict(spec_b, bounds=fixed), g, "pspl_xi_fixed")
        c_free, n_free, s_free = fit("BSPL", dict(spec_b, bounds=free), g, "pspl_xi_free")
        out["periods"].append({
            "period_days": P, "lin_BSPL_xi0": runner.chi2_linear_fluxes(bspl_model, g),
            "fit_BSPL_xi_fixed": c_fix, "nfev_xi_fixed": n_fix, "seconds_xi_fixed": s_fix,
            "fit_BSPL_free": c_free, "nfev_free": n_free, "seconds_free": s_free,
            "q_free": chi2_pspl - c_free})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, nargs="+", required=True, help="runner JSONs (.npz next to each)")
    ap.add_argument("--fit-policy", type=Path, required=True)
    ap.add_argument("--periods", type=float, nargs="+", default=[35.9, 118.6, 1000.0])
    ap.add_argument("--out", type=Path, required=True, help="output prefix (.json)")
    args = ap.parse_args()
    policy = runner.load_policy(args.fit_policy)
    events = []
    print("event            P[d]   refit-PSPL  linPSPL-PSPL  linBSPL0-linPSPL  xifixed-PSPL   q_free"
          "   nfev(fixed/free)  s(fixed/free)")
    for p in args.json:
        e = event_check(p, policy, args.periods)
        events.append(e)
        for r in e["periods"]:
            print(f"{p.stem[:15]:15s} {r['period_days']:7.1f}  {e['chi2_PSPL_refit'] - e['chi2_PSPL']:10.4f}  "
                  f"{e['lin_PSPL'] - e['chi2_PSPL']:12.4f}  {r['lin_BSPL_xi0'] - e['lin_PSPL']:16.4f}  "
                  f"{r['fit_BSPL_xi_fixed'] - e['chi2_PSPL']:12.4f}  {r['q_free']:8.3f}   "
                  f"{r['nfev_xi_fixed']}/{r['nfev_free']}   {r['seconds_xi_fixed']:.0f}/{r['seconds_free']:.0f}",
                  flush=True)
    out = {**code_commits(), "fit_policy": str(args.fit_policy), "periods": args.periods, "events": events}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"saved {args.out}.json")


if __name__ == "__main__":
    main()
