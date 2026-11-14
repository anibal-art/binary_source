#!/usr/bin/env python3
"""
Linear-theory null distribution of the xallarap profile statistic, without fits.

In the small-amplitude limit the BSPL at fixed omega adds four linear directions
to the PSPL (start_bank.xallarap_columns: dm/dt0 and dm/du0 modulated by cos and
sin of omega t, PSPL directions projected out). For whitened noise z ~ N(0, I),
    q_lin(omega) = |U(omega)^T z|^2,   U = orthonormal basis of those columns,
is a chi2_4 process in omega. This script gives, for the real cadence and errors
of each event (runner JSON + .npz) and the best-PSPL Jacobian:

1. Davies / Rice upcrossings (the analytic formula behind Baluev 2008):
       E[N(c)] = c^((k-1)/2) e^(-c/2) / (2^((k-2)/2) Gamma(k/2)) * L,
       L = (2 pi)^(-1/2) Int domega  E_v |(I - U U^T) U'(omega) v|,
   v uniform on the unit sphere of R^k (k = 4). L is evaluated on a fine omega
   grid with U' from the projector derivative (independent of the basis choice).
   Critical value: P(chi2_k > c) + E[N(c)] = alpha.
2. Linear Monte Carlo: n_real noise vectors, q_lin on (a) the grid of the pilot
   profiles (lrt_omega_profile.omega_grid, same oversample) and (b) the fine grid;
   maximum, upcrossings of each level, quantile 1 - alpha of the maximum.

Linear theory keeps k = 4 at every period. In the fitted profiles the local mean
of q falls at long P (the PSPL absorbs the orbit and large amplitudes leave the
linear regime), so this prediction is expected to be conservative there; the
comparison with the fitted pilot (lrt_gross_vitells.py) measures by how much.
The window |t - t0| < window_tE tE is the one of the start bank.

Usage (on the pilot runner JSONs, .npz next to each):
    python binary_source/analysis/lrt_linear_null.py \
        --json results/roman_lrt/gv_pilot_v1_dark/json/*.json \
        --fit-policy catalogs/roman_bspl_lrt_v1/fit_policy_test_a_dark_v1.json \
        --out results/roman_lrt/gv_pilot_v1_dark/linear_null
Outputs <out>.json (per event and summary) and a printed table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

from binary_source.analysis import roman_bspl_lrt_catalog_runner as runner
from binary_source.analysis import roman_lrt_start_bank as start_bank
from binary_source.analysis.lrt_omega_profile import code_commits, omega_grid

K = 4                       # linear xallarap directions at fixed omega
N_SPHERE = 400              # directions v for the sphere average in L


def davies_factor(c, k=K):
    """c^((k-1)/2) e^(-c/2) / (2^((k-2)/2) Gamma(k/2)); E[N(c)] = factor * L."""
    return np.exp((k - 1) / 2 * np.log(c) - c / 2 - (k - 2) / 2 * np.log(2)
                  - special.gammaln(k / 2))


def davies_critical(L, alpha, k=K):
    f = lambda c: stats.chi2.sf(c, k) + davies_factor(c, k) * L - alpha
    return optimize.brentq(f, stats.chi2.isf(alpha, k), 200.0)


def orthonormal(X):
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U, s


def rice_length(design, omegas, rng):
    """L = (2 pi)^(-1/2) Int E_v |(I - P) U' v| domega (trapezoid on omegas)."""
    v = rng.standard_normal((K, N_SPHERE))
    v /= np.linalg.norm(v, axis=0)
    rate = np.empty(len(omegas))
    cond = np.empty(len(omegas))
    for j, om in enumerate(omegas):
        h = 1e-4 * om
        U, s = orthonormal(start_bank.xallarap_columns(design, om))
        Up, _ = orthonormal(start_bank.xallarap_columns(design, om + h))
        Um, _ = orthonormal(start_bank.xallarap_columns(design, om - h))
        dPU = (Up @ (Up.T @ U) - Um @ (Um.T @ U)) / (2 * h)      # P' U
        dPU = dPU - U @ (U.T @ dPU)                               # (I - P) P' U = (I - P) U'
        rate[j] = np.mean(np.linalg.norm(dPU @ v, axis=0))
        cond[j] = s[-1] / s[0]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed trapz
    L = trapezoid(rate, omegas) / np.sqrt(2 * np.pi)
    return L, rate, cond


def mc_profiles(design, omegas, n_real, rng):
    """q_lin(omega_j) for n_real whitened noise vectors: array (n_real, len(omegas))."""
    n = len(design["sw"])
    Z = rng.standard_normal((n, n_real))
    q = np.empty((n_real, len(omegas)))
    for j, om in enumerate(omegas):
        U, _ = orthonormal(start_bank.xallarap_columns(design, om))
        q[:, j] = np.sum((U.T @ Z) ** 2, axis=0)
    return q


def upcrossings(q, levels):
    return {float(c): float(np.mean(np.sum((q[:, :-1] < c) & (q[:, 1:] >= c), axis=1)
                                     + (q[:, 0] >= c)))
            for c in levels}


def event_linear_null(json_path, policy, oversample, fine_oversample, n_real, levels, alpha, seed):
    d = json.loads(Path(json_path).read_text())
    a = np.load(Path(json_path).with_suffix(".npz"))
    lc = np.column_stack([a["time"], a["mag"], a["err_mag"]])
    rule = policy["BSPL"]["start_rule"]
    ra, dec, ref = d["event_ra_deg"], d["event_dec_deg"], d["fit_reference"]
    pspl = d["fits"]["PSPL"]["best"]
    model = runner.build_fit_model(lc, ra, dec, ref, False)
    jac, names = runner.magnitude_jacobian(model, pspl["best_model"])
    tE = float(pspl["best_model"][names.index("tE")])
    window = float(rule.get("window_tE", 3.0))
    design = start_bank.xallarap_design(lc[:, 0], lc[:, 2], jac, names.index("t0"),
                                        names.index("u0"), tE, ref["t0"], window)
    pmin, pmax = float(rule["period_min_days"]), float(rule["period_max_days"])
    om_pilot = omega_grid(tE, window, oversample, pmin, pmax)
    om_fine = omega_grid(tE, window, fine_oversample, pmin, pmax)
    rng = np.random.default_rng(seed)

    L, rate, cond = rice_length(design, om_fine, rng)
    out = {"event_json": str(json_path), "simulation_seed": d["simulation_seed"],
           "config_id": d.get("catalog_row", {}).get("config_id"), "tE_PSPL": tE,
           "n_points_window": int(len(design["sw"])),
           "n_nodes_pilot": len(om_pilot), "n_nodes_fine": len(om_fine),
           "rice_L": float(L), "min_condition_number_ratio": float(np.min(cond)),
           "analytic_EN": {float(c): float(davies_factor(c) * L) for c in levels},
           "analytic_critical": float(davies_critical(L, alpha))}
    for tag, om in (("pilot_grid", om_pilot), ("fine_grid", om_fine)):
        q = mc_profiles(design, om, n_real, rng)
        T = q.max(axis=1)
        P = 2 * np.pi / om
        out[tag] = {"critical": float(np.quantile(T, 1 - alpha)),
                    "P_T_above": {float(c): float(np.mean(T > c)) for c in levels},
                    "EN": upcrossings(q, levels),
                    "mean_q_bands": {"P<30": float(q[:, P < 30].mean()),
                                     "30<=P<150": float(q[:, (P >= 30) & (P < 150)].mean()),
                                     "P>=150": float(q[:, P >= 150].mean())}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, nargs="+", required=True,
                    help="runner JSONs (.npz next to each); *_omega_profile.json are ignored")
    ap.add_argument("--fit-policy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output prefix (.json)")
    ap.add_argument("--oversample", type=float, default=2.0, help="pilot profile grid")
    ap.add_argument("--fine-oversample", type=float, default=20.0)
    ap.add_argument("--n-real", type=int, default=2000)
    ap.add_argument("--levels", type=float, nargs="+", default=[8, 10, 11.07, 12, 14, 15, 16, 18, 20])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    policy = runner.load_policy(args.fit_policy)
    inputs = [p for p in args.json if not p.name.endswith("_omega_profile.json")]
    inputs = inputs[:args.max_events] if args.max_events else inputs
    events = []
    for i, p in enumerate(inputs):
        e = event_linear_null(p, policy, args.oversample, args.fine_oversample, args.n_real,
                              args.levels, args.alpha, args.seed + i)
        events.append(e)
        print(f"{p.stem}: L = {e['rice_L']:.3f}  analytic c95 = {e['analytic_critical']:.2f}  "
              f"MC c95 pilot grid = {e['pilot_grid']['critical']:.2f}  "
              f"fine grid = {e['fine_grid']['critical']:.2f}  "
              f"E[N(10)] analytic {e['analytic_EN'][10.0]:.3f} / MC pilot grid "
              f"{e['pilot_grid']['EN'][10.0]:.3f}", flush=True)

    med = lambda key: float(np.median([key(e) for e in events]))
    summary = {
        "n_events": len(events), "alpha": args.alpha, "k": K,
        "rice_L_median": med(lambda e: e["rice_L"]),
        "analytic_critical_median": med(lambda e: e["analytic_critical"]),
        "analytic_EN10_median": med(lambda e: e["analytic_EN"][10.0]),
        "mc_pilot_grid_critical_median": med(lambda e: e["pilot_grid"]["critical"]),
        "mc_fine_grid_critical_median": med(lambda e: e["fine_grid"]["critical"]),
        "mc_pilot_grid_EN10_median": med(lambda e: e["pilot_grid"]["EN"][10.0]),
        "mc_fine_grid_EN10_median": med(lambda e: e["fine_grid"]["EN"][10.0]),
        "mc_pilot_grid_P_T_above_11.07_median": med(lambda e: e["pilot_grid"]["P_T_above"][11.07]),
        "wilks_chi2_5_critical": float(stats.chi2.isf(args.alpha, 5)),
    }
    print("\nsummary (median over events):")
    for k_, v_ in summary.items():
        print(f"    {k_}: {v_}")
    print("compare with the fitted pilot: lrt_gross_vitells.py (E[N(10)], GV and empirical c95)")
    out = {**code_commits(), "fit_policy": str(args.fit_policy), "args": {
        k_: (str(v_) if isinstance(v_, Path) else v_) for k_, v_ in vars(args).items()
        if k_ != "json"}, "summary": summary, "events": events}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.out) + ".json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"saved {args.out}.json")


if __name__ == "__main__":
    main()
