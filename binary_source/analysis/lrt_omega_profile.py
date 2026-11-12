#!/usr/bin/env python3
"""
Profile likelihood of the BSPL in the orbital angular frequency omega, for
events already processed by roman_bspl_lrt_catalog_runner (runner JSON plus
the .npz written with --save-arrays).

For every omega_k of a grid uniform in f = 1/P (step 1/(oversample * T),
T = 2 window_tE tE, the width of one profile peak divided by oversample):
  * omega is fixed (TRF bounds omega_k (1 +/- 1e-9); roman_rubin cannot fix a
    parameter), all other BSPL parameters are free within the policy bounds;
  * the start is the linear optimum at omega_k (joint_coefficients +
    candidates_at_omega), the amplitude candidate with the lowest full-model
    chi2; for P > --all-starts-above-days every amplitude candidate is fitted
    and the lowest chi2 is kept;
  * with --search sweep the nodes are visited from long to short P and every
    node also starts from the solution of the previous node (continuation; the
    longest-P node, close to the PSPL limit, fits every amplitude candidate);
    --sweep-back adds a pass from short to long P with the same continuation;
  * the TRF fit is the runner's run_one_fit (same roman_rubin fitter).
q(omega) = chi2_PSPL - min chi2_BSPL(omega). The grid maximum is refined by one
TRF with omega free between the two neighbouring nodes:
    T_profile = max(max_k q(omega_k), q_refined).

Single event (plot):
    python binary_source/analysis/lrt_omega_profile.py \
        --json results/roman_lrt/explore/h0a_118900_identity.json \
        --fit-policy catalogs/roman_bspl_lrt_v1/fit_policy_test_a_data_start_v1.json \
        --out results/roman_lrt/explore/h0a_118900_omega_profile --check-top 3
Many events (runner JSONs of a task run, <json>.npz next to each):
    python binary_source/analysis/lrt_omega_profile.py \
        --json results/roman_lrt/gv_pilot_v1/json/*.json \
        --fit-policy catalogs/roman_bspl_lrt_v1/fit_policy_test_a_data_start_v1.json \
        --out-dir results/roman_lrt/gv_pilot_v1/omega_profile --workers 40 --no-plot
Events whose output JSON exists are skipped.
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from binary_source.analysis import roman_bspl_lrt_catalog_runner as runner
from binary_source.analysis import roman_lrt_start_bank as start_bank


def omega_grid(tE, window_tE, oversample, period_min, period_max):
    df = 1.0 / (oversample * 2.0 * window_tE * abs(tE))
    f = np.arange(1.0 / period_max, 1.0 / period_min + 0.5 * df, df)
    return 2.0 * np.pi * f


def event_profile(json_path, arrays_path, policy_path, oversample=2.0,
                  period_min=None, period_max=None, all_starts_above_days=None,
                  check_top=0, refine=True, search="grid", sweep_back=False, tag=""):
    d = json.loads(Path(json_path).read_text())
    a = np.load(arrays_path)
    lc = np.column_stack([a["time"], a["mag"], a["err_mag"]])
    policy = runner.load_policy(Path(policy_path))
    spec = dict(policy["BSPL"])
    rule, bounds = spec["start_rule"], spec["bounds"]
    ra, dec, ref = d["event_ra_deg"], d["event_dec_deg"], d["fit_reference"]
    pspl = d["fits"]["PSPL"]["best"]
    chi2_pspl = float(pspl["chi2"])

    # Linear ingredients at the best PSPL (same as the runner).
    model = runner.build_fit_model(lc, ra, dec, ref, False)
    jac, names = runner.magnitude_jacobian(model, pspl["best_model"])
    resid = lc[:, 1] - runner.model_magnitude(model, pspl["best_model"])
    i_t0, i_u0 = names.index("t0"), names.index("u0")
    tE = float(pspl["best_model"][names.index("tE")])
    window = float(rule.get("window_tE", 3.0))
    bspl_model = runner.build_fit_model(lc, ra, dec, ref, True)
    pmin = float(period_min if period_min is not None else rule["period_min_days"])
    pmax = float(period_max if period_max is not None else rule["period_max_days"])
    omegas = omega_grid(tE, window, oversample, pmin, pmax)
    print(f"{tag}{len(omegas)} omega nodes, P = {2 * np.pi / omegas.max():.4g} ... "
          f"{2 * np.pi / omegas.min():.4g} d", flush=True)

    def fit(lo, hi, guess, label):
        s = dict(spec)
        s["bounds"] = dict(bounds, xi_angular_velocity=[lo, hi])
        t_start = time.perf_counter()
        row = runner.run_one_fit(
            source_id=int(d["simulation_seed"]), data={"W149": lc}, ra=ra, dec=dec,
            fit_key="BSPL", spec=s, start={"label": label, "initial_guess": guess},
            fit_reference=ref, optimizer_options=policy.get("optimizer_options"))
        row["wall_seconds"] = time.perf_counter() - t_start
        return row

    def candidates(omega):
        aa, cc = start_bank.joint_coefficients(
            lc[:, 0], resid, lc[:, 2], jac, i_t0, i_u0, tE, ref["t0"], omega, window)
        cands = start_bank.candidates_at_omega(names, pspl["best_model"], aa, cc, omega, rule, bounds)
        for c in cands:
            c["chi2_linear_fluxes"] = runner.chi2_linear_fluxes(bspl_model, c["initial_guess"])
        return sorted(cands, key=lambda c: c["chi2_linear_fluxes"])

    def fit_fixed(omega, cands):
        """Fit every candidate in cands at fixed omega; return best row and all chi2."""
        best, tried = None, []
        for c in cands:
            g = dict(c["initial_guess"], xi_angular_velocity=omega)
            try:
                row = fit(omega * (1 - 1e-9), omega * (1 + 1e-9), g, c["label"])
            except Exception as exc:                   # keep the node, mark it
                tried.append({"label": c["label"], "alpha": c["alpha"], "chi2": None,
                              "error": repr(exc)})
                continue
            tried.append({"label": c["label"], "alpha": c["alpha"], "chi2": float(row["chi2"])})
            if best is None or row["chi2"] < best["chi2"]:
                row["start_alpha"], row["start_label"], row["start_guess"] = c["alpha"], c["label"], g
                best = row
        return best, tried

    def from_row(row, omega, label):
        """Start at omega from the solution of another node (continuation)."""
        g = dict(row["start_guess"])
        g.update({n: v for n, v in zip(row["fit_parameter_names"], row["best_model"]) if n in g})
        g["xi_angular_velocity"] = float(omega)
        return {"label": label, "alpha": None, "initial_guess": g}

    def record(om, row, tried):
        return {
            "omega": float(om), "period_days": float(2 * np.pi / om),
            "delta_chi2": None if row is None else chi2_pspl - float(row["chi2"]),
            "chi2": None if row is None else float(row["chi2"]), "starts": tried,
            "start_alpha": None if row is None else row["start_alpha"],
            "start_label": None if row is None else row["start_label"],
            "nfev": None if row is None else row.get("optimizer_nfev"),
            "best_model": None if row is None else row["best_model"],
            "fit_parameter_names": None if row is None else row["fit_parameter_names"]}

    def show(k, rec, n_starts, seconds, prefix=""):
        dchi2 = np.nan if rec["delta_chi2"] is None else rec["delta_chi2"]
        print(f"{tag}{prefix}[{k + 1:3d}/{len(omegas)}] P = {rec['period_days']:9.3f} d  "
              f"starts {n_starts}  Delta chi2 = {dchi2:8.3f}  best start "
              f"{rec['start_label']}  {seconds:.1f} s", flush=True)

    # omegas ascending = periods descending: the sweep goes from long to short P.
    profile, rows, prev = [], [], None
    for k, om in enumerate(omegas):
        P = 2 * np.pi / om
        cands = candidates(om)
        many = all_starts_above_days is not None and P > all_starts_above_days
        use = cands if (many or (search == "sweep" and k == 0)) else cands[:1]
        if search == "sweep" and prev is not None:
            use = [from_row(prev, om, "continuation")] + use
        t_node = time.perf_counter()
        row, tried = fit_fixed(om, use)
        rows.append(row)
        prev = row if row is not None else prev
        profile.append(record(om, row, tried))
        show(k, profile[-1], len(tried), time.perf_counter() - t_node)

    if search == "sweep" and sweep_back:              # short to long P
        for k in range(len(omegas) - 2, -1, -1):
            if rows[k + 1] is None:
                continue
            t_node = time.perf_counter()
            row, tried = fit_fixed(omegas[k], [from_row(rows[k + 1], omegas[k], "continuation_back")])
            starts = profile[k]["starts"] + tried
            if row is not None and (rows[k] is None or row["chi2"] < rows[k]["chi2"]):
                rows[k] = row
                profile[k] = record(omegas[k], row, starts)
            else:
                profile[k]["starts"] = starts
            show(k, profile[k], len(tried), time.perf_counter() - t_node, prefix="back ")

    q = np.array([np.nan if p["delta_chi2"] is None else p["delta_chi2"] for p in profile])
    if not np.any(np.isfinite(q)):
        raise RuntimeError("every omega node failed")
    j = int(np.nanargmax(q))
    T_profile, refined = float(q[j]), None
    if refine:
        lo = omegas[j - 1] if j > 0 else omegas[j] * (1 - 1e-9)
        hi = omegas[j + 1] if j < len(omegas) - 1 else omegas[j] * (1 + 1e-9)
        g = dict(rows[j]["start_guess"])
        g.update({n: v for n, v in zip(rows[j]["fit_parameter_names"], rows[j]["best_model"])
                  if n in g})
        g["xi_angular_velocity"] = float(omegas[j])
        try:
            r = fit(lo, hi, g, "refine_free_omega")
            om_r = float(r["best_model"][r["fit_parameter_names"].index("xi_angular_velocity")])
            refined = {"delta_chi2": chi2_pspl - float(r["chi2"]), "omega": om_r,
                       "period_days": 2 * np.pi / om_r, "nfev": r.get("optimizer_nfev"),
                       "best_model": r["best_model"],
                       "fit_parameter_names": r["fit_parameter_names"]}
            T_profile = max(T_profile, refined["delta_chi2"])
        except Exception as exc:
            refined = {"error": repr(exc)}

    checks = []
    for jj in np.argsort(-np.nan_to_num(q, nan=-np.inf))[:check_top]:
        _, tried = fit_fixed(omegas[jj], candidates(omegas[jj]))
        checks.append({"period_days": profile[jj]["period_days"],
                       "delta_chi2_by_alpha": [
                           [t["alpha"], None if t["chi2"] is None else chi2_pspl - t["chi2"]]
                           for t in tried]})
        print(f"{tag}check P = {profile[jj]['period_days']:.3f} d: " + ", ".join(
            f"alpha {a_} -> {v:.3f}" if v is not None else f"alpha {a_} -> failed"
            for a_, v in checks[-1]["delta_chi2_by_alpha"]), flush=True)

    print(f"{tag}T_profile = {T_profile:.3f}  (grid max {q[j]:.3f} at P = "
          f"{profile[j]['period_days']:.3f} d; runner T_A = {d['statistics']['T_A']:.3f})",
          flush=True)
    return {"event_json": str(json_path), "fit_policy": str(policy_path),
            "config_id": d.get("catalog_row", {}).get("config_id"),
            "simulation_seed": d["simulation_seed"], "truth_kind": d.get("truth_kind"),
            "tE_PSPL": tE, "window_tE": window, "oversample": oversample,
            "period_min_days": pmin, "period_max_days": pmax,
            "all_starts_above_days": all_starts_above_days,
            "search": search, "sweep_back": sweep_back,
            "chi2_PSPL": chi2_pspl, "T_A_runner": d["statistics"]["T_A"],
            "T_profile": T_profile, "grid_max_delta_chi2": float(q[j]),
            "grid_max_period_days": profile[j]["period_days"], "refined": refined,
            "profile": profile, "checks": checks}


def plot_profile(out, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = [p["period_days"] for p in out["profile"]]
    y = [np.nan if p["delta_chi2"] is None else p["delta_chi2"] for p in out["profile"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(P, y, "o-", color="C3", ms=3, lw=1,
            label="profile: omega fixed, all other parameters fitted")
    ref = out.get("refined") or {}
    if "delta_chi2" in ref:
        ax.plot([ref["period_days"]], [ref["delta_chi2"]], "*", color="k", ms=10,
                label=f"refined maximum (omega free) = {ref['delta_chi2']:.2f}")
    ax.axhline(out["T_A_runner"], color="C0", ls="--", lw=1,
               label=f"T_A of the runner fit = {out['T_A_runner']:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("period [d]")
    ax.set_ylabel("chi2_PSPL - chi2_BSPL")
    ax.set_title(f"{out['config_id']}  seed {out['simulation_seed']}", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(png, dpi=150)
    plt.close(fig)


def _run(job):
    json_path, arrays_path, out_prefix, kw, plot = job
    out_json = Path(str(out_prefix) + ".json")
    if out_json.exists():
        print(f"[skip] {out_json} exists", flush=True)
        return
    try:
        out = event_profile(json_path, arrays_path, tag=f"{Path(json_path).stem} ", **kw)
    except Exception as exc:
        print(f"[FAIL] {json_path}: {exc!r}", flush=True)
        return
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2) + "\n")
    if plot:
        plot_profile(out, str(out_prefix) + ".png")
    print(f"[ok]   {out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, nargs="+", required=True,
                    help="runner JSON(s); inputs ending in _omega_profile.json are ignored")
    ap.add_argument("--arrays", type=Path, default=None,
                    help="only with one --json (default: the .npz next to the JSON)")
    ap.add_argument("--fit-policy", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="output prefix, only with one --json")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="outputs <out-dir>/<json stem>_omega_profile.json (default: next to the JSON)")
    ap.add_argument("--oversample", type=float, default=2.0,
                    help="omega nodes per profile-peak width 1/(2 window_tE tE)")
    ap.add_argument("--period-min", type=float, default=None, help="default: policy start_rule")
    ap.add_argument("--period-max", type=float, default=None, help="default: policy start_rule")
    ap.add_argument("--all-starts-above-days", type=float, default=None,
                    help="fit every amplitude candidate at nodes with P above this")
    ap.add_argument("--check-top", type=int, default=0,
                    help="fit every amplitude candidate at the N best nodes (diagnostic)")
    ap.add_argument("--search", choices=("grid", "sweep"), default="grid",
                    help="grid: independent start(s) per node; sweep: long to short P, each node "
                         "also starts from the previous node's solution (continuation)")
    ap.add_argument("--sweep-back", action="store_true",
                    help="with --search sweep, a second pass from short to long P")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    inputs = [p for p in args.json if not p.name.endswith("_omega_profile.json")]
    if (args.arrays or args.out) and len(inputs) != 1:
        raise SystemExit("--arrays and --out need exactly one --json")
    kw = dict(policy_path=args.fit_policy, oversample=args.oversample,
              period_min=args.period_min, period_max=args.period_max,
              all_starts_above_days=args.all_starts_above_days,
              check_top=args.check_top, refine=not args.no_refine,
              search=args.search, sweep_back=args.sweep_back)
    jobs = []
    for p in inputs:
        arrays = args.arrays or p.with_suffix(".npz")
        if args.out:
            prefix = args.out
        else:
            prefix = (args.out_dir or p.parent) / f"{p.stem}_omega_profile"
        jobs.append((p, arrays, prefix, kw, not args.no_plot))
    print(f"{len(jobs)} events, {args.workers} workers", flush=True)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            pool.map(_run, jobs, chunksize=1)
    else:
        for job in jobs:
            _run(job)


if __name__ == "__main__":
    main()
