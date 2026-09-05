#!/usr/bin/env python3
"""
Validation suite for the BSPL -> PSPL manuscript.

This script addresses five referee-style numerical checks in one run:

1) Same-objective optimizer check:
   Nelder-Mead vs scipy least_squares(method='trf') on the exact same
   intrinsic magnification-space trapezoidal objective.

2) Cross-pipeline check:
   intrinsic Nelder-Mead vs the Roman/pyLIMA TRF flux-space fit on the
   same noiseless, uniformly sampled BSPL curves.  The pyLIMA solution is
   then evaluated with the intrinsic D metric on the same uniform grid.

3) Roman correlation statistics:
   Pearson r and Spearman rho between log10(D_intrinsic) and
   log10(D_Roman,w), separately for each W149 magnitude.

4) Grid-wide production audit:
   SUCCESS flags, finite D, finite best-fit parameters, parameter ranges,
   large adjacent jumps in log D, and a stratified independent re-fit
   subset of the one-luminous production grid.

5) Roman nonlinear-bound audit:
   verifies whether fitted tE values approach/saturate the adopted
   [0.02, 20] * tE,true bounds.

Run from repository root, e.g.

    python binary_source/validation/validate_optimizer_and_grid_audit.py

Outputs are written to

    results/validation_optimizer_grid_audit/

The script does not modify production files.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.stats import pearsonr, spearmanr

from binary_source.analysis import roman_bspl_pspl_asimov as roman


# =============================================================================
# Constants used by the manuscript production family
# =============================================================================

T0_INTRINSIC = 50.0
TE_TRUE = 150.0
Q_MASS = 0.5
Q_FLUX = 0.0
MTOT_MSUN = 3.0
REHAT_AU = 5.0
THETA = 0.0
PHI = 0.0
INCLINATION = np.pi / 2.0
WINDOW_TE = 3.5
N_TIME = 10_000
ROMAN_T0_JD = 2461849.0
ROMAN_SOURCE_MAG_FOR_CROSSCHECK = 21.0

TRAPZ = getattr(np, "trapezoid", np.trapz)


# =============================================================================
# Small utilities
# =============================================================================


def section(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def first_existing_key(files, aliases):
    for key in aliases:
        if key in files:
            return key
    return None


def trapezoid_weights(t: np.ndarray) -> np.ndarray:
    """Weights w such that sum(w * y) == trapezoidal integral of y."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be a 1-D array with at least two samples")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly increasing")

    w = np.empty_like(t)
    w[0] = 0.5 * dt[0]
    w[-1] = 0.5 * dt[-1]
    if len(t) > 2:
        w[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    return w


def pspl_magnification(t, t0, u0, tE):
    t = np.asarray(t, dtype=float)
    if not np.isfinite(tE) or tE <= 0:
        return np.full_like(t, np.nan)

    tau = (t - float(t0)) / float(tE)
    u = np.sqrt(float(u0) ** 2 + tau**2)
    u = np.maximum(u, 1e-300)
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def intrinsic_D(t, A_truth, params):
    t0, u0, tE = np.asarray(params, dtype=float)
    A_fit = pspl_magnification(t, t0, u0, tE)
    if np.any(~np.isfinite(A_fit)):
        return np.nan, np.nan

    J = float(TRAPZ((A_truth - A_fit) ** 2, t))
    denominator = float(TRAPZ((A_truth - 1.0) ** 2, t))
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan, J
    return float(np.sqrt(max(J, 0.0) / denominator)), J


# =============================================================================
# Intrinsic optimizers
# =============================================================================


def fit_intrinsic_nm(t, A_truth, x0):
    """Production-like unconstrained Nelder-Mead on the trapezoidal J."""
    t = np.asarray(t, dtype=float)
    A_truth = np.asarray(A_truth, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    def objective(p):
        t0, u0, tE = p
        if (not np.all(np.isfinite(p))) or tE <= 0:
            return 1e300
        A = pspl_magnification(t, t0, u0, tE)
        if np.any(~np.isfinite(A)):
            return 1e300
        return float(TRAPZ((A_truth - A) ** 2, t))

    res = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        options={
            "maxiter": 200_000,
            "xatol": 1e-10,
            "fatol": 1e-14,
        },
    )
    D, J = intrinsic_D(t, A_truth, res.x)
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(getattr(res, "nfev", -1)),
        "params": np.asarray(res.x, dtype=float),
        "D": D,
        "J": J,
    }


def fit_intrinsic_trf_same_objective(t, A_truth, x0):
    """
    TRF minimization of exactly the same trapezoidal objective as NM.

    residual_i = sqrt(w_i) * [A_truth - A_PSPL], therefore
    sum residual_i^2 == trapezoidal J.
    """
    t = np.asarray(t, dtype=float)
    A_truth = np.asarray(A_truth, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    w = trapezoid_weights(t)
    sqrt_w = np.sqrt(w)

    t0_ref, u0_ref, tE_ref = x0
    lower = np.array(
        [t0_ref - 2.0 * tE_ref, -20.0, 0.02 * tE_ref],
        dtype=float,
    )
    upper = np.array(
        [t0_ref + 2.0 * tE_ref, +20.0, 20.0 * tE_ref],
        dtype=float,
    )

    def residuals(p):
        A = pspl_magnification(t, p[0], p[1], p[2])
        return sqrt_w * (A_truth - A)

    res = least_squares(
        residuals,
        x0=x0,
        method="trf",
        bounds=(lower, upper),
        max_nfev=50_000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        x_scale="jac",
    )
    D, J = intrinsic_D(t, A_truth, res.x)
    return {
        "success": bool(res.success),
        "message": str(res.message),
        "nfev": int(getattr(res, "nfev", -1)),
        "params": np.asarray(res.x, dtype=float),
        "D": D,
        "J": J,
        "active_mask": np.asarray(getattr(res, "active_mask", [0, 0, 0]), dtype=int),
    }


# =============================================================================
# Truth generation
# =============================================================================


def make_truth(t, t0, u0, tE, P_days):
    out = roman.bspl_truth_magnification(
        t=np.asarray(t, dtype=float),
        t0_true=float(t0),
        u0_true=float(u0),
        tE_true=float(tE),
        P_days=float(P_days),
        q_mass=Q_MASS,
        qflux=Q_FLUX,
        Mtot_Msun=MTOT_MSUN,
        rEhat_AU=REHAT_AU,
        theta=THETA,
        phi=PHI,
        inclination=INCLINATION,
    )
    return np.asarray(out["A_bspl"], dtype=float)


# =============================================================================
# Load the one-luminous production grid
# =============================================================================


@dataclass
class MainGrid:
    files: list[Path]
    t: np.ndarray
    u0: np.ndarray
    P_days: np.ndarray
    D: np.ndarray
    success: np.ndarray
    best: np.ndarray
    t0_true: float
    tE_true: float
    rows: pd.DataFrame


def load_main_grid(pattern: str) -> MainGrid:
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No production files found with pattern:\n  {pattern}"
        )

    records = []
    P_ref = None
    t_ref = None
    t0_ref = None
    tE_ref = None

    for fn in files:
        with np.load(fn, allow_pickle=False) as d:
            required = {"truth", "P_grid", "D", "SUCCESS", "BEST_T0U0TE"}
            missing = sorted(required.difference(d.files))
            if missing:
                raise KeyError(f"{fn}: missing keys {missing}; available={d.files}")

            truth = np.asarray(d["truth"], dtype=float)
            P = np.asarray(d["P_grid"], dtype=float)
            D = np.asarray(d["D"], dtype=float)
            success = np.asarray(d["SUCCESS"], dtype=bool)
            best = np.asarray(d["BEST_T0U0TE"], dtype=float)
            t = np.asarray(d["t"], dtype=float) if "t" in d.files else None

        if D.shape != P.shape or success.shape != P.shape:
            raise ValueError(f"{fn}: incompatible P/D/SUCCESS shapes")
        if best.shape != (len(P), 3):
            raise ValueError(f"{fn}: expected BEST_T0U0TE shape {(len(P), 3)}, got {best.shape}")

        if P_ref is None:
            P_ref = P.copy()
        elif not np.allclose(P, P_ref, rtol=0, atol=1e-12):
            raise ValueError(f"{fn}: P_grid differs from reference")

        if t is not None:
            if t_ref is None:
                t_ref = t.copy()
            elif not np.allclose(t, t_ref, rtol=0, atol=1e-12):
                raise ValueError(f"{fn}: time grid differs from reference")

        t0_true = float(truth[0])
        u0_true = float(truth[1])
        tE_true = float(truth[2])

        if t0_ref is None:
            t0_ref = t0_true
            tE_ref = tE_true
        else:
            if not np.isclose(t0_true, t0_ref, rtol=0, atol=1e-12):
                raise ValueError(f"{fn}: t0_true differs from reference")
            if not np.isclose(tE_true, tE_ref, rtol=0, atol=1e-12):
                raise ValueError(f"{fn}: tE_true differs from reference")

        records.append(
            {
                "file": fn,
                "u0": u0_true,
                "D": D,
                "success": success,
                "best": best,
            }
        )

    records = sorted(records, key=lambda r: r["u0"])
    u0_grid = np.array([r["u0"] for r in records], dtype=float)
    D_map = np.vstack([r["D"] for r in records])
    success_map = np.vstack([r["success"] for r in records])
    best_map = np.stack([r["best"] for r in records], axis=0)

    if t_ref is None:
        t_ref = np.linspace(
            t0_ref - WINDOW_TE * tE_ref,
            t0_ref + WINDOW_TE * tE_ref,
            N_TIME,
        )

    row_list = []
    for iu, u0_true in enumerate(u0_grid):
        for ip, P_days in enumerate(P_ref):
            b = best_map[iu, ip]
            dval = D_map[iu, ip]
            ok = bool(success_map[iu, ip])
            finite_best = bool(np.all(np.isfinite(b)))
            row_list.append(
                {
                    "iu": iu,
                    "ip": ip,
                    "u0_true": u0_true,
                    "P_days": float(P_days),
                    "P_over_tE": float(P_days / tE_ref),
                    "D_stored": float(dval),
                    "success": ok,
                    "finite_D": bool(np.isfinite(dval)),
                    "finite_best": finite_best,
                    "best_t0": float(b[0]) if finite_best else np.nan,
                    "best_u0": float(b[1]) if finite_best else np.nan,
                    "best_tE": float(b[2]) if finite_best else np.nan,
                    "dt0_over_tE": float((b[0] - t0_ref) / tE_ref) if finite_best else np.nan,
                    "du0_over_u0": float((b[1] - u0_true) / u0_true) if finite_best and u0_true != 0 else np.nan,
                    "dtE_over_tE": float((b[2] - tE_ref) / tE_ref) if finite_best else np.nan,
                }
            )

    return MainGrid(
        files=[r["file"] for r in records],
        t=t_ref,
        u0=u0_grid,
        P_days=P_ref,
        D=D_map,
        success=success_map,
        best=best_map,
        t0_true=float(t0_ref),
        tE_true=float(tE_ref),
        rows=pd.DataFrame(row_list),
    )


# =============================================================================
# Full-grid audit
# =============================================================================


def compute_neighbor_jumps(grid: MainGrid) -> pd.DataFrame:
    rows = []
    D = grid.D
    ok = grid.success & np.isfinite(D) & (D > 0)

    # Along P at fixed u0
    for iu, u0 in enumerate(grid.u0):
        for ip in range(len(grid.P_days) - 1):
            if ok[iu, ip] and ok[iu, ip + 1]:
                jump = abs(np.log10(D[iu, ip + 1]) - np.log10(D[iu, ip]))
                rows.append(
                    {
                        "direction": "P",
                        "jump_abs_dex": float(jump),
                        "u0_a": float(u0),
                        "u0_b": float(u0),
                        "P_a_days": float(grid.P_days[ip]),
                        "P_b_days": float(grid.P_days[ip + 1]),
                        "D_a": float(D[iu, ip]),
                        "D_b": float(D[iu, ip + 1]),
                        "iu_a": iu,
                        "ip_a": ip,
                        "iu_b": iu,
                        "ip_b": ip + 1,
                    }
                )

    # Along u0 at fixed P
    for iu in range(len(grid.u0) - 1):
        for ip, P in enumerate(grid.P_days):
            if ok[iu, ip] and ok[iu + 1, ip]:
                jump = abs(np.log10(D[iu + 1, ip]) - np.log10(D[iu, ip]))
                rows.append(
                    {
                        "direction": "u0",
                        "jump_abs_dex": float(jump),
                        "u0_a": float(grid.u0[iu]),
                        "u0_b": float(grid.u0[iu + 1]),
                        "P_a_days": float(P),
                        "P_b_days": float(P),
                        "D_a": float(D[iu, ip]),
                        "D_b": float(D[iu + 1, ip]),
                        "iu_a": iu,
                        "ip_a": ip,
                        "iu_b": iu + 1,
                        "ip_b": ip,
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("jump_abs_dex", ascending=False)


def audit_main_grid(grid: MainGrid, output_dir: Path):
    df = grid.rows.copy()
    total = len(df)
    n_success = int(df["success"].sum())
    bad_success = df[
        df["success"]
        & (~df["finite_D"] | ~df["finite_best"] | (df["D_stored"] < 0))
    ].copy()

    summary = {
        "n_files": len(grid.files),
        "n_points": total,
        "n_success": n_success,
        "success_fraction": n_success / total if total else np.nan,
        "n_failed": total - n_success,
        "n_success_with_nonfinite_or_negative_output": len(bad_success),
        "D_min_success": float(df.loc[df.success & df.finite_D, "D_stored"].min()),
        "D_max_success": float(df.loc[df.success & df.finite_D, "D_stored"].max()),
        "best_tE_min_success": float(df.loc[df.success & df.finite_best, "best_tE"].min()),
        "best_tE_max_success": float(df.loc[df.success & df.finite_best, "best_tE"].max()),
        "max_abs_dt0_over_tE": float(df.loc[df.success, "dt0_over_tE"].abs().max()),
        "max_abs_du0_over_u0": float(df.loc[df.success, "du0_over_u0"].abs().max()),
        "max_abs_dtE_over_tE": float(df.loc[df.success, "dtE_over_tE"].abs().max()),
    }

    df.to_csv(output_dir / "intrinsic_grid_all_points.csv", index=False)
    bad_success.to_csv(output_dir / "intrinsic_grid_invalid_success_points.csv", index=False)

    jumps = compute_neighbor_jumps(grid)
    if not jumps.empty:
        jumps.to_csv(output_dir / "intrinsic_neighbor_jumps_all.csv", index=False)
        jumps.head(100).to_csv(output_dir / "intrinsic_neighbor_jumps_top100.csv", index=False)
        summary["largest_adjacent_jump_dex"] = float(jumps.iloc[0]["jump_abs_dex"])
    else:
        summary["largest_adjacent_jump_dex"] = np.nan

    return summary, jumps


def audit_summary_npz(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}

    with np.load(path, allow_pickle=False) as d:
        files = list(d.files)
        D_key = first_existing_key(files, ["D", "D_MAP", "D_grid"])
        S_key = first_existing_key(files, ["SUCCESS", "success"])
        if D_key is None:
            return {
                "path": str(path),
                "exists": True,
                "audited": False,
                "reason": "no D-like key",
                "keys": files,
            }
        D = np.asarray(d[D_key], dtype=float)
        success = np.asarray(d[S_key], dtype=bool) if S_key is not None else np.isfinite(D)

    if success.shape != D.shape:
        return {
            "path": str(path),
            "exists": True,
            "audited": False,
            "reason": f"SUCCESS shape {success.shape} != D shape {D.shape}",
            "keys": files,
        }

    good = success & np.isfinite(D)
    return {
        "path": str(path),
        "exists": True,
        "audited": True,
        "D_key": D_key,
        "success_key": S_key,
        "shape": D.shape,
        "n_points": int(D.size),
        "n_success": int(np.count_nonzero(success)),
        "n_failed": int(D.size - np.count_nonzero(success)),
        "n_success_nonfinite_D": int(np.count_nonzero(success & ~np.isfinite(D))),
        "D_min_success": float(np.nanmin(D[good])) if np.any(good) else np.nan,
        "D_max_success": float(np.nanmax(D[good])) if np.any(good) else np.nan,
    }


# =============================================================================
# Stratified independent re-fit audit
# =============================================================================


def select_stratified_points(grid: MainGrid, n_target: int, seed: int) -> pd.DataFrame:
    df = grid.rows[
        grid.rows.success
        & grid.rows.finite_D
        & grid.rows.finite_best
        & (grid.rows.D_stored > 0)
    ].copy()

    if df.empty:
        return df

    chosen = set()

    def add_indices(indices):
        for idx in indices:
            chosen.add(int(idx))

    # D quantiles: sample across the full mismatch distribution.
    qs = np.linspace(0.0, 1.0, 21)
    logD = np.log10(df.D_stored.to_numpy())
    for q in qs:
        target = np.quantile(logD, q)
        idx = int(df.index[np.argmin(np.abs(logD - target))])
        chosen.add(idx)

    # Points closest to manuscript-relevant D levels.
    for d0 in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.3]:
        idx = int(df.index[np.argmin(np.abs(np.log10(df.D_stored) - np.log10(d0)))])
        chosen.add(idx)

    # Largest parameter excursions in each fitted PSPL parameter.
    for col in ["dt0_over_tE", "du0_over_u0", "dtE_over_tE"]:
        add_indices(df[col].abs().nlargest(8).index)

    # Extremes in u0 and P.
    for col in ["u0_true", "P_days"]:
        add_indices(df[col].nsmallest(5).index)
        add_indices(df[col].nlargest(5).index)

    # Fill remaining positions with deterministic random points.
    rng = np.random.default_rng(seed)
    remaining = np.array(sorted(set(df.index) - chosen), dtype=int)
    need = max(0, n_target - len(chosen))
    if need > 0 and len(remaining) > 0:
        add_indices(rng.choice(remaining, size=min(need, len(remaining)), replace=False))

    selected = df.loc[sorted(chosen)].copy()

    # If deterministic extremes produced more than requested, retain a broad
    # deterministic subset by evenly spaced positions in sorted log D.
    if len(selected) > n_target:
        selected = selected.sort_values("D_stored")
        keep = np.unique(np.round(np.linspace(0, len(selected) - 1, n_target)).astype(int))
        selected = selected.iloc[keep]

    return selected.reset_index(drop=True)


def run_stratified_refits(grid: MainGrid, selected: pd.DataFrame, output_dir: Path):
    results = []

    section(f"STRATIFIED RE-FIT AUDIT ({len(selected)} production points)")

    for k, row in selected.iterrows():
        u0 = float(row.u0_true)
        P_days = float(row.P_days)
        x0 = np.array([grid.t0_true, u0, grid.tE_true], dtype=float)
        A_truth = make_truth(grid.t, grid.t0_true, u0, grid.tE_true, P_days)

        nm = fit_intrinsic_nm(grid.t, A_truth, x0)
        trf = fit_intrinsic_trf_same_objective(grid.t, A_truth, x0)

        stored = float(row.D_stored)
        rel_nm_stored = (nm["D"] - stored) / stored if stored > 0 else np.nan
        rel_trf_stored = (trf["D"] - stored) / stored if stored > 0 else np.nan
        rel_trf_nm = (trf["D"] - nm["D"]) / nm["D"] if nm["D"] > 0 else np.nan

        rec = {
            "u0_true": u0,
            "P_days": P_days,
            "P_over_tE": P_days / grid.tE_true,
            "D_stored": stored,
            "D_nm_fresh": nm["D"],
            "D_trf_same_objective": trf["D"],
            "rel_nm_minus_stored": rel_nm_stored,
            "rel_trf_minus_stored": rel_trf_stored,
            "rel_trf_minus_nm": rel_trf_nm,
            "nm_success": nm["success"],
            "trf_success": trf["success"],
            "nm_t0": nm["params"][0],
            "nm_u0": nm["params"][1],
            "nm_tE": nm["params"][2],
            "trf_t0": trf["params"][0],
            "trf_u0": trf["params"][1],
            "trf_tE": trf["params"][2],
            "trf_active_any": bool(np.any(trf["active_mask"] != 0)),
        }
        results.append(rec)

        print(
            f"[{k+1:02d}/{len(selected):02d}] "
            f"u0={u0:.5g} P/tE={P_days/grid.tE_true:.5g} "
            f"Dstored={stored:.6e} Dnm={nm['D']:.6e} Dtrf={trf['D']:.6e} "
            f"(TRF-NM)/NM={rel_trf_nm:+.3e}"
        )

    out = pd.DataFrame(results)
    out.to_csv(output_dir / "stratified_refit_audit.csv", index=False)

    finite = out[np.isfinite(out.rel_trf_minus_nm)]
    summary = {
        "n_points": len(out),
        "n_nm_success": int(out.nm_success.sum()),
        "n_trf_success": int(out.trf_success.sum()),
        "n_trf_active_at_bound": int(out.trf_active_any.sum()),
        "max_abs_rel_nm_minus_stored": float(out.rel_nm_minus_stored.abs().max()),
        "median_abs_rel_nm_minus_stored": float(out.rel_nm_minus_stored.abs().median()),
        "max_abs_rel_trf_minus_nm": float(finite.rel_trf_minus_nm.abs().max()) if len(finite) else np.nan,
        "median_abs_rel_trf_minus_nm": float(finite.rel_trf_minus_nm.abs().median()) if len(finite) else np.nan,
    }
    return summary


# =============================================================================
# Systematic optimizer cross-check on a simple 5 x 6 subset
# =============================================================================


def run_optimizer_crosscheck(output_dir: Path, do_flux_trf: bool):
    t = np.linspace(
        T0_INTRINSIC - WINDOW_TE * TE_TRUE,
        T0_INTRINSIC + WINDOW_TE * TE_TRUE,
        N_TIME,
    )

    u0_values = [0.01, 0.03, 0.1, 0.3, 1.0]
    P_days_values = [10.0, 45.0, 150.0, 450.0, 6000.0, 100000.0]

    rows = []
    section("SYSTEMATIC OPTIMIZER CROSS-CHECK")

    for k, (u0, P_days) in enumerate(itertools.product(u0_values, P_days_values), start=1):
        A_truth = make_truth(t, T0_INTRINSIC, u0, TE_TRUE, P_days)
        x0 = np.array([T0_INTRINSIC, u0, TE_TRUE], dtype=float)

        nm = fit_intrinsic_nm(t, A_truth, x0)
        trf = fit_intrinsic_trf_same_objective(t, A_truth, x0)

        rel_same = (trf["D"] - nm["D"]) / nm["D"] if nm["D"] > 0 else np.nan

        rec = {
            "u0_true": u0,
            "P_days": P_days,
            "P_over_tE": P_days / TE_TRUE,
            "D_nm": nm["D"],
            "D_trf_same_objective": trf["D"],
            "rel_trf_same_minus_nm": rel_same,
            "J_nm": nm["J"],
            "J_trf_same": trf["J"],
            "nm_success": nm["success"],
            "trf_same_success": trf["success"],
            "nm_t0": nm["params"][0],
            "nm_u0": nm["params"][1],
            "nm_tE": nm["params"][2],
            "trf_same_t0": trf["params"][0],
            "trf_same_u0": trf["params"][1],
            "trf_same_tE": trf["params"][2],
            "trf_same_active_any": bool(np.any(trf["active_mask"] != 0)),
        }

        if do_flux_trf:
            shift = ROMAN_T0_JD - T0_INTRINSIC
            t_jd = t + shift
            asimov = roman.make_roman_asimov_event(
                t=t_jd,
                A_bspl=A_truth,
                source_mag=ROMAN_SOURCE_MAG_FOR_CROSSCHECK,
            )
            fit_flux = roman.fit_pspl_roman(
                ev=asimov["event"],
                t0_guess=ROMAN_T0_JD,
                u0_guess=u0,
                tE_guess=TE_TRUE,
            )

            p_flux_centered = np.array(
                [
                    float(fit_flux["best_t0"]) - shift,
                    float(fit_flux["best_u0"]),
                    float(fit_flux["best_tE"]),
                ]
            )
            D_flux, J_flux = intrinsic_D(t, A_truth, p_flux_centered)
            rel_flux = (D_flux - nm["D"]) / nm["D"] if nm["D"] > 0 else np.nan

            rec.update(
                {
                    "D_flux_trf_evaluated_intrinsically": D_flux,
                    "J_flux_trf_evaluated_intrinsically": J_flux,
                    "rel_flux_trf_minus_nm": rel_flux,
                    "flux_trf_t0_centered": p_flux_centered[0],
                    "flux_trf_u0": p_flux_centered[1],
                    "flux_trf_tE": p_flux_centered[2],
                    "flux_trf_chi2": float(fit_flux.get("chi2", np.nan)),
                }
            )

        rows.append(rec)
        extra = ""
        if do_flux_trf:
            extra = f" Dflux={rec['D_flux_trf_evaluated_intrinsically']:.6e}"
        print(
            f"[{k:02d}/30] u0={u0:.3g} P/tE={P_days/TE_TRUE:.4g} "
            f"Dnm={nm['D']:.6e} Dtrf={trf['D']:.6e} "
            f"rel={rel_same:+.3e}{extra}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "optimizer_crosscheck_5x6.csv", index=False)

    summary = {
        "n_cases": len(df),
        "same_objective_nm_success": int(df.nm_success.sum()),
        "same_objective_trf_success": int(df.trf_same_success.sum()),
        "same_objective_trf_bound_active": int(df.trf_same_active_any.sum()),
        "same_objective_max_abs_fractional_D_difference": float(df.rel_trf_same_minus_nm.abs().max()),
        "same_objective_median_abs_fractional_D_difference": float(df.rel_trf_same_minus_nm.abs().median()),
    }
    if do_flux_trf:
        summary.update(
            {
                "flux_trf_max_abs_fractional_D_difference_vs_nm": float(df.rel_flux_trf_minus_nm.abs().max()),
                "flux_trf_median_abs_fractional_D_difference_vs_nm": float(df.rel_flux_trf_minus_nm.abs().median()),
                "flux_trf_spearman_D_vs_nm": float(spearmanr(df.D_nm, df.D_flux_trf_evaluated_intrinsically)[0]),
                "flux_trf_pearson_logD_vs_nm": float(
                    pearsonr(
                        np.log10(df.D_nm),
                        np.log10(df.D_flux_trf_evaluated_intrinsically),
                    )[0]
                ),
            }
        )
    return summary


# =============================================================================
# Roman production NPZ discovery and axis normalization
# =============================================================================


ROMAN_D_ALIASES = ["D_ROMAN_EFF", "D_roman_eff", "D_ROMAN_W", "D_ROMAN"]
ROMAN_SUCCESS_ALIASES = ["SUCCESS", "success"]
ROMAN_MAG_ALIASES = [
    "w149_magnitudes",
    "W149_MAGNITUDES",
    "SOURCE_MAG_GRID",
    "SOURCE_MAGS",
    "MAGNITUDES",
    "MAGS",
    "MAG_GRID",
    "W149_MAG_GRID",
    "source_mags",
]
ROMAN_U0_ALIASES = ["U0_GRID", "u0_grid", "U0_VALUES", "u0_values", "U0"]
ROMAN_P_ALIASES = ["P_GRID", "P_grid", "P_DAYS_GRID", "P_VALUES", "P_days"]
ROMAN_POVERTE_ALIASES = ["P_OVER_TE_GRID", "P_OVER_TE", "P_over_tE_grid", "P_over_tE"]
ROMAN_BEST_TE_ALIASES = ["BEST_TE", "BEST_tE", "BEST_TE_GRID", "BEST_TE_DAYS"]
ROMAN_BEST_ALL_ALIASES = ["BEST_T0U0TE", "BEST_PARAMS", "BEST_PSPL"]


def discover_roman_npz(roman_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = []
    for path in sorted(roman_dir.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as d:
                dkey = first_existing_key(d.files, ROMAN_D_ALIASES)
                if dkey is None:
                    continue
                arr = np.asarray(d[dkey], dtype=float)
                nfinite = int(np.count_nonzero(np.isfinite(arr)))
                # Prefer large production files; weak penalty for obvious test/smoke names.
                penalty = 0.25 if any(s in path.name.lower() for s in ["test", "smoke"]) else 1.0
                candidates.append((nfinite * penalty, nfinite, path))
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"No NPZ containing one of {ROMAN_D_ALIASES} found in {roman_dir}"
        )

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]



def discover_roman_npzs(roman_dir: Path, explicit: str | None) -> list[Path]:
    """
    Return the Roman production NPZ files needed for the audit.

    The production currently stores one file per source magnitude,
    e.g. roman_intrinsic_grid_W149_19.npz, _21.npz, and _23.npz.
    Prefer that complete per-magnitude family when present.  Fall back
    to the older single-file discovery logic for a combined product.
    """
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]

    per_mag = []
    for path in sorted(roman_dir.glob("roman_intrinsic_grid_W149_*.npz")):
        if any(tag in path.name.lower() for tag in ["test", "smoke"]):
            continue
        try:
            with np.load(path, allow_pickle=False) as d:
                if first_existing_key(d.files, ROMAN_D_ALIASES) is not None:
                    per_mag.append(path)
        except Exception:
            continue

    if per_mag:
        return per_mag

    return [discover_roman_npz(roman_dir, explicit=None)]


def combine_roman_products(products: list[dict]) -> dict:
    """Combine one or more normalized Roman products along magnitude."""
    if not products:
        raise ValueError("No Roman products supplied")
    if len(products) == 1:
        out = dict(products[0])
        out["paths"] = [products[0]["path"]]
        return out

    ref = products[0]
    for prod in products[1:]:
        if len(prod["u0"]) != len(ref["u0"]) or not np.allclose(
            prod["u0"], ref["u0"], rtol=0, atol=1e-12
        ):
            raise ValueError(
                f"Roman u0 grids differ between {ref['path']} and {prod['path']}"
            )
        if len(prod["P_days"]) != len(ref["P_days"]) or not np.allclose(
            prod["P_days"], ref["P_days"], rtol=0, atol=1e-8
        ):
            raise ValueError(
                f"Roman P grids differ between {ref['path']} and {prod['path']}"
            )

    mags = np.concatenate([np.asarray(prod["mags"], dtype=float) for prod in products])
    D = np.concatenate([np.asarray(prod["D"], dtype=float) for prod in products], axis=0)
    success = np.concatenate(
        [np.asarray(prod["success"], dtype=bool) for prod in products], axis=0
    )

    if all(prod["best_te"] is not None for prod in products):
        best_te = np.concatenate(
            [np.asarray(prod["best_te"], dtype=float) for prod in products], axis=0
        )
        best_te_source = "+".join(
            sorted({str(prod["best_te_source"]) for prod in products})
        )
    else:
        best_te = None
        best_te_source = None

    order = np.argsort(mags)
    mags = mags[order]
    D = D[order]
    success = success[order]
    if best_te is not None:
        best_te = best_te[order]

    if len(np.unique(np.round(mags, 10))) != len(mags):
        raise ValueError(
            "Duplicate W149 magnitudes found while combining Roman products: "
            f"{mags.tolist()}"
        )

    return {
        "path": None,
        "paths": [prod["path"] for prod in products],
        "keys": sorted({key for prod in products for key in prod["keys"]}),
        "D_key": "+".join(sorted({str(prod["D_key"]) for prod in products})),
        "success_key": "+".join(
            sorted({str(prod["success_key"]) for prod in products})
        ),
        "mag_key": "+".join(sorted({str(prod["mag_key"]) for prod in products})),
        "u0_key": "+".join(sorted({str(prod["u0_key"]) for prod in products})),
        "P_key": "+".join(sorted({str(prod["P_key"]) for prod in products})),
        "best_te_source": best_te_source,
        "mags": mags,
        "u0": np.asarray(ref["u0"], dtype=float),
        "P_days": np.asarray(ref["P_days"], dtype=float),
        "D": D,
        "success": success,
        "best_te": best_te,
    }


def find_axis_permutation(shape, lengths):
    """Return permutation mapping current 3-D shape -> desired lengths."""
    if len(shape) != 3:
        raise ValueError(f"Expected a 3-D array, got shape={shape}")
    matches = []
    for perm in itertools.permutations(range(3)):
        if tuple(shape[p] for p in perm) == tuple(lengths):
            matches.append(perm)
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely infer axes: shape={shape}, desired lengths={lengths}, matches={matches}"
        )
    return matches[0]


def transpose_to_m_u_p(arr, nmag, nu0, np_):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D Roman array, got {arr.shape}")
    perm = find_axis_permutation(arr.shape, (nmag, nu0, np_))
    return np.transpose(arr, perm)


def load_roman_product(path: Path, grid: MainGrid):
    with np.load(path, allow_pickle=False) as d:
        files = list(d.files)
        dkey = first_existing_key(files, ROMAN_D_ALIASES)
        skey = first_existing_key(files, ROMAN_SUCCESS_ALIASES)
        mkey = first_existing_key(files, ROMAN_MAG_ALIASES)
        ukey = first_existing_key(files, ROMAN_U0_ALIASES)
        pkey = first_existing_key(files, ROMAN_P_ALIASES)
        pokey = first_existing_key(files, ROMAN_POVERTE_ALIASES)
        tekey = first_existing_key(files, ROMAN_BEST_TE_ALIASES)
        allkey = first_existing_key(files, ROMAN_BEST_ALL_ALIASES)

        if dkey is None:
            raise KeyError(f"No Roman D key found. Available={files}")
        D_raw = np.asarray(d[dkey], dtype=float)

        # Coordinates: prefer explicit arrays; infer only when safe.
        mags = np.asarray(d[mkey], dtype=float).ravel() if mkey else None
        u0 = np.asarray(d[ukey], dtype=float).ravel() if ukey else None

        if pkey:
            P_days = np.asarray(d[pkey], dtype=float).ravel()
        elif pokey:
            P_days = np.asarray(d[pokey], dtype=float).ravel() * grid.tE_true
        else:
            P_days = None

        if D_raw.ndim != 3:
            raise ValueError(
                f"Roman production array must be 3-D for this audit. {dkey} has shape {D_raw.shape}. "
                f"Available keys={files}"
            )

        # If coordinates are absent, infer from known production structure only when unambiguous.
        if mags is None:
            axes3 = [n for n in D_raw.shape if n == 3]
            if len(axes3) == 1:
                mags = np.array([19.0, 21.0, 23.0])
            else:
                raise KeyError(f"No magnitude coordinate key found. Available={files}")

        if u0 is None:
            candidates = [n for n in D_raw.shape if n <= len(grid.u0) and n != len(mags)]
            possible = [grid.u0[grid.u0 <= 1.0 + 1e-12]]
            if len(possible[0]) in D_raw.shape:
                u0 = possible[0]
            else:
                raise KeyError(f"No u0 coordinate key found. Available={files}")

        if P_days is None:
            if len(grid.P_days) in D_raw.shape:
                P_days = grid.P_days.copy()
            else:
                raise KeyError(f"No P coordinate key found. Available={files}")

        nmag, nu0, np_ = len(mags), len(u0), len(P_days)
        D = transpose_to_m_u_p(D_raw, nmag, nu0, np_)

        if skey:
            success_raw = np.asarray(d[skey], dtype=bool)
            success = transpose_to_m_u_p(success_raw, nmag, nu0, np_)
        else:
            success = np.isfinite(D)

        best_te = None
        best_te_source = None
        if tekey:
            best_te = transpose_to_m_u_p(np.asarray(d[tekey], dtype=float), nmag, nu0, np_)
            best_te_source = tekey
        elif allkey:
            allarr = np.asarray(d[allkey], dtype=float)
            # Common shape: (Nmag, Nu0, NP, 3), possibly with first 3 axes permuted.
            if allarr.ndim == 4 and allarr.shape[-1] >= 3:
                perm3 = find_axis_permutation(allarr.shape[:3], (nmag, nu0, np_))
                allarr = np.transpose(allarr, perm3 + (3,))
                best_te = allarr[..., 2]
                best_te_source = allkey + "[...,2]"

    return {
        "path": path,
        "keys": files,
        "D_key": dkey,
        "success_key": skey,
        "mag_key": mkey,
        "u0_key": ukey,
        "P_key": pkey or pokey,
        "best_te_source": best_te_source,
        "mags": mags,
        "u0": u0,
        "P_days": P_days,
        "D": D,
        "success": success,
        "best_te": best_te,
    }


# =============================================================================
# Correlation + Roman bound audit
# =============================================================================


def map_intrinsic_to_roman(grid: MainGrid, u0_roman, P_roman):
    out = np.full((len(u0_roman), len(P_roman)), np.nan, dtype=float)

    for i, u in enumerate(u0_roman):
        iu = int(np.argmin(np.abs(grid.u0 - u)))
        if not np.isclose(grid.u0[iu], u, rtol=1e-8, atol=1e-12):
            raise ValueError(f"Roman u0={u} not matched in intrinsic grid; nearest={grid.u0[iu]}")

        for j, P in enumerate(P_roman):
            ip = int(np.argmin(np.abs(grid.P_days - P)))
            if not np.isclose(grid.P_days[ip], P, rtol=1e-8, atol=1e-8):
                raise ValueError(f"Roman P={P} not matched in intrinsic grid; nearest={grid.P_days[ip]}")
            if grid.success[iu, ip] and np.isfinite(grid.D[iu, ip]) and grid.D[iu, ip] > 0:
                out[i, j] = grid.D[iu, ip]
    return out


def roman_correlation_and_bounds(roman_product, grid: MainGrid, output_dir: Path):
    Dintr = map_intrinsic_to_roman(grid, roman_product["u0"], roman_product["P_days"])

    corr_rows = []
    bound_rows = []

    for im, mag in enumerate(roman_product["mags"]):
        Dr = roman_product["D"][im]
        succ = roman_product["success"][im]
        mask = succ & np.isfinite(Dr) & (Dr > 0) & np.isfinite(Dintr) & (Dintr > 0)

        if np.count_nonzero(mask) >= 3:
            x = np.log10(Dintr[mask])
            y = np.log10(Dr[mask])
            pr = pearsonr(x, y)
            sr = spearmanr(x, y)
            corr_rows.append(
                {
                    "W149": float(mag),
                    "N": int(len(x)),
                    "pearson_r_log10": float(pr[0]),
                    "pearson_pvalue": float(pr[1]),
                    "spearman_rho": float(sr[0]),
                    "spearman_pvalue": float(sr[1]),
                }
            )

        best_te = roman_product["best_te"]
        if best_te is not None:
            te = best_te[im]
            valid = succ & np.isfinite(te)
            vals = te[valid]
            lo = 0.02 * grid.tE_true
            hi = 20.0 * grid.tE_true
            atol = 1e-6 * grid.tE_true
            if len(vals):
                bound_rows.append(
                    {
                        "W149": float(mag),
                        "N_valid": int(len(vals)),
                        "tE_min": float(np.min(vals)),
                        "tE_max": float(np.max(vals)),
                        "lower_bound": lo,
                        "upper_bound": hi,
                        "N_at_lower_bound": int(np.count_nonzero(np.isclose(vals, lo, rtol=0, atol=atol))),
                        "N_at_upper_bound": int(np.count_nonzero(np.isclose(vals, hi, rtol=0, atol=atol))),
                        "N_within_1pct_of_lower": int(np.count_nonzero(vals <= lo * 1.01)),
                        "N_within_1pct_of_upper": int(np.count_nonzero(vals >= hi * 0.99)),
                        "min_distance_to_any_bound_days": float(
                            np.min(np.minimum(np.abs(vals - lo), np.abs(hi - vals)))
                        ),
                    }
                )

    corr_df = pd.DataFrame(corr_rows)
    bound_df = pd.DataFrame(bound_rows)
    corr_df.to_csv(output_dir / "roman_correlations_pearson_spearman.csv", index=False)
    bound_df.to_csv(output_dir / "roman_tE_bound_audit.csv", index=False)

    section("ROMAN CORRELATIONS")
    if len(corr_df):
        print(corr_df.to_string(index=False))
    else:
        print("No valid correlation rows could be computed.")

    section("ROMAN tE BOUND AUDIT")
    if roman_product["best_te"] is None:
        print("No best-tE array found in the Roman NPZ; bound audit skipped.")
        print("Available keys:")
        print("  " + "\n  ".join(roman_product["keys"]))
    elif len(bound_df):
        print(bound_df.to_string(index=False))

    return {
        "correlations": corr_df.to_dict(orient="records"),
        "bounds": bound_df.to_dict(orient="records"),
    }


# =============================================================================
# Main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--main-grid",
        default="results/scan_many_tE_200x200/scan_u0_tE150/scan_kepler_u0_*.npz",
        help="Glob for final tE=150 one-luminous intrinsic production files.",
    )
    p.add_argument(
        "--two-luminous",
        default="results/final_a1b8a9b31002/qmass_qflux_tE150/summary_qM_qf.npz",
        help="Final two-luminous summary NPZ.",
    )
    p.add_argument(
        "--roman-dir",
        default="results/roman_asimov",
        help="Directory containing Roman production NPZ files.",
    )
    p.add_argument(
        "--roman-npz",
        default=None,
        help="Explicit Roman production NPZ. If omitted, auto-detect the largest production-like file.",
    )
    p.add_argument(
        "--output-dir",
        default="results/validation_optimizer_grid_audit",
    )
    p.add_argument("--n-stratified", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument(
        "--skip-flux-crosscheck",
        action="store_true",
        help="Skip the 30 pyLIMA flux-space TRF cross-pipeline fits.",
    )
    p.add_argument(
        "--skip-stratified-refits",
        action="store_true",
        help="Audit all stored points but skip fresh stratified re-fits.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "git_commit": git_commit(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "pandas": pd.__version__,
        "pyLIMA": package_version("pyLIMA"),
        "main_grid_pattern": args.main_grid,
        "roman_npz_requested": args.roman_npz,
        "seed": args.seed,
    }
    save_json(outdir / "provenance.json", provenance)

    section("LOAD FINAL ONE-LUMINOUS PRODUCTION GRID")
    grid = load_main_grid(args.main_grid)
    print(f"files       = {len(grid.files)}")
    print(f"shape       = {grid.D.shape}")
    print(f"u0 range    = {grid.u0.min():.6g} .. {grid.u0.max():.6g}")
    print(f"P range [d] = {grid.P_days.min():.6g} .. {grid.P_days.max():.6g}")
    print(f"t0, tE      = {grid.t0_true:.6g}, {grid.tE_true:.6g} d")
    print(f"N time      = {len(grid.t)}")

    grid_summary, jumps = audit_main_grid(grid, outdir)
    section("GRID-WIDE INTRINSIC AUDIT SUMMARY")
    for k, v in grid_summary.items():
        print(f"{k:45s}: {v}")

    two_summary = audit_summary_npz(Path(args.two_luminous))
    section("TWO-LUMINOUS PRODUCTION SUMMARY AUDIT")
    for k, v in two_summary.items():
        print(f"{k:35s}: {v}")

    optimizer_summary = run_optimizer_crosscheck(
        outdir,
        do_flux_trf=not args.skip_flux_crosscheck,
    )

    stratified_summary = None
    if not args.skip_stratified_refits:
        selected = select_stratified_points(grid, args.n_stratified, args.seed)
        selected.to_csv(outdir / "stratified_refit_selected_points.csv", index=False)
        stratified_summary = run_stratified_refits(grid, selected, outdir)

    roman_paths = discover_roman_npzs(Path(args.roman_dir), args.roman_npz)
    section("ROMAN PRODUCTION FILES")
    for path in roman_paths:
        print(path)

    roman_products = [load_roman_product(path, grid) for path in roman_paths]
    roman_product = combine_roman_products(roman_products)
    print(f"D key          = {roman_product['D_key']}")
    print(f"SUCCESS key    = {roman_product['success_key']}")
    print(f"magnitude key  = {roman_product['mag_key']}")
    print(f"u0 key         = {roman_product['u0_key']}")
    print(f"P key          = {roman_product['P_key']}")
    print(f"best tE source = {roman_product['best_te_source']}")
    print(f"magnitudes     = {roman_product['mags'].tolist()}")
    print(f"D shape        = {roman_product['D'].shape}")

    roman_summary = roman_correlation_and_bounds(roman_product, grid, outdir)

    final_summary = {
        "provenance": provenance,
        "intrinsic_grid_audit": grid_summary,
        "two_luminous_audit": two_summary,
        "optimizer_crosscheck": optimizer_summary,
        "stratified_refit_audit": stratified_summary,
        "roman_files": {
            "paths": [str(path) for path in roman_paths],
            "D_key": roman_product["D_key"],
            "success_key": roman_product["success_key"],
            "mag_key": roman_product["mag_key"],
            "u0_key": roman_product["u0_key"],
            "P_key": roman_product["P_key"],
            "best_te_source": roman_product["best_te_source"],
            "magnitudes": roman_product["mags"],
            "shape": roman_product["D"].shape,
        },
        "roman": roman_summary,
    }
    save_json(outdir / "summary.json", final_summary)

    section("DONE")
    print(f"Results written to: {outdir}")
    print()
    print("Main files to inspect:")
    for name in [
        "summary.json",
        "optimizer_crosscheck_5x6.csv",
        "stratified_refit_audit.csv",
        "roman_correlations_pearson_spearman.csv",
        "roman_tE_bound_audit.csv",
        "intrinsic_grid_all_points.csv",
        "intrinsic_grid_invalid_success_points.csv",
        "intrinsic_neighbor_jumps_top100.csv",
    ]:
        path = outdir / name
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    # pyLIMA/astropy may emit the known ERFA dubious-year warning for the
    # synthetic Roman epochs.  We leave warnings visible by default because
    # this is a validation script rather than a production runner.
    main()
