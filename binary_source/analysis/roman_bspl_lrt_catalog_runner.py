#!/usr/bin/env python3
"""
Catalog-driven Roman BSPL LRT reference runner.

For every numerical dataset, run exactly the same three observational fits:

    PSPL      : PSPL, no parallax, no double source
    PSPL_PI   : PSPL + Full parallax, no double source
    BSPL      : PSPL + Circular double source, no parallax

and construct

    Test A:
        raw_delta_chi2_A = chi2_PSPL - chi2_BSPL
        T_A = max(0, raw_delta_chi2_A)

    Test B:
        T_B = chi2_PSPL_PI - chi2_BSPL

T_B is signed. No Wilks p-value is computed here.

roman_rubin owns simulation/Event/model/TRF mechanics.
binary_source owns hypotheses, a common data-derived fit reference, starts, bounds and test statistics.

The fit-start policy is intentionally external and loaded from JSON so this
integration runner does not silently freeze the scientific search policy.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


ROMAN_RUBIN_REPO = Path(
    os.environ.get(
        "ROMAN_RUBIN_REPO",
        "/share/storage3/rubin/microlensing/romanrubin/roman_rubin",
    )
).resolve()

if not ROMAN_RUBIN_REPO.exists():
    raise RuntimeError(
        f"roman_rubin repository not found: {ROMAN_RUBIN_REPO}"
    )

sys.path.insert(0, str(ROMAN_RUBIN_REPO))

from catalog.event_reader import load_event_from_catalog  # noqa: E402
import fit_lc  # noqa: E402
import functions_roman_rubin as frr  # noqa: E402


EPHEMERIDES = (
    ROMAN_RUBIN_REPO
    / "ephemerides"
    / "Roman_positions_2027_2032_periodic_v1.npy"
)

EXPECTED_EPHEMERIS_SHA256 = (
    "eeb7bcf5300d2f3d4be3595898170975b9da20b5d47a26791af0c5258f307d58"
)

FIT_KEYS = ("PSPL", "PSPL_PI", "BSPL")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a) -> str:
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    h = hashlib.sha256()
    h.update(arr.dtype.str.encode("ascii"))
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def as_float(x) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got shape={arr.shape}")
    return float(arr.reshape(-1)[0])


def require_current_api() -> None:
    sim_sig = inspect.signature(frr.simulate_event_for_fit)
    fit_sig = inspect.signature(fit_lc.fit_rubin_roman)

    required_sim = {
        "truth_parallax",
        "truth_double_source",
        "add_photometric_noise",
    }
    required_fit = {
        "fit_parallax",
        "fit_double_source",
        "fit_bounds",
        "initial_guess",
        "optimizer_options",
        "event_ra",
        "event_dec",
    }

    missing_sim = required_sim - set(sim_sig.parameters)
    missing_fit = required_fit - set(fit_sig.parameters)

    if missing_sim:
        raise RuntimeError(
            "roman_rubin simulate_event_for_fit API is too old; "
            f"missing={sorted(missing_sim)}"
        )

    if missing_fit:
        raise RuntimeError(
            "roman_rubin fit_lc.fit_rubin_roman API is too old; "
            f"missing={sorted(missing_fit)}"
        )


def load_policy(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        policy = json.load(f)

    fit_reference = policy.get("fit_reference")
    if not isinstance(fit_reference, dict):
        raise KeyError("policy must contain fit_reference")

    for name in ("t0", "u0", "tE"):
        if name not in fit_reference:
            raise KeyError(f"fit_reference missing {name!r}")
        fit_reference[name] = float(fit_reference[name])

    missing = [key for key in FIT_KEYS if key not in policy]
    if missing:
        raise KeyError(
            "Fit policy is missing required model blocks: "
            f"{missing}"
        )

    for key in FIT_KEYS:
        starts = policy[key].get("starts", [])
        if not starts:
            raise ValueError(f"Fit policy {key!r} has no starts.")

        for i, start in enumerate(starts):
            if "initial_guess" not in start:
                raise KeyError(
                    f"{key}.starts[{i}] lacks initial_guess."
                )

    return policy


def truth_switches(truth_kind: str) -> tuple[bool, bool]:
    """Return (truth_parallax, truth_double_source)."""
    mapping = {
        "h0a": (False, False),
        "h0b": (True, False),
        "h1": (False, True),
    }
    return mapping[truth_kind]


def extract_fit_data(true_model) -> dict:
    lc_to_fit, _ = frr.extract_lightcurves_for_fit(true_model)

    roman = np.asarray(lc_to_fit["W149"], dtype=float)
    if roman.ndim != 2 or roman.shape[1] != 3:
        raise RuntimeError(
            "Expected W149 columns time, mag, err_mag; "
            f"got shape={roman.shape}"
        )
    if len(roman) == 0:
        raise RuntimeError("Roman W149 light curve is empty.")

    return {
        "W149": roman.copy(),
        "u": [],
        "g": [],
        "r": [],
        "i": [],
        "z": [],
        "y": [],
    }


def get_event_coordinates(true_model) -> tuple[float, float]:
    return float(true_model.event.ra), float(true_model.event.dec)


def best_model_vector(fit) -> np.ndarray:
    if "best_model" not in fit.fit_results:
        raise RuntimeError("Fit returned no best_model.")
    best = np.asarray(
        fit.fit_results["best_model"],
        dtype=float,
    ).reshape(-1)
    if not np.all(np.isfinite(best)):
        raise RuntimeError("Fit returned non-finite best_model.")
    return best


def run_one_fit(
    *,
    source_id: int,
    data: dict,
    ra: float,
    dec: float,
    fit_key: str,
    spec: dict,
    start: dict,
    fit_reference: dict,
    optimizer_options: dict | None,
) -> dict:
    """
    Run one TRF start through roman_rubin fit_lc.fit_rubin_roman.

    The fit event_params are the same data-derived fit_reference for every
    model family and every start. The simulation truth dictionary is never
    passed to the fitter. In particular, the parallax reference epoch is common.
    """
    fit_parallax = bool(spec.get("fit_parallax", False))
    fit_double_source = bool(spec.get("fit_double_source", False))

    initial = {
        str(k): float(v)
        for k, v in start["initial_guess"].items()
    }

    for required in ("t0", "u0", "tE"):
        if required not in initial:
            raise KeyError(
                f"{fit_key} start lacks required parameter {required!r}"
            )

    fit_event_params = {
        "t0": float(fit_reference["t0"]),
        "u0": float(fit_reference["u0"]),
        "tE": float(fit_reference["tE"]),
    }

    fit_defaults = spec.get("fit_defaults", None)

    # Circular BSPL extra parameters must exist when the model is built.
    # If no separate defaults block is supplied, initialize them from this
    # start. They remain free parameters.
    if fit_double_source and fit_defaults is None:
        fit_defaults = {
            k: v
            for k, v in initial.items()
            if k not in {"t0", "u0", "tE"}
        }

    fit_bounds = spec.get("bounds", None)
    if fit_bounds is None:
        raise ValueError(
            f"{fit_key}: explicit bounds are required by this project."
        )

    label = str(start.get("label", "start"))

    with tempfile.TemporaryDirectory(
        prefix=f"binary_source_{fit_key.lower()}_{source_id}_"
    ) as tmpdir:
        fit, event_fit, model_fit = fit_lc.fit_rubin_roman(
            Source=int(source_id),
            event_params=fit_event_params,
            path_save=Path(tmpdir),
            path_ephemerides=str(EPHEMERIDES),
            model="PSPL",
            algo="TRF",
            Origin=None,
            rango=0,
            wfirst_lc=data["W149"].copy(),
            lsst_u=[],
            lsst_g=[],
            lsst_r=[],
            lsst_i=[],
            lsst_z=[],
            lsst_y=[],
            fit_model="PSPL",
            fit_parallax=fit_parallax,
            fit_double_source=fit_double_source,
            fit_defaults=fit_defaults,
            fit_bounds=fit_bounds,
            random_state=None,
            initial_guess=initial,
            optimizer_options=optimizer_options,
            event_ra=float(ra),
            event_dec=float(dec),
        )

    best = best_model_vector(fit)
    chi2 = as_float(fit.fit_results["chi2"])

    model_ra = float(model_fit.event.ra)
    model_dec = float(model_fit.event.dec)
    if not (
        np.isclose(model_ra, ra, rtol=0.0, atol=1e-10)
        and np.isclose(model_dec, dec, rtol=0.0, atol=1e-10)
    ):
        raise RuntimeError(
            f"{fit_key}/{label}: fit coordinates changed. "
            f"expected=({ra},{dec}), got=({model_ra},{model_dec})"
        )

    expected_parallax = "Full" if fit_parallax else "None"
    got_parallax = getattr(model_fit, "parallax_model", [None])[0]
    if got_parallax != expected_parallax:
        raise RuntimeError(
            f"{fit_key}/{label}: unexpected parallax model "
            f"{got_parallax!r}; expected {expected_parallax!r}"
        )

    got_ds = getattr(model_fit, "double_source_model", None)
    if fit_double_source:
        if got_ds is None or got_ds[0] != "Circular":
            raise RuntimeError(
                f"{fit_key}/{label}: fit_double_source=True did not "
                "produce Circular double source."
            )
    else:
        if got_ds is not None and got_ds[0] not in {"None", None}:
            raise RuntimeError(
                f"{fit_key}/{label}: double source unexpectedly active: "
                f"{got_ds!r}"
            )

    fit_results = fit.fit_results

    return {
        "fit_key": fit_key,
        "start_label": label,
        "status": "fitted",
        "chi2": float(chi2),
        "best_model": [float(x) for x in best],
        "fit_parallax": fit_parallax,
        "fit_double_source": fit_double_source,
        "fit_reference": dict(fit_reference),
        "initial_guess": initial,
        "initial_guess_source": fit_results.get(
            "initial_guess_source"
        ),
        "initial_guess_parameter_order": fit_results.get(
            "initial_guess_parameter_order"
        ),
        "fit_bounds": fit_results.get("fit_bounds", fit_bounds),
        "optimizer_success": fit_results.get("optimizer_success"),
        "optimizer_status": fit_results.get("optimizer_status"),
        "optimizer_message": fit_results.get("optimizer_message"),
        "optimizer_nfev": fit_results.get("optimizer_nfev"),
    }


def run_fit_bank(
    *,
    source_id: int,
    data: dict,
    ra: float,
    dec: float,
    fit_key: str,
    spec: dict,
    fit_reference: dict,
    optimizer_options: dict | None,
) -> dict:
    results = []

    for start in spec["starts"]:
        try:
            row = run_one_fit(
                source_id=source_id,
                data=data,
                ra=ra,
                dec=dec,
                fit_key=fit_key,
                spec=spec,
                start=start,
                fit_reference=fit_reference,
                optimizer_options=optimizer_options,
            )
        except Exception as exc:
            row = {
                "fit_key": fit_key,
                "start_label": str(start.get("label", "start")),
                "status": "error",
                "error": repr(exc),
            }

        results.append(row)

    successful = [
        row
        for row in results
        if (
            row.get("status") == "fitted"
            and np.isfinite(row.get("chi2", np.nan))
        )
    ]

    if not successful:
        raise RuntimeError(f"All starts failed for {fit_key}.")

    best = min(successful, key=lambda row: row["chi2"])

    return {
        "best": best,
        "starts": results,
    }


def run_catalog_event(args) -> dict:
    require_current_api()

    if not EPHEMERIDES.exists():
        raise FileNotFoundError(EPHEMERIDES)

    ephem_sha = sha256_file(EPHEMERIDES)
    if ephem_sha != EXPECTED_EPHEMERIS_SHA256:
        raise RuntimeError(
            "Unexpected Roman ephemeris SHA256.\n"
            f"expected={EXPECTED_EPHEMERIS_SHA256}\n"
            f"actual  ={ephem_sha}"
        )

    policy = load_policy(args.fit_policy)
    fit_reference = dict(policy["fit_reference"])

    event = load_event_from_catalog(
        catalog_path=args.catalog,
        catalog_event_index=args.catalog_event_index,
    )

    simulation_seed = int(
        args.simulation_seed
        if args.simulation_seed is not None
        else event.get("simulation_seed", args.catalog_event_index)
    )

    truth_parallax, truth_double_source = truth_switches(
        args.truth_kind
    )

    np.random.seed(simulation_seed)

    true_model, true_params, decision = frr.simulate_event_for_fit(
        i=simulation_seed,
        event_params=event,
        path_ephemerides=str(EPHEMERIDES),
        model="PSPL",
        time_window=None,
        use_roman=True,
        use_rubin=False,
        truth_parallax=truth_parallax,
        truth_double_source=truth_double_source,
        add_photometric_noise=True,
        apply_detection_criteria=bool(
            args.apply_detection_criteria
        ),
        apply_photometric_filter=True,
    )

    if true_model is None:
        raise RuntimeError("roman_rubin returned no simulated model.")

    data = extract_fit_data(true_model)
    data_hash = sha256_array(data["W149"])
    ra, dec = get_event_coordinates(true_model)

    optimizer_options = policy.get("optimizer_options", None)

    fits = {}

    # Same exact numerical dataset for all three fit families.
    for fit_key in FIT_KEYS:
        before_hash = sha256_array(data["W149"])

        fits[fit_key] = run_fit_bank(
            source_id=simulation_seed,
            data=data,
            ra=ra,
            dec=dec,
            fit_key=fit_key,
            spec=policy[fit_key],
            fit_reference=fit_reference,
            optimizer_options=optimizer_options,
        )

        after_hash = sha256_array(data["W149"])

        if before_hash != data_hash or after_hash != data_hash:
            raise RuntimeError(
                f"Dataset changed during {fit_key} fit bank."
            )

    chi2_pspl = float(fits["PSPL"]["best"]["chi2"])
    chi2_pspl_pi = float(fits["PSPL_PI"]["best"]["chi2"])
    chi2_bspl = float(fits["BSPL"]["best"]["chi2"])

    raw_a = chi2_pspl - chi2_bspl
    t_a = max(0.0, raw_a)
    t_b = chi2_pspl_pi - chi2_bspl

    return {
        "runner_schema_version": "roman_bspl_lrt_catalog_runner_v1",
        "catalog_path": str(args.catalog),
        "catalog_event_index": int(args.catalog_event_index),
        "simulation_seed": int(simulation_seed),
        "truth_kind": args.truth_kind,
        "truth_parallax": bool(truth_parallax),
        "truth_double_source": bool(truth_double_source),
        "decision": bool(decision),
        "event_ra_deg": float(ra),
        "event_dec_deg": float(dec),
        "ephemeris_path": str(EPHEMERIDES),
        "ephemeris_sha256": ephem_sha,
        "n_roman_points": int(len(data["W149"])),
        "data_sha256": data_hash,
        "fit_reference": fit_reference,
        "fits": fits,
        "statistics": {
            "chi2_PSPL": chi2_pspl,
            "chi2_PSPL_PI": chi2_pspl_pi,
            "chi2_BSPL": chi2_bspl,
            "delta_chi2_A_raw": float(raw_a),
            "T_A": float(t_a),
            "T_B": float(t_b),
            "test_A_definition": (
                "T_A=max(0, chi2_PSPL-chi2_BSPL)"
            ),
            "test_B_definition": (
                "T_B=chi2_PSPL_PI-chi2_BSPL"
            ),
        },
        "fit_policy_path": str(args.fit_policy),
        "fit_policy_schema_version": policy.get("schema_version"),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--catalog-event-index", type=int, required=True)
    p.add_argument(
        "--truth-kind",
        choices=("h0a", "h0b", "h1"),
        required=True,
        help=(
            "h0a=PSPL/no-parallax; "
            "h0b=PSPL+parallax; "
            "h1=BSPL/no-parallax"
        ),
    )
    p.add_argument(
        "--fit-policy",
        type=Path,
        required=True,
        help=(
            "JSON defining data-driven starts/bounds for "
            "PSPL, PSPL_PI and BSPL."
        ),
    )
    p.add_argument("--simulation-seed", type=int, default=None)
    p.add_argument(
        "--apply-detection-criteria",
        action="store_true",
        help=(
            "Enable Roman detection selection. Leave off for wiring tests."
        ),
    )
    p.add_argument("--output", type=Path, required=True)

    return p


def main() -> None:
    args = build_parser().parse_args()

    payload = run_catalog_event(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )
        f.write("\n")

    s = payload["statistics"]

    print("=" * 90)
    print("ROMAN BSPL LRT CATALOG RUNNER")
    print("=" * 90)
    print("catalog_event_index =", payload["catalog_event_index"])
    print("truth_kind          =", payload["truth_kind"])
    print("seed                =", payload["simulation_seed"])
    print("N Roman             =", payload["n_roman_points"])
    print("data sha256         =", payload["data_sha256"])
    print()
    print("chi2 PSPL           =", s["chi2_PSPL"])
    print("chi2 PSPL+pi        =", s["chi2_PSPL_PI"])
    print("chi2 BSPL           =", s["chi2_BSPL"])
    print()
    print("Delta chi2 A raw    =", s["delta_chi2_A_raw"])
    print("T_A                 =", s["T_A"])
    print("T_B                 =", s["T_B"])
    print()
    print("output              =", args.output)


if __name__ == "__main__":
    main()
