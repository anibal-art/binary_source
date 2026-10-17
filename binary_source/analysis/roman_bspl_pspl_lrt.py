#!/usr/bin/env python3

"""
Roman BSPL-vs-PSPL likelihood-ratio test.

First implementation:
    H0 = PSPL
    H1 = physical circular BSPL family used by the paper

Free H1 physical parameters:
    t0, u0, tE, omega

At each omega,

    P = 2*pi/omega
    a_rel = Kepler(P, Mtot)
    xi_rel = a_rel / rEhat

so xi and omega remain physically coupled.

The orbital geometry, q_M and q_f are kept at the fiducial
paper values for this first calibrated LRT.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scipy.optimize import least_squares, lsq_linear

from pyLIMA.models import PSPL_model

from binary_source.analysis import (
    roman_bspl_pspl_asimov as roman
)

from binary_source.analysis import (
    roman_f146_backend as f146_backend
)


# ============================================================
# Array helpers
# ============================================================

def plain(x):
    if hasattr(x, "value"):
        x = x.value

    return np.asarray(
        x,
        dtype=float,
    )


# ============================================================
# PSPL magnification
# ============================================================

def pspl_magnification(
    t,
    t0,
    u0,
    tE,
):
    t = np.asarray(
        t,
        dtype=float,
    )

    tau = (
        (t - float(t0))
        / float(tE)
    )

    u = np.sqrt(
        float(u0)**2
        + tau**2
    )

    u = np.maximum(
        u,
        1.0e-12,
    )

    return (
        (u**2 + 2.0)
        /
        (
            u
            * np.sqrt(
                u**2 + 4.0
            )
        )
    )


# ============================================================
# Flux profiling
#
# blend_flux_parameter = "ftotal":
#
#     F(t) = Fs * (A - 1) + Ftotal
#
# Fs and Ftotal are linear nuisance parameters.
# ============================================================

def profile_flux(
    A,
    flux,
    err_flux,
):
    A = np.asarray(
        A,
        dtype=float,
    )

    flux = np.asarray(
        flux,
        dtype=float,
    )

    err_flux = np.asarray(
        err_flux,
        dtype=float,
    )

    if not (
        A.shape
        == flux.shape
        == err_flux.shape
    ):
        raise ValueError(
            "A, flux and err_flux must have "
            "identical shapes."
        )

    valid = (
        np.isfinite(A)
        & np.isfinite(flux)
        & np.isfinite(err_flux)
        & (err_flux > 0.0)
    )

    if np.count_nonzero(valid) < 3:
        raise RuntimeError(
            "Too few valid photometric points."
        )

    A = A[valid]
    y = flux[valid]
    sigma = err_flux[valid]

    x = A - 1.0

    w = 1.0 / sigma**2

    s_xx = np.sum(
        w * x * x
    )
    s_x = np.sum(
        w * x
    )
    s_1 = np.sum(
        w
    )

    s_yx = np.sum(
        w * y * x
    )
    s_y = np.sum(
        w * y
    )

    matrix = np.array(
        [
            [s_xx, s_x],
            [s_x, s_1],
        ],
        dtype=float,
    )

    rhs = np.array(
        [
            s_yx,
            s_y,
        ],
        dtype=float,
    )

    try:
        fs, ft = np.linalg.solve(
            matrix,
            rhs,
        )

    except np.linalg.LinAlgError:
        fs = np.nan
        ft = np.nan

    fmax = float(
        np.nanmax(y)
    )

    # Match the spirit of pyLIMA's non-negative
    # source/total-flux bounds.
    if (
        not np.isfinite(fs)
        or not np.isfinite(ft)
        or fs < 0.0
        or ft < 0.0
        or fs > fmax
        or ft > fmax
    ):
        design = np.column_stack(
            [
                x,
                np.ones_like(x),
            ]
        )

        weighted_design = (
            design
            / sigma[:, None]
        )

        weighted_y = (
            y / sigma
        )

        solution = lsq_linear(
            weighted_design,
            weighted_y,
            bounds=(
                [0.0, 0.0],
                [fmax, fmax],
            ),
            method="trf",
        )

        fs, ft = solution.x

    model_flux = (
        fs * x
        + ft
    )

    residual = (
        y - model_flux
    ) / sigma

    chi2 = float(
        np.sum(
            residual**2
        )
    )

    return {
        "fs": float(fs),
        "ft": float(ft),
        "chi2": chi2,
        "residual": residual,
    }


# ============================================================
# Physical BSPL evaluator
# ============================================================

class PhysicalCircularBSPL:

    def __init__(
        self,
        t,
        t0_reference,
    ):
        self.t = np.asarray(
            t,
            dtype=float,
        )

        self.t0_reference = float(
            t0_reference
        )

        ev, tel = (
            roman.build_truth_event(
                self.t
            )
        )

        self.event = ev
        self.telescope = tel

        self.model = (
            PSPL_model.PSPLmodel(
                ev,
                parallax=[
                    "None",
                    self.t0_reference,
                ],
                double_source=[
                    "Circular",
                    self.t0_reference,
                ],
            )
        )

        self.model.define_model_parameters()

    def magnification(
        self,
        t0,
        u0,
        tE,
        omega,
    ):
        omega = float(
            omega
        )

        if omega <= 0.0:
            raise ValueError(
                "omega must be positive."
            )

        P_days = (
            2.0
            * np.pi
            / omega
        )

        a_rel_AU = float(
            roman.a_from_P_kepler_days(
                P_days,
                roman.MTOT_MSUN,
            )
        )

        xi_rel = (
            a_rel_AU
            / roman.REHAT_AU
        )

        xi_para = (
            xi_rel
            * np.cos(
                roman.THETA_TRUE
            )
        )

        xi_perp = (
            xi_rel
            * np.sin(
                roman.THETA_TRUE
            )
        )

        params = [
            float(t0),
            float(u0),
            float(tE),

            float(xi_para),
            float(xi_perp),
            float(omega),

            float(roman.PHI_TRUE),
            float(
                roman.INCLINATION_TRUE
            ),

            float(roman.Q_MASS),
            float(roman.QFLUX_TRUE),

            1.0,
            1.0,
        ]

        py_params = (
            self.model.compute_pyLIMA_parameters(
                params
            )
        )

        A = (
            self.model.model_magnification(
                self.telescope,
                py_params,
            )
        )

        A = plain(A)

        A = (
            A
            / (
                1.0
                + roman.QFLUX_TRUE
            )
        )

        if np.any(
            ~np.isfinite(A)
        ):
            raise RuntimeError(
                "Non-finite BSPL magnification."
            )

        if np.any(
            A <= 0.0
        ):
            raise RuntimeError(
                "Non-positive BSPL magnification."
            )

        return A


# ============================================================
# Data realization
# ============================================================

def make_noisy_dataset(
    t,
    mag_truth,
    err_mag,
    seed,
):
    mag_truth = np.asarray(
        mag_truth,
        dtype=float,
    )

    err_mag = np.asarray(
        err_mag,
        dtype=float,
    )

    rng = np.random.default_rng(
        int(seed)
    )

    mag_obs = (
        mag_truth
        + rng.normal(
            loc=0.0,
            scale=err_mag,
            size=len(mag_truth),
        )
    )

    ev, tel = (
        roman.build_event_from_magnitudes(
            t=t,
            magnitude=mag_obs,
            err_mag=err_mag,
            name="Roman",
        )
    )

    flux = plain(
        tel.lightcurve["flux"]
    )

    err_flux = plain(
        tel.lightcurve["err_flux"]
    )

    return {
        "event": ev,
        "telescope": tel,
        "mag_obs": mag_obs,
        "flux": flux,
        "err_flux": err_flux,
    }


# ============================================================
# Common nonlinear parameterization
#
# x[0] = (t0 - t0_ref) / tE_ref
# x[1] = u0
# x[2] = log(tE/tE_ref)
# ============================================================

def unpack_shape_parameters(
    x,
    t0_ref,
    tE_ref,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    t0 = (
        float(t0_ref)
        + x[0]
        * float(tE_ref)
    )

    u0 = float(
        x[1]
    )

    tE = (
        float(tE_ref)
        * np.exp(
            x[2]
        )
    )

    return t0, u0, tE


def shape_bounds():
    return (
        np.array(
            [
                -2.0,
                -20.0,
                np.log(0.02),
            ],
            dtype=float,
        ),
        np.array(
            [
                2.0,
                20.0,
                np.log(20.0),
            ],
            dtype=float,
        ),
    )


# ============================================================
# H0 fit
# ============================================================

def fit_h0_pspl(
    t,
    flux,
    err_flux,
    t0_ref,
    u0_guess,
    tE_ref,
):
    lower, upper = (
        shape_bounds()
    )

    x0 = np.array(
        [
            0.0,
            float(u0_guess),
            0.0,
        ],
        dtype=float,
    )

    def residuals(x):
        t0, u0, tE = (
            unpack_shape_parameters(
                x,
                t0_ref,
                tE_ref,
            )
        )

        A = pspl_magnification(
            t,
            t0,
            u0,
            tE,
        )

        return profile_flux(
            A,
            flux,
            err_flux,
        )["residual"]

    opt = least_squares(
        residuals,
        x0,
        bounds=(
            lower,
            upper,
        ),
        method="trf",
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=5000,
    )

    t0, u0, tE = (
        unpack_shape_parameters(
            opt.x,
            t0_ref,
            tE_ref,
        )
    )

    A = pspl_magnification(
        t,
        t0,
        u0,
        tE,
    )

    prof = profile_flux(
        A,
        flux,
        err_flux,
    )

    return {
        "success": bool(
            opt.success
        ),
        "chi2": float(
            prof["chi2"]
        ),
        "t0": float(t0),
        "u0": float(u0),
        "tE": float(tE),
        "fs": prof["fs"],
        "ft": prof["ft"],
        "nfev": int(opt.nfev),
        "x": np.asarray(
            opt.x,
            dtype=float,
        ),
    }


# ============================================================
# H1 fit at fixed omega
# ============================================================

def fit_h1_fixed_omega(
    evaluator,
    flux,
    err_flux,
    omega,
    h0_fit,
    t0_ref,
    tE_ref,
):
    lower, upper = (
        shape_bounds()
    )

    x0 = np.asarray(
        h0_fit["x"],
        dtype=float,
    ).copy()

    def residuals(x):
        t0, u0, tE = (
            unpack_shape_parameters(
                x,
                t0_ref,
                tE_ref,
            )
        )

        A = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            omega=omega,
        )

        return profile_flux(
            A,
            flux,
            err_flux,
        )["residual"]

    opt = least_squares(
        residuals,
        x0,
        bounds=(
            lower,
            upper,
        ),
        method="trf",
        xtol=1.0e-9,
        ftol=1.0e-9,
        gtol=1.0e-9,
        max_nfev=3000,
    )

    t0, u0, tE = (
        unpack_shape_parameters(
            opt.x,
            t0_ref,
            tE_ref,
        )
    )

    A = evaluator.magnification(
        t0=t0,
        u0=u0,
        tE=tE,
        omega=omega,
    )

    prof = profile_flux(
        A,
        flux,
        err_flux,
    )

    P_days = (
        2.0
        * np.pi
        / float(omega)
    )

    return {
        "success": bool(
            opt.success
        ),
        "chi2": float(
            prof["chi2"]
        ),
        "omega": float(omega),
        "P_days": float(P_days),
        "t0": float(t0),
        "u0": float(u0),
        "tE": float(tE),
        "fs": prof["fs"],
        "ft": prof["ft"],
        "nfev": int(opt.nfev),
    }


def fit_h1_scan(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    h0_fit,
    t0_ref,
    tE_ref,
):
    fits = []

    for omega in omega_grid:
        try:
            result = (
                fit_h1_fixed_omega(
                    evaluator=evaluator,
                    flux=flux,
                    err_flux=err_flux,
                    omega=float(omega),
                    h0_fit=h0_fit,
                    t0_ref=t0_ref,
                    tE_ref=tE_ref,
                )
            )

        except Exception as exc:
            result = {
                "success": False,
                "chi2": np.inf,
                "omega": float(omega),
                "P_days": float(
                    2.0
                    * np.pi
                    / omega
                ),
                "error": repr(exc),
            }

        fits.append(
            result
        )

    finite = [
        x
        for x in fits
        if (
            x.get(
                "success",
                False,
            )
            and np.isfinite(
                x["chi2"]
            )
        )
    ]

    if len(finite) == 0:
        raise RuntimeError(
            "All H1 omega fits failed."
        )

    best = min(
        finite,
        key=lambda x: x["chi2"],
    )

    return best, fits


# ============================================================
# Continuous omega profiling
#
# The fixed-omega grid is used only to identify basins.
# The final H1 likelihood is continuously optimized in omega.
#
# We optimize log(omega) for numerical stability.
# ============================================================

def fit_h1_free_omega(
    evaluator,
    flux,
    err_flux,
    start_fit,
    t0_ref,
    tE_ref,
    omega_min,
    omega_max,
):
    lower_shape, upper_shape = (
        shape_bounds()
    )

    omega_min = float(
        omega_min
    )

    omega_max = float(
        omega_max
    )

    if not (
        0.0
        < omega_min
        < omega_max
    ):
        raise ValueError(
            "Invalid omega bounds."
        )

    x0 = np.array(
        [
            (
                start_fit["t0"]
                - float(t0_ref)
            )
            / float(tE_ref),

            float(
                start_fit["u0"]
            ),

            np.log(
                float(
                    start_fit["tE"]
                )
                / float(tE_ref)
            ),

            np.log(
                float(
                    start_fit["omega"]
                )
            ),
        ],
        dtype=float,
    )

    lower = np.concatenate(
        [
            lower_shape,
            [
                np.log(
                    omega_min
                )
            ],
        ]
    )

    upper = np.concatenate(
        [
            upper_shape,
            [
                np.log(
                    omega_max
                )
            ],
        ]
    )

    # Guard against tiny floating-point excursions.
    x0 = np.maximum(
        x0,
        lower + 1.0e-12,
    )

    x0 = np.minimum(
        x0,
        upper - 1.0e-12,
    )

    def residuals(x):

        t0, u0, tE = (
            unpack_shape_parameters(
                x[:3],
                t0_ref,
                tE_ref,
            )
        )

        omega = float(
            np.exp(
                x[3]
            )
        )

        A = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            omega=omega,
        )

        return profile_flux(
            A,
            flux,
            err_flux,
        )["residual"]

    opt = least_squares(
        residuals,
        x0,
        bounds=(
            lower,
            upper,
        ),
        method="trf",
        xtol=1.0e-9,
        ftol=1.0e-9,
        gtol=1.0e-9,
        max_nfev=5000,
    )

    t0, u0, tE = (
        unpack_shape_parameters(
            opt.x[:3],
            t0_ref,
            tE_ref,
        )
    )

    omega = float(
        np.exp(
            opt.x[3]
        )
    )

    A = evaluator.magnification(
        t0=t0,
        u0=u0,
        tE=tE,
        omega=omega,
    )

    prof = profile_flux(
        A,
        flux,
        err_flux,
    )

    P_days = (
        2.0
        * np.pi
        / omega
    )

    return {
        "success": bool(
            opt.success
        ),
        "chi2": float(
            prof["chi2"]
        ),
        "omega": omega,
        "P_days": float(
            P_days
        ),
        "t0": float(t0),
        "u0": float(u0),
        "tE": float(tE),
        "fs": prof["fs"],
        "ft": prof["ft"],
        "nfev": int(
            opt.nfev
        ),
        "refined_omega": True,
    }


def fit_h1_profile_omega(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    h0_fit,
    t0_ref,
    tE_ref,
    n_refine=6,
):
    """
    Global-ish H1 profile:

    1. Fit t0/u0/tE at every fixed omega node.
    2. Identify local minima of chi2(omega).
    3. Use the best minima as starting points.
    4. Jointly optimize t0/u0/tE/log(omega).
    """

    omega_grid = np.asarray(
        omega_grid,
        dtype=float,
    )

    scan_best, scan = (
        fit_h1_scan(
            evaluator=evaluator,
            flux=flux,
            err_flux=err_flux,
            omega_grid=omega_grid,
            h0_fit=h0_fit,
            t0_ref=t0_ref,
            tE_ref=tE_ref,
        )
    )

    chi = np.array(
        [
            (
                row["chi2"]
                if (
                    row.get(
                        "success",
                        False,
                    )
                    and np.isfinite(
                        row["chi2"]
                    )
                )
                else np.inf
            )
            for row in scan
        ],
        dtype=float,
    )

    candidate_indices = []

    n = len(
        chi
    )

    # Local minima, including boundaries.
    for i in range(n):

        if not np.isfinite(
            chi[i]
        ):
            continue

        left = (
            chi[i - 1]
            if i > 0
            else np.inf
        )

        right = (
            chi[i + 1]
            if i < n - 1
            else np.inf
        )

        if (
            chi[i] <= left
            and chi[i] <= right
        ):
            candidate_indices.append(
                i
            )

    candidate_indices = sorted(
        candidate_indices,
        key=lambda i: chi[i],
    )

    # Fill with globally best nodes if necessary.
    for i in np.argsort(
        chi
    ):

        i = int(i)

        if not np.isfinite(
            chi[i]
        ):
            continue

        if (
            i
            not in candidate_indices
        ):
            candidate_indices.append(
                i
            )

        if (
            len(candidate_indices)
            >= int(n_refine)
        ):
            break

    candidate_indices = (
        candidate_indices[
            : int(n_refine)
        ]
    )

    omega_min = float(
        np.min(
            omega_grid
        )
    )

    omega_max = float(
        np.max(
            omega_grid
        )
    )

    refined = []

    for i in candidate_indices:

        start = scan[i]

        try:

            fit = (
                fit_h1_free_omega(
                    evaluator=evaluator,
                    flux=flux,
                    err_flux=err_flux,
                    start_fit=start,
                    t0_ref=t0_ref,
                    tE_ref=tE_ref,
                    omega_min=omega_min,
                    omega_max=omega_max,
                )
            )

        except Exception as exc:

            fit = {
                "success": False,
                "chi2": np.inf,
                "omega": float(
                    start["omega"]
                ),
                "P_days": float(
                    start["P_days"]
                ),
                "error": repr(exc),
                "refined_omega": True,
            }

        refined.append(
            fit
        )

    valid_refined = [
        row
        for row in refined
        if (
            row.get(
                "success",
                False,
            )
            and np.isfinite(
                row["chi2"]
            )
        )
    ]

    # Keep the discrete solution as a fallback. A continuous
    # refinement should never make the global result worse.
    candidates = (
        [scan_best]
        + valid_refined
    )

    best = min(
        candidates,
        key=lambda row: row[
            "chi2"
        ],
    )

    return (
        best,
        scan,
        refined,
    )



# ============================================================
# H0 truth
# ============================================================

def make_h0_truth(
    t,
    t0_true,
    u0_true,
    tE_true,
    source_mag,
    blend_ratio=0.0,
):
    """
    Construct the deterministic PSPL truth and the fixed Roman
    instrumental observation mask.

    The saturation/validity decision is made ONCE from the
    truth light curve, before adding noise.  H0 and H1 are then
    fitted to exactly the same retained epochs.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    A = pspl_magnification(
        t=t,
        t0=t0_true,
        u0=u0_true,
        tE=tE_true,
    )

    blend_ratio = float(
        blend_ratio
    )

    if (
        not np.isfinite(
            blend_ratio
        )
        or blend_ratio < 0.0
    ):
        raise ValueError(
            "blend_ratio = Fb/Fs must be finite and >= 0."
        )

    # source_mag is the unmagnified SOURCE magnitude.
    #
    # With b = Fb/Fs,
    #
    #     F(t) / Fs = A(t) + b
    #
    # so the true observed magnitude, including the blend,
    # must be passed to Pandeia before computing uncertainties
    # and saturation.
    mag_truth = (
        float(source_mag)
        - 2.5
        * np.log10(
            A + blend_ratio
        )
    )

    baseline_mag = (
        float(source_mag)
        - 2.5
        * np.log10(
            1.0 + blend_ratio
        )
    )

    instrument = (
        f146_backend.evaluate_f146_pandeia_ab(
            mag_truth
        )
    )

    err_mag = np.asarray(
        instrument["sigma_mag"],
        dtype=float,
    )

    valid = np.asarray(
        instrument["valid"],
        dtype=bool,
    )

    full_saturated = np.asarray(
        instrument["full_saturated"],
        dtype=bool,
    )

    partial_saturated = np.asarray(
        instrument["partial_saturated"],
        dtype=bool,
    )

    keep = (
        valid
        & np.isfinite(err_mag)
        & (err_mag > 0.0)
    )

    n_total = int(
        len(t)
    )

    n_used = int(
        np.count_nonzero(
            keep
        )
    )

    if n_used < 10:
        raise RuntimeError(
            "Too few valid Roman F146 epochs after "
            "instrumental saturation masking: "
            f"{n_used}/{n_total}"
        )

    return {
        "A":
            np.asarray(
                A,
                dtype=float,
            ),

        "blend_ratio":
            float(
                blend_ratio
            ),

        "source_mag":
            float(
                source_mag
            ),

        "baseline_mag":
            float(
                baseline_mag
            ),

        "mag_truth":
            np.asarray(
                mag_truth,
                dtype=float,
            ),

        "err_mag":
            err_mag,

        "keep_mask":
            keep,

        "valid":
            valid,

        "full_saturated":
            full_saturated,

        "partial_saturated":
            partial_saturated,

        "n_epochs_total":
            n_total,

        "n_epochs_used":
            n_used,

        "n_epochs_rejected":
            int(
                n_total
                - n_used
            ),

        "n_full_saturated":
            int(
                np.count_nonzero(
                    full_saturated
                )
            ),

        "n_partial_saturated":
            int(
                np.count_nonzero(
                    partial_saturated
                )
            ),

        "fraction_rejected":
            float(
                1.0
                - n_used
                / n_total
            ),
    }


def prepare_h0_configuration(
    t,
    t0_true,
    u0_true,
    tE_true,
    source_mag,
    evaluator=None,
    blend_ratio=0.0,
):
    """
    Prepare all deterministic quantities shared by every
    bootstrap realization of one H0 configuration.

    This includes:
        - PSPL truth
        - Pandeia saturation mask
        - retained timestamps
        - retained uncertainties
        - BSPL evaluator on exactly those timestamps

    The returned object can be cached and reused for all seeds.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    truth = make_h0_truth(
        t=t,
        t0_true=t0_true,
        u0_true=u0_true,
        tE_true=tE_true,
        source_mag=source_mag,
        blend_ratio=blend_ratio,
    )

    keep = np.asarray(
        truth["keep_mask"],
        dtype=bool,
    )

    t_used = t[
        keep
    ]

    mag_truth_used = (
        truth["mag_truth"][
            keep
        ]
    )

    err_mag_used = (
        truth["err_mag"][
            keep
        ]
    )

    evaluator_ok = False

    if evaluator is not None:

        try:

            evaluator_t = np.asarray(
                evaluator.t,
                dtype=float,
            )

            evaluator_ok = (
                evaluator_t.shape
                == t_used.shape
                and np.array_equal(
                    evaluator_t,
                    t_used,
                )
            )

        except Exception:
            evaluator_ok = False

    if not evaluator_ok:

        evaluator = (
            PhysicalCircularBSPL(
                t=t_used,
                t0_reference=t0_true,
            )
        )

    return {
        "truth": truth,
        "t": t_used,
        "mag_truth": mag_truth_used,
        "err_mag": err_mag_used,
        "evaluator": evaluator,
    }


# ============================================================
# One null realization
# ============================================================

def run_h0_realization(
    t,
    t0_true,
    u0_true,
    tE_true,
    source_mag,
    seed,
    evaluator,
    omega_grid,
    validate_pylima=False,
    prepared=None,
):
    """
    Run one parametric-bootstrap realization under H0.

    The Pandeia observation mask is deterministic and is
    derived from the truth light curve before noise is added.
    Both H0 and H1 use exactly the same retained epochs.
    """

    if prepared is None:

        prepared = (
            prepare_h0_configuration(
                t=t,
                t0_true=t0_true,
                u0_true=u0_true,
                tE_true=tE_true,
                source_mag=source_mag,
                evaluator=evaluator,
            )
        )

    truth = prepared[
        "truth"
    ]

    t_used = np.asarray(
        prepared["t"],
        dtype=float,
    )

    evaluator_used = (
        prepared["evaluator"]
    )

    data = make_noisy_dataset(
        t=t_used,
        mag_truth=prepared[
            "mag_truth"
        ],
        err_mag=prepared[
            "err_mag"
        ],
        seed=seed,
    )

    h0 = fit_h0_pspl(
        t=t_used,
        flux=data["flux"],
        err_flux=data["err_flux"],
        t0_ref=t0_true,
        u0_guess=u0_true,
        tE_ref=tE_true,
    )

    (
        h1_best,
        h1_scan,
        h1_refined,
    ) = fit_h1_profile_omega(
        evaluator=evaluator_used,
        flux=data["flux"],
        err_flux=data["err_flux"],
        omega_grid=omega_grid,
        h0_fit=h0,
        t0_ref=t0_true,
        tE_ref=tE_true,
        n_refine=6,
    )

    raw_delta = (
        h0["chi2"]
        - h1_best["chi2"]
    )

    # H1* = H0 union the searched BSPL family.
    T = max(
        0.0,
        raw_delta,
    )

    result = {
        "seed":
            int(seed),

        "source_mag":
            float(
                source_mag
            ),

        "u0_true":
            float(
                u0_true
            ),

        "tE_true":
            float(
                tE_true
            ),

        "n_epochs_total":
            int(
                truth[
                    "n_epochs_total"
                ]
            ),

        "n_epochs_used":
            int(
                truth[
                    "n_epochs_used"
                ]
            ),

        "n_epochs_rejected":
            int(
                truth[
                    "n_epochs_rejected"
                ]
            ),

        "n_full_saturated":
            int(
                truth[
                    "n_full_saturated"
                ]
            ),

        "n_partial_saturated":
            int(
                truth[
                    "n_partial_saturated"
                ]
            ),

        "fraction_rejected":
            float(
                truth[
                    "fraction_rejected"
                ]
            ),

        "chi2_h0":
            float(
                h0["chi2"]
            ),

        "chi2_h1_bspl":
            float(
                h1_best["chi2"]
            ),

        "delta_chi2_raw":
            float(
                raw_delta
            ),

        "T":
            float(T),

        "best_P_days":
            float(
                h1_best[
                    "P_days"
                ]
            ),

        "best_omega":
            float(
                h1_best[
                    "omega"
                ]
            ),

        "h0_t0":
            float(
                h0["t0"]
            ),

        "h0_u0":
            float(
                h0["u0"]
            ),

        "h0_tE":
            float(
                h0["tE"]
            ),

        "h1_t0":
            float(
                h1_best[
                    "t0"
                ]
            ),

        "h1_u0":
            float(
                h1_best[
                    "u0"
                ]
            ),

        "h1_tE":
            float(
                h1_best[
                    "tE"
                ]
            ),

        "h0_success":
            bool(
                h0["success"]
            ),

        "h1_success":
            bool(
                h1_best[
                    "success"
                ]
            ),
    }

    if validate_pylima:

        pyfit = (
            roman.fit_pspl_roman(
                ev=data[
                    "event"
                ],
                t0_guess=t0_true,
                u0_guess=u0_true,
                tE_guess=tE_true,
            )
        )

        result[
            "chi2_h0_pylima"
        ] = float(
            pyfit["chi2"]
        )

        result[
            "chi2_h0_profiled_minus_pylima"
        ] = float(
            h0["chi2"]
            - pyfit["chi2"]
        )

    return result


# ============================================================
# Empirical threshold
# ============================================================

def empirical_threshold(
    values,
    alpha,
):
    values = np.sort(
        np.asarray(
            values,
            dtype=float,
        )
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n == 0:
        return np.nan

    # Conservative finite-sample empirical upper quantile.
    k = int(
        np.ceil(
            (1.0 - alpha)
            * (n + 1)
        )
    )

    k = min(
        max(k, 1),
        n,
    )

    return float(
        values[k - 1]
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-null",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--source-mag",
        type=float,
        default=21.0,
    )

    parser.add_argument(
        "--u0",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--tE",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--P-min",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--P-max",
        type=float,
        default=1.0e5,
    )

    parser.add_argument(
        "--n-omega",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--seed0",
        type=int,
        default=20261016,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/roman_lrt/"
            "h0_smoke.csv"
        ),
    )

    args = parser.parse_args()

    roman.set_roman_noise_backend(
        "pandeia"
    )

    t, t0_true = (
        roman.build_roman_times(
            tE_true=args.tE,
            anchor_season_index=2,
            include_off_seasons=True,
        )
    )

    # Period nodes are logarithmic, as in the physical scans.
    P_grid = np.logspace(
        np.log10(
            args.P_min
        ),
        np.log10(
            args.P_max
        ),
        args.n_omega,
    )

    omega_grid = (
        2.0
        * np.pi
        / P_grid
    )

    evaluator = (
        PhysicalCircularBSPL(
            t=t,
            t0_reference=t0_true,
        )
    )

    rows = []

    print()
    print("=" * 88)
    print("ROMAN BSPL-vs-PSPL H0 LRT SMOKE")
    print("=" * 88)
    print(
        "N epochs       =",
        len(t),
    )
    print(
        "F146 AB        =",
        args.source_mag,
    )
    print(
        "u0 true        =",
        args.u0,
    )
    print(
        "tE true [d]    =",
        args.tE,
    )
    print(
        "omega nodes    =",
        args.n_omega,
    )
    print(
        "P range [d]    =",
        args.P_min,
        args.P_max,
    )
    print(
        "null draws     =",
        args.n_null,
    )
    print("=" * 88)

    for i in range(
        args.n_null
    ):
        seed = (
            args.seed0
            + i
        )

        result = (
            run_h0_realization(
                t=t,
                t0_true=t0_true,
                u0_true=args.u0,
                tE_true=args.tE,
                source_mag=args.source_mag,
                seed=seed,
                evaluator=evaluator,
                omega_grid=omega_grid,
                validate_pylima=(
                    i == 0
                ),
            )
        )

        rows.append(
            result
        )

        print(
            f"[{i+1:03d}/{args.n_null:03d}] "
            f"seed={seed}  "
            f"chi2_H0={result['chi2_h0']:.3f}  "
            f"chi2_BSPL={result['chi2_h1_bspl']:.3f}  "
            f"T={result['T']:.6f}  "
            f"Pbest={result['best_P_days']:.3g} d"
        )

        if (
            i == 0
            and "chi2_h0_pylima"
            in result
        ):
            print(
                "    H0 validation: "
                f"profiled={result['chi2_h0']:.9e}  "
                f"pyLIMA={result['chi2_h0_pylima']:.9e}  "
                f"delta="
                f"{result['chi2_h0_profiled_minus_pylima']:.3e}"
            )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
        }
    )

    with args.output.open(
        "w",
        newline="",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    T = np.array(
        [
            row["T"]
            for row in rows
        ],
        dtype=float,
    )

    print()
    print("=" * 88)
    print("NULL SUMMARY")
    print("=" * 88)

    print(
        "T min/median/max =",
        f"{np.min(T):.6g}",
        f"{np.median(T):.6g}",
        f"{np.max(T):.6g}",
    )

    print(
        "fraction T=0      =",
        f"{np.mean(T == 0.0):.4f}",
    )

    for alpha in [
        0.10,
        0.05,
        0.01,
    ]:
        c = empirical_threshold(
            T,
            alpha,
        )

        tail = np.mean(
            T > c
        )

        print(
            f"alpha={alpha:.3f}  "
            f"c_alpha={c:.6g}  "
            f"empirical strict tail={tail:.4f}"
        )

    metadata = {
        "noise_backend": "pandeia",
        "hypothesis_h0": "PSPL",
        "hypothesis_h1": (
            "physical circular BSPL, "
            "fixed geometry/qM/qf, "
            "profiled over t0/u0/tE/omega; coarse omega grid used as multistart"
        ),
        "source_mag_ab": args.source_mag,
        "u0_true": args.u0,
        "tE_true_days": args.tE,
        "P_min_days": args.P_min,
        "P_max_days": args.P_max,
        "n_omega": args.n_omega,
        "n_null": args.n_null,
        "seed0": args.seed0,
    }

    metadata_path = (
        args.output.with_suffix(
            ".json"
        )
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "CSV      =",
        args.output,
    )
    print(
        "metadata =",
        metadata_path,
    )


if __name__ == "__main__":
    main()
