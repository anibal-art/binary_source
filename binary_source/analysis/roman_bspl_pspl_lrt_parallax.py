"""
Roman BSPL-vs-PSPL LRT with Roman/L2 parallax.

This module is deliberately additive: it does not modify the validated
no-parallax LRT implementation.

First controlled experiment:
    truth: piEN = piEE = 0
    H0:    PSPL + parallax
    H1:    circular BSPL + parallax

The Roman spacecraft ephemeris is loaded directly from the roman_rubin
repository. Telescope source/total fluxes remain profiled linear nuisance
parameters, exactly as in the current LRT reduced problem.
"""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from scipy.optimize import least_squares

from pyLIMA.models import PSPL_model

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base,
)
from binary_source.analysis import (
    roman_bspl_pspl_lrt_levels as levels,
)


# ======================================================================
# Configuration
# ======================================================================

PIE_MIN = -0.5
PIE_MAX = +0.5

DEFAULT_RR_ROOTS = (
    Path(
        "/share/storage3/rubin/microlensing/romanrubin/"
        "roman_rubin_lrt_pandeia_20261016"
    ),
    Path(
        "/export/storage3/rubin/microlensing/romanrubin/"
        "roman_rubin_lrt_pandeia_20261016"
    ),
)


# ======================================================================
# Roman ephemeris
# ======================================================================

def resolve_roman_ephemerides(
    path=None,
):
    """
    Resolve the CURRENT Roman ephemeris from the roman_rubin repo.

    Priority:
      1. explicit path
      2. ROMAN_EPHEMERIDES environment variable
      3. ROMAN_RUBIN_ROOT/ephemerides/Roman_positions.npy
      4. known CHE roman_rubin roots
    """

    candidates = []

    if path is not None:
        candidates.append(
            Path(path)
        )

    env_eph = os.environ.get(
        "ROMAN_EPHEMERIDES"
    )

    if env_eph:
        candidates.append(
            Path(env_eph)
        )

    env_root = os.environ.get(
        "ROMAN_RUBIN_ROOT"
    )

    if env_root:
        candidates.append(
            Path(env_root)
            / "ephemerides"
            / "Roman_positions.npy"
        )

    for root in DEFAULT_RR_ROOTS:
        candidates.append(
            root
            / "ephemerides"
            / "Roman_positions.npy"
        )

    for candidate in candidates:

        candidate = (
            candidate.expanduser()
            .resolve()
        )

        if candidate.is_file():
            return candidate

    attempted = "\n".join(
        str(x)
        for x in candidates
    )

    raise FileNotFoundError(
        "Could not resolve Roman ephemerides. "
        "Attempted:\n"
        + attempted
    )


@lru_cache(maxsize=8)
def _load_ephemerides_cached(
    path_string,
):

    path = Path(
        path_string
    )

    eph = np.asarray(
        np.load(path),
        dtype=float,
    )

    if (
        eph.ndim != 2
        or eph.shape[1] != 4
    ):
        raise ValueError(
            "Roman ephemerides must have shape (N, 4): "
            "[JD, RA, Dec, distance_AU]. "
            f"Got {eph.shape}."
        )

    if not np.all(
        np.isfinite(eph)
    ):
        raise ValueError(
            "Roman ephemerides contain non-finite values."
        )

    if np.any(
        np.diff(
            eph[:, 0]
        )
        <= 0.0
    ):
        raise ValueError(
            "Roman ephemeris JD column must be strictly increasing."
        )

    return eph


def load_roman_ephemerides(
    path=None,
):

    resolved = (
        resolve_roman_ephemerides(
            path
        )
    )

    eph = (
        _load_ephemerides_cached(
            str(resolved)
        )
    )

    return resolved, eph


# ======================================================================
# pyLIMA event with Roman/L2 geometry
# ======================================================================

def build_roman_parallax_event(
    t,
    ephemerides_path=None,
    name="Roman_parallax",
):
    """
    Build the same Roman event geometry used by the LRT, but attach the
    externally maintained Roman/L2 ephemeris before constructing a
    pyLIMA Full-parallax model.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    ev, tel = (
        base.roman.build_truth_event(
            t
        )
    )

    tel.name = str(name)

    path, eph = (
        load_roman_ephemerides(
            ephemerides_path
        )
    )

    if getattr(
        tel,
        "location",
        None,
    ) != "Space":
        raise RuntimeError(
            "Roman parallax telescope must have location='Space'."
        )

    # Same convention already used in roman_rubin.
    tel.spacecraft_name = "L2"

    tel.spacecraft_positions = {
        "astrometry": [],
        "photometry": eph,
    }

    return ev, tel, path


# ======================================================================
# Helpers
# ======================================================================

def _plain(
    value,
):

    return np.asarray(
        getattr(
            value,
            "value",
            value,
        ),
        dtype=float,
    )


def _check_magnification(
    A,
    label,
):

    A = _plain(A)

    if np.any(
        ~np.isfinite(A)
    ):
        raise RuntimeError(
            f"{label}: non-finite magnification."
        )

    if np.any(
        A <= 0.0
    ):
        raise RuntimeError(
            f"{label}: non-positive magnification."
        )

    return A


# ======================================================================
# PSPL + parallax evaluator
# ======================================================================

class PSPLParallaxEvaluator:

    def __init__(
        self,
        t,
        t0_reference,
        ephemerides_path=None,
    ):

        self.t = np.asarray(
            t,
            dtype=float,
        )

        self.t0_reference = float(
            t0_reference
        )

        (
            self.event,
            self.telescope,
            self.ephemerides_path,
        ) = build_roman_parallax_event(
            self.t,
            ephemerides_path=ephemerides_path,
            name="Roman_PSPL_parallax",
        )

        self.model = (
            PSPL_model.PSPLmodel(
                self.event,
                blend_flux_parameter="ftotal",
                parallax=[
                    "Full",
                    self.t0_reference,
                ],
                double_source=[
                    "None",
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
        piEN,
        piEE,
    ):

        # Parameter order verified against pyLIMA:
        # t0, u0, tE, piEN, piEE, fsource, ftotal.
        params = [
            float(t0),
            float(u0),
            float(tE),
            float(piEN),
            float(piEE),

            # Dummy fluxes. They do not enter model_magnification.
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

        return _check_magnification(
            A,
            "PSPL+parallax",
        )


# ======================================================================
# Circular BSPL + parallax evaluator
# ======================================================================

class PhysicalCircularBSPLParallax:

    def __init__(
        self,
        t,
        t0_reference,
        ephemerides_path=None,
    ):

        self.t = np.asarray(
            t,
            dtype=float,
        )

        self.t0_reference = float(
            t0_reference
        )

        (
            self.event,
            self.telescope,
            self.ephemerides_path,
        ) = build_roman_parallax_event(
            self.t,
            ephemerides_path=ephemerides_path,
            name="Roman_BSPL_parallax",
        )

        self.model = (
            PSPL_model.PSPLmodel(
                self.event,
                blend_flux_parameter="ftotal",
                parallax=[
                    "Full",
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
        piEN,
        piEE,
        omega,
        q_mass=None,
        q_flux=None,
        theta=None,
        phi=None,
        inclination=None,
    ):

        fid = (
            levels.fiducial_values()
        )

        if q_mass is None:
            q_mass = fid[
                "q_mass"
            ]

        if q_flux is None:
            q_flux = fid[
                "q_flux"
            ]

        if theta is None:
            theta = fid[
                "theta"
            ]

        if phi is None:
            phi = fid[
                "phi"
            ]

        if inclination is None:
            inclination = fid[
                "inclination"
            ]

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
            base.roman.a_from_P_kepler_days(
                P_days,
                base.roman.MTOT_MSUN,
            )
        )

        xi_rel = (
            a_rel_AU
            / base.roman.REHAT_AU
        )

        xi_para = (
            xi_rel
            * np.cos(
                float(theta)
            )
        )

        xi_perp = (
            xi_rel
            * np.sin(
                float(theta)
            )
        )

        # Parameter order verified against pyLIMA:
        #
        # t0, u0, tE,
        # piEN, piEE,
        # xi_para, xi_perp, omega, phi, inclination,
        # q_mass, q_flux,
        # fsource, ftotal.
        params = [
            float(t0),
            float(u0),
            float(tE),

            float(piEN),
            float(piEE),

            float(xi_para),
            float(xi_perp),
            float(omega),
            float(phi),
            float(inclination),
            float(q_mass),
            float(q_flux),

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

        A = _plain(A)

        # pyLIMA circular double-source model returns
        # A1 + q_flux * A2.
        A = (
            A
            / (
                1.0
                + float(q_flux)
            )
        )

        return _check_magnification(
            A,
            "BSPL+parallax",
        )


# ======================================================================
# H0: PSPL + parallax
# ======================================================================

def fit_h0_pspl_parallax(
    evaluator,
    flux,
    err_flux,
    t0_ref,
    u0_guess,
    tE_ref,
    piEN_start=0.0,
    piEE_start=0.0,
):

    lower_shape, upper_shape = (
        base.shape_bounds()
    )

    x0 = np.array(
        [
            0.0,
            float(u0_guess),
            0.0,
            float(piEN_start),
            float(piEE_start),
        ],
        dtype=float,
    )

    lower = np.concatenate(
        [
            np.asarray(
                lower_shape,
                dtype=float,
            ),
            [
                PIE_MIN,
                PIE_MIN,
            ],
        ]
    )

    upper = np.concatenate(
        [
            np.asarray(
                upper_shape,
                dtype=float,
            ),
            [
                PIE_MAX,
                PIE_MAX,
            ],
        ]
    )

    x0 = np.clip(
        x0,
        lower + 1.0e-10,
        upper - 1.0e-10,
    )

    def decode(x):

        t0, u0, tE = (
            base.unpack_shape_parameters(
                x[:3],
                t0_ref,
                tE_ref,
            )
        )

        return (
            float(t0),
            float(u0),
            float(tE),
            float(x[3]),
            float(x[4]),
        )

    def residuals(x):

        (
            t0,
            u0,
            tE,
            piEN,
            piEE,
        ) = decode(x)

        A = evaluator.magnification(
            t0=t0,
            u0=u0,
            tE=tE,
            piEN=piEN,
            piEE=piEE,
        )

        # Same reduced/profiled flux problem used by the
        # validated LRT and by the TRF/polyfit strategy:
        # F = fs * (A - 1) + ftotal.
        return base.profile_flux(
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
        max_nfev=10000,
    )

    (
        t0,
        u0,
        tE,
        piEN,
        piEE,
    ) = decode(
        opt.x
    )

    A = evaluator.magnification(
        t0=t0,
        u0=u0,
        tE=tE,
        piEN=piEN,
        piEE=piEE,
    )

    prof = base.profile_flux(
        A,
        flux,
        err_flux,
    )

    return {
        "success":
            bool(opt.success),

        "chi2":
            float(
                prof["chi2"]
            ),

        "t0":
            t0,

        "u0":
            u0,

        "tE":
            tE,

        "piEN":
            piEN,

        "piEE":
            piEE,

        "fs":
            float(
                prof["fs"]
            ),

        "ft":
            float(
                prof["ft"]
            ),

        "x":
            np.asarray(
                opt.x,
                dtype=float,
            ),

        "nfev":
            int(
                opt.nfev
            ),
    }


# ======================================================================
# H1 optimization helpers
# ======================================================================

def _fit_h1_fixed_omega_parallax(
    evaluator,
    flux,
    err_flux,
    omega,
    start_fit,
    t0_ref,
    tE_ref,
):

    fid = (
        levels.fiducial_values()
    )

    lower_shape, upper_shape = (
        base.shape_bounds()
    )

    x0 = np.array(
        [
            (
                float(
                    start_fit["t0"]
                )
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

            float(
                start_fit.get(
                    "piEN",
                    0.0,
                )
            ),

            float(
                start_fit.get(
                    "piEE",
                    0.0,
                )
            ),
        ],
        dtype=float,
    )

    lower = np.concatenate(
        [
            np.asarray(
                lower_shape,
                dtype=float,
            ),
            [
                PIE_MIN,
                PIE_MIN,
            ],
        ]
    )

    upper = np.concatenate(
        [
            np.asarray(
                upper_shape,
                dtype=float,
            ),
            [
                PIE_MAX,
                PIE_MAX,
            ],
        ]
    )

    x0 = np.clip(
        x0,
        lower + 1.0e-10,
        upper - 1.0e-10,
    )

    def decode(x):

        t0, u0, tE = (
            base.unpack_shape_parameters(
                x[:3],
                t0_ref,
                tE_ref,
            )
        )

        return {
            "t0":
                float(t0),

            "u0":
                float(u0),

            "tE":
                float(tE),

            "piEN":
                float(x[3]),

            "piEE":
                float(x[4]),
        }

    def residuals(x):

        p = decode(x)

        A = evaluator.magnification(
            t0=p["t0"],
            u0=p["u0"],
            tE=p["tE"],
            piEN=p["piEN"],
            piEE=p["piEE"],
            omega=omega,
            q_mass=fid["q_mass"],
            q_flux=fid["q_flux"],
            theta=fid["theta"],
            phi=fid["phi"],
            inclination=fid[
                "inclination"
            ],
        )

        return base.profile_flux(
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
        max_nfev=10000,
    )

    p = decode(
        opt.x
    )

    A = evaluator.magnification(
        t0=p["t0"],
        u0=p["u0"],
        tE=p["tE"],
        piEN=p["piEN"],
        piEE=p["piEE"],
        omega=omega,
        q_mass=fid["q_mass"],
        q_flux=fid["q_flux"],
        theta=fid["theta"],
        phi=fid["phi"],
        inclination=fid[
            "inclination"
        ],
    )

    prof = base.profile_flux(
        A,
        flux,
        err_flux,
    )

    return {
        **p,

        "success":
            bool(opt.success),

        "chi2":
            float(
                prof["chi2"]
            ),

        "omega":
            float(omega),

        "P_days":
            float(
                2.0
                * np.pi
                / float(omega)
            ),

        "q_mass":
            float(
                fid["q_mass"]
            ),

        "q_flux":
            float(
                fid["q_flux"]
            ),

        "theta":
            float(
                fid["theta"]
            ),

        "phi":
            float(
                fid["phi"]
            ),

        "inclination":
            float(
                fid["inclination"]
            ),

        "fs":
            float(
                prof["fs"]
            ),

        "ft":
            float(
                prof["ft"]
            ),

        "nfev":
            int(
                opt.nfev
            ),
    }


def fit_h1_joint_parallax(
    evaluator,
    flux,
    err_flux,
    start_fit,
    h1_level,
    t0_ref,
    tE_ref,
    omega_min,
    omega_max,
    q_mass_start=None,
    q_flux_start=None,
):
    """
    Joint H1 fit for ladder levels 1--3 with piEN/piEE shared
    as free nuisance parameters.
    """

    levels.validate_level(
        h1_level
    )

    if (
        levels.level_number(
            h1_level
        )
        > 3
    ):
        raise NotImplementedError(
            "Parallax implementation currently supports "
            "the production ladder through level 3."
        )

    fid = (
        levels.fiducial_values()
    )

    lower_shape, upper_shape = (
        base.shape_bounds()
    )

    if q_mass_start is None:
        q_mass_start = float(
            start_fit.get(
                "q_mass",
                fid["q_mass"],
            )
        )

    if q_flux_start is None:
        q_flux_start = float(
            start_fit.get(
                "q_flux",
                fid["q_flux"],
            )
        )

    x0 = [
        (
            float(
                start_fit["t0"]
            )
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

        float(
            start_fit.get(
                "piEN",
                0.0,
            )
        ),

        float(
            start_fit.get(
                "piEE",
                0.0,
            )
        ),

        np.log(
            float(
                start_fit["omega"]
            )
        ),
    ]

    lower = list(
        lower_shape
    )

    upper = list(
        upper_shape
    )

    lower.extend(
        [
            PIE_MIN,
            PIE_MIN,
            np.log(
                float(
                    omega_min
                )
            ),
        ]
    )

    upper.extend(
        [
            PIE_MAX,
            PIE_MAX,
            np.log(
                float(
                    omega_max
                )
            ),
        ]
    )

    if (
        levels.level_number(
            h1_level
        )
        >= 2
    ):

        x0.append(
            float(
                q_mass_start
            )
        )

        lower.append(
            levels.QMASS_MIN
        )

        upper.append(
            levels.QMASS_MAX
        )

    if (
        levels.level_number(
            h1_level
        )
        >= 3
    ):

        x0.append(
            float(
                q_flux_start
            )
        )

        lower.append(
            levels.QFLUX_MIN
        )

        upper.append(
            levels.QFLUX_MAX
        )

    x0 = np.asarray(
        x0,
        dtype=float,
    )

    lower = np.asarray(
        lower,
        dtype=float,
    )

    upper = np.asarray(
        upper,
        dtype=float,
    )

    x0 = np.clip(
        x0,
        lower + 1.0e-10,
        upper - 1.0e-10,
    )

    def decode(x):

        t0, u0, tE = (
            base.unpack_shape_parameters(
                x[:3],
                t0_ref,
                tE_ref,
            )
        )

        piEN = float(
            x[3]
        )

        piEE = float(
            x[4]
        )

        omega = float(
            np.exp(
                x[5]
            )
        )

        index = 6

        q_mass = float(
            fid["q_mass"]
        )

        q_flux = float(
            fid["q_flux"]
        )

        if (
            levels.level_number(
                h1_level
            )
            >= 2
        ):

            q_mass = float(
                x[index]
            )

            index += 1

        if (
            levels.level_number(
                h1_level
            )
            >= 3
        ):

            q_flux = float(
                x[index]
            )

        return {
            "t0":
                float(t0),

            "u0":
                float(u0),

            "tE":
                float(tE),

            "piEN":
                piEN,

            "piEE":
                piEE,

            "omega":
                omega,

            "q_mass":
                q_mass,

            "q_flux":
                q_flux,

            "theta":
                float(
                    fid["theta"]
                ),

            "phi":
                float(
                    fid["phi"]
                ),

            "inclination":
                float(
                    fid[
                        "inclination"
                    ]
                ),
        }

    def residuals(x):

        p = decode(x)

        A = evaluator.magnification(
            **p
        )

        return base.profile_flux(
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
        max_nfev=15000,
    )

    p = decode(
        opt.x
    )

    A = evaluator.magnification(
        **p
    )

    prof = base.profile_flux(
        A,
        flux,
        err_flux,
    )

    return {
        **p,

        "success":
            bool(opt.success),

        "chi2":
            float(
                prof["chi2"]
            ),

        "P_days":
            float(
                2.0
                * np.pi
                / p["omega"]
            ),

        "fs":
            float(
                prof["fs"]
            ),

        "ft":
            float(
                prof["ft"]
            ),

        "nfev":
            int(
                opt.nfev
            ),

        "h1_level":
            h1_level,
    }


# ======================================================================
# Ladder levels 1--3
# ======================================================================

def _valid_fit(
    row,
):
    return (
        bool(
            row.get(
                "success",
                False,
            )
        )
        and np.isfinite(
            row.get(
                "chi2",
                np.inf,
            )
        )
    )


def _select_best_starts(
    rows,
    n,
):

    valid = [
        row
        for row in rows
        if _valid_fit(
            row
        )
    ]

    valid = sorted(
        valid,
        key=lambda row:
            float(
                row["chi2"]
            ),
    )

    out = []
    seen = set()

    for row in valid:

        key = (
            round(
                np.log(
                    float(
                        row["omega"]
                    )
                ),
                8,
            ),
            round(
                float(
                    row.get(
                        "piEN",
                        0.0,
                    )
                ),
                7,
            ),
            round(
                float(
                    row.get(
                        "piEE",
                        0.0,
                    )
                ),
                7,
            ),
            round(
                float(
                    row.get(
                        "q_mass",
                        0.0,
                    )
                ),
                7,
            ),
            round(
                float(
                    row.get(
                        "q_flux",
                        0.0,
                    )
                ),
                7,
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(row)

        if len(out) >= int(n):
            break

    return out


def fit_level_1_parallax(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    h0_fit,
    t0_ref,
    tE_ref,
    n_joint_starts=6,
):

    fixed_fits = []

    for omega in omega_grid:

        try:

            fit = (
                _fit_h1_fixed_omega_parallax(
                    evaluator=evaluator,
                    flux=flux,
                    err_flux=err_flux,
                    omega=float(
                        omega
                    ),
                    start_fit=h0_fit,
                    t0_ref=t0_ref,
                    tE_ref=tE_ref,
                )
            )

        except Exception as exc:

            fit = {
                "success":
                    False,
                "chi2":
                    np.inf,
                "omega":
                    float(
                        omega
                    ),
                "error":
                    repr(exc),
            }

        fixed_fits.append(
            fit
        )

    starts = (
        _select_best_starts(
            fixed_fits,
            n_joint_starts,
        )
    )

    if not starts:
        raise RuntimeError(
            "No valid level-1 parallax scan fits."
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

    joint_fits = []

    for start in starts:

        try:

            fit = (
                fit_h1_joint_parallax(
                    evaluator=evaluator,
                    flux=flux,
                    err_flux=err_flux,
                    start_fit=start,
                    h1_level=levels.LEVEL_1,
                    t0_ref=t0_ref,
                    tE_ref=tE_ref,
                    omega_min=omega_min,
                    omega_max=omega_max,
                )
            )

            joint_fits.append(
                fit
            )

        except Exception:
            pass

    candidates = (
        fixed_fits
        + joint_fits
    )

    valid = [
        row
        for row in candidates
        if _valid_fit(
            row
        )
    ]

    best = min(
        valid,
        key=lambda row:
            float(
                row["chi2"]
            ),
    )

    best = dict(
        best
    )

    best["h1_level"] = (
        levels.LEVEL_1
    )

    return (
        best,
        {
            "fits":
                valid,

            "scan":
                fixed_fits,

            "joint_refinements":
                joint_fits,
        },
    )


def fit_level_2_parallax(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    t0_ref,
    tE_ref,
    previous_profile,
    n_previous_starts=6,
):

    prev_best, prev_diag = (
        previous_profile
    )

    starts = (
        _select_best_starts(
            prev_diag["fits"],
            n_previous_starts,
        )
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

    fits = []

    for start in starts:

        for q_mass in (
            levels.QMASS_STARTS
        ):

            try:

                fit = (
                    fit_h1_joint_parallax(
                        evaluator=evaluator,
                        flux=flux,
                        err_flux=err_flux,
                        start_fit=start,
                        h1_level=levels.LEVEL_2,
                        t0_ref=t0_ref,
                        tE_ref=tE_ref,
                        omega_min=omega_min,
                        omega_max=omega_max,
                        q_mass_start=q_mass,
                    )
                )

                fits.append(
                    fit
                )

            except Exception:
                pass

    fallback = dict(
        prev_best
    )

    fallback[
        "h1_level"
    ] = levels.LEVEL_2

    candidates = [
        fallback
    ] + [
        row
        for row in fits
        if _valid_fit(
            row
        )
    ]

    best = min(
        candidates,
        key=lambda row:
            float(
                row["chi2"]
            ),
    )

    return (
        best,
        {
            "fits":
                candidates,

            "new_fits":
                fits,

            "previous_best":
                prev_best,
        },
    )


def fit_level_3_parallax(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    t0_ref,
    tE_ref,
    previous_profile,
    n_previous_starts=6,
):

    prev_best, prev_diag = (
        previous_profile
    )

    starts = (
        _select_best_starts(
            prev_diag["fits"],
            n_previous_starts,
        )
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

    fits = []

    for start in starts:

        for q_flux in (
            levels.QFLUX_STARTS
        ):

            try:

                fit = (
                    fit_h1_joint_parallax(
                        evaluator=evaluator,
                        flux=flux,
                        err_flux=err_flux,
                        start_fit=start,
                        h1_level=levels.LEVEL_3,
                        t0_ref=t0_ref,
                        tE_ref=tE_ref,
                        omega_min=omega_min,
                        omega_max=omega_max,
                        q_mass_start=start.get(
                            "q_mass",
                            base.roman.Q_MASS,
                        ),
                        q_flux_start=q_flux,
                    )
                )

                fits.append(
                    fit
                )

            except Exception:
                pass

    fallback = dict(
        prev_best
    )

    fallback[
        "h1_level"
    ] = levels.LEVEL_3

    candidates = [
        fallback
    ] + [
        row
        for row in fits
        if _valid_fit(
            row
        )
    ]

    best = min(
        candidates,
        key=lambda row:
            float(
                row["chi2"]
            ),
    )

    return (
        best,
        {
            "fits":
                candidates,

            "new_fits":
                fits,

            "previous_best":
                prev_best,
        },
    )


def fit_h1_ladder_parallax(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    h0_fit,
    t0_ref,
    tE_ref,
    max_level=levels.LEVEL_3,
):

    if (
        levels.level_number(
            max_level
        )
        > 3
    ):
        raise NotImplementedError(
            "This first parallax implementation supports "
            "the production ladder through level 3."
        )

    profiles = {}

    profiles[
        levels.LEVEL_1
    ] = fit_level_1_parallax(
        evaluator=evaluator,
        flux=flux,
        err_flux=err_flux,
        omega_grid=omega_grid,
        h0_fit=h0_fit,
        t0_ref=t0_ref,
        tE_ref=tE_ref,
    )

    if (
        levels.level_number(
            max_level
        )
        >= 2
    ):

        profiles[
            levels.LEVEL_2
        ] = fit_level_2_parallax(
            evaluator=evaluator,
            flux=flux,
            err_flux=err_flux,
            omega_grid=omega_grid,
            t0_ref=t0_ref,
            tE_ref=tE_ref,
            previous_profile=profiles[
                levels.LEVEL_1
            ],
        )

    if (
        levels.level_number(
            max_level
        )
        >= 3
    ):

        profiles[
            levels.LEVEL_3
        ] = fit_level_3_parallax(
            evaluator=evaluator,
            flux=flux,
            err_flux=err_flux,
            omega_grid=omega_grid,
            t0_ref=t0_ref,
            tE_ref=tE_ref,
            previous_profile=profiles[
                levels.LEVEL_2
            ],
        )

    return profiles


# ======================================================================
# One complete H0 realization: truth piE=0, fits piE free
# ======================================================================

def run_h0_lrt_ladder_parallax(
    t,
    t0_true,
    u0_true,
    tE_true,
    source_mag,
    seed,
    omega_grid,
    blend_ratio=0.0,
    ephemerides_path=None,
    max_level=levels.LEVEL_3,
):

    prepared = (
        base.prepare_h0_configuration(
            t=t,
            t0_true=t0_true,
            u0_true=u0_true,
            tE_true=tE_true,
            source_mag=source_mag,
            evaluator=None,
            blend_ratio=blend_ratio,
        )
    )

    t_used = np.asarray(
        prepared["t"],
        dtype=float,
    )

    data = (
        base.make_noisy_dataset(
            t=t_used,
            mag_truth=prepared[
                "mag_truth"
            ],
            err_mag=prepared[
                "err_mag"
            ],
            seed=seed,
        )
    )

    h0_evaluator = (
        PSPLParallaxEvaluator(
            t=t_used,
            t0_reference=t0_true,
            ephemerides_path=ephemerides_path,
        )
    )

    h1_evaluator = (
        PhysicalCircularBSPLParallax(
            t=t_used,
            t0_reference=t0_true,
            ephemerides_path=ephemerides_path,
        )
    )

    h0 = (
        fit_h0_pspl_parallax(
            evaluator=h0_evaluator,
            flux=data["flux"],
            err_flux=data[
                "err_flux"
            ],
            t0_ref=t0_true,
            u0_guess=u0_true,
            tE_ref=tE_true,
            piEN_start=0.0,
            piEE_start=0.0,
        )
    )

    profiles = (
        fit_h1_ladder_parallax(
            evaluator=h1_evaluator,
            flux=data["flux"],
            err_flux=data[
                "err_flux"
            ],
            omega_grid=omega_grid,
            h0_fit=h0,
            t0_ref=t0_true,
            tE_ref=tE_true,
            max_level=max_level,
        )
    )

    rows = []

    for level in levels.LEVELS:

        if level not in profiles:
            continue

        best = (
            profiles[level][0]
        )

        raw_delta = (
            float(
                h0["chi2"]
            )
            - float(
                best["chi2"]
            )
        )

        T = max(
            0.0,
            raw_delta,
        )

        rows.append(
            {
                "seed":
                    int(seed),

                "h1_level":
                    level,

                "level_number":
                    int(
                        levels.level_number(
                            level
                        )
                    ),

                "blend_ratio":
                    float(
                        blend_ratio
                    ),

                # First controlled experiment:
                # truth parallax is exactly zero.
                "true_piEN":
                    0.0,

                "true_piEE":
                    0.0,

                "chi2_h0":
                    float(
                        h0["chi2"]
                    ),

                "chi2_h1":
                    float(
                        best["chi2"]
                    ),

                "delta_chi2_raw":
                    float(
                        raw_delta
                    ),

                "T":
                    float(T),

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

                "h0_piEN":
                    float(
                        h0["piEN"]
                    ),

                "h0_piEE":
                    float(
                        h0["piEE"]
                    ),

                "h1_t0":
                    float(
                        best["t0"]
                    ),

                "h1_u0":
                    float(
                        best["u0"]
                    ),

                "h1_tE":
                    float(
                        best["tE"]
                    ),

                "h1_piEN":
                    float(
                        best["piEN"]
                    ),

                "h1_piEE":
                    float(
                        best["piEE"]
                    ),

                "best_P_days":
                    float(
                        best["P_days"]
                    ),

                "best_omega":
                    float(
                        best["omega"]
                    ),

                "best_q_mass":
                    float(
                        best["q_mass"]
                    ),

                "best_q_flux":
                    float(
                        best["q_flux"]
                    ),

                "h0_fs":
                    float(
                        h0["fs"]
                    ),

                "h0_ft":
                    float(
                        h0["ft"]
                    ),

                "h1_fs":
                    float(
                        best["fs"]
                    ),

                "h1_ft":
                    float(
                        best["ft"]
                    ),
            }
        )

    # Ladder nesting is still mandatory.
    for i in range(
        1,
        len(rows),
    ):

        if (
            rows[i][
                "chi2_h1"
            ]
            >
            rows[i - 1][
                "chi2_h1"
            ]
            + 1.0e-8
        ):
            raise RuntimeError(
                "Parallax ladder nesting violation."
            )

        if (
            rows[i]["T"]
            <
            rows[i - 1]["T"]
            - 1.0e-8
        ):
            raise RuntimeError(
                "Parallax ladder T monotonicity violation."
            )

    return {
        "rows":
            rows,

        "h0":
            h0,

        "profiles":
            profiles,

        "prepared":
            prepared,

        "data":
            data,

        "h0_evaluator":
            h0_evaluator,

        "h1_evaluator":
            h1_evaluator,

        "ephemerides_path":
            str(
                h0_evaluator.ephemerides_path
            ),
    }


# ======================================================================
# CLI: one realization
# ======================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-mag",
        type=float,
        default=21.0,
    )

    parser.add_argument(
        "--blend-ratio",
        type=float,
        default=0.0,
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
        "--seed",
        type=int,
        default=2033101601,
    )

    parser.add_argument(
        "--p-min",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--p-max",
        type=float,
        default=1.0e5,
    )

    parser.add_argument(
        "--n-omega",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--max-level",
        choices=[
            levels.LEVEL_1,
            levels.LEVEL_2,
            levels.LEVEL_3,
        ],
        default=levels.LEVEL_3,
    )

    parser.add_argument(
        "--ephemerides",
        default=None,
    )

    args = parser.parse_args()

    t, t0 = (
        base.roman.build_roman_times(
            tE_true=args.tE
        )
    )

    P_grid = np.logspace(
        np.log10(
            args.p_min
        ),
        np.log10(
            args.p_max
        ),
        args.n_omega,
    )

    omega_grid = (
        2.0
        * np.pi
        / P_grid
    )

    result = (
        run_h0_lrt_ladder_parallax(
            t=t,
            t0_true=t0,
            u0_true=args.u0,
            tE_true=args.tE,
            source_mag=args.source_mag,
            seed=args.seed,
            omega_grid=omega_grid,
            blend_ratio=args.blend_ratio,
            ephemerides_path=args.ephemerides,
            max_level=args.max_level,
        )
    )

    print()
    print("=" * 120)
    print("ROMAN BSPL-vs-PSPL LRT WITH PARALLAX")
    print("=" * 120)

    print(
        "ephemerides =",
        result[
            "ephemerides_path"
        ],
    )

    print(
        "H0 piEN/piEE =",
        result["h0"]["piEN"],
        result["h0"]["piEE"],
    )

    print(
        "H0 chi2      =",
        result["h0"]["chi2"],
    )

    print()

    for row in result["rows"]:

        print(
            f"{row['h1_level']:24s} "
            f"T={row['T']:12.6f} "
            f"chi2={row['chi2_h1']:14.6f} "
            f"P={row['best_P_days']:12.4f} "
            f"qM={row['best_q_mass']:.6g} "
            f"qf={row['best_q_flux']:.6g} "
            f"piEN={row['h1_piEN']:+.6g} "
            f"piEE={row['h1_piEE']:+.6g}"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
