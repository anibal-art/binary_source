#!/usr/bin/env python3

"""
LRT model ladder for Roman BSPL-vs-PSPL.

Validated base model
--------------------
H1(1): omega

Extended ladder
---------------
H1(1): omega
H1(2): omega + q_mass
H1(3): omega + q_mass + q_flux
H1(4): omega + q_mass + q_flux + geometry

Geometry means:
    theta
    phi
    inclination

Common nuisance parameters:
    t0, u0, tE
    source/blend fluxes

The photometric flux parameters remain analytically profiled with
the same profile_flux() implementation used by the validated LRT.

Physical convention
-------------------
q_mass = M2 / M1

Mtot is fixed to the fiducial Roman analysis value:

    Mtot = 3 Msun

so at fixed P, varying q_mass does not change a_rel(P).

Important pyLIMA limitation
---------------------------
pyLIMA's Circular xallarap implementation does not support
q_mass = 0 exactly because it divides by xi_mass_ratio internally.

Therefore the BSPL branch uses:

    q_mass >= 1e-4

The exact PSPL model remains part of the test through:

    H1* = H0 union H1_BSPL

and the statistic remains:

    T = max(0, chi2_H0 - chi2_H1)
"""

from __future__ import annotations

import argparse

import numpy as np

from scipy.optimize import least_squares

from binary_source.analysis import (
    roman_bspl_pspl_lrt as base
)


# ============================================================
# LRT levels
# ============================================================

LEVEL_1 = "omega"

LEVEL_2 = "omega_qmass"

LEVEL_3 = "omega_qmass_qflux"

LEVEL_4 = "omega_qmass_qflux_geometry"


LEVELS = (
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    LEVEL_4,
)


# ============================================================
# Physical bounds
# ============================================================

QMASS_MIN = 1.0e-4
QMASS_MAX = 1.0

QFLUX_MIN = 0.0
QFLUX_MAX = 1.0

THETA_MIN = 0.0
THETA_MAX = 2.0 * np.pi

PHI_MIN = 0.0
PHI_MAX = 2.0 * np.pi

INCLINATION_MIN = -0.5 * np.pi
INCLINATION_MAX = +0.5 * np.pi


# ============================================================
# Multistart configuration
# ============================================================

QMASS_STARTS = (
    1.0e-4,
    0.01,
    0.10,
    0.50,
    0.90,
)

QFLUX_STARTS = (
    0.0,
    0.01,
    0.10,
    0.50,
    0.90,
)


# ============================================================
# Helpers
# ============================================================

def plain(x):

    if hasattr(
        x,
        "value",
    ):
        return np.asarray(
            x.value,
            dtype=float,
        )

    return np.asarray(
        x,
        dtype=float,
    )


def wrap_angle(x):

    return float(
        np.mod(
            float(x),
            2.0 * np.pi,
        )
    )


def validate_level(
    h1_level,
):

    if (
        h1_level
        not in LEVELS
    ):
        raise ValueError(
            f"Unknown H1 level "
            f"{h1_level!r}. "
            f"Expected one of "
            f"{LEVELS}."
        )

    return h1_level


def level_number(
    h1_level,
):

    validate_level(
        h1_level
    )

    return (
        LEVELS.index(
            h1_level
        )
        + 1
    )


def fiducial_values():

    return {
        "q_mass":
            float(
                base.roman.Q_MASS
            ),

        "q_flux":
            float(
                base.roman.QFLUX_TRUE
            ),

        "theta":
            wrap_angle(
                base.roman.THETA_TRUE
            ),

        "phi":
            wrap_angle(
                base.roman.PHI_TRUE
            ),

        "inclination":
            float(
                base.roman.INCLINATION_TRUE
            ),
    }


def promote_candidate(
    row,
    target_level,
):
    """
    Add missing parameters at their fiducial values.

    This allows the exact best fit of level N to be retained
    as a candidate of level N+1.
    """

    out = dict(
        row
    )

    fid = (
        fiducial_values()
    )

    for key, value in fid.items():

        if key not in out:

            out[key] = (
                float(value)
            )

    out[
        "h1_level"
    ] = target_level

    return out


def is_valid_fit(
    row,
):

    return bool(
        row.get(
            "success",
            False,
        )
        and np.isfinite(
            row.get(
                "chi2",
                np.inf,
            )
        )
    )


def fit_key(
    row,
):
    """
    Numerical key used only to remove duplicate multistarts.
    """

    fid = (
        fiducial_values()
    )

    omega = float(
        row[
            "omega"
        ]
    )

    return (
        round(
            np.log(omega),
            8,
        ),

        round(
            float(
                row.get(
                    "q_mass",
                    fid[
                        "q_mass"
                    ],
                )
            ),
            7,
        ),

        round(
            float(
                row.get(
                    "q_flux",
                    fid[
                        "q_flux"
                    ],
                )
            ),
            7,
        ),

        round(
            wrap_angle(
                row.get(
                    "theta",
                    fid[
                        "theta"
                    ],
                )
            ),
            7,
        ),

        round(
            wrap_angle(
                row.get(
                    "phi",
                    fid[
                        "phi"
                    ],
                )
            ),
            7,
        ),

        round(
            float(
                row.get(
                    "inclination",
                    fid[
                        "inclination"
                    ],
                )
            ),
            7,
        ),
    )


def select_best_starts(
    rows,
    n,
):

    valid = [
        row
        for row
        in rows
        if is_valid_fit(
            row
        )
    ]

    valid = sorted(
        valid,
        key=lambda row:
            float(
                row[
                    "chi2"
                ]
            ),
    )

    selected = []

    seen = set()

    for row in valid:

        key = (
            fit_key(
                row
            )
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        selected.append(
            row
        )

        if (
            len(selected)
            >= int(n)
        ):
            break

    if len(
        selected
    ) == 0:

        raise RuntimeError(
            "No valid starting "
            "solutions found."
        )

    return selected


def geometry_starts():
    """
    Small physically representative geometry multistart set.

    The fiducial geometry is always the first entry.
    """

    fid = (
        fiducial_values()
    )

    th = fid[
        "theta"
    ]

    ph = fid[
        "phi"
    ]

    inc = fid[
        "inclination"
    ]

    raw = [
        (
            th,
            ph,
            inc,
        ),

        (
            th + 0.5 * np.pi,
            ph,
            inc,
        ),

        (
            th + np.pi,
            ph,
            inc,
        ),

        (
            th + 1.5 * np.pi,
            ph,
            inc,
        ),

        (
            th,
            ph + 0.5 * np.pi,
            inc,
        ),

        (
            th,
            ph + np.pi,
            inc,
        ),

        (
            th,
            ph + 1.5 * np.pi,
            inc,
        ),

        (
            th,
            ph,
            0.0,
        ),

        (
            th,
            ph,
            np.pi / 3.0,
        ),

        (
            th,
            ph,
            -np.pi / 3.0,
        ),

        (
            th + 0.5 * np.pi,
            ph + 0.5 * np.pi,
            np.pi / 3.0,
        ),

        (
            th + 0.5 * np.pi,
            ph + 0.5 * np.pi,
            -np.pi / 3.0,
        ),
    ]

    result = []

    seen = set()

    for theta, phi, inclination in raw:

        row = (
            wrap_angle(
                theta
            ),
            wrap_angle(
                phi
            ),
            float(
                np.clip(
                    inclination,
                    INCLINATION_MIN,
                    INCLINATION_MAX,
                )
            ),
        )

        key = tuple(
            round(
                float(x),
                10,
            )
            for x
            in row
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            row
        )

    return tuple(
        result
    )


# ============================================================
# General physical BSPL evaluator
# ============================================================

class PhysicalCircularBSPLLadder:
    """
    Physical circular BSPL evaluator for all H1 levels.

    At the fiducial values this is exactly the same
    parametrization as the validated PhysicalCircularBSPL.
    """

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
            base.roman.build_truth_event(
                self.t
            )
        )

        self.event = ev
        self.telescope = tel

        self.model = (
            base.PSPL_model.PSPLmodel(
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
        q_mass=None,
        q_flux=None,
        theta=None,
        phi=None,
        inclination=None,
    ):

        fid = (
            fiducial_values()
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

        q_mass = float(
            q_mass
        )

        q_flux = float(
            q_flux
        )

        theta = wrap_angle(
            theta
        )

        phi = wrap_angle(
            phi
        )

        inclination = float(
            inclination
        )

        if not (
            np.isfinite(
                omega
            )
            and omega > 0.0
        ):
            raise ValueError(
                "omega must be positive."
            )

        if not (
            QMASS_MIN
            <= q_mass
            <= QMASS_MAX
        ):
            raise ValueError(
                "q_mass outside allowed "
                "BSPL range."
            )

        if not (
            QFLUX_MIN
            <= q_flux
            <= QFLUX_MAX
        ):
            raise ValueError(
                "q_flux outside allowed range."
            )

        if not (
            INCLINATION_MIN
            <= inclination
            <= INCLINATION_MAX
        ):
            raise ValueError(
                "inclination outside "
                "allowed range."
            )

        P_days = (
            2.0
            * np.pi
            / omega
        )

        # Fixed total source mass.
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
                theta
            )
        )

        xi_perp = (
            xi_rel
            * np.sin(
                theta
            )
        )

        params = [
            float(t0),
            float(u0),
            float(tE),

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

        A = plain(
            A
        )

        # pyLIMA returns A1 + qf A2.
        # Convert to magnification relative to total
        # baseline source flux F1 + F2.
        A = (
            A
            / (
                1.0
                + q_flux
            )
        )

        if np.any(
            ~np.isfinite(
                A
            )
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


# Backward-compatible name from the first controlled test.
PhysicalCircularBSPLQMass = (
    PhysicalCircularBSPLLadder
)


# ============================================================
# Generic joint optimizer
# ============================================================

def fit_h1_joint(
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
    theta_start=None,
    phi_start=None,
    inclination_start=None,
):
    """
    Jointly optimize all parameters free at the requested level.

    Shape parameters t0,u0,tE are always free.

    omega is represented as log(omega).
    """

    validate_level(
        h1_level
    )

    fid = (
        fiducial_values()
    )

    lower_shape, upper_shape = (
        base.shape_bounds()
    )

    omega_min = float(
        omega_min
    )

    omega_max = float(
        omega_max
    )

    if q_mass_start is None:

        q_mass_start = float(
            start_fit.get(
                "q_mass",
                fid[
                    "q_mass"
                ],
            )
        )

    if q_flux_start is None:

        q_flux_start = float(
            start_fit.get(
                "q_flux",
                fid[
                    "q_flux"
                ],
            )
        )

    if theta_start is None:

        theta_start = float(
            start_fit.get(
                "theta",
                fid[
                    "theta"
                ],
            )
        )

    if phi_start is None:

        phi_start = float(
            start_fit.get(
                "phi",
                fid[
                    "phi"
                ],
            )
        )

    if inclination_start is None:

        inclination_start = float(
            start_fit.get(
                "inclination",
                fid[
                    "inclination"
                ],
            )
        )

    x0 = [
        (
            float(
                start_fit[
                    "t0"
                ]
            )
            - float(
                t0_ref
            )
        )
        / float(
            tE_ref
        ),

        float(
            start_fit[
                "u0"
            ]
        ),

        np.log(
            float(
                start_fit[
                    "tE"
                ]
            )
            / float(
                tE_ref
            )
        ),

        np.log(
            float(
                start_fit[
                    "omega"
                ]
            )
        ),
    ]

    lower = list(
        lower_shape
    )

    upper = list(
        upper_shape
    )

    lower.append(
        np.log(
            omega_min
        )
    )

    upper.append(
        np.log(
            omega_max
        )
    )

    if (
        level_number(
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
            QMASS_MIN
        )

        upper.append(
            QMASS_MAX
        )

    if (
        level_number(
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
            QFLUX_MIN
        )

        upper.append(
            QFLUX_MAX
        )

    if (
        level_number(
            h1_level
        )
        >= 4
    ):

        x0.extend(
            [
                wrap_angle(
                    theta_start
                ),

                wrap_angle(
                    phi_start
                ),

                float(
                    inclination_start
                ),
            ]
        )

        lower.extend(
            [
                THETA_MIN,
                PHI_MIN,
                INCLINATION_MIN,
            ]
        )

        upper.extend(
            [
                THETA_MAX,
                PHI_MAX,
                INCLINATION_MAX,
            ]
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

    eps = 1.0e-10

    x0 = np.maximum(
        x0,
        lower + eps,
    )

    x0 = np.minimum(
        x0,
        upper - eps,
    )

    def decode(
        x,
    ):

        t0, u0, tE = (
            base.unpack_shape_parameters(
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

        index = 4

        q_mass = fid[
            "q_mass"
        ]

        q_flux = fid[
            "q_flux"
        ]

        theta = fid[
            "theta"
        ]

        phi = fid[
            "phi"
        ]

        inclination = fid[
            "inclination"
        ]

        if (
            level_number(
                h1_level
            )
            >= 2
        ):

            q_mass = float(
                x[index]
            )

            index += 1

        if (
            level_number(
                h1_level
            )
            >= 3
        ):

            q_flux = float(
                x[index]
            )

            index += 1

        if (
            level_number(
                h1_level
            )
            >= 4
        ):

            theta = float(
                x[index]
            )

            phi = float(
                x[
                    index + 1
                ]
            )

            inclination = float(
                x[
                    index + 2
                ]
            )

        return {
            "t0":
                float(
                    t0
                ),

            "u0":
                float(
                    u0
                ),

            "tE":
                float(
                    tE
                ),

            "omega":
                float(
                    omega
                ),

            "q_mass":
                float(
                    q_mass
                ),

            "q_flux":
                float(
                    q_flux
                ),

            "theta":
                wrap_angle(
                    theta
                ),

            "phi":
                wrap_angle(
                    phi
                ),

            "inclination":
                float(
                    inclination
                ),
        }

    def residuals(
        x,
    ):

        p = decode(
            x
        )

        A = evaluator.magnification(
            t0=p[
                "t0"
            ],

            u0=p[
                "u0"
            ],

            tE=p[
                "tE"
            ],

            omega=p[
                "omega"
            ],

            q_mass=p[
                "q_mass"
            ],

            q_flux=p[
                "q_flux"
            ],

            theta=p[
                "theta"
            ],

            phi=p[
                "phi"
            ],

            inclination=p[
                "inclination"
            ],
        )

        return base.profile_flux(
            A,
            flux,
            err_flux,
        )[
            "residual"
        ]

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
        t0=p[
            "t0"
        ],

        u0=p[
            "u0"
        ],

        tE=p[
            "tE"
        ],

        omega=p[
            "omega"
        ],

        q_mass=p[
            "q_mass"
        ],

        q_flux=p[
            "q_flux"
        ],

        theta=p[
            "theta"
        ],

        phi=p[
            "phi"
        ],

        inclination=p[
            "inclination"
        ],
    )

    prof = (
        base.profile_flux(
            A,
            flux,
            err_flux,
        )
    )

    result = dict(
        p
    )

    result.update(
        {
            "success":
                bool(
                    opt.success
                ),

            "chi2":
                float(
                    prof[
                        "chi2"
                    ]
                ),

            "P_days":
                float(
                    2.0
                    * np.pi
                    / p[
                        "omega"
                    ]
                ),

            "fs":
                prof[
                    "fs"
                ],

            "ft":
                prof[
                    "ft"
                ],

            "nfev":
                int(
                    opt.nfev
                ),

            "h1_level":
                h1_level,
        }
    )

    return result


# ============================================================
# Level 1
# ============================================================

def fit_level_1(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    h0_fit,
    t0_ref,
    tE_ref,
):
    """
    EXACTLY reuse the validated omega-only fitter.
    """

    (
        best,
        scan,
        refined,
    ) = (
        base.fit_h1_profile_omega(
            evaluator=evaluator,
            flux=flux,
            err_flux=err_flux,
            omega_grid=omega_grid,
            h0_fit=h0_fit,
            t0_ref=t0_ref,
            tE_ref=tE_ref,
            n_refine=6,
        )
    )

    best = promote_candidate(
        best,
        LEVEL_1,
    )

    rows = []

    for row in (
        [
            best
        ]
        + list(
            scan
        )
        + list(
            refined
        )
    ):

        if is_valid_fit(
            row
        ):

            rows.append(
                promote_candidate(
                    row,
                    LEVEL_1,
                )
            )

    return (
        best,
        {
            "fits":
                rows,

            "scan":
                scan,

            "refined":
                refined,
        },
    )


# ============================================================
# Level 2: omega + q_mass
# ============================================================

def fit_level_2(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    h0_fit,
    t0_ref,
    tE_ref,
    previous_profile,
    n_previous_starts=6,
):

    prev_best, prev_diag = (
        previous_profile
    )

    starts = (
        select_best_starts(
            prev_diag[
                "fits"
            ],
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

        for q_mass in QMASS_STARTS:

            try:

                fit = fit_h1_joint(
                    evaluator=evaluator,
                    flux=flux,
                    err_flux=err_flux,
                    start_fit=start,
                    h1_level=LEVEL_2,
                    t0_ref=t0_ref,
                    tE_ref=tE_ref,
                    omega_min=omega_min,
                    omega_max=omega_max,
                    q_mass_start=q_mass,
                )

            except Exception as exc:

                fit = {
                    "success":
                        False,

                    "chi2":
                        np.inf,

                    "error":
                        repr(
                            exc
                        ),
                }

            fits.append(
                fit
            )

    # Exact previous-level solution retained.
    fallback = (
        promote_candidate(
            prev_best,
            LEVEL_2,
        )
    )

    valid = [
        row
        for row
        in fits
        if is_valid_fit(
            row
        )
    ]

    candidates = (
        [
            fallback
        ]
        + valid
    )

    best = min(
        candidates,
        key=lambda row:
            float(
                row[
                    "chi2"
                ]
            ),
    )

    best = promote_candidate(
        best,
        LEVEL_2,
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


# ============================================================
# Level 3: omega + q_mass + q_flux
# ============================================================

def fit_level_3(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    h0_fit,
    t0_ref,
    tE_ref,
    previous_profile,
    n_previous_starts=6,
):

    prev_best, prev_diag = (
        previous_profile
    )

    starts = (
        select_best_starts(
            prev_diag[
                "fits"
            ],
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

        for q_flux in QFLUX_STARTS:

            try:

                fit = fit_h1_joint(
                    evaluator=evaluator,
                    flux=flux,
                    err_flux=err_flux,
                    start_fit=start,
                    h1_level=LEVEL_3,
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

            except Exception as exc:

                fit = {
                    "success":
                        False,

                    "chi2":
                        np.inf,

                    "error":
                        repr(
                            exc
                        ),
                }

            fits.append(
                fit
            )

    fallback = (
        promote_candidate(
            prev_best,
            LEVEL_3,
        )
    )

    valid = [
        row
        for row
        in fits
        if is_valid_fit(
            row
        )
    ]

    candidates = (
        [
            fallback
        ]
        + valid
    )

    best = min(
        candidates,
        key=lambda row:
            float(
                row[
                    "chi2"
                ]
            ),
    )

    best = promote_candidate(
        best,
        LEVEL_3,
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


# ============================================================
# Level 4: full geometry
# ============================================================

def fit_level_4(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    h0_fit,
    t0_ref,
    tE_ref,
    previous_profile,
    n_previous_starts=4,
):

    prev_best, prev_diag = (
        previous_profile
    )

    starts = (
        select_best_starts(
            prev_diag[
                "fits"
            ],
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

    geometries = (
        geometry_starts()
    )

    for start in starts:

        for (
            theta,
            phi,
            inclination,
        ) in geometries:

            try:

                fit = fit_h1_joint(
                    evaluator=evaluator,
                    flux=flux,
                    err_flux=err_flux,
                    start_fit=start,
                    h1_level=LEVEL_4,
                    t0_ref=t0_ref,
                    tE_ref=tE_ref,
                    omega_min=omega_min,
                    omega_max=omega_max,
                    q_mass_start=start.get(
                        "q_mass",
                        base.roman.Q_MASS,
                    ),
                    q_flux_start=start.get(
                        "q_flux",
                        base.roman.QFLUX_TRUE,
                    ),
                    theta_start=theta,
                    phi_start=phi,
                    inclination_start=inclination,
                )

            except Exception as exc:

                fit = {
                    "success":
                        False,

                    "chi2":
                        np.inf,

                    "error":
                        repr(
                            exc
                        ),
                }

            fits.append(
                fit
            )

    fallback = (
        promote_candidate(
            prev_best,
            LEVEL_4,
        )
    )

    valid = [
        row
        for row
        in fits
        if is_valid_fit(
            row
        )
    ]

    candidates = (
        [
            fallback
        ]
        + valid
    )

    best = min(
        candidates,
        key=lambda row:
            float(
                row[
                    "chi2"
                ]
            ),
    )

    best = promote_candidate(
        best,
        LEVEL_4,
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

            "geometry_starts":
                geometries,
        },
    )


# ============================================================
# Complete H1 ladder
# ============================================================

def fit_h1_ladder(
    evaluator,
    flux,
    err_flux,
    omega_grid,
    h0_fit,
    t0_ref,
    tE_ref,
    max_level=LEVEL_4,
):

    validate_level(
        max_level
    )

    profiles = {}

    profiles[
        LEVEL_1
    ] = fit_level_1(
        evaluator=evaluator,
        flux=flux,
        err_flux=err_flux,
        omega_grid=omega_grid,
        h0_fit=h0_fit,
        t0_ref=t0_ref,
        tE_ref=tE_ref,
    )

    if (
        level_number(
            max_level
        )
        >= 2
    ):

        profiles[
            LEVEL_2
        ] = fit_level_2(
            evaluator=evaluator,
            flux=flux,
            err_flux=err_flux,
            omega_grid=omega_grid,
            h0_fit=h0_fit,
            t0_ref=t0_ref,
            tE_ref=tE_ref,
            previous_profile=profiles[
                LEVEL_1
            ],
        )

    if (
        level_number(
            max_level
        )
        >= 3
    ):

        profiles[
            LEVEL_3
        ] = fit_level_3(
            evaluator=evaluator,
            flux=flux,
            err_flux=err_flux,
            omega_grid=omega_grid,
            h0_fit=h0_fit,
            t0_ref=t0_ref,
            tE_ref=tE_ref,
            previous_profile=profiles[
                LEVEL_2
            ],
        )

    if (
        level_number(
            max_level
        )
        >= 4
    ):

        profiles[
            LEVEL_4
        ] = fit_level_4(
            evaluator=evaluator,
            flux=flux,
            err_flux=err_flux,
            omega_grid=omega_grid,
            h0_fit=h0_fit,
            t0_ref=t0_ref,
            tE_ref=tE_ref,
            previous_profile=profiles[
                LEVEL_3
            ],
        )

    return profiles


# ============================================================
# One H0 realization through the whole ladder
# ============================================================

def run_h0_lrt_ladder(
    t,
    t0_true,
    u0_true,
    tE_true,
    source_mag,
    seed,
    omega_grid,
    max_level=LEVEL_4,
    blend_ratio=0.0,
):
    """
    Generate ONE noisy H0 realization and fit every requested
    H1 level to exactly the same data.

    This is the controlled comparison needed for the LRT ladder.
    """

    # Use the validated evaluator only for construction of the
    # deterministic Pandeia mask.
    legacy_evaluator = (
        base.PhysicalCircularBSPL(
            t=t,
            t0_reference=t0_true,
        )
    )

    prepared = (
        base.prepare_h0_configuration(
            t=t,
            t0_true=t0_true,
            u0_true=u0_true,
            tE_true=tE_true,
            source_mag=source_mag,
            evaluator=legacy_evaluator,
            blend_ratio=blend_ratio,
        )
    )

    t_used = np.asarray(
        prepared[
            "t"
        ],
        dtype=float,
    )

    # General evaluator on exactly the retained timestamps.
    evaluator = (
        PhysicalCircularBSPLLadder(
            t=t_used,
            t0_reference=t0_true,
        )
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

    h0 = (
        base.fit_h0_pspl(
            t=t_used,
            flux=data[
                "flux"
            ],
            err_flux=data[
                "err_flux"
            ],
            t0_ref=t0_true,
            u0_guess=u0_true,
            tE_ref=tE_true,
        )
    )

    profiles = (
        fit_h1_ladder(
            evaluator=evaluator,
            flux=data[
                "flux"
            ],
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

    truth = prepared[
        "truth"
    ]

    rows = []

    for level in LEVELS:

        if level not in profiles:
            continue

        best = (
            profiles[
                level
            ][0]
        )

        raw_delta = (
            float(
                h0[
                    "chi2"
                ]
            )
            - float(
                best[
                    "chi2"
                ]
            )
        )

        T = max(
            0.0,
            raw_delta,
        )

        rows.append(
            {
                "seed":
                    int(
                        seed
                    ),

                "source_mag":
                    float(
                        source_mag
                    ),

                "blend_ratio":
                    float(
                        truth[
                            "blend_ratio"
                        ]
                    ),

                "baseline_mag":
                    float(
                        truth[
                            "baseline_mag"
                        ]
                    ),

                "h1_level":
                    level,

                "level_number":
                    int(
                        level_number(
                            level
                        )
                    ),

                "chi2_h0":
                    float(
                        h0[
                            "chi2"
                        ]
                    ),

                "chi2_h1":
                    float(
                        best[
                            "chi2"
                        ]
                    ),

                "delta_chi2_raw":
                    float(
                        raw_delta
                    ),

                "T":
                    float(
                        T
                    ),

                "best_P_days":
                    float(
                        best[
                            "P_days"
                        ]
                    ),

                "best_omega":
                    float(
                        best[
                            "omega"
                        ]
                    ),

                "best_q_mass":
                    float(
                        best[
                            "q_mass"
                        ]
                    ),

                "best_q_flux":
                    float(
                        best[
                            "q_flux"
                        ]
                    ),

                "best_theta":
                    float(
                        best[
                            "theta"
                        ]
                    ),

                "best_phi":
                    float(
                        best[
                            "phi"
                        ]
                    ),

                "best_inclination":
                    float(
                        best[
                            "inclination"
                        ]
                    ),

                "h1_t0":
                    float(
                        best[
                            "t0"
                        ]
                    ),

                "h1_u0":
                    float(
                        best[
                            "u0"
                        ]
                    ),

                "h1_tE":
                    float(
                        best[
                            "tE"
                        ]
                    ),

                "h0_fs":
                    float(
                        h0[
                            "fs"
                        ]
                    ),

                "h0_ft":
                    float(
                        h0[
                            "ft"
                        ]
                    ),

                "h1_fs":
                    float(
                        best[
                            "fs"
                        ]
                    ),

                "h1_ft":
                    float(
                        best[
                            "ft"
                        ]
                    ),
            }
        )

    # Mandatory nesting check.
    for i in range(
        1,
        len(
            rows
        ),
    ):

        previous = (
            rows[
                i - 1
            ]
        )

        current = (
            rows[
                i
            ]
        )

        if (
            current[
                "chi2_h1"
            ]
            >
            previous[
                "chi2_h1"
            ]
            + 1.0e-8
        ):
            raise RuntimeError(
                "LRT ladder nesting violation: "
                f"{current['h1_level']} has "
                "larger chi2 than previous level."
            )

        if (
            current[
                "T"
            ]
            <
            previous[
                "T"
            ]
            - 1.0e-8
        ):
            raise RuntimeError(
                "LRT ladder T monotonicity "
                "violation."
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
    }


# ============================================================
# CLI: controlled one-realization ladder
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--source-mag",
        type=float,
        default=21.0,
    )

    parser.add_argument(
        "--blend-ratio",
        type=float,
        default=0.0,
        help="True blending ratio Fb/Fs.",
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
        choices=LEVELS,
        default=LEVEL_4,
    )

    args = (
        parser.parse_args()
    )

    t, t0 = (
        base.roman.build_roman_times(
            tE_true=args.tE
        )
    )

    P_grid = (
        np.logspace(
            np.log10(
                args.p_min
            ),
            np.log10(
                args.p_max
            ),
            args.n_omega,
        )
    )

    omega_grid = (
        2.0
        * np.pi
        / P_grid
    )

    result = (
        run_h0_lrt_ladder(
            t=t,
            t0_true=t0,
            u0_true=args.u0,
            tE_true=args.tE,
            source_mag=args.source_mag,
            seed=args.seed,
            omega_grid=omega_grid,
            max_level=args.max_level,
            blend_ratio=args.blend_ratio,
        )
    )

    print()
    print(
        "=" * 150
    )

    print(
        "ROMAN BSPL-vs-PSPL LRT LADDER"
    )

    print(
        "=" * 150
    )

    print(
        "seed       =",
        args.seed,
    )

    print(
        "F146       =",
        args.source_mag,
    )

    print(
        "Fb/Fs      =",
        args.blend_ratio,
    )

    print(
        "u0         =",
        args.u0,
    )

    print(
        "tE [d]     =",
        args.tE,
    )

    print()

    header = (
        f"{'level':30s} "
        f"{'chi2_H1':>14s} "
        f"{'T':>12s} "
        f"{'P[d]':>12s} "
        f"{'qM':>10s} "
        f"{'qf':>10s} "
        f"{'theta':>10s} "
        f"{'phi':>10s} "
        f"{'inc':>10s}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for row in result[
        "rows"
    ]:

        print(
            f"{row['h1_level']:30s} "
            f"{row['chi2_h1']:14.6f} "
            f"{row['T']:12.6f} "
            f"{row['best_P_days']:12.5g} "
            f"{row['best_q_mass']:10.5f} "
            f"{row['best_q_flux']:10.5f} "
            f"{row['best_theta']:10.5f} "
            f"{row['best_phi']:10.5f} "
            f"{row['best_inclination']:10.5f}"
        )

    print(
        "=" * 150
    )


if __name__ == "__main__":
    main()
