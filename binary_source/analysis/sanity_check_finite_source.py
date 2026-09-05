#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finite-source sanity check for the BSPL -> PSPL degeneracy study.

Scientific question
-------------------
Do finite-source effects substantially modify the localized deviations
produced by extreme close approaches of the secondary source?

Truth models
------------
1. Point-source binary source:
       PSPLmodel + double_source=["Circular", t0]

2. Finite-source binary source:
       FSPLarge + double_source=["Circular", t0]
       with rho1 = rho and rho2 = rho_2

Fit model
---------
Single-source PSPL in ALL cases.

Thus we continue measuring:

               int [A_truth - A_PSPL,best]^2 dt
D^2 = ------------------------------------------------
                  int [A_truth - 1]^2 dt

No survey cadence/noise is introduced.

Important
---------
The current upstream pyLIMA FSPLarge binary-source implementation
contains a typo in the source-2 branch when sqrt limb darkening is zero:
it recomputes source1_magnification instead of defining
source2_magnification.

This script implements a minimal local subclass correcting only that bug.
The installed pyLIMA package is NOT modified.
"""

from __future__ import annotations

import csv
import inspect
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

from pyLIMA import event, telescopes

from pyLIMA.models import (
    PSPL_model,
    FSPLarge_model,
)

from pyLIMA.magnification import (
    magnification_VBB,
)


# ============================================================
# PATHS
# ============================================================

SCRIPT = Path(__file__).resolve()

ANALYSIS_DIR = SCRIPT.parent
SOURCE_DIR = ANALYSIS_DIR.parent
REPO_ROOT = SOURCE_DIR.parent


OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "finite_source_sanity"
)


FIGURE_DIR = (
    REPO_ROOT
    / "figures"
    / "finite_source_sanity"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPERIMENT
# ============================================================

T0 = 50.0
U0 = 0.1
TE = 150.0

P_DAYS = 150.0

P_OVER_TE = (
    P_DAYS
    / TE
)


MTOT_MSUN = 3.0

REHAT_AU = 5.0


# We deliberately keep qf away from qf=qM so that
# photocenter cancellation is not mixed with this test.

QFLUX = 0.01


# Same orbital geometry used in the qM-qf experiment.

THETA = 0.0

PHI = 0.0

INCLINATION = (
    np.pi
    / 2.0
)


# ============================================================
# GEOMETRIES
#
# extreme:
# qM=0.09 showed the very close secondary-source approach
# for exactly u0=0.1.
#
# control:
# a configuration away from that localized feature.
# ============================================================

GEOMETRIES = {

    "extreme":
        0.09,

    "control":
        0.30,
}


# ============================================================
# FINITE-SOURCE RADII
#
# rho1 is fixed because source 1 is nowhere near a
# source-crossing configuration here.
#
# rho2 is the physically interesting quantity.
# ============================================================

RHO1 = 1.0e-3


RHO2_CASES = np.array(
    [
        1.0e-4,
        3.0e-4,
        1.0e-3,
        3.0e-3,
    ],
    dtype=float,
)


# ============================================================
# LIMB DARKENING
#
# For this sanity check use uniform sources.
#
# This is deliberate:
# we want to isolate finite-source size, not explore
# limb-darkening systematics.
# ============================================================

LINEAR_LD = 0.0

SQRT_LD = 0.0


# ============================================================
# TIME GRID
#
# Full intrinsic-fit interval = +-3.5 tE.
#
# The close-approach feature can be much narrower than the
# normal 10,000-point grid. Therefore we use a non-uniform
# grid:
#
# 1. coarse full-event grid
# 2. dense region around source-2 closest approach
# 3. ultra-dense core
#
# All objective functions and metrics are integrated using
# np.trapezoid, so non-uniform sampling is handled correctly.
# ============================================================

FIT_WINDOW_TE = 3.5


N_COARSE = 8000


BROAD_HALF_WIDTH_TE = 0.03

N_BROAD = 6001


CORE_HALF_WIDTH_TE = 5.0e-4

N_CORE = 5001


# ============================================================
# FIT OPTIONS
# ============================================================

MAXITER = 50000

XATOL = 1.0e-10

FATOL = 1.0e-12


# ============================================================
# PLOT STYLE
# ============================================================

def set_paper_style():

    plt.rcParams.update({

        "font.size": 11,

        "axes.labelsize": 12,
        "axes.titlesize": 12,

        "legend.fontsize": 9,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "axes.linewidth": 0.8,

        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.top": True,
        "ytick.right": True,

        "xtick.major.size": 5,
        "ytick.major.size": 5,

        "xtick.minor.size": 3,
        "ytick.minor.size": 3,

        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",

        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


# ============================================================
# LOCAL FIX FOR pyLIMA FSPLarge
#
# Upstream bug:
#
# in the source-2 branch without sqrt limb darkening,
# FSPLarge currently does:
#
#     source1_magnification = magnification_FSPL(
#         source1_x,
#         source1_y,
#         rho_2,
#         ...
#     )
#
# It should do:
#
#     source2_magnification = magnification_FSPL(
#         source2_x,
#         source2_y,
#         rho_2,
#         ...
#     )
#
# Everything else below follows the upstream implementation.
# ============================================================

class FSPLargeBinaryFixed(
    FSPLarge_model.FSPLargemodel
):

    def model_magnification(
        self,
        telescope,
        pyLIMA_parameters,
        return_impact_parameter=False,
    ):

        rho = (
            pyLIMA_parameters[
                "rho"
            ]
        )


        linear_limb_darkening = (
            telescope.ld_a1
        )


        sqrt_limb_darkening = (
            telescope.ld_a2
        )


        (
            source1_trajectory_x,
            source1_trajectory_y,

            source2_trajectory_x,
            source2_trajectory_y,

            dseparation,
            dalpha,

        ) = self.sources_trajectory(

            telescope,

            pyLIMA_parameters,

            data_type="photometry",
        )


        # ====================================================
        # SOURCE 1
        # ====================================================

        if (
            sqrt_limb_darkening
            is not None
            and
            sqrt_limb_darkening > 0.0
        ):

            source1_magnification = (
                magnification_VBB.magnification_FSPL(

                    source1_trajectory_x,

                    source1_trajectory_y,

                    rho,

                    linear_limb_darkening,

                    sqrt_limb_darkening,
                )
            )


        else:

            source1_magnification = (
                magnification_VBB.magnification_FSPL(

                    source1_trajectory_x,

                    source1_trajectory_y,

                    rho,

                    linear_limb_darkening,
                )
            )


        # ====================================================
        # SOURCE 2
        # ====================================================

        if (
            source2_trajectory_x
            is not None
        ):

            rho_2 = (
                pyLIMA_parameters[
                    "rho_2"
                ]
            )


            if (
                sqrt_limb_darkening
                is not None
                and
                sqrt_limb_darkening > 0.0
            ):

                source2_magnification = (
                    magnification_VBB.magnification_FSPL(

                        source2_trajectory_x,

                        source2_trajectory_y,

                        rho_2,

                        linear_limb_darkening,

                        sqrt_limb_darkening,
                    )
                )


            else:

                # ============================================
                # LOCAL BUG FIX
                # ============================================

                source2_magnification = (
                    magnification_VBB.magnification_FSPL(

                        source2_trajectory_x,

                        source2_trajectory_y,

                        rho_2,

                        linear_limb_darkening,
                    )
                )


            blend_magnification_factor = (
                pyLIMA_parameters[
                    "q_flux_"
                    + telescope.filter
                ]
            )


            magnification = (

                source1_magnification

                +

                source2_magnification
                * blend_magnification_factor
            )


        else:

            magnification = (
                source1_magnification
            )


        return (
            magnification
        )


# ============================================================
# BASIC HELPERS
# ============================================================

def _arr(
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


def build_event(
    t,
):

    t = np.asarray(
        t,
        dtype=float,
    )


    ev = event.Event()

    ev.name = (
        "FiniteSourceSanity"
    )

    ev.ra = 170.0
    ev.dec = -70.0


    lightcurve = np.c_[

        t,

        np.full_like(
            t,
            19.0,
        ),

        np.full_like(
            t,
            1.0e-9,
        ),
    ]


    telescope = telescopes.Telescope(

        name="Simulation",

        camera_filter="G",

        lightcurve=(
            lightcurve.astype(
                float
            )
        ),

        lightcurve_names=[
            "time",
            "mag",
            "err_mag",
        ],

        lightcurve_units=[
            "JD",
            "mag",
            "mag",
        ],

        location="Earth",
    )


    # ========================================================
    # Uniform-brightness sources
    # ========================================================

    telescope.ld_a1 = (
        LINEAR_LD
    )

    telescope.ld_a2 = (
        SQRT_LD
    )


    ev.telescopes.append(
        telescope
    )


    return ev


# ============================================================
# KEPLER
# ============================================================

def compute_relative_orbit():

    P_yr = (
        P_DAYS
        / 365.25
    )


    a_rel_AU = (
        MTOT_MSUN
        * P_yr**2
    )**(
        1.0
        / 3.0
    )


    xi_rel = (
        a_rel_AU
        / REHAT_AU
    )


    return (
        a_rel_AU,
        xi_rel,
    )


A_REL_AU, XI_REL = (
    compute_relative_orbit()
)


# ============================================================
# PARAMETER VECTOR
#
# Build vectors from pyLIMA's own model dictionary rather
# than assuming a parameter order.
# ============================================================

def build_parameter_vector(
    model,
    values,
):

    model.define_model_parameters()


    dictionary = (
        model.model_dictionnary
    )


    if not dictionary:

        raise RuntimeError(
            "Empty pyLIMA model dictionary."
        )


    n_parameters = (

        max(
            int(
                index
            )

            for index
            in dictionary.values()
        )

        + 1
    )


    vector = np.zeros(
        n_parameters,
        dtype=float,
    )


    unknown = []


    for (
        name,
        index,
    ) in dictionary.items():


        index = int(
            index
        )


        name_lower = (
            str(
                name
            ).lower()
        )


        if name in values:

            value = (
                values[
                    name
                ]
            )


        elif name_lower.startswith(
            "q_flux"
        ):

            value = (
                values[
                    "q_flux"
                ]
            )


        elif name_lower.startswith(
            "fsource"
        ):

            value = (
                values.get(
                    "fsource",
                    1.0,
                )
            )


        elif name_lower.startswith(
            "ftotal"
        ):

            value = (
                values.get(
                    "ftotal",
                    1.0,
                )
            )


        elif name_lower.startswith(
            "fblend"
        ):

            value = (
                values.get(
                    "fblend",
                    0.0,
                )
            )


        else:

            unknown.append(
                name
            )

            continue


        vector[
            index
        ] = float(
            value
        )


    if unknown:

        print()
        print(
            "pyLIMA model dictionary:"
        )

        print(
            dictionary
        )


        raise KeyError(

            "No values supplied for "
            f"parameters: {unknown}"
        )


    return vector


# ============================================================
# BINARY-SOURCE PARAMETER VALUES
# ============================================================

def binary_source_values(
    q_mass,
    rho1=None,
    rho2=None,
):

    omega = (
        2.0
        * np.pi
        / P_DAYS
    )


    values = {

        "t0":
            T0,

        "u0":
            U0,

        "tE":
            TE,

        "xi_para":
            XI_REL
            * np.cos(
                THETA
            ),

        "xi_perp":
            XI_REL
            * np.sin(
                THETA
            ),

        "xi_angular_velocity":
            omega,

        "xi_phase":
            PHI,

        "xi_inclination":
            INCLINATION,

        "xi_mass_ratio":
            float(
                q_mass
            ),

        "q_flux":
            QFLUX,

        "fsource":
            1.0,

        "ftotal":
            1.0,

        "fblend":
            0.0,
    }


    if rho1 is not None:

        values[
            "rho"
        ] = float(
            rho1
        )


    if rho2 is not None:

        values[
            "rho_2"
        ] = float(
            rho2
        )


    return values


# ============================================================
# TRAJECTORY DIAGNOSTICS
# ============================================================

def trajectory_diagnostics(
    t,
    trajectories,
):

    s1x = _arr(
        trajectories[
            0
        ]
    )

    s1y = _arr(
        trajectories[
            1
        ]
    )


    s2x = _arr(
        trajectories[
            2
        ]
    )

    s2y = _arr(
        trajectories[
            3
        ]
    )


    u1 = np.hypot(
        s1x,
        s1y,
    )


    u2 = np.hypot(
        s2x,
        s2y,
    )


    i1 = int(
        np.nanargmin(
            u1
        )
    )


    i2 = int(
        np.nanargmin(
            u2
        )
    )


    return dict(

        s1x=s1x,
        s1y=s1y,

        s2x=s2x,
        s2y=s2y,

        u1=u1,
        u2=u2,

        u1_min=float(
            u1[
                i1
            ]
        ),

        u2_min=float(
            u2[
                i2
            ]
        ),

        t_u1_min=float(
            t[
                i1
            ]
        ),

        t_u2_min=float(
            t[
                i2
            ]
        ),
    )


# ============================================================
# POINT-SOURCE BSPL
# ============================================================

def build_point_bspl_state(
    t,
    q_mass,
):

    ev = build_event(
        t
    )


    model = (
        PSPL_model.PSPLmodel(

            ev,

            parallax=[
                "None",
                0.0,
            ],

            double_source=[
                "Circular",
                T0,
            ],
        )
    )


    values = binary_source_values(
        q_mass=q_mass,
    )


    vector = build_parameter_vector(
        model,
        values,
    )


    py_params = (
        model.compute_pyLIMA_parameters(
            vector
        )
    )


    A = (

        model.model_magnification(

            ev.telescopes[
                0
            ],

            py_params,
        )

        /
        (
            1.0
            + QFLUX
        )
    )


    trajectories = (
        model.sources_trajectory(

            ev.telescopes[
                0
            ],

            py_params,

            data_type="photometry",
        )
    )


    geometry = (
        trajectory_diagnostics(

            t=t,

            trajectories=trajectories,
        )
    )


    return dict(

        event=ev,

        model=model,

        py_params=py_params,

        A=_arr(
            A
        ),

        **geometry,
    )


# ============================================================
# FINITE-SOURCE BSPL
# ============================================================

def build_finite_bspl_state(
    t,
    q_mass,
    rho1,
    rho2,
):

    ev = build_event(
        t
    )


    model = (
        FSPLargeBinaryFixed(

            ev,

            parallax=[
                "None",
                0.0,
            ],

            double_source=[
                "Circular",
                T0,
            ],
        )
    )


    values = binary_source_values(

        q_mass=q_mass,

        rho1=rho1,

        rho2=rho2,
    )


    vector = build_parameter_vector(
        model,
        values,
    )


    py_params = (
        model.compute_pyLIMA_parameters(
            vector
        )
    )


    A = (

        model.model_magnification(

            ev.telescopes[
                0
            ],

            py_params,
        )

        /
        (
            1.0
            + QFLUX
        )
    )


    trajectories = (
        model.sources_trajectory(

            ev.telescopes[
                0
            ],

            py_params,

            data_type="photometry",
        )
    )


    geometry = (
        trajectory_diagnostics(

            t=t,

            trajectories=trajectories,
        )
    )


    return dict(

        event=ev,

        model=model,

        py_params=py_params,

        A=_arr(
            A
        ),

        **geometry,
    )


# ============================================================
# FSPLARGE PREFLIGHT
# ============================================================

def preflight_fsplarge():

    print()
    print("=" * 80)
    print("FSPLarge preflight")
    print("=" * 80)


    print(
        "Installed FSPLarge module:"
    )

    print(
        inspect.getfile(
            FSPLarge_model
        )
    )


    t = np.linspace(

        T0
        - 1.0,

        T0
        + 1.0,

        101,
    )


    ev = build_event(
        t
    )


    model = (
        FSPLargeBinaryFixed(

            ev,

            parallax=[
                "None",
                0.0,
            ],

            double_source=[
                "Circular",
                T0,
            ],
        )
    )


    model.define_model_parameters()


    print()
    print(
        "model_dictionnary:"
    )

    print(
        model.model_dictionnary
    )


    required = [
        "rho",
        "rho_2",
    ]


    missing = [

        key

        for key
        in required

        if key not in (
            model.model_dictionnary
        )
    ]


    if missing:

        raise RuntimeError(

            "FSPLarge binary-source model does not "
            "expose expected parameters "
            f"{missing}."
        )


    # ========================================================
    # Smoke test
    # ========================================================

    print()
    print(
        "Running corrected finite-source "
        "binary-source smoke test..."
    )


    state = build_finite_bspl_state(

        t=t,

        q_mass=(
            GEOMETRIES[
                "extreme"
            ]
        ),

        rho1=RHO1,

        rho2=(
            RHO2_CASES[
                0
            ]
        ),
    )


    if not np.all(
        np.isfinite(
            state[
                "A"
            ]
        )
    ):

        raise RuntimeError(

            "Corrected FSPLarge returned "
            "non-finite magnifications."
        )


    print(
        "Corrected FSPLarge smoke test: PASS"
    )


    # ========================================================
    # Point-source limit
    #
    # Test away from the extreme crossing.
    # ========================================================

    print()
    print(
        "Checking point-source limit..."
    )


    t_validation = np.linspace(

        T0
        - 0.5
        * TE,

        T0
        + 0.5
        * TE,

        5001,
    )


    qM_validation = (
        GEOMETRIES[
            "control"
        ]
    )


    point_state = (
        build_point_bspl_state(

            t=t_validation,

            q_mass=qM_validation,
        )
    )


    finite_small_state = (
        build_finite_bspl_state(

            t=t_validation,

            q_mass=qM_validation,

            rho1=1.0e-6,

            rho2=1.0e-6,
        )
    )


    delta = (

        finite_small_state[
            "A"
        ]

        - point_state[
            "A"
        ]
    )


    max_abs_difference = float(
        np.nanmax(
            np.abs(
                delta
            )
        )
    )


    rms_difference = float(
        np.sqrt(
            np.nanmean(
                delta**2
            )
        )
    )


    print(
        "max |A_FS - A_PS| =",
        f"{max_abs_difference:.8e}",
    )


    print(
        "RMS difference     =",
        f"{rms_difference:.8e}",
    )


    if (
        not np.isfinite(
            max_abs_difference
        )
        or
        not np.isfinite(
            rms_difference
        )
    ):

        raise RuntimeError(

            "Point-source-limit validation "
            "returned non-finite values."
        )


    print(
        "Point-source limit validation: PASS"
    )


    print("=" * 80)


# ============================================================
# ADAPTIVE TIME GRID
# ============================================================

def build_time_grid(
    q_mass,
):

    # ========================================================
    # 1. Full event
    # ========================================================

    t_coarse = np.linspace(

        T0
        - FIT_WINDOW_TE
        * TE,

        T0
        + FIT_WINDOW_TE
        * TE,

        N_COARSE,
    )


    coarse_state = (
        build_point_bspl_state(

            t=t_coarse,

            q_mass=q_mass,
        )
    )


    t2_rough = (
        coarse_state[
            "t_u2_min"
        ]
    )


    # ========================================================
    # 2. Broad refinement
    # ========================================================

    broad_half = (
        BROAD_HALF_WIDTH_TE
        * TE
    )


    t_broad = np.linspace(

        t2_rough
        - broad_half,

        t2_rough
        + broad_half,

        N_BROAD,
    )


    broad_state = (
        build_point_bspl_state(

            t=t_broad,

            q_mass=q_mass,
        )
    )


    t2_refined = (
        broad_state[
            "t_u2_min"
        ]
    )


    # ========================================================
    # 3. Ultra-dense core
    # ========================================================

    core_half = (
        CORE_HALF_WIDTH_TE
        * TE
    )


    t_core = np.linspace(

        t2_refined
        - core_half,

        t2_refined
        + core_half,

        N_CORE,
    )


    # ========================================================
    # Merge
    # ========================================================

    t = np.unique(

        np.concatenate(
            [
                t_coarse,

                t_broad,

                t_core,

                np.array(
                    [
                        T0,
                        t2_refined,
                    ]
                ),
            ]
        )
    )


    t_left = (
        T0
        - FIT_WINDOW_TE
        * TE
    )


    t_right = (
        T0
        + FIT_WINDOW_TE
        * TE
    )


    t = t[
        (
            t >= t_left
        )
        &
        (
            t <= t_right
        )
    ]


    return (
        t,
        t2_refined,
    )


# ============================================================
# SINGLE-SOURCE PSPL FIT
# ============================================================

def pspl_objective(
    fit_params,
    model,
    A_truth,
    t,
):

    values = {

        "t0":
            fit_params[
                0
            ],

        "u0":
            fit_params[
                1
            ],

        "tE":
            fit_params[
                2
            ],

        "fsource":
            1.0,

        "ftotal":
            1.0,

        "fblend":
            0.0,
    }


    vector = build_parameter_vector(
        model,
        values,
    )


    py_params = (
        model.compute_pyLIMA_parameters(
            vector
        )
    )


    A_model = (
        model.model_magnification(

            model.event.telescopes[
                0
            ],

            py_params,
        )
    )


    residual = (
        A_truth
        - A_model
    )


    return float(
        np.trapezoid(

            residual**2,

            x=t,
        )
    )


def fit_pspl(
    t,
    A_truth,
):

    ev = build_event(
        t
    )


    model = (
        PSPL_model.PSPLmodel(

            ev,

            parallax=[
                "None",
                0.0,
            ],

            double_source=[
                "None",
                0.0,
            ],
        )
    )


    model.define_model_parameters()


    x0 = np.array(
        [
            T0,
            U0,
            TE,
        ],
        dtype=float,
    )


    result = so.minimize(

        pspl_objective,

        x0=x0,

        args=(
            model,
            A_truth,
            t,
        ),

        method="Nelder-Mead",

        options=dict(

            maxiter=MAXITER,

            xatol=XATOL,

            fatol=FATOL,
        ),
    )


    if not result.success:

        raise RuntimeError(

            "PSPL fit failed: "
            f"{result.message}"
        )


    best = np.asarray(
        result.x,
        dtype=float,
    )


    values = {

        "t0":
            best[
                0
            ],

        "u0":
            best[
                1
            ],

        "tE":
            best[
                2
            ],

        "fsource":
            1.0,

        "ftotal":
            1.0,

        "fblend":
            0.0,
    }


    vector = build_parameter_vector(
        model,
        values,
    )


    py_params = (
        model.compute_pyLIMA_parameters(
            vector
        )
    )


    A_fit = (
        model.model_magnification(

            ev.telescopes[
                0
            ],

            py_params,
        )
    )


    return dict(

        best=best,

        A_fit=_arr(
            A_fit
        ),

        objective=float(
            result.fun
        ),
    )


# ============================================================
# TDEV
# ============================================================

def measure_residual_fwhm(
    t,
    residual,
):

    t = np.asarray(
        t,
        dtype=float,
    )


    y = np.abs(

        np.asarray(
            residual,
            dtype=float,
        )
    )


    finite = (
        np.isfinite(t)
        &
        np.isfinite(y)
    )


    if np.count_nonzero(
        finite
    ) < 3:

        return np.nan


    safe = np.where(
        finite,
        y,
        -np.inf,
    )


    i_max = int(
        np.argmax(
            safe
        )
    )


    Rmax = float(
        y[
            i_max
        ]
    )


    if (
        not np.isfinite(
            Rmax
        )
        or
        Rmax <= 0.0
    ):

        return 0.0


    half = (
        0.5
        * Rmax
    )


    # ========================================================
    # LEFT
    # ========================================================

    i_left = (
        i_max
    )


    while (
        i_left > 0
        and
        np.isfinite(
            y[
                i_left - 1
            ]
        )
        and
        y[
            i_left - 1
        ] >= half
    ):

        i_left -= 1


    if i_left == 0:

        t_left = (
            t[
                0
            ]
        )


    else:

        ia = (
            i_left - 1
        )

        ib = (
            i_left
        )


        ta = (
            t[
                ia
            ]
        )

        tb = (
            t[
                ib
            ]
        )


        ya = (
            y[
                ia
            ]
        )

        yb = (
            y[
                ib
            ]
        )


        if np.isclose(
            ya,
            yb,
        ):

            t_left = (
                tb
            )


        else:

            t_left = (

                ta

                +

                (
                    half
                    - ya
                )

                *
                (
                    tb
                    - ta
                )

                /

                (
                    yb
                    - ya
                )
            )


    # ========================================================
    # RIGHT
    # ========================================================

    i_right = (
        i_max
    )


    while (
        i_right
        < len(
            y
        ) - 1
        and
        np.isfinite(
            y[
                i_right + 1
            ]
        )
        and
        y[
            i_right + 1
        ] >= half
    ):

        i_right += 1


    if i_right == len(y) - 1:

        t_right = (
            t[
                -1
            ]
        )


    else:

        ia = (
            i_right
        )

        ib = (
            i_right + 1
        )


        ta = (
            t[
                ia
            ]
        )

        tb = (
            t[
                ib
            ]
        )


        ya = (
            y[
                ia
            ]
        )

        yb = (
            y[
                ib
            ]
        )


        if np.isclose(
            ya,
            yb,
        ):

            t_right = (
                ta
            )


        else:

            t_right = (

                ta

                +

                (
                    half
                    - ya
                )

                *
                (
                    tb
                    - ta
                )

                /

                (
                    yb
                    - ya
                )
            )


    return float(
        t_right
        - t_left
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    t,
    A_truth,
    A_fit,
):

    residual = (
        A_fit
        - A_truth
    )


    numerator = np.trapezoid(

        residual**2,

        x=t,
    )


    denominator = np.trapezoid(

        (
            A_truth
            - 1.0
        )**2,

        x=t,
    )


    if (
        np.isfinite(
            denominator
        )
        and
        denominator > 0.0
    ):

        D = np.sqrt(
            numerator
            / denominator
        )


    else:

        D = np.nan


    total_duration = (
        t[
            -1
        ]
        - t[
            0
        ]
    )


    RMS = np.sqrt(
        numerator
        / total_duration
    )


    Rmax = float(
        np.nanmax(
            np.abs(
                residual
            )
        )
    )


    tdev = (
        measure_residual_fwhm(

            t=t,

            residual=residual,
        )
    )


    return dict(

        D=float(
            D
        ),

        RMS=float(
            RMS
        ),

        Rmax=Rmax,

        tdev=float(
            tdev
        ),

        residual=residual,
    )


# ============================================================
# EVALUATE ONE TRUTH
# ============================================================

def evaluate_truth(
    case_name,
    q_mass,
    t,
    truth_kind,
    rho1=None,
    rho2=None,
):

    if truth_kind == "point":

        truth = (
            build_point_bspl_state(

                t=t,

                q_mass=q_mass,
            )
        )


    elif truth_kind == "finite":

        truth = (
            build_finite_bspl_state(

                t=t,

                q_mass=q_mass,

                rho1=rho1,

                rho2=rho2,
            )
        )


    else:

        raise ValueError(
            truth_kind
        )


    fit = fit_pspl(

        t=t,

        A_truth=truth[
            "A"
        ],
    )


    metrics = compute_metrics(

        t=t,

        A_truth=truth[
            "A"
        ],

        A_fit=fit[
            "A_fit"
        ],
    )


    if (
        rho2 is not None
        and
        rho2 > 0.0
    ):

        u2_over_rho2 = (

            truth[
                "u2_min"
            ]

            / rho2
        )


    else:

        u2_over_rho2 = (
            np.nan
        )


    return dict(

        case=case_name,

        qM=float(
            q_mass
        ),

        truth_kind=(
            truth_kind
        ),

        rho1=(
            np.nan
            if rho1 is None
            else float(
                rho1
            )
        ),

        rho2=(
            np.nan
            if rho2 is None
            else float(
                rho2
            )
        ),

        u1_min=float(
            truth[
                "u1_min"
            ]
        ),

        u2_min=float(
            truth[
                "u2_min"
            ]
        ),

        u2_over_rho2=float(
            u2_over_rho2
        ),

        t_u2_min=float(
            truth[
                "t_u2_min"
            ]
        ),

        dt_u2_min_over_tE=float(

            (
                truth[
                    "t_u2_min"
                ]

                - T0
            )

            / TE
        ),

        D=metrics[
            "D"
        ],

        RMS=metrics[
            "RMS"
        ],

        Rmax=metrics[
            "Rmax"
        ],

        tdev=metrics[
            "tdev"
        ],

        best_t0=float(
            fit[
                "best"
            ][
                0
            ]
        ),

        best_u0=float(
            fit[
                "best"
            ][
                1
            ]
        ),

        best_tE=float(
            fit[
                "best"
            ][
                2
            ]
        ),

        t=t,

        A_truth=truth[
            "A"
        ],

        A_fit=fit[
            "A_fit"
        ],

        residual=metrics[
            "residual"
        ],
    )


# ============================================================
# RATIOS TO POINT-SOURCE CASE
# ============================================================

def add_point_ratios(
    rows,
):

    for case_name in (
        GEOMETRIES.keys()
    ):

        case_rows = [

            row

            for row
            in rows

            if row[
                "case"
            ] == case_name
        ]


        point = next(

            row

            for row
            in case_rows

            if row[
                "truth_kind"
            ] == "point"
        )


        for row in case_rows:

            row[
                "D_over_point"
            ] = (

                row[
                    "D"
                ]

                / point[
                    "D"
                ]
            )


            row[
                "RMS_over_point"
            ] = (

                row[
                    "RMS"
                ]

                / point[
                    "RMS"
                ]
            )


            row[
                "Rmax_over_point"
            ] = (

                row[
                    "Rmax"
                ]

                / point[
                    "Rmax"
                ]
            )


            row[
                "tdev_over_point"
            ] = (

                row[
                    "tdev"
                ]

                / point[
                    "tdev"
                ]
            )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    rows,
):

    print()
    print("=" * 145)

    print(
        "FINITE-SOURCE SANITY CHECK"
    )

    print("=" * 145)


    print(

        "case       "
        "kind      "
        "qM         "
        "rho2       "
        "u2/rho2    "
        "D          "
        "D/Dpoint   "
        "Rmax       "
        "R/Rpoint   "
        "tdev[d]    "
        "t/tpoint"
    )


    print("-" * 145)


    for row in rows:

        if np.isfinite(
            row[
                "rho2"
            ]
        ):

            rho2_text = (
                f"{row['rho2']:.1e}"
            )


        else:

            rho2_text = (
                "-"
            )


        if np.isfinite(
            row[
                "u2_over_rho2"
            ]
        ):

            ratio_text = (
                f"{row['u2_over_rho2']:.3e}"
            )


        else:

            ratio_text = (
                "-"
            )


        print(

            f"{row['case']:<10s} "

            f"{row['truth_kind']:<9s} "

            f"{row['qM']:9.6f} "

            f"{rho2_text:>9s} "

            f"{ratio_text:>10s} "

            f"{row['D']:10.3e} "

            f"{row['D_over_point']:10.3e} "

            f"{row['Rmax']:10.3e} "

            f"{row['Rmax_over_point']:10.3e} "

            f"{row['tdev']:10.3e} "

            f"{row['tdev_over_point']:10.3e}"
        )


    print("=" * 145)


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    rows,
):

    filename = (
        OUTPUT_DIR
        / "finite_source_sanity_metrics.csv"
    )


    fields = [

        "case",

        "qM",

        "truth_kind",

        "rho1",

        "rho2",

        "u1_min",

        "u2_min",

        "u2_over_rho2",

        "dt_u2_min_over_tE",

        "D",

        "D_over_point",

        "RMS",

        "RMS_over_point",

        "Rmax",

        "Rmax_over_point",

        "tdev",

        "tdev_over_point",

        "best_t0",

        "best_u0",

        "best_tE",
    ]


    with open(

        filename,

        "w",

        newline="",

        encoding="utf-8",

    ) as handle:


        writer = csv.DictWriter(

            handle,

            fieldnames=fields,
        )


        writer.writeheader()


        for row in rows:

            writer.writerow({

                key:
                    row.get(
                        key,
                        np.nan,
                    )

                for key
                in fields
            })


    print(
        "CSV saved:",
        filename,
    )


# ============================================================
# SAVE NPZ
# ============================================================

def save_npz(
    rows,
):

    filename = (
        OUTPUT_DIR
        / "finite_source_sanity_summary.npz"
    )


    np.savez_compressed(

        filename,

        case=np.array(
            [
                row[
                    "case"
                ]

                for row
                in rows
            ]
        ),

        qM=np.array(
            [
                row[
                    "qM"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        truth_kind=np.array(
            [
                row[
                    "truth_kind"
                ]

                for row
                in rows
            ]
        ),

        rho1=np.array(
            [
                row[
                    "rho1"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        rho2=np.array(
            [
                row[
                    "rho2"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        u1_min=np.array(
            [
                row[
                    "u1_min"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        u2_min=np.array(
            [
                row[
                    "u2_min"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        u2_over_rho2=np.array(
            [
                row[
                    "u2_over_rho2"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        D=np.array(
            [
                row[
                    "D"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        D_over_point=np.array(
            [
                row[
                    "D_over_point"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        RMS=np.array(
            [
                row[
                    "RMS"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        RMS_over_point=np.array(
            [
                row[
                    "RMS_over_point"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        Rmax=np.array(
            [
                row[
                    "Rmax"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        Rmax_over_point=np.array(
            [
                row[
                    "Rmax_over_point"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        tdev=np.array(
            [
                row[
                    "tdev"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        tdev_over_point=np.array(
            [
                row[
                    "tdev_over_point"
                ]

                for row
                in rows
            ],
            dtype=float,
        ),

        P_days=np.float64(
            P_DAYS
        ),

        tE=np.float64(
            TE
        ),

        u0=np.float64(
            U0
        ),

        qf=np.float64(
            QFLUX
        ),

        rho1_fixed=np.float64(
            RHO1
        ),

        Mtot_Msun=np.float64(
            MTOT_MSUN
        ),

        rEhat_AU=np.float64(
            REHAT_AU
        ),

        a_rel_AU=np.float64(
            A_REL_AU
        ),

        xi_rel=np.float64(
            XI_REL
        ),

        linear_limb_darkening=np.float64(
            LINEAR_LD
        ),

        sqrt_limb_darkening=np.float64(
            SQRT_LD
        ),
    )


    print(
        "NPZ saved:",
        filename,
    )


# ============================================================
# PLOT: CLOSE APPROACH CURVES
# ============================================================

def plot_case_comparison(
    case_name,
    case_rows,
):

    set_paper_style()


    point = next(

        row

        for row
        in case_rows

        if row[
            "truth_kind"
        ] == "point"
    )


    tau2 = (

        point[
            "t_u2_min"
        ]

        - T0

    ) / TE


    # A few percent of tE around the encounter.
    zoom_half_tau = 0.025


    fig, axes = plt.subplots(

        2,
        1,

        figsize=(
            8.0,
            6.0,
        ),

        sharex=True,

        gridspec_kw=dict(

            height_ratios=[
                2.2,
                1.0,
            ],

            hspace=0.05,
        ),
    )


    axA = (
        axes[
            0
        ]
    )


    axR = (
        axes[
            1
        ]
    )


    for row in case_rows:

        tau = (

            row[
                "t"
            ]

            - T0

        ) / TE


        if (
            row[
                "truth_kind"
            ]
            == "point"
        ):

            label = (
                "point source"
            )


        else:

            label = (

                rf"$\rho_2="
                rf"{row['rho2']:.0e}$"
            )


        axA.plot(

            tau,

            row[
                "A_truth"
            ],

            linewidth=1.3,

            label=label,
        )


        axR.plot(

            tau,

            np.abs(
                row[
                    "residual"
                ]
            ),

            linewidth=1.1,
        )


    axA.set_yscale(
        "log"
    )


    axR.set_yscale(
        "log"
    )


    axA.set_ylabel(
        r"$A_{\rm BSPL}$"
    )


    axR.set_ylabel(
        r"$|\Delta A|$"
    )


    axR.set_xlabel(
        r"$(t-t_0)/t_E$"
    )


    axA.set_xlim(

        tau2
        - zoom_half_tau,

        tau2
        + zoom_half_tau,
    )


    for ax in (
        axA,
        axR,
    ):

        ax.axvline(

            tau2,

            linestyle=":",

            linewidth=0.8,

            alpha=0.65,
        )


        ax.grid(

            alpha=0.10,

            linewidth=0.5,
        )


    axA.legend(

        frameon=False,

        ncol=2,
    )


    axA.set_title(

        rf"{case_name}: "
        rf"$q_M={point['qM']:.3g}$, "
        rf"$q_f={QFLUX:g}$"
    )


    png = (

        FIGURE_DIR

        / (
            f"finite_source_"
            f"{case_name}.png"
        )
    )


    pdf = (

        FIGURE_DIR

        / (
            f"finite_source_"
            f"{case_name}.pdf"
        )
    )


    fig.savefig(

        png,

        dpi=600,
    )


    fig.savefig(
        pdf
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        pdf,
    )


# ============================================================
# PLOT: METRIC RATIOS
# ============================================================

def plot_metric_summary(
    rows,
):

    set_paper_style()


    fig, axes = plt.subplots(

        1,
        3,

        figsize=(
            11.0,
            3.6,
        ),

        constrained_layout=True,
    )


    specs = [

        (
            "D_over_point",
            r"$D_{\rm FS}/D_{\rm point}$",
        ),

        (
            "Rmax_over_point",
            r"$R_{\rm max,FS}/R_{\rm max,point}$",
        ),

        (
            "tdev_over_point",
            r"$t_{\rm dev,FS}/t_{\rm dev,point}$",
        ),
    ]


    for case_name in (
        GEOMETRIES.keys()
    ):

        finite_rows = [

            row

            for row
            in rows

            if (
                row[
                    "case"
                ] == case_name
                and
                row[
                    "truth_kind"
                ] == "finite"
            )
        ]


        finite_rows = sorted(

            finite_rows,

            key=lambda row:
                row[
                    "rho2"
                ],
        )


        rho2 = np.array(
            [
                row[
                    "rho2"
                ]

                for row
                in finite_rows
            ],
            dtype=float,
        )


        for (
            ax,
            (
                key,
                ylabel,
            ),
        ) in zip(

            axes,

            specs,
        ):

            values = np.array(
                [
                    row[
                        key
                    ]

                    for row
                    in finite_rows
                ],
                dtype=float,
            )


            ax.plot(

                rho2,

                values,

                marker="o",

                linewidth=1.3,

                label=case_name,
            )


            ax.axhline(

                1.0,

                linestyle="--",

                linewidth=0.8,

                alpha=0.6,
            )


            ax.set_xscale(
                "log"
            )


            ax.set_xlabel(
                r"$\rho_2$"
            )


            ax.set_ylabel(
                ylabel
            )


            ax.grid(

                alpha=0.10,

                linewidth=0.5,
            )


    axes[
        0
    ].legend(
        frameon=False
    )


    png = (
        FIGURE_DIR
        / "finite_source_metric_ratios.png"
    )


    pdf = (
        FIGURE_DIR
        / "finite_source_metric_ratios.pdf"
    )


    fig.savefig(

        png,

        dpi=600,
    )


    fig.savefig(
        pdf
    )


    plt.close(
        fig
    )


    print(
        "Saved:",
        pdf,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("FINITE-SOURCE SANITY CHECK")
    print("=" * 80)


    print(
        "P/tE       =",
        P_OVER_TE,
    )


    print(
        "P [d]      =",
        P_DAYS,
    )


    print(
        "tE [d]     =",
        TE,
    )


    print(
        "u0         =",
        U0,
    )


    print(
        "qf         =",
        QFLUX,
    )


    print(
        "Mtot       =",
        MTOT_MSUN,
        "Msun",
    )


    print(
        "rEhat      =",
        REHAT_AU,
        "AU",
    )


    print(
        "a_rel      =",
        A_REL_AU,
        "AU",
    )


    print(
        "xi_rel     =",
        XI_REL,
    )


    print(
        "rho1       =",
        RHO1,
    )


    print(
        "rho2 cases =",
        RHO2_CASES,
    )


    print(
        "limb dark. = uniform source"
    )


    print("=" * 80)


    # ========================================================
    # Validate corrected FSPLarge
    # ========================================================

    preflight_fsplarge()


    rows = []


    # ========================================================
    # Run geometries
    # ========================================================

    for (
        case_name,
        q_mass,
    ) in GEOMETRIES.items():


        print()
        print("=" * 80)

        print(
            f"CASE: {case_name}"
        )


        print(
            f"qM = {q_mass:.10f}"
        )


        print("=" * 80)


        # ====================================================
        # Common time grid for all rho2 at this geometry
        # ====================================================

        t, t2_est = (
            build_time_grid(
                q_mass=q_mass
            )
        )


        print(
            "N time =",
            len(
                t
            ),
        )


        print(
            "estimated "
            "dt(u2,min)/tE =",
            (
                t2_est
                - T0
            )
            / TE,
        )


        # ====================================================
        # Point-source reference
        # ====================================================

        point = (
            evaluate_truth(

                case_name=case_name,

                q_mass=q_mass,

                t=t,

                truth_kind="point",
            )
        )


        rows.append(
            point
        )


        print()
        print(
            "POINT SOURCE"
        )


        print(
            "  u2,min =",
            f"{point['u2_min']:.8e}",
        )


        print(
            "  D      =",
            f"{point['D']:.8e}",
        )


        print(
            "  RMS    =",
            f"{point['RMS']:.8e}",
        )


        print(
            "  Rmax   =",
            f"{point['Rmax']:.8e}",
        )


        print(
            "  tdev   =",
            f"{point['tdev']:.8e}",
        )


        # ====================================================
        # Finite-source cases
        # ====================================================

        for rho2 in (
            RHO2_CASES
        ):


            finite = (
                evaluate_truth(

                    case_name=case_name,

                    q_mass=q_mass,

                    t=t,

                    truth_kind="finite",

                    rho1=RHO1,

                    rho2=float(
                        rho2
                    ),
                )
            )


            rows.append(
                finite
            )


            print()
            print(
                "FINITE SOURCE"
            )


            print(
                f"  rho2     = "
                f"{rho2:.3e}"
            )


            print(
                "  u2/rho2  =",
                f"{finite['u2_over_rho2']:.8e}",
            )


            print(
                "  D        =",
                f"{finite['D']:.8e}",
            )


            print(
                "  RMS      =",
                f"{finite['RMS']:.8e}",
            )


            print(
                "  Rmax     =",
                f"{finite['Rmax']:.8e}",
            )


            print(
                "  tdev     =",
                f"{finite['tdev']:.8e}",
            )


    # ========================================================
    # Ratios
    # ========================================================

    add_point_ratios(
        rows
    )


    # ========================================================
    # Results
    # ========================================================

    print_summary(
        rows
    )


    save_csv(
        rows
    )


    save_npz(
        rows
    )


    # ========================================================
    # Figures
    # ========================================================

    for case_name in (
        GEOMETRIES.keys()
    ):


        case_rows = [

            row

            for row
            in rows

            if row[
                "case"
            ] == case_name
        ]


        plot_case_comparison(

            case_name=case_name,

            case_rows=case_rows,
        )


    plot_metric_summary(
        rows
    )


    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":

    main()
