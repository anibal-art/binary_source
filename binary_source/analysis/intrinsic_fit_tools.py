'Ideal/noiseless fitting helpers for intrinsic BSPL/PSPL studies. This is not the Roman production fitter.'


from __future__ import annotations


import numpy as np


from scipy.optimize import least_squares, lsq_linear


from pyLIMA.models import PSPL_model


from binary_source.analysis import (
    intrinsic_xallarap as roman
)


def plain(x):
    if hasattr(x, "value"):
        x = x.value

    return np.asarray(
        x,
        dtype=float,
    )


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
