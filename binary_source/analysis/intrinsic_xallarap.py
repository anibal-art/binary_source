'Intrinsic/noiseless BSPL-xallarap physics. No Roman cadence, Pandeia photometry, noise realization or observational fitter belongs in this module.'


import numpy as np


from pyLIMA import event, telescopes


from pyLIMA.models import PSPL_model


from binary_source.functions_aux import a_from_P_kepler_days


M1_MSUN = 2.0


M2_MSUN = 1.0


Q_MASS = M2_MSUN / M1_MSUN


MTOT_MSUN = M1_MSUN + M2_MSUN


REHAT_AU = 5.0


PHI_TRUE = 0.0


INCLINATION_TRUE = np.pi / 2.0


THETA_TRUE = 0.0


QFLUX_TRUE = 0.0


def as_array(x):
    if hasattr(x, "value"):
        return np.asarray(x.value, dtype=float)

    return np.asarray(x, dtype=float)


def build_event_from_magnitudes(
    t,
    magnitude,
    err_mag,
    name="Roman",
):
    """
    Build the Event used for the fit.

    Parallax is disabled in this first experiment.
    Therefore, the physical Telescope location does not enter
    the trajectory. We use location='Space' to preserve
    the Roman semantics.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    magnitude = np.asarray(
        magnitude,
        dtype=float,
    )

    err_mag = np.asarray(
        err_mag,
        dtype=float,
    )

    if not (
        len(t)
        == len(magnitude)
        == len(err_mag)
    ):
        raise ValueError(
            "t, magnitude y err_mag "
            "must have the same length."
        )

    ev = event.Event()

    ev.name = (
        "Roman_BSPL_Asimov"
    )

    ev.ra = 267.92497054815516
    ev.dec = -29.152232510353276

    lc = np.column_stack(
        [
            t,
            magnitude,
            err_mag,
        ]
    ).astype(float)

    tel = telescopes.Telescope(
        name=name,
        camera_filter="F146",
        lightcurve=lc,
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
        location="Space",
    )

    ev.telescopes.append(
        tel
    )

    return ev, tel


def build_truth_event(
    t,
):
    """
    Auxiliary event used to compute the magnification.

    The photometric values here are placeholders.
    """

    t = np.asarray(
        t,
        dtype=float,
    )

    mag0 = np.full_like(
        t,
        20.0,
    )

    emag = np.full_like(
        t,
        0.01,
    )

    return build_event_from_magnitudes(
        t=t,
        magnitude=mag0,
        err_mag=emag,
        name="Roman_truth",
    )


def bspl_truth_magnification(
    t,
    t0_true,
    u0_true,
    tE_true,
    P_days,
    q_mass=Q_MASS,
    qflux=QFLUX_TRUE,
    Mtot_Msun=MTOT_MSUN,
    rEhat_AU=REHAT_AU,
    theta=THETA_TRUE,
    phi=PHI_TRUE,
    inclination=INCLINATION_TRUE,
):
    """
    Same physical parameterization used in the intrinsic scans.

    xi_rel = a_rel / rEhat

    and pyLIMA distributes the orbital motion between the two sources
    mediante q_mass.
    """

    P_days = float(P_days)

    if P_days <= 0.0:
        raise ValueError(
            "P_days must be positive."
        )

    a_rel_AU = float(
        a_from_P_kepler_days(
            P_days,
            Mtot_Msun,
        )
    )

    xi_rel = (
        a_rel_AU
        / float(rEhat_AU)
    )

    omega = (
        2.0
        * np.pi
        / P_days
    )

    xi_para = (
        xi_rel
        * np.cos(theta)
    )

    xi_perp = (
        xi_rel
        * np.sin(theta)
    )

    ev_truth, tel_truth = (
        build_truth_event(t)
    )

    model_bspl = (
        PSPL_model.PSPLmodel(
            ev_truth,
            parallax=[
                "None",
                t0_true,
            ],
            double_source=[
                "Circular",
                t0_true,
            ],
        )
    )

    model_bspl.define_model_parameters()

    # Only required to construct pyLIMA_parameters.
    # No intervienen en model_magnification.
    fsource_dummy = 1.0
    ftotal_dummy = 1.0

    params_bspl = [
        t0_true,
        u0_true,
        tE_true,

        xi_para,
        xi_perp,
        omega,

        phi,
        inclination,

        q_mass,
        qflux,

        fsource_dummy,
        ftotal_dummy,
    ]

    py_params_bspl = (
        model_bspl.compute_pyLIMA_parameters(
            params_bspl
        )
    )

    A_bspl = (
        model_bspl.model_magnification(
            tel_truth,
            py_params_bspl,
        )
    )

    A_bspl = as_array(
        A_bspl
    )

    # pyLIMA returns A1 + qf A2.
    # We want the magnification relative to the total unlensed flux
    # total no magnificado.
    A_bspl = (
        A_bspl
        / (1.0 + qflux)
    )

    if np.any(
        ~np.isfinite(A_bspl)
    ):
        raise RuntimeError(
            "A_BSPL contiene valores no finitos."
        )

    if np.any(
        A_bspl <= 0.0
    ):
        raise RuntimeError(
            "A_BSPL contiene valores <= 0."
        )

    return {
        "A_bspl": A_bspl,
        "xi_rel": xi_rel,
        "a_rel_AU": a_rel_AU,
        "omega": omega,
    }
