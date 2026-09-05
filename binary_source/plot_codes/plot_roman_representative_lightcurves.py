#!/usr/bin/env python3

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from binary_source.analysis import roman_bspl_pspl_asimov as roman


# ============================================================
# Configuration
# ============================================================

SOURCE_MAG = 21.0

ROMAN_FILE = Path(
    "results/roman_asimov/"
    "roman_intrinsic_grid_W149_21.npz"
)

COMBINED_FILE = Path(
    "results/roman_asimov/"
    "combined_intrinsic_grid/"
    "roman_intrinsic_grid_W149_19_21_23.npz"
)

OUTDIR = Path(
    "figures/roman_asimov/"
    "representative_lightcurves"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Number of points used only to draw smooth curves.
N_DENSE = 60000

# Silence verbose pyLIMA fitting output.
QUIET_PYLIMA = True


# ============================================================
# Error-bar display
# ============================================================

SHOW_ERRORBARS = True

# Plot one error bar every N Roman epochs.
# This affects visualization only; the fit uses every epoch.
ERRORBAR_EVERY = 25

ERRORBAR_ALPHA = 0.45
ERRORBAR_ELINEWIDTH = 0.6
ERRORBAR_MARKERSIZE = 1.8
ERRORBAR_CAPSIZE = 0.0


# ============================================================
# Representative cases
#
# indices correspond exactly to the combined intrinsic grid
# ============================================================

CASES = [
    {
        "key": "A_hidden",
        "label": "A: extreme hidden",
        "iu0": 0,
        "iP": 59,
    },
    {
        "key": "B_deltaChi2_100",
        "label": (
            r"B: intrinsically degenerate, "
            r"$\Delta\chi^2\simeq100$"
        ),
        "iu0": 45,
        "iP": 46,
    },
    {
        "key": "C_deltaChi2_500",
        "label": (
            r"C: intrinsically degenerate, "
            r"$\Delta\chi^2\simeq500$"
        ),
        "iu0": 47,
        "iP": 42,
    },
    {
        "key": "D_clear",
        "label": "D: clear intrinsic mismatch",
        "iu0": 46,
        "iP": 0,
    },
]


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }
)


# ============================================================
# Utilities
# ============================================================

def as_array(x):
    return np.asarray(
        x,
        dtype=float,
    ).reshape(-1)


def pspl_magnification(
    t,
    t0,
    u0,
    tE,
):
    """
    Standard point-source point-lens magnification.
    """

    tau = (
        (np.asarray(t, dtype=float) - float(t0))
        / float(tE)
    )

    u = np.sqrt(
        float(u0) ** 2
        + tau ** 2
    )

    return (
        (u ** 2 + 2.0)
        / (
            u
            * np.sqrt(
                u ** 2 + 4.0
            )
        )
    )


def weighted_linear_flux_fit(
    A,
    flux,
    sigma,
):
    """
    For fixed PSPL lens parameters solve

        F = Fs * A + Fb

    using weighted least squares.

    This is equivalent to fitting the linear photometric
    nuisance parameters for a one-telescope PSPL model.
    """

    A = as_array(A)
    flux = as_array(flux)
    sigma = as_array(sigma)

    valid = (
        np.isfinite(A)
        & np.isfinite(flux)
        & np.isfinite(sigma)
        & (sigma > 0.0)
    )

    A = A[valid]
    flux = flux[valid]
    sigma = sigma[valid]

    X = np.column_stack(
        [
            A,
            np.ones_like(A),
        ]
    )

    w = (
        1.0
        / sigma ** 2
    )

    XT_W = (
        X.T
        * w
    )

    beta = np.linalg.solve(
        XT_W @ X,
        XT_W @ flux,
    )

    Fs = float(beta[0])
    Fb = float(beta[1])

    return Fs, Fb


def run_fit_quietly(
    event,
    t0,
    u0,
    tE,
):
    """
    Run the exact Roman PSPL fitter used in the production
    experiment while optionally suppressing pyLIMA stdout.
    """

    if not QUIET_PYLIMA:

        return roman.fit_pspl_roman(
            ev=event,
            t0_guess=t0,
            u0_guess=u0,
            tE_guess=tE,
        )

    buffer = StringIO()

    with (
        redirect_stdout(buffer),
        redirect_stderr(buffer),
    ):

        result = roman.fit_pspl_roman(
            ev=event,
            t0_guess=t0,
            u0_guess=u0,
            tE_guess=tE,
        )

    return result


# ============================================================
# Load production results
# ============================================================

if not ROMAN_FILE.exists():
    raise FileNotFoundError(
        ROMAN_FILE
    )

if not COMBINED_FILE.exists():
    raise FileNotFoundError(
        COMBINED_FILE
    )


roman_data = np.load(
    ROMAN_FILE,
    allow_pickle=False,
)

combined = np.load(
    COMBINED_FILE,
    allow_pickle=False,
)


# ============================================================
# Common physical / observational setup
# ============================================================

t = as_array(
    roman_data[
        "roman_times"
    ]
)

t0_true = float(
    roman_data[
        "t0_true"
    ]
)

tE_true = float(
    roman_data[
        "tE_true"
    ]
)

q_mass = float(
    roman_data[
        "q_mass"
    ]
)

qflux = float(
    roman_data[
        "qflux"
    ]
)


u0_grid = as_array(
    combined[
        "u0_grid"
    ]
)

P_grid = as_array(
    combined[
        "P_grid"
    ]
)

P_over_tE = as_array(
    combined[
        "P_over_tE"
    ]
)

D_intrinsic = np.asarray(
    combined[
        "D_INTRINSIC"
    ],
    dtype=float,
)

magnitudes = as_array(
    combined[
        "w149_magnitudes"
    ]
)

DELTA_CHI2 = np.asarray(
    combined[
        "DELTA_CHI2"
    ],
    dtype=float,
)


# Locate W149 = 21 plane.
im = int(
    np.where(
        np.isclose(
            magnitudes,
            SOURCE_MAG,
        )
    )[0][0]
)


# ============================================================
# Dense plotting grid
# ============================================================

t_dense = np.linspace(
    np.min(t),
    np.max(t),
    N_DENSE,
)

x_dense = (
    (t_dense - t0_true)
    / tE_true
)

x_obs = (
    (t - t0_true)
    / tE_true
)


# ============================================================
# Compute all representative cases
# ============================================================

results = []


for case in CASES:

    iu = case["iu0"]
    ip = case["iP"]

    u0_true = float(
        u0_grid[iu]
    )

    P_days = float(
        P_grid[ip]
    )

    Pte = float(
        P_over_tE[ip]
    )

    D_value = float(
        D_intrinsic[
            iu,
            ip,
        ]
    )

    chi2_stored = float(
        DELTA_CHI2[
            im,
            iu,
            ip,
        ]
    )

    print()
    print("=" * 90)
    print(case["label"])
    print("=" * 90)

    print(
        f"u0        = {u0_true:.10g}"
    )

    print(
        f"P/tE      = {Pte:.10g}"
    )

    print(
        f"P [d]     = {P_days:.10g}"
    )

    print(
        f"D         = {D_value:.8e}"
    )

    print(
        f"stored DeltaChi2 = "
        f"{chi2_stored:.8e}"
    )


    # ========================================================
    # BSPL truth at Roman epochs
    # ========================================================

    truth_obs = (
        roman.bspl_truth_magnification(
            t=t,
            t0_true=t0_true,
            u0_true=u0_true,
            tE_true=tE_true,
            P_days=P_days,
            q_mass=q_mass,
            qflux=qflux,
        )
    )

    A_bspl_obs = as_array(
        truth_obs[
            "A_bspl"
        ]
    )


    # ========================================================
    # Roman Asimov event
    # ========================================================

    asimov = (
        roman.make_roman_asimov_event(
            t=t,
            A_bspl=A_bspl_obs,
            source_mag=SOURCE_MAG,
        )
    )

    event = asimov[
        "event"
    ]

    telescope = asimov[
        "telescope"
    ]


    # ========================================================
    # Re-run exact production PSPL fit
    # ========================================================

    fit_result = run_fit_quietly(
        event=event,
        t0=t0_true,
        u0=u0_true,
        tE=tE_true,
    )

    best_t0 = float(
        fit_result[
            "best_t0"
        ]
    )

    best_u0 = float(
        fit_result[
            "best_u0"
        ]
    )

    best_tE = float(
        fit_result[
            "best_tE"
        ]
    )

    chi2_fit = float(
        fit_result[
            "chi2"
        ]
    )


    # ========================================================
    # Exact pyLIMA model at Roman epochs
    # ========================================================

    model_pspl = fit_result[
        "model"
    ]

    best_model = fit_result[
        "best_model"
    ]

    py_best = (
        model_pspl.compute_pyLIMA_parameters(
            best_model
        )
    )

    F_pspl_obs = as_array(
        model_pspl.compute_the_microlensing_model(
            telescope,
            py_best,
        )[
            "photometry"
        ]
    )


    # ========================================================
    # Asimov flux and uncertainty
    # ========================================================

    F_bspl_obs = as_array(
        telescope.lightcurve[
            "flux"
        ]
    )

    sigma_F = as_array(
        telescope.lightcurve[
            "err_flux"
        ]
    )


    # ========================================================
    # Baseline truth flux
    #
    # qflux=0 and no truth blend:
    #
    #     F_BSPL = F_base * A_BSPL
    #
    # ========================================================

    F_base_samples = (
        F_bspl_obs
        / A_bspl_obs
    )

    F_base = float(
        np.median(
            F_base_samples[
                np.isfinite(
                    F_base_samples
                )
            ]
        )
    )


    # ========================================================
    # Residuals at Roman epochs
    # ========================================================

    delta_F_obs = (
        F_bspl_obs
        - F_pspl_obs
    )

    residual_relative = (
        delta_F_obs
        / F_base
    )

    residual_sigma = (
        delta_F_obs
        / sigma_F
    )

    chi2_from_residuals = float(
        np.sum(
            residual_sigma ** 2
        )
    )


    # ========================================================
    # Independent linear photometric solution
    #
    # Used only to construct the smooth PSPL curve.
    # This should reproduce the pyLIMA observed model.
    # ========================================================

    A_pspl_obs = pspl_magnification(
        t=t,
        t0=best_t0,
        u0=best_u0,
        tE=best_tE,
    )

    Fs_linear, Fb_linear = (
        weighted_linear_flux_fit(
            A=A_pspl_obs,
            flux=F_bspl_obs,
            sigma=sigma_F,
        )
    )

    F_pspl_linear_obs = (
        Fs_linear
        * A_pspl_obs
        + Fb_linear
    )

    max_linear_difference = float(
        np.max(
            np.abs(
                F_pspl_linear_obs
                - F_pspl_obs
            )
        )
    )

    rms_linear_difference = float(
        np.sqrt(
            np.mean(
                (
                    F_pspl_linear_obs
                    - F_pspl_obs
                ) ** 2
            )
        )
    )


    # ========================================================
    # Dense BSPL truth
    # ========================================================

    truth_dense = (
        roman.bspl_truth_magnification(
            t=t_dense,
            t0_true=t0_true,
            u0_true=u0_true,
            tE_true=tE_true,
            P_days=P_days,
            q_mass=q_mass,
            qflux=qflux,
        )
    )

    A_bspl_dense = as_array(
        truth_dense[
            "A_bspl"
        ]
    )

    F_bspl_dense = (
        F_base
        * A_bspl_dense
    )


    # ========================================================
    # Dense PSPL
    # ========================================================

    A_pspl_dense = pspl_magnification(
        t=t_dense,
        t0=best_t0,
        u0=best_u0,
        tE=best_tE,
    )

    F_pspl_dense = (
        Fs_linear
        * A_pspl_dense
        + Fb_linear
    )


    # ========================================================
    # Dense physical residual
    # ========================================================

    relative_dense = (
        F_bspl_dense
        - F_pspl_dense
    ) / F_base


    # ========================================================
    # Cumulative Delta chi2
    # ========================================================

    order = np.argsort(t)

    x_sorted = (
        x_obs[order]
    )

    chi2_cumulative = np.cumsum(
        residual_sigma[
            order
        ] ** 2
    )


    # ========================================================
    # Validation
    # ========================================================

    rel_chi2_stored = (
        abs(
            chi2_from_residuals
            - chi2_stored
        )
        / max(
            abs(chi2_stored),
            1.0e-300,
        )
    )

    rel_chi2_fit = (
        abs(
            chi2_from_residuals
            - chi2_fit
        )
        / max(
            abs(chi2_fit),
            1.0e-300,
        )
    )

    print()
    print("Best PSPL:")
    print(
        f"  t0 = {best_t0:.10f}"
    )
    print(
        f"  u0 = {best_u0:.10g}"
    )
    print(
        f"  tE = {best_tE:.10g}"
    )

    print()
    print("Photometric solution:")
    print(
        f"  Fs = {Fs_linear:.10e}"
    )
    print(
        f"  Fb = {Fb_linear:.10e}"
    )
    print(
        f"  Fb/Fs = "
        f"{Fb_linear/Fs_linear:.6e}"
    )

    print()
    print("Delta chi2 validation:")
    print(
        f"  stored       = "
        f"{chi2_stored:.12e}"
    )
    print(
        f"  fit           = "
        f"{chi2_fit:.12e}"
    )
    print(
        f"  residual sum  = "
        f"{chi2_from_residuals:.12e}"
    )
    print(
        f"  rel diff stored = "
        f"{rel_chi2_stored:.3e}"
    )
    print(
        f"  rel diff fit    = "
        f"{rel_chi2_fit:.3e}"
    )

    print()
    print(
        "Dense photometric reconstruction check:"
    )
    print(
        f"  max |F_linear-F_pyLIMA| = "
        f"{max_linear_difference:.6e}"
    )
    print(
        f"  RMS difference           = "
        f"{rms_linear_difference:.6e}"
    )

    print()
    print(
        "Residual significance:"
    )
    print(
        f"  max |r_i| = "
        f"{np.max(np.abs(residual_sigma)):.6e}"
    )

    # We do not hard-fail on tiny optimizer reproducibility
    # differences, but large disagreement should be investigated.
    if rel_chi2_stored > 1.0e-4:

        print()
        print(
            "WARNING: rerun chi2 differs from stored "
            "value by more than 1e-4 relative."
        )


    results.append(
        {
            **case,

            "u0": u0_true,
            "P_days": P_days,
            "P_over_tE": Pte,
            "D": D_value,

            "chi2_stored": (
                chi2_stored
            ),

            "chi2_fit": (
                chi2_fit
            ),

            "chi2_residual": (
                chi2_from_residuals
            ),

            "best_t0": (
                best_t0
            ),

            "best_u0": (
                best_u0
            ),

            "best_tE": (
                best_tE
            ),

            "F_base": (
                F_base
            ),

            "x_dense": (
                x_dense
            ),

            "x_obs": (
                x_obs
            ),

            "F_bspl_dense_norm": (
                F_bspl_dense
                / F_base
            ),

            "F_pspl_dense_norm": (
                F_pspl_dense
                / F_base
            ),

            "F_bspl_obs_norm": (
                F_bspl_obs
                / F_base
            ),

            "F_pspl_obs_norm": (
                F_pspl_obs
                / F_base
            ),

            "sigma_F_norm": (
                sigma_F
                / F_base
            ),

            "relative_dense": (
                relative_dense
            ),

            "residual_relative": (
                residual_relative
            ),

            "residual_sigma": (
                residual_sigma
            ),

            "x_sorted": (
                x_sorted
            ),

            "chi2_cumulative": (
                chi2_cumulative
            ),
        }
    )


# ============================================================
# Plot one publication-style figure per case
# ============================================================

for r in results:

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.0, 9.2),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                2.5,
                1.25,
                1.25,
                1.25,
            ],
            "hspace": 0.06,
        },
    )

    ax_lc, ax_df, ax_sig, ax_chi = axes


    # --------------------------------------------------------
    # Light curve
    # --------------------------------------------------------

    ax_lc.plot(
        r["x_dense"],
        r["F_bspl_dense_norm"],
        lw=2.0,
        label="BSPL truth",
        zorder=2,
    )

    ax_lc.plot(
        r["x_dense"],
        r["F_pspl_dense_norm"],
        lw=1.0,
        label="Best PSPL",
        zorder=3,
    )

    # --------------------------------------------------------
    # Roman sampling and representative uncertainties
    # --------------------------------------------------------

    # All Roman epochs as faint points.
    ax_lc.scatter(
        r["x_obs"],
        r["F_bspl_obs_norm"],
        s=1.5,
        alpha=0.12,
        rasterized=True,
        zorder=1,
        label="Roman epochs",
    )

    # The Asimov data are noiseless, so the points lie on the
    # BSPL truth. Error bars show the 1-sigma photometric
    # uncertainties actually used in the Roman chi2.
    #
    # Only a subset is drawn for readability; all epochs enter
    # the fit and Delta chi2 calculation.
    if SHOW_ERRORBARS:

        idx_err = np.arange(
            0,
            len(r["x_obs"]),
            ERRORBAR_EVERY,
        )

        ax_lc.errorbar(
            r["x_obs"][idx_err],
            r["F_bspl_obs_norm"][idx_err],
            yerr=r["sigma_F_norm"][idx_err],
            fmt="o",
            markersize=ERRORBAR_MARKERSIZE,
            linestyle="none",
            elinewidth=ERRORBAR_ELINEWIDTH,
            capsize=ERRORBAR_CAPSIZE,
            alpha=ERRORBAR_ALPHA,
            rasterized=True,
            zorder=4,
            label=(
                rf"$1\sigma$ uncertainty "
                rf"(every {ERRORBAR_EVERY}th epoch)"
            ),
        )

    ax_lc.set_ylabel(
        r"$F/F_{\rm base}$"
    )

    ax_lc.legend(
        loc="best",
        frameon=False,
        ncol=2,
    )

    ax_lc.set_title(
        (
            f"{r['label']}"
            "\n"
            rf"$u_0={r['u0']:.4g}$, "
            rf"$P/t_E={r['P_over_tE']:.3g}$, "
            rf"$D={r['D']:.2e}$, "
            rf"$\Delta\chi^2_{{\rm Roman}}="
            rf"{r['chi2_stored']:.3g}$"
        )
    )


    # --------------------------------------------------------
    # Physical relative residual
    # --------------------------------------------------------

    ax_df.axhline(
        0.0,
        lw=0.8,
        color="0.5",
    )

    ax_df.plot(
        r["x_dense"],
        r["relative_dense"],
        lw=1.2,
    )

    ax_df.set_ylabel(
        (
            r"$(F_{\rm BSPL}-F_{\rm PSPL})"
            "\n"
            r"/F_{\rm base}$"
        )
    )


    # --------------------------------------------------------
    # Error-normalized residuals
    # --------------------------------------------------------

    ax_sig.axhline(
        0.0,
        lw=0.8,
        color="0.5",
    )

    ax_sig.axhline(
        1.0,
        lw=0.7,
        ls="--",
        color="0.65",
    )

    ax_sig.axhline(
        -1.0,
        lw=0.7,
        ls="--",
        color="0.65",
    )

    ax_sig.scatter(
        r["x_obs"],
        r["residual_sigma"],
        s=2.5,
        alpha=0.45,
        rasterized=True,
    )

    ax_sig.set_ylabel(
        (
            r"$(F_{\rm BSPL}-F_{\rm PSPL})"
            "\n"
            r"/\sigma_F$"
        )
    )


    # --------------------------------------------------------
    # Cumulative chi2
    # --------------------------------------------------------

    ax_chi.step(
        r["x_sorted"],
        r["chi2_cumulative"],
        where="post",
        lw=1.4,
    )

    ax_chi.axhline(
        r["chi2_stored"],
        lw=0.8,
        ls="--",
        color="0.5",
    )

    ax_chi.set_ylabel(
        r"$\Delta\chi^2(<t)$"
    )

    ax_chi.set_xlabel(
        r"$(t-t_0)/t_E$"
    )


    # --------------------------------------------------------
    # Common axes formatting
    # --------------------------------------------------------

    for ax in axes:

        ax.tick_params(
            which="both",
            direction="in",
            top=True,
            right=True,
        )

        ax.grid(
            alpha=0.15,
        )

        ax.set_xlim(
            np.min(x_obs),
            np.max(x_obs),
        )


    fig.subplots_adjust(
        left=0.16,
        right=0.97,
        bottom=0.08,
        top=0.91,
    )

    pdf = (
        OUTDIR
        / f"{r['key']}.pdf"
    )

    png = (
        OUTDIR
        / f"{r['key']}.png"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        "Saved:",
        pdf,
    )

    print(
        "Saved:",
        png,
    )


# ============================================================
# Combined appendix figure
#
# Rows = representative systems
# Columns =
#   1. light curve
#   2. physical residual
#   3. normalized residual
#   4. cumulative Delta chi2
# ============================================================

ncase = len(results)

fig, axes = plt.subplots(
    ncase,
    4,
    figsize=(14.5, 10.5),
    sharex="col",
)

for ir, r in enumerate(results):

    ax0 = axes[ir, 0]
    ax1 = axes[ir, 1]
    ax2 = axes[ir, 2]
    ax3 = axes[ir, 3]


    # --------------------------------------------------------
    # Column 1: light curve
    # --------------------------------------------------------

    ax0.plot(
        r["x_dense"],
        r["F_bspl_dense_norm"],
        lw=2.0,
        label="BSPL truth",
    )

    ax0.plot(
        r["x_dense"],
        r["F_pspl_dense_norm"],
        lw=1.0,
        label="Best PSPL",
    )

    ax0.scatter(
        r["x_obs"],
        r["F_bspl_obs_norm"],
        s=1.2,
        alpha=0.10,
        rasterized=True,
        zorder=1,
    )

    if SHOW_ERRORBARS:

        idx_err = np.arange(
            0,
            len(r["x_obs"]),
            ERRORBAR_EVERY,
        )

        ax0.errorbar(
            r["x_obs"][idx_err],
            r["F_bspl_obs_norm"][idx_err],
            yerr=r["sigma_F_norm"][idx_err],
            fmt="o",
            markersize=1.3,
            linestyle="none",
            elinewidth=0.45,
            capsize=0.0,
            alpha=0.35,
            rasterized=True,
            zorder=4,
        )

    ax0.set_ylabel(
        (
            f"{r['key'][0]}\n"
            r"$F/F_{\rm base}$"
        )
    )


    # --------------------------------------------------------
    # Column 2: physical residual
    # --------------------------------------------------------

    ax1.axhline(
        0.0,
        lw=0.7,
        color="0.5",
    )

    ax1.plot(
        r["x_dense"],
        r["relative_dense"],
        lw=1.1,
    )


    # --------------------------------------------------------
    # Column 3: normalized residual
    # --------------------------------------------------------

    ax2.axhline(
        0.0,
        lw=0.7,
        color="0.5",
    )

    ax2.axhline(
        1.0,
        lw=0.6,
        ls="--",
        color="0.65",
    )

    ax2.axhline(
        -1.0,
        lw=0.6,
        ls="--",
        color="0.65",
    )

    ax2.scatter(
        r["x_obs"],
        r["residual_sigma"],
        s=1.8,
        alpha=0.4,
        rasterized=True,
    )


    # --------------------------------------------------------
    # Column 4: cumulative chi2
    # --------------------------------------------------------

    ax3.step(
        r["x_sorted"],
        r["chi2_cumulative"],
        where="post",
        lw=1.2,
    )

    ax3.axhline(
        r["chi2_stored"],
        lw=0.7,
        ls="--",
        color="0.5",
    )


    # --------------------------------------------------------
    # Compact annotation
    # --------------------------------------------------------

    ax0.text(
        0.03,
        0.95,
        (
            rf"$u_0={r['u0']:.3g}$"
            "\n"
            rf"$P/t_E={r['P_over_tE']:.3g}$"
            "\n"
            rf"$D={r['D']:.1e}$"
            "\n"
            rf"$\Delta\chi^2={r['chi2_stored']:.2g}$"
        ),
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )


# ============================================================
# Column titles
# ============================================================

axes[0, 0].set_title(
    "Roman light curve"
)

axes[0, 1].set_title(
    r"Relative flux residual"
)

axes[0, 2].set_title(
    r"Error-normalized residual"
)

axes[0, 3].set_title(
    r"Cumulative $\Delta\chi^2$"
)


# ============================================================
# Labels
# ============================================================

for ir in range(ncase):

    axes[ir, 1].set_ylabel(
        r"$\Delta F/F_{\rm base}$"
    )

    axes[ir, 2].set_ylabel(
        r"$\Delta F/\sigma_F$"
    )

    axes[ir, 3].set_ylabel(
        r"$\Delta\chi^2(<t)$"
    )


for ic in range(4):

    axes[-1, ic].set_xlabel(
        r"$(t-t_0)/t_E$"
    )


# ============================================================
# Shared formatting
# ============================================================

for ax in axes.flat:

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    ax.grid(
        alpha=0.12,
    )

    ax.set_xlim(
        np.min(x_obs),
        np.max(x_obs),
    )


axes[0, 0].legend(
    loc="best",
    frameon=False,
    fontsize=8,
)


fig.subplots_adjust(
    left=0.07,
    right=0.985,
    bottom=0.07,
    top=0.94,
    hspace=0.18,
    wspace=0.30,
)


combined_pdf = (
    OUTDIR
    / "roman_representative_lightcurves_appendix.pdf"
)

combined_png = (
    OUTDIR
    / "roman_representative_lightcurves_appendix.png"
)

fig.savefig(
    combined_pdf,
    bbox_inches="tight",
)

fig.savefig(
    combined_png,
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("=" * 90)
print("FINAL OUTPUTS")
print("=" * 90)

print(
    combined_pdf
)

print(
    combined_png
)

print("=" * 90)
