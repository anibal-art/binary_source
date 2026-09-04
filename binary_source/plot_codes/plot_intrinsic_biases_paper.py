# ============================================================
# PAPER-QUALITY FIGURES
#
# FIGURE 1
# --------
# Relative Einstein-timescale bias:
#
#   (a) u0 scan
#   (b) q_M scan
#
# Color:
#       Delta tE / tE
#
# Black contours:
#       D_BSPL-PSPL
#
#
# FIGURE 2
# --------
# Correlated PSPL parameter shifts:
#
#       Delta u0 / u0,true
#       Delta tE / tE
#
# versus P/tE at:
#
#       u0 = 0.1
#       q_M = 0.5
#
#
# OUTPUT:
#   - PDF vector/raster hybrid
#   - PNG at 600 dpi
#
# ============================================================


import os
import re
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

from matplotlib.ticker import LogLocator
from scipy.stats import spearmanr


# ============================================================
# GLOBAL PAPER STYLE
# ============================================================

PAPER_DPI = 600

plt.rcParams.update({

    # --------------------------------------------------------
    # Fonts
    # --------------------------------------------------------
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",

    "font.size": 9,

    "axes.labelsize": 10,
    "axes.titlesize": 10,

    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,

    "legend.fontsize": 8,

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------
    "axes.linewidth": 0.8,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,

    "xtick.minor.size": 2.3,
    "ytick.minor.size": 2.3,

    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,

    "xtick.minor.width": 0.55,
    "ytick.minor.width": 0.55,

    # --------------------------------------------------------
    # Lines
    # --------------------------------------------------------
    "lines.linewidth": 1.4,

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------
    "savefig.dpi": PAPER_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# CONFIGURATION
# ============================================================

tE_plot = 30.0

N_u0 = 100
N_q = 100

u0_grid_input = np.logspace(
    -2,
    1,
    N_u0,
)

q_grid_input = np.logspace(
    -4,
    0,
    N_q,
)


# Fixed u0 in q_M scan
u0_q_scan = 0.1


# ============================================================
# D CONTOURS
# ============================================================

D_LEVELS = np.array([
    1e-4,
    1e-3,
    1e-2,
    1e-1,
])

LABEL_D_CONTOURS = True


# ============================================================
# COLOR SCALE
# ============================================================

# Use 99 percentile to avoid a tiny number of pathological
# fits defining the entire color scale.
#
# Set 100.0 for absolute extrema.

ROBUST_PERCENTILE = 99.0


# ============================================================
# FIGURE 2 CONFIGURATION
# ============================================================

u0_slice_target = 0.1
q_slice_target = 0.5

# Usually False for the paper figure.
# True can be useful as a consistency check.
SHOW_Q_CONSISTENCY = False


# ============================================================
# PATHS
# ============================================================

home = os.path.expanduser("~")

results_root = os.path.join(
    home,
    "binary_source",
    "results",
)


# ============================================================
# FIND u0 SCAN
# ============================================================

u0_candidates = [

    os.path.join(
        results_root,
        f"scan_u0_tE{int(tE_plot)}",
    ),

    os.path.join(
        results_root,
        "scan_many_tE_200x200",
        f"scan_u0_tE{int(tE_plot)}",
    ),

]


u0_directory = None


for candidate in u0_candidates:

    files_test = glob.glob(
        os.path.join(
            candidate,
            "scan_kepler_u0_*.npz",
        )
    )

    if len(files_test) > 0:

        u0_directory = candidate
        break


if u0_directory is None:

    raise FileNotFoundError(
        "Could not find the u0 scan.\n\n"
        "Searched:\n"
        + "\n".join(u0_candidates)
    )


# ============================================================
# MASS-RATIO SCAN
# ============================================================

q_candidates = [

    # Final production used for the paper
    os.path.join(
        results_root,
        "final_6b888737a3c3",
        f"qmass_fixed_mtot_tE{int(tE_plot)}",
    ),

    # Legacy location, kept for backward compatibility
    os.path.join(
        results_root,
        f"scan_q_Mtotfixed_tE{int(tE_plot)}",
    ),

]


q_directory = None


for candidate in q_candidates:

    q_files_test = glob.glob(
        os.path.join(
            candidate,
            "scan_kepler_q_*.npz",
        )
    )

    if len(q_files_test) > 0:

        q_directory = candidate
        break


if q_directory is None:

    raise FileNotFoundError(
        "Could not find the mass-ratio scan.\n\n"
        "Searched:\n"
        + "\n".join(q_candidates)
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_directory = os.path.join(
    results_root,
    "bias_paper_figures",
)

os.makedirs(
    output_directory,
    exist_ok=True,
)


print("=" * 80)
print("u0 scan:")
print(u0_directory)
print()
print("q_M scan:")
print(q_directory)
print()
print("Output:")
print(output_directory)
print("=" * 80)


# ============================================================
# FILE INDEX
# ============================================================

def get_file_index(filename, parameter):

    pattern = (
        rf"scan_kepler_{parameter}_(\d+)\.npz$"
    )

    match = re.search(
        pattern,
        os.path.basename(filename),
    )

    if match is None:

        raise ValueError(
            f"Could not extract index from:\n{filename}"
        )

    return int(match.group(1))


# ============================================================
# LOAD SCAN
# ============================================================

def load_scan(
    directory,
    file_parameter,
    parameter_grid,
    tE_true,
):

    pattern = os.path.join(
        directory,
        f"scan_kepler_{file_parameter}_*.npz",
    )

    files = sorted(
        glob.glob(pattern)
    )

    if len(files) == 0:
        raise FileNotFoundError(pattern)


    # --------------------------------------------------------
    # Period grid
    # --------------------------------------------------------

    with np.load(
        files[0],
        allow_pickle=False,
    ) as d:

        P_grid = d["P_grid"].astype(float)


    shape = (
        len(parameter_grid),
        len(P_grid),
    )


    DT0 = np.full(shape, np.nan)
    DU0 = np.full(shape, np.nan)
    DTE = np.full(shape, np.nan)

    D = np.full(shape, np.nan)
    RMS = np.full(shape, np.nan)

    SUCCESS = np.zeros(
        shape,
        dtype=bool,
    )


    # --------------------------------------------------------
    # Load all files
    # --------------------------------------------------------

    for filename in files:

        k = get_file_index(
            filename,
            file_parameter,
        )

        if not (
            0 <= k < len(parameter_grid)
        ):
            continue


        with np.load(
            filename,
            allow_pickle=False,
        ) as d:


            required = [
                "P_grid",
                "DT0",
                "DU0",
                "DTE",
                "D",
                "RMS",
                "SUCCESS",
            ]


            for key in required:

                if key not in d.files:

                    raise KeyError(
                        f"\nMissing key '{key}' in\n"
                        f"{filename}\n\n"
                        f"Available keys:\n{d.files}"
                    )


            P_this = d[
                "P_grid"
            ].astype(float)


            if not np.allclose(
                P_this,
                P_grid,
            ):

                raise ValueError(
                    f"P_grid differs in:\n{filename}"
                )


            dt0 = d[
                "DT0"
            ].astype(float)

            du0 = d[
                "DU0"
            ].astype(float)

            dte = d[
                "DTE"
            ].astype(float)

            dmetric = d[
                "D"
            ].astype(float)

            rms = d[
                "RMS"
            ].astype(float)

            success = d[
                "SUCCESS"
            ].astype(bool)


        # ----------------------------------------------------
        # Masks
        # ----------------------------------------------------

        valid_bias = (
            success
            & np.isfinite(dt0)
            & np.isfinite(du0)
            & np.isfinite(dte)
        )

        valid_D = (
            success
            & np.isfinite(dmetric)
            & (dmetric > 0)
        )

        valid_RMS = (
            success
            & np.isfinite(rms)
            & (rms > 0)
        )


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        DT0[k, valid_bias] = dt0[valid_bias]
        DU0[k, valid_bias] = du0[valid_bias]
        DTE[k, valid_bias] = dte[valid_bias]

        D[k, valid_D] = dmetric[valid_D]

        RMS[k, valid_RMS] = rms[valid_RMS]

        SUCCESS[k, :] = success


    return {

        "parameter":
            np.asarray(
                parameter_grid,
                dtype=float,
            ),

        "P_grid":
            P_grid,

        "P_over_tE":
            P_grid / tE_true,

        "DT0":
            DT0,

        "DU0":
            DU0,

        "DTE":
            DTE,

        "D":
            D,

        "RMS":
            RMS,

        "SUCCESS":
            SUCCESS,

    }


# ============================================================
# LOAD BOTH SCANS
# ============================================================

u0_data = load_scan(
    directory=u0_directory,
    file_parameter="u0",
    parameter_grid=u0_grid_input,
    tE_true=tE_plot,
)


q_data = load_scan(
    directory=q_directory,
    file_parameter="q",
    parameter_grid=q_grid_input,
    tE_true=tE_plot,
)


# ============================================================
# NORMALIZED BIASES
# ============================================================

# u0 scan

u0_DT0_norm = (
    u0_data["DT0"]
    /
    tE_plot
)


u0_DU0_norm = (
    u0_data["DU0"]
    /
    u0_data["parameter"][:, None]
)


u0_DTE_norm = (
    u0_data["DTE"]
    /
    tE_plot
)


# q_M scan

q_DT0_norm = (
    q_data["DT0"]
    /
    tE_plot
)


q_DU0_norm = (
    q_data["DU0"]
    /
    u0_q_scan
)


q_DTE_norm = (
    q_data["DTE"]
    /
    tE_plot
)


# ============================================================
# LOGARITHMIC BIN EDGES
# ============================================================

def log_edges(x):

    x = np.asarray(
        x,
        dtype=float,
    )

    lx = np.log10(x)

    ledges = np.empty(
        len(x) + 1
    )

    ledges[1:-1] = (
        lx[:-1]
        +
        lx[1:]
    ) / 2.0


    ledges[0] = (
        lx[0]
        -
        0.5
        *
        (
            lx[1]
            -
            lx[0]
        )
    )


    ledges[-1] = (
        lx[-1]
        +
        0.5
        *
        (
            lx[-1]
            -
            lx[-2]
        )
    )


    return 10**ledges


# ============================================================
# D CONTOURS
# ============================================================

def add_D_contours(
    ax,
    x,
    y,
    D_map,
    levels=D_LEVELS,
):

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )


    valid_values = D_map[
        np.isfinite(D_map)
        &
        (D_map > 0)
    ]


    if len(valid_values) == 0:
        return None


    dmin = np.nanmin(
        valid_values
    )

    dmax = np.nanmax(
        valid_values
    )


    levels_here = [
        level
        for level in levels
        if dmin < level < dmax
    ]


    if len(levels_here) == 0:
        return None


    cs = ax.contour(

        X,
        Y,

        np.ma.masked_invalid(
            D_map
        ),

        levels=levels_here,

        colors="black",

        linewidths=0.90,

        linestyles="-",

        zorder=10,
    )


    # --------------------------------------------------------
    # White halo
    # --------------------------------------------------------

    try:

        for collection in cs.collections:

            collection.set_path_effects([

                pe.Stroke(
                    linewidth=1.8,
                    foreground="white",
                ),

                pe.Normal(),

            ])

    except AttributeError:

        pass


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if LABEL_D_CONTOURS:

        fmt = {
            level:
            rf"$D=10^{{{int(np.log10(level))}}}$"
            for level in levels_here
        }


        labels = ax.clabel(

            cs,

            fmt=fmt,

            fontsize=7.0,

            inline=True,

            inline_spacing=5,

        )


        for text in labels:

            text.set_path_effects([

                pe.Stroke(
                    linewidth=2.0,
                    foreground="white",
                ),

                pe.Normal(),

            ])


    return cs


# ============================================================
# COMMON AXIS STYLE
# ============================================================

def format_log_panel(ax):

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    ax.xaxis.set_major_locator(
        LogLocator(
            base=10.0
        )
    )

    ax.yaxis.set_major_locator(
        LogLocator(
            base=10.0
        )
    )

    ax.grid(False)


# ============================================================
# ============================================================
#
# FIGURE 1
#
# tE BIAS MAPS + D CONTOURS
#
# ============================================================
# ============================================================


# ============================================================
# SHARED COLOR RANGE
# ============================================================

all_DTE_bias = np.concatenate([

    np.abs(
        u0_DTE_norm[
            np.isfinite(
                u0_DTE_norm
            )
        ]
    ),

    np.abs(
        q_DTE_norm[
            np.isfinite(
                q_DTE_norm
            )
        ]
    ),

])


vmax_DTE = np.nanpercentile(
    all_DTE_bias,
    ROBUST_PERCENTILE,
)


norm_DTE = mcolors.TwoSlopeNorm(
    vmin=-vmax_DTE,
    vcenter=0.0,
    vmax=vmax_DTE,
)


# ============================================================
# GRID EDGES
# ============================================================

u0_edges = log_edges(
    u0_data["parameter"]
)

q_edges = log_edges(
    q_data["parameter"]
)

P_u0_edges = log_edges(
    u0_data["P_over_tE"]
)

P_q_edges = log_edges(
    q_data["P_over_tE"]
)


# ============================================================
# FIGURE 1
#
# ~7.2 inch wide:
# appropriate for a two-column paper figure.
# ============================================================

fig1, axes = plt.subplots(

    1,
    2,

    figsize=(
        7.2,
        3.25,
    ),

)


fig1.subplots_adjust(

    left=0.085,

    right=0.985,

    bottom=0.16,

    top=0.78,

    wspace=0.20,

)


cmap = plt.get_cmap(
    "RdBu_r"
)


# ============================================================
# PANEL (a) — u0 SCAN
# ============================================================

ax = axes[0]


pcm = ax.pcolormesh(

    u0_edges,
    P_u0_edges,

    np.ma.masked_invalid(
        u0_DTE_norm
    ).T,

    cmap=cmap,

    norm=norm_DTE,

    shading="auto",

    # Important:
    # rasterize only the dense heatmap
    rasterized=True,

    zorder=1,

)


add_D_contours(

    ax=ax,

    x=u0_data["parameter"],

    y=u0_data["P_over_tE"],

    D_map=u0_data["D"],

)


format_log_panel(ax)


ax.set_xlabel(
    r"$u_0$"
)

ax.set_ylabel(
    r"$P/t_E$"
)


ax.text(

    0.045,
    0.95,

    "(a)",

    transform=ax.transAxes,

    ha="left",
    va="top",

    fontsize=9.5,

    zorder=20,

)


# ============================================================
# PANEL (b) — q_M SCAN
# ============================================================

ax = axes[1]


ax.pcolormesh(

    q_edges,
    P_q_edges,

    np.ma.masked_invalid(
        q_DTE_norm
    ).T,

    cmap=cmap,

    norm=norm_DTE,

    shading="auto",

    rasterized=True,

    zorder=1,

)


add_D_contours(

    ax=ax,

    x=q_data["parameter"],

    y=q_data["P_over_tE"],

    D_map=q_data["D"],

)


format_log_panel(ax)


ax.set_xlabel(
    r"$q_M=M_2/M_1$"
)


ax.text(

    0.045,
    0.95,

    "(b)",

    transform=ax.transAxes,

    ha="left",
    va="top",

    fontsize=9.5,

    zorder=20,

)


# ============================================================
# COMMON TOP COLORBAR
# ============================================================

pos_left = axes[0].get_position()
pos_right = axes[1].get_position()


cax = fig1.add_axes([

    pos_left.x0,

    0.835,

    pos_right.x1
    -
    pos_left.x0,

    0.028,

])


cbar = fig1.colorbar(

    pcm,

    cax=cax,

    orientation="horizontal",

    extend="both",

)


cbar.ax.xaxis.set_ticks_position(
    "top"
)

cbar.ax.xaxis.set_label_position(
    "top"
)


cbar.ax.tick_params(
    direction="in",
    labelsize=8,
    pad=1.5,
)


cbar.set_label(

    r"$\Delta t_E/t_E$",

    fontsize=10,

    labelpad=3,

)


# ============================================================
# SAVE FIGURE 1
# ============================================================

figure1_png = os.path.join(
    output_directory,
    f"paper_tE_bias_D_contours_tE{int(tE_plot)}.png",
)

figure1_pdf = os.path.join(
    output_directory,
    f"paper_tE_bias_D_contours_tE{int(tE_plot)}.pdf",
)


fig1.savefig(

    figure1_png,

    dpi=PAPER_DPI,

    bbox_inches="tight",

    facecolor="white",

)


fig1.savefig(

    figure1_pdf,

    bbox_inches="tight",

    facecolor="white",

)


print()
print("=" * 80)
print("FIGURE 1 SAVED")
print(figure1_png)
print(figure1_pdf)
print("=" * 80)


plt.show()


# ============================================================
# ============================================================
#
# FIGURE 2
#
# CORRELATED u0 -- tE PARAMETER SHIFTS
#
# ============================================================
# ============================================================


# ============================================================
# FIND COMMON u0 = 0.1 SLICE
# ============================================================

iu = np.argmin(

    np.abs(

        np.log10(
            u0_data["parameter"]
        )

        -

        np.log10(
            u0_slice_target
        )

    )

)


u0_used = (
    u0_data[
        "parameter"
    ][iu]
)


# ============================================================
# FIND q_M ~ 0.5 SLICE
# ============================================================

iq = np.argmin(

    np.abs(

        np.log10(
            q_data["parameter"]
        )

        -

        np.log10(
            q_slice_target
        )

    )

)


q_used = (
    q_data[
        "parameter"
    ][iq]
)


# ============================================================
# EXTRACT u0-SCAN SLICE
# ============================================================

x_u0 = (
    u0_data[
        "P_over_tE"
    ]
)


du0_u0 = (
    u0_DU0_norm[
        iu,
        :
    ]
)


dtE_u0 = (
    u0_DTE_norm[
        iu,
        :
    ]
)


# ============================================================
# q_M-SCAN CONSISTENCY SLICE
# ============================================================

x_q = (
    q_data[
        "P_over_tE"
    ]
)


du0_q = (
    q_DU0_norm[
        iq,
        :
    ]
)


dtE_q = (
    q_DTE_norm[
        iq,
        :
    ]
)


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

valid_slice = (

    np.isfinite(
        du0_u0
    )

    &

    np.isfinite(
        dtE_u0
    )

)


rho_slice, p_slice = spearmanr(

    du0_u0[
        valid_slice
    ],

    dtE_u0[
        valid_slice
    ],

)


print()
print("=" * 80)
print("FIGURE 2")
print(f"u0 used                 = {u0_used:.8f}")
print(f"q_M closest to 0.5      = {q_used:.8f}")
print(f"Spearman rho            = {rho_slice:.5f}")
print(f"Spearman p-value        = {p_slice:.5e}")
print("=" * 80)


# ============================================================
# FIGURE 2
#
# ~3.5 inch:
# suitable for single-column placement.
# ============================================================

fig2, ax = plt.subplots(

    figsize=(
        3.65,
        3.0,
    ),

)


# ============================================================
# MAIN CURVES
# ============================================================

ax.plot(

    x_u0,
    du0_u0,

    lw=1.7,

    label=
    r"$\Delta u_0/u_{0,\rm true}$",

)


ax.plot(

    x_u0,
    dtE_u0,

    lw=1.7,

    label=
    r"$\Delta t_E/t_E$",

)


# ============================================================
# OPTIONAL CONSISTENCY OVERLAY
# ============================================================

if SHOW_Q_CONSISTENCY:


    ax.plot(

        x_q,
        du0_q,

        "--",

        lw=1.0,

        alpha=0.7,

        label=(
            rf"$\Delta u_0/u_0$, "
            rf"$q_M={q_used:.3f}$ scan"
        ),

    )


    ax.plot(

        x_q,
        dtE_q,

        "--",

        lw=1.0,

        alpha=0.7,

        label=(
            rf"$\Delta t_E/t_E$, "
            rf"$q_M={q_used:.3f}$ scan"
        ),

    )


# ============================================================
# ZERO LINE
# ============================================================

ax.axhline(

    0.0,

    lw=0.75,

    linestyle=":",

    color="0.35",

    zorder=0,

)


# ============================================================
# FORMAT
# ============================================================

ax.set_xscale(
    "log"
)


ax.set_xlabel(
    r"$P/t_E$"
)


ax.set_ylabel(
    r"Relative PSPL parameter bias"
)


ax.tick_params(

    axis="both",

    which="both",

    direction="in",

    top=True,

    right=True,

)


ax.xaxis.set_major_locator(
    LogLocator(
        base=10
    )
)


ax.grid(
    False
)


# ============================================================
# ANNOTATION
# ============================================================

ax.text(

    0.96,
    0.95,

    (
        rf"$u_0={u0_used:.2f}$"
        "\n"
        rf"$q_M=0.5$"
        "\n"
        rf"$\rho_s={rho_slice:.3f}$"
    ),

    transform=ax.transAxes,

    ha="right",
    va="top",

    fontsize=8.2,

)


# ============================================================
# LEGEND
# ============================================================

ax.legend(

    frameon=False,

    loc="lower right",

    handlelength=2.3,

    borderaxespad=0.5,

)


# ============================================================
# SAVE FIGURE 2
# ============================================================

figure2_png = os.path.join(
    output_directory,
    f"paper_u0_tE_correlated_bias_tE{int(tE_plot)}.png",
)

figure2_pdf = os.path.join(
    output_directory,
    f"paper_u0_tE_correlated_bias_tE{int(tE_plot)}.pdf",
)


fig2.savefig(

    figure2_png,

    dpi=PAPER_DPI,

    bbox_inches="tight",

    facecolor="white",

)


fig2.savefig(

    figure2_pdf,

    bbox_inches="tight",

    facecolor="white",

)


print()
print("=" * 80)
print("FIGURE 2 SAVED")
print(figure2_png)
print(figure2_pdf)
print("=" * 80)

plt.show()
