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
# PAPER STYLE (shared)
# ============================================================

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply_paper_style, fig_width, heatmap, panel_label,
                         save_figure, halo, pow10_label, BIAS_CMAP,
                         FS_ANNOT, LW_THIN, C_BLUE, C_VERMILION)

apply_paper_style()



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
    home,
    "binary_source",
    "figures",
    "current",
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
#
# IMPORTANT:
#
# Do NOT assume the number or spacing of parameter nodes.
#
# The production grids have changed:
#
#   u0 scan : 200 nodes
#   qM scan : 300 nodes
#
# The physical parameter is therefore recovered directly
# from each NPZ file:
#
#   u0 = truth[1]
#   qM = M2/M1 = truth[6]/truth[5]
#
# This makes the plotting independent of the production-grid
# resolution and prevents file-index -> coordinate mismatches.
# ============================================================

def load_scan(

    directory,
    file_parameter,
    parameter_grid=None,   # retained only for backward compatibility
    tE_true=None,

):

    if tE_true is None:

        raise ValueError(
            "tE_true must be provided."
        )


    pattern = os.path.join(

        directory,

        f"scan_kepler_{file_parameter}_*.npz",

    )


    files = sorted(
        glob.glob(
            pattern
        )
    )


    if len(files) == 0:

        raise FileNotFoundError(
            pattern
        )


    rows = []

    P_grid_ref = None


    # ========================================================
    # FILE LOOP
    # ========================================================

    for filename in files:


        with np.load(
            filename,
            allow_pickle=False,
        ) as d:


            required = [

                "truth",
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
                        f"Available keys:\n"
                        f"{d.files}"
                    )


            truth = np.asarray(
                d["truth"],
                dtype=float,
            )


            if truth.size < 7:

                raise RuntimeError(

                    f"truth array too short in:\n"
                    f"{filename}"
                )


            # ------------------------------------------------
            # Verify tE
            # ------------------------------------------------

            file_tE = float(
                truth[2]
            )


            if not np.isclose(

                file_tE,

                float(tE_true),

                rtol=0.0,

                atol=1e-10,

            ):

                raise RuntimeError(

                    f"Unexpected tE in:\n"
                    f"{filename}\n"
                    f"file tE = {file_tE}\n"
                    f"expected = {tE_true}"
                )


            # ------------------------------------------------
            # Recover the ACTUAL scanned parameter
            # ------------------------------------------------

            if file_parameter == "u0":

                parameter_value = float(
                    truth[1]
                )


            elif file_parameter == "q":

                M1 = float(
                    truth[5]
                )

                M2 = float(
                    truth[6]
                )


                if M1 <= 0:

                    raise RuntimeError(

                        f"Invalid M1={M1} in:\n"
                        f"{filename}"
                    )


                parameter_value = (
                    M2 / M1
                )


                # Optional consistency check against the
                # explicitly stored q_mass_true scalar.
                if "q_mass_true" in d.files:

                    q_stored = float(
                        np.asarray(
                            d["q_mass_true"]
                        ).reshape(-1)[0]
                    )


                    if not np.isclose(

                        parameter_value,

                        q_stored,

                        rtol=1e-10,

                        atol=1e-14,

                    ):

                        raise RuntimeError(

                            f"qM inconsistency in:\n"
                            f"{filename}\n"
                            f"M2/M1 = {parameter_value}\n"
                            f"stored = {q_stored}"
                        )


            else:

                raise ValueError(

                    f"Unknown file_parameter="
                    f"{file_parameter!r}"
                )


            # ------------------------------------------------
            # Period grid
            # ------------------------------------------------

            P_this = np.asarray(
                d["P_grid"],
                dtype=float,
            )


            if P_grid_ref is None:

                P_grid_ref = (
                    P_this.copy()
                )


            elif not np.allclose(

                P_this,

                P_grid_ref,

            ):

                raise ValueError(

                    f"P_grid differs in:\n"
                    f"{filename}"
                )


            # ------------------------------------------------
            # Numerical products
            # ------------------------------------------------

            dt0 = np.asarray(
                d["DT0"],
                dtype=float,
            )

            du0 = np.asarray(
                d["DU0"],
                dtype=float,
            )

            dte = np.asarray(
                d["DTE"],
                dtype=float,
            )

            dmetric = np.asarray(
                d["D"],
                dtype=float,
            )

            rms = np.asarray(
                d["RMS"],
                dtype=float,
            )

            success = np.asarray(
                d["SUCCESS"],
                dtype=bool,
            )


        # ====================================================
        # MASK INVALID VALUES
        # ====================================================

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


        dt0_clean = np.full_like(
            dt0,
            np.nan,
            dtype=float,
        )

        du0_clean = np.full_like(
            du0,
            np.nan,
            dtype=float,
        )

        dte_clean = np.full_like(
            dte,
            np.nan,
            dtype=float,
        )

        D_clean = np.full_like(
            dmetric,
            np.nan,
            dtype=float,
        )

        RMS_clean = np.full_like(
            rms,
            np.nan,
            dtype=float,
        )


        dt0_clean[
            valid_bias
        ] = dt0[
            valid_bias
        ]

        du0_clean[
            valid_bias
        ] = du0[
            valid_bias
        ]

        dte_clean[
            valid_bias
        ] = dte[
            valid_bias
        ]

        D_clean[
            valid_D
        ] = dmetric[
            valid_D
        ]

        RMS_clean[
            valid_RMS
        ] = rms[
            valid_RMS
        ]


        rows.append(

            (
                parameter_value,
                dt0_clean,
                du0_clean,
                dte_clean,
                D_clean,
                RMS_clean,
                success,
            )

        )


    if len(rows) == 0:

        raise RuntimeError(

            f"No valid scan rows in:\n"
            f"{directory}"
        )


    # ========================================================
    # SORT BY THE ACTUAL PHYSICAL PARAMETER
    # ========================================================

    rows.sort(
        key=lambda row: row[0]
    )


    parameter = np.asarray(

        [
            row[0]
            for row in rows
        ],

        dtype=float,

    )


    DT0 = np.vstack(
        [
            row[1]
            for row in rows
        ]
    )


    DU0 = np.vstack(
        [
            row[2]
            for row in rows
        ]
    )


    DTE = np.vstack(
        [
            row[3]
            for row in rows
        ]
    )


    D = np.vstack(
        [
            row[4]
            for row in rows
        ]
    )


    RMS = np.vstack(
        [
            row[5]
            for row in rows
        ]
    )


    SUCCESS = np.vstack(
        [
            row[6]
            for row in rows
        ]
    )


    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    print()

    print(
        f"Loaded {file_parameter} scan:"
    )

    print(
        f"  files       = {len(files)}"
    )

    print(
        f"  rows        = {len(parameter)}"
    )

    print(
        f"  parameter   = "
        f"{parameter.min():.8e} "
        f"... "
        f"{parameter.max():.8e}"
    )

    print(
        f"  P nodes     = {len(P_grid_ref)}"
    )

    print(
        f"  shape       = {D.shape}"
    )


    return {

        "parameter":
            parameter,

        "P_grid":
            P_grid_ref,

        "P_over_tE":
            P_grid_ref
            /
            float(tE_true),

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


# ============================================================
# BEGIN INDEPENDENT BIAS NORMS
#
# At tE=30 d the amplitudes of Delta tE/tE in the u0 and
# qM experiments differ substantially.  Keep the physical
# quantity in both panels, but determine a robust symmetric
# color range independently for each one.
# ============================================================

def make_bias_norm(
    Z,
    percentile=ROBUST_PERCENTILE,
):

    values = np.abs(
        Z[
            np.isfinite(Z)
        ]
    )

    if values.size == 0:
        raise RuntimeError(
            "No finite bias values."
        )

    vmax = float(
        np.nanpercentile(
            values,
            percentile,
        )
    )

    if (
        not np.isfinite(vmax)
        or
        vmax <= 0.0
    ):
        vmax = 1.0e-12

    return (
        mcolors.TwoSlopeNorm(
            vmin=-vmax,
            vcenter=0.0,
            vmax=vmax,
        ),
        vmax,
    )


norm_DTE_u0, vmax_DTE_u0 = make_bias_norm(
    u0_DTE_norm
)

norm_DTE_q, vmax_DTE_q = make_bias_norm(
    q_DTE_norm
)


print()
print("=" * 80)
print("INDEPENDENT BIAS COLOR SCALES")
print(
    f"u0 panel : +/- {vmax_DTE_u0:.8e}"
)
print(
    f"qM panel : +/- {vmax_DTE_q:.8e}"
)
print("=" * 80)

# END INDEPENDENT BIAS NORMS

# ============================================================
# PLOTTING (restyled; loaded arrays and colour norms above are unchanged)
# ============================================================
#
# Fig. 3: paper_tE_bias_D_contours_tE30.{pdf,png}   printed width 6.14 in
# Fig. 4: paper_u0_tE_correlated_bias_tE30.{pdf,png}  printed width 3.07 in
#         paper_u0_tE_correlated_bias_tE30_alt.{pdf,png} (+ top axis P [d])
# ============================================================

import json
import matplotlib.ticker as mticker

verify_directory = os.path.join(home, "binary_source", "figures", "_verify")
os.makedirs(verify_directory, exist_ok=True)

def _percent(v, _pos=None):
    s = f"{100.0 * v:.1f}"
    return "$0.0$" if s in ("-0.0", "0.0") else f"${s}$"


percent_fmt = mticker.FuncFormatter(_percent)


def D_contours_labelled(ax, x, y, D_map, label_xfrac, levels=D_LEVELS):
    """Black D contours with inline labels at chosen positions.

    label_xfrac: {level: [fraction of log-x axis span, ...]} giving where
    along each branch the label goes (one entry per branch, ordered by
    decreasing mean y of the branch).
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    valid_values = D_map[np.isfinite(D_map) & (D_map > 0)]
    dmin, dmax = np.nanmin(valid_values), np.nanmax(valid_values)
    levels_here = [lv for lv in levels if dmin < lv < dmax]
    cs = ax.contour(X, Y, np.ma.masked_invalid(D_map), levels=levels_here,
                    colors="black", linewidths=LW_THIN, linestyles="-",
                    zorder=10)

    lx0, lx1 = np.log10(ax.get_xlim())
    manual = []
    for i, lv in enumerate(cs.levels):
        segs = [s for s in cs.allsegs[i] if len(s) > 5]
        segs.sort(key=lambda s: -np.mean(np.log10(s[:, 1])))
        for seg, frac in zip(segs, label_xfrac.get(lv, [])):
            if frac is None:
                continue
            lxs = np.log10(seg[:, 0])
            target = lx0 + frac * (lx1 - lx0)
            j = int(np.argmin(np.abs(lxs - target)))
            manual.append((seg[j, 0], seg[j, 1]))
    fmt = {lv: pow10_label(lv) for lv in cs.levels}
    texts = ax.clabel(cs, fmt=fmt, fontsize=FS_ANNOT, inline=True,
                      inline_spacing=3, manual=manual)
    for t in texts:
        t.set_path_effects(halo(1.6))
    return cs, levels_here


def log_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


# ------------------------------------------------------------
# FIGURE 3
# ------------------------------------------------------------

def draw_figure3(alt=False):
    fig = plt.figure(figsize=(fig_width(1.0), 2.85))
    left, right, gap = 0.085, 0.985, 0.075
    w = (right - left - gap) / 2.0
    ax_bottom, ax_top = 0.135, 0.765
    axes = [fig.add_axes([left + k * (w + gap), ax_bottom, w, ax_top - ax_bottom])
            for k in range(2)]
    caxes = [fig.add_axes([left + k * (w + gap), 0.815, w, 0.03])
             for k in range(2)]

    panels = [
        (axes[0], caxes[0], u0_edges, P_u0_edges, u0_DTE_norm, norm_DTE_u0,
         u0_data, r"$u_0$", {1e-4: [0.12], 1e-3: [0.90], 1e-2: [0.50, 0.85]}),
        (axes[1], caxes[1], q_edges, P_q_edges, q_DTE_norm, norm_DTE_q,
         q_data, r"$q_M=M_2/M_1$", {1e-4: [0.40], 1e-3: [0.62], 1e-2: [0.86]}),
    ]
    contour_levels = {}
    for k, (ax, cax, xe, ye, Z, norm, data, xlabel, lab) in enumerate(panels):
        pcm = heatmap(ax, xe, ye, np.ma.masked_invalid(Z).T, cmap=BIAS_CMAP,
                      norm=norm, zorder=1)
        log_axes(ax)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        _, lv = D_contours_labelled(ax, data["parameter"], data["P_over_tE"],
                                    data["D"], lab)
        contour_levels["u0" if k == 0 else "qM"] = lv
        if alt:
            X, Y = np.meshgrid(data["parameter"], data["P_over_tE"], indexing="ij")
            if np.nanmax(np.abs(Z)) > 0.01:
                ax.contour(X, Y, np.abs(Z), levels=[0.01], colors="k",
                           linewidths=LW_THIN, linestyles=":", zorder=11)
        ax.set_xlabel(xlabel)
        if k == 0:
            ax.set_ylabel(r"$P/t_E$")
        panel_label(ax, "ab"[k])

        cbar = fig.colorbar(pcm, cax=cax, orientation="horizontal", extend="both")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(which="both", direction="in", pad=1.5)
        lim = norm.vmax
        step = 0.1 / 100 if lim < 0.004 else 0.2 / 100
        if lim > 0.008:
            step = 0.5 / 100
        cbar.set_ticks(np.arange(-np.floor(lim / step), np.floor(lim / step) + 1) * step)
        cbar.ax.xaxis.set_major_formatter(percent_fmt)
        cbar.ax.xaxis.set_minor_locator(mticker.NullLocator())
        cbar.set_label(r"$\Delta t_E/t_E$ (\%)", labelpad=3)

    stem = os.path.join(output_directory,
                        f"paper_tE_bias_D_contours_tE{int(tE_plot)}" + ("_alt" if alt else ""))
    out = save_figure(fig, stem)
    plt.close(fig)
    return out, contour_levels


out3, contour_levels3 = draw_figure3(alt=False)
max_abs_DTE = max(np.nanmax(np.abs(u0_DTE_norm)), np.nanmax(np.abs(q_DTE_norm)))
print("FIGURE 3:", *out3)
print(f"max |DtE/tE| over both scans = {max_abs_DTE:.3e}")
# |DtE/tE| = 1% is reached only in a few cells at the (u0, P/tE) corner
# (u0 ~ 0.01, P/tE ~ 0.3); a 1% contour adds no information, so the
# alternative version is not written by default.
MAKE_FIG3_ALT = False
if MAKE_FIG3_ALT:
    out3_alt, _ = draw_figure3(alt=True)
    print("FIGURE 3 alt:", *out3_alt)

np.savez(os.path.join(verify_directory, f"paper_tE_bias_tE{int(tE_plot)}_plotted_values.npz"),
         u0_param=u0_data["parameter"], u0_P_over_tE=u0_data["P_over_tE"],
         u0_DTE_norm=u0_DTE_norm, u0_D=u0_data["D"],
         q_param=q_data["parameter"], q_P_over_tE=q_data["P_over_tE"],
         q_DTE_norm=q_DTE_norm, q_D=q_data["D"],
         vmax_u0=vmax_DTE_u0, vmax_q=vmax_DTE_q)

# ------------------------------------------------------------
# FIGURE 4
# ------------------------------------------------------------

iu = np.argmin(np.abs(np.log10(u0_data["parameter"]) - np.log10(u0_slice_target)))
u0_used = u0_data["parameter"][iu]
x_u0 = u0_data["P_over_tE"]
du0_u0 = u0_DU0_norm[iu, :]
dtE_u0 = u0_DTE_norm[iu, :]

valid_slice = np.isfinite(du0_u0) & np.isfinite(dtE_u0)
rho_slice, p_slice = spearmanr(du0_u0[valid_slice], dtE_u0[valid_slice])
print(f"FIGURE 4: u0 used = {u0_used:.8f}, Spearman rho = {rho_slice:.5f}, p = {p_slice:.5e}")


def draw_figure4(alt=False):
    fig = plt.figure(figsize=(fig_width(0.5), 2.45 if not alt else 2.65))
    top = 0.97 if not alt else 0.82
    ax = fig.add_axes([0.205, 0.165 if not alt else 0.15, 0.765, top - (0.165 if not alt else 0.15)])
    ax.axhline(0.0, lw=0.5, ls=":", color="0.4", zorder=0)
    ax.plot(x_u0, du0_u0, color=C_BLUE, ls="-", label=r"$\Delta u_0/u_0$")
    ax.plot(x_u0, dtE_u0, color=C_VERMILION, ls="--", label=r"$\Delta t_E/t_E$")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.set_xlim(x_u0.min(), x_u0.max())
    ax.yaxis.set_major_formatter(percent_fmt)
    ax.set_xlabel(r"$P/t_E$")
    ax.set_ylabel(r"Relative PSPL bias (\%)")
    ax.legend(loc="lower right")
    if alt:
        ax.tick_params(top=False, which="both")
        sec = ax.secondary_xaxis("top", functions=(lambda r: r * tE_plot,
                                                   lambda p: p / tE_plot))
        sec.set_xlabel(r"$P$ [d]")
        sec.tick_params(which="both", direction="in")
        sec.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    stem = os.path.join(output_directory,
                        f"paper_u0_tE_correlated_bias_tE{int(tE_plot)}" + ("_alt" if alt else ""))
    out = save_figure(fig, stem)
    plt.close(fig)
    return out


print("FIGURE 4:", *draw_figure4(False))
print("FIGURE 4 alt:", *draw_figure4(True))
np.savez(os.path.join(verify_directory, f"paper_u0_tE_correlated_bias_tE{int(tE_plot)}_plotted_values.npz"),
         P_over_tE=x_u0, du0_u0=du0_u0, dtE_tE=dtE_u0, u0_used=u0_used,
         spearman_rho=rho_slice, spearman_p=p_slice)
with open(os.path.join(verify_directory, "fig3_contour_levels.json"), "w") as fh:
    json.dump({k: [float(v) for v in lv] for k, lv in contour_levels3.items()}, fh)
