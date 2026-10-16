# ============================================================
# D_BSPL-PSPL -- PAPER FIGURES
#
# FIGURE 1
# --------
# Heatmap D(u0, P/tE) for the fiducial tE
# + ONLY the red D = 1e-2 contour.
#
# IMPORTANT:
#
# Instead of plotting the raw matplotlib contour directly,
# we reconstruct the D=1e-2 level from the simulated D field:
#
#   1. interpolate log10(D) on a fine log-log grid
#   2. for every u0, locate crossings of D = 1e-2
#   3. identify upper and lower branches
#   4. interpolate missing INTERNAL parts of each branch
#      with PCHIP in log(u0)-log(P/tE)
#
# This completes the missing red segment using neighboring
# simulated values, without plotting additional contours.
#
#
# FIGURE 2
# --------
# D = 1e-2 contour for several tE values.
#
# Both figures are saved independently:
#
#   PNG 600 dpi
#   PDF
#
# ============================================================


import os
import re
import glob

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patheffects as pe

from matplotlib.ticker import LogLocator

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt


# ============================================================
# PAPER STYLE
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (apply_paper_style, fig_width, heatmap, panel_label,
                         save_figure, d_norm, halo, D_CMAP, D_LABEL,
                         FS_ANNOT, LW_MAIN, LW_THIN)

apply_paper_style()


# ============================================================
# CONFIGURATION
# ============================================================

tE_reference = 30.0


tE_values = [

    10,
    20,
    30,
    50,
    75,
    100,
    150,
    200,
    300,
    500,
    750,
    1000,

]


# ============================================================
# REFERENCE LEVEL
# ============================================================

D_REF = 1e-2


# ============================================================
# HEATMAP RANGE
# ============================================================

D_VMIN = 1e-5
D_VMAX = 1e0


# ============================================================
# INTERPOLATION RESOLUTION
# ============================================================

N_FINE_U0 = 1200
N_FINE_P = 1200


# ============================================================
# COMPLETE BRANCHES
# ============================================================
#
# True:
#
#   if a D=1e-2 branch disappears over an internal interval
#   in u0 but exists on both sides, reconstruct that missing
#   section using PCHIP interpolation.
#
# No extrapolation is performed outside the first and last
# actual crossings.
#
# ============================================================

COMPLETE_INTERNAL_GAPS = True


# ============================================================
# PATHS
# ============================================================

home = os.path.expanduser("~")


results_root = os.path.join(

    home,

    "binary_source",

    "results",

)


multi_tE_root = os.path.join(

    results_root,

    "scan_many_tE_200x200",

)


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


# ============================================================
# FIND SCAN
# ============================================================

def find_scan_directory(tE):


    label = int(tE)


    candidates = [

        os.path.join(

            multi_tE_root,

            f"scan_u0_tE{label}",

        ),

        os.path.join(

            results_root,

            f"scan_u0_tE{label}",

        ),

    ]


    for directory in candidates:


        files = glob.glob(

            os.path.join(

                directory,

                "scan_kepler_u0_*.npz",

            )

        )


        if len(files) > 0:

            return directory


    return None


# ============================================================
# LOAD ONE D SCAN
# ============================================================

def load_D_scan(tE):


    directory = find_scan_directory(tE)


    if directory is None:

        raise FileNotFoundError(

            f"No scan found for tE={tE}"

        )


    files = glob.glob(

        os.path.join(

            directory,

            "scan_kepler_u0_*.npz",

        )

    )


    # --------------------------------------------------------
    # Sort numerically
    # --------------------------------------------------------

    def get_index(filename):


        match = re.search(

            r"scan_kepler_u0_(\d+)\.npz$",

            os.path.basename(filename),

        )


        if match is None:

            return 10**9


        return int(

            match.group(1)

        )


    files = sorted(

        files,

        key=get_index,

    )


    # --------------------------------------------------------
    # Reference P grid
    # --------------------------------------------------------

    with np.load(

        files[0],

        allow_pickle=False,

    ) as d:


        P_grid = d[

            "P_grid"

        ].astype(float)


    rows = []


    # --------------------------------------------------------
    # Read u0 scans
    # --------------------------------------------------------

    for filename in files:


        with np.load(

            filename,

            allow_pickle=False,

        ) as d:


            if (
                "D" not in d.files
                or
                "truth" not in d.files
            ):

                continue


            truth = d[

                "truth"

            ].astype(float)


            u0_true = float(

                truth[1]

            )


            P_this = d[

                "P_grid"

            ].astype(float)


            if not np.allclose(

                P_this,

                P_grid,

            ):

                raise ValueError(

                    f"P_grid differs in:\n"
                    f"{filename}"

                )


            D_values = d[

                "D"

            ].astype(float)


            if "SUCCESS" in d.files:

                success = d[

                    "SUCCESS"

                ].astype(bool)

            else:

                success = np.ones_like(

                    D_values,

                    dtype=bool,

                )


            D_values = D_values.copy()


            D_values[

                ~success

            ] = np.nan


            D_values[

                ~np.isfinite(D_values)

            ] = np.nan


            D_values[

                D_values <= 0

            ] = np.nan


            rows.append(

                (
                    u0_true,
                    D_values,
                )

            )


    if len(rows) == 0:

        raise RuntimeError(

            f"No valid data in {directory}"

        )


    # --------------------------------------------------------
    # Sort by u0
    # --------------------------------------------------------

    rows.sort(

        key=lambda x:
            x[0]

    )


    u0_grid = np.array(

        [

            row[0]

            for row in rows

        ],

        dtype=float,

    )


    D_map = np.vstack(

        [

            row[1]

            for row in rows

        ]

    )


    P_over_tE = (

        P_grid

        /

        float(tE)

    )


    return {

        "directory":
            directory,

        "u0":
            u0_grid,

        "P":
            P_grid,

        "P_over_tE":
            P_over_tE,

        "D":
            D_map,

    }


# ============================================================
# LOG BIN EDGES
# ============================================================

def log_edges(x):


    x = np.asarray(

        x,

        dtype=float,

    )


    lx = np.log10(x)


    edges = np.empty(

        len(x) + 1

    )


    edges[1:-1] = (

        lx[:-1]

        +

        lx[1:]

    ) / 2.0


    edges[0] = (

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


    edges[-1] = (

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


    return 10**edges


# ============================================================
# FILL NaNs ON ORIGINAL REGULAR GRID
# ============================================================
#
# Only needed for constructing the interpolation surface.
#
# The heatmap itself still uses the original D values.
#
# ============================================================

def nearest_fill_nan(array):


    arr = np.asarray(

        array,

        dtype=float,

    ).copy()


    invalid = ~np.isfinite(arr)


    if not np.any(invalid):

        return arr


    if np.all(invalid):

        raise RuntimeError(

            "All values are NaN."

        )


    # --------------------------------------------------------
    # Get indices of nearest valid cells
    # --------------------------------------------------------

    nearest_indices = distance_transform_edt(

        invalid,

        return_distances=False,

        return_indices=True,

    )


    nearest_values = arr[

        tuple(nearest_indices)

    ]


    arr[

        invalid

    ] = nearest_values[

        invalid

    ]


    return arr


# ============================================================
# BUILD FINE D SURFACE
# ============================================================
#
# Regular-grid interpolation in:
#
#   log10(u0)
#   log10(P/tE)
#   log10(D)
#
# ============================================================

def make_fine_D_surface(

    u0,

    P_over_tE,

    D_map,

):


    u0 = np.asarray(

        u0,

        dtype=float,

    )


    P_over_tE = np.asarray(

        P_over_tE,

        dtype=float,

    )


    D_map = np.asarray(

        D_map,

        dtype=float,

    )


    # ========================================================
    # log coordinates
    # ========================================================

    logu = np.log10(

        u0

    )


    logP = np.log10(

        P_over_tE

    )


    logD = np.full_like(

        D_map,

        np.nan,

        dtype=float,

    )


    valid = (

        np.isfinite(D_map)

        &

        (D_map > 0)

    )


    logD[

        valid

    ] = np.log10(

        D_map[

            valid

        ]

    )


    # ========================================================
    # Fill isolated failed fits
    # ========================================================

    logD_filled = nearest_fill_nan(

        logD

    )


    # ========================================================
    # Fine coordinates
    # ========================================================

    logu_fine = np.linspace(

        logu.min(),

        logu.max(),

        N_FINE_U0,

    )


    logP_fine = np.linspace(

        logP.min(),

        logP.max(),

        N_FINE_P,

    )


    # ========================================================
    # Bilinear interpolation
    #
    # kx=1, ky=1:
    # avoids spline overshooting and remains conservative.
    # ========================================================

    interpolator = RectBivariateSpline(

        logu,

        logP,

        logD_filled,

        kx=1,

        ky=1,

        s=0,

    )


    logD_fine = interpolator(

        logu_fine,

        logP_fine,

    )


    D_fine = (

        10**logD_fine

    )


    return (

        logu_fine,

        logP_fine,

        D_fine,

    )


# ============================================================
# FIND D=LEVEL CROSSINGS
# ============================================================
#
# For every u0:
#
#     find all P/tE where D crosses D_REF.
#
# Usually there are:
#
#     0 crossings
#     1 crossing
#     2 crossings
#
# If there is one crossing, it is assigned to the upper branch.
#
# If there are >= 2:
#
#     smallest P/tE -> lower branch
#     largest  P/tE -> upper branch
#
# ============================================================

def find_level_branches(

    logu,

    logP,

    D_fine,

    level,

):


    target = np.log10(

        level

    )


    logD = np.log10(

        D_fine

    )


    lower = np.full(

        len(logu),

        np.nan,

    )


    upper = np.full(

        len(logu),

        np.nan,

    )


    n_crossings = np.zeros(

        len(logu),

        dtype=int,

    )


    # ========================================================
    # Loop over u0
    # ========================================================

    for i in range(

        len(logu)

    ):


        z = (

            logD[i, :]

            -

            target

        )


        valid = np.isfinite(z)


        if np.sum(valid) < 2:

            continue


        crossings = []


        # ====================================================
        # Search adjacent P cells
        # ====================================================

        for j in range(

            len(logP) - 1

        ):


            if not (

                np.isfinite(z[j])

                and

                np.isfinite(z[j + 1])

            ):

                continue


            z1 = z[j]

            z2 = z[j + 1]


            # ------------------------------------------------
            # Exact crossing
            # ------------------------------------------------

            if z1 == 0:


                crossings.append(

                    logP[j]

                )


                continue


            # ------------------------------------------------
            # Sign change
            # ------------------------------------------------

            if (

                z1 * z2

                <

                0

            ):


                # Linear interpolation in logD-logP

                fraction = (

                    -z1

                    /

                    (
                        z2
                        -
                        z1
                    )

                )


                logP_cross = (

                    logP[j]

                    +

                    fraction
                    *
                    (
                        logP[j + 1]

                        -

                        logP[j]
                    )

                )


                crossings.append(

                    logP_cross

                )


        # ====================================================
        # Remove duplicates
        # ====================================================

        if len(crossings) == 0:

            continue


        crossings = np.array(

            sorted(

                crossings

            )

        )


        crossings = np.unique(

            np.round(

                crossings,

                decimals=10,

            )

        )


        n_crossings[i] = len(

            crossings

        )


        # ====================================================
        # Assign branches
        # ====================================================

        if len(crossings) == 1:


            upper[i] = (

                crossings[0]

            )


        else:


            lower[i] = (

                crossings[0]

            )


            upper[i] = (

                crossings[-1]

            )


    return (

        lower,

        upper,

        n_crossings,

    )


# ============================================================
# COMPLETE INTERNAL GAPS
# ============================================================
#
# This is the part that completes the missing red segment.
#
# We interpolate the P/tE coordinate of the contour as a
# function of log10(u0).
#
# IMPORTANT:
#
# PCHIP is only evaluated BETWEEN the first and last
# actual crossing.
#
# There is NO extrapolation beyond the simulated crossing
# range.
#
# ============================================================

def complete_branch(

    logu,

    branch,

):


    branch = np.asarray(

        branch,

        dtype=float,

    )


    valid = np.isfinite(

        branch

    )


    if np.sum(valid) < 2:

        return branch.copy()


    result = branch.copy()


    x_valid = logu[

        valid

    ]


    y_valid = branch[

        valid

    ]


    # ========================================================
    # PCHIP interpolation
    # ========================================================

    interpolator = PchipInterpolator(

        x_valid,

        y_valid,

        extrapolate=False,

    )


    # ========================================================
    # Only inside actual domain
    # ========================================================

    inside = (

        (logu >= x_valid.min())

        &

        (logu <= x_valid.max())

    )


    missing_inside = (

        inside

        &

        ~valid

    )


    result[

        missing_inside

    ] = interpolator(

        logu[

            missing_inside

        ]

    )


    return result


# ============================================================
# CONSTRUCT RED CONTOUR BRANCHES
# ============================================================

def reconstruct_D_contour(

    u0,

    P_over_tE,

    D_map,

    level=D_REF,

):


    # ========================================================
    # Fine D field
    # ========================================================

    logu_fine, logP_fine, D_fine = make_fine_D_surface(

        u0,

        P_over_tE,

        D_map,

    )


    # ========================================================
    # Raw crossings
    # ========================================================

    lower, upper, n_crossings = find_level_branches(

        logu_fine,

        logP_fine,

        D_fine,

        level,

    )


    print()

    print(

        f"D={level:.2e}:"

    )


    print(

        "u0 columns with crossings =",

        np.sum(

            n_crossings > 0

        ),

        "/",

        len(

            n_crossings

        ),

    )


    print(

        "u0 columns with >=2 crossings =",

        np.sum(

            n_crossings >= 2

        ),

    )


    # ========================================================
    # Complete internal missing regions
    # ========================================================

    if COMPLETE_INTERNAL_GAPS:


        upper_complete = complete_branch(

            logu_fine,

            upper,

        )


        lower_complete = complete_branch(

            logu_fine,

            lower,

        )


    else:


        upper_complete = upper


        lower_complete = lower


    # ========================================================
    # Convert to physical coordinates
    # ========================================================

    u_fine = (

        10**logu_fine

    )


    P_upper = np.full_like(

        upper_complete,

        np.nan,

    )


    valid_upper = np.isfinite(

        upper_complete

    )


    P_upper[

        valid_upper

    ] = (

        10**

        upper_complete[

            valid_upper

        ]

    )


    P_lower = np.full_like(

        lower_complete,

        np.nan,

    )


    valid_lower = np.isfinite(

        lower_complete

    )


    P_lower[

        valid_lower

    ] = (

        10**

        lower_complete[

            valid_lower

        ]

    )


    return {

        "u0":
            u_fine,

        "upper":
            P_upper,

        "lower":
            P_lower,

        "n_crossings":
            n_crossings,

    }


# ============================================================
# PLOT RECONSTRUCTED CONTOUR
# ============================================================

def plot_reconstructed_contour(

    ax,

    contour,

    color,

    linewidth=1.7,

    zorder=20,

):


    u = contour[

        "u0"

    ]


    upper = contour[

        "upper"

    ]


    lower = contour[

        "lower"

    ]


    # ========================================================
    # Upper branch
    # ========================================================

    valid = np.isfinite(

        upper

    )


    if np.sum(valid) > 1:


        ax.plot(

            u[valid],

            upper[valid],

            color=color,

            lw=linewidth,

            zorder=zorder,

        )


    # ========================================================
    # Lower branch
    # ========================================================

    valid = np.isfinite(

        lower

    )


    if np.sum(valid) > 1:


        ax.plot(

            u[valid],

            lower[valid],

            color=color,

            lw=linewidth,

            zorder=zorder,

        )


# ============================================================
# COMMON AXIS FORMAT
# ============================================================

def format_log_axes(ax):


    ax.set_xscale(

        "log"

    )


    ax.set_yscale(

        "log"

    )


    ax.tick_params(

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


    ax.yaxis.set_major_locator(

        LogLocator(

            base=10

        )

    )


    ax.grid(

        False

    )


# ============================================================
# PLOTTING (restyled; data processing above is unchanged)
# ============================================================
#
# Outputs (figures/current/):
#   D_heatmap_tE30_complete_contour.{pdf,png}      Fig. 2a
#   D_heatmap_tE30_complete_contour_alt.{pdf,png}  Fig. 2a + reference scales
#   D_contours_many_tE_complete.{pdf,png}          Fig. 2b
#
# Both panels are drawn at the printed subfigure width
# (0.48 \linewidth = 2.95 in) with identical axes geometry.
# ============================================================

import json

verify_directory = os.path.join(home, "binary_source", "figures", "_verify")
os.makedirs(verify_directory, exist_ok=True)
import matplotlib.ticker as mticker

PANEL_W = fig_width(0.48)
PANEL_H = 2.75
AX_RECT = [0.175, 0.135, 0.785, 0.665]
CAX_RECT = [0.175, 0.875, 0.785, 0.028]

# Fiducial physical family of the scan (stored in the npz 'truth' vector
# and checked below): M1 = 2, M2 = 1 Msun, rEhat = 5 AU.
M_TOT_MSUN = 3.0
Q_MASS = 0.5
REHAT_AU = 5.0
YEAR_D = 365.25


def new_panel():
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    ax = fig.add_axes(AX_RECT)
    cax = fig.add_axes(CAX_RECT)
    return fig, ax, cax


def finish_axes(ax, ylim):
    format_log_axes(ax)
    ax.set_xlim(1e-2, 1e1)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r"$u_0$")
    ax.set_ylabel(r"$P/t_E$")


def top_colorbar(fig, cax, mappable, label):
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.tick_params(which="both", direction="in", pad=1.5)
    cbar.set_label(label, labelpad=3)
    return cbar


def label_on_curve(ax, x, y, x_at, text, color, dy_pts=0.0,
                   above=True):
    """Place a text label along a log-log curve at abscissa x_at."""
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    i = int(np.argmin(np.abs(np.log10(x) - np.log10(x_at))))
    j = min(i + 8, len(x) - 1)
    k = max(i - 8, 0)
    p1 = ax.transData.transform((x[k], y[k]))
    p2 = ax.transData.transform((x[j], y[j]))
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    offset = mtransforms.offset_copy(
        ax.transData, fig=ax.figure, x=0.0,
        y=(dy_pts if above else -dy_pts), units="points")
    return ax.text(x[i], y[i], text, color=color, fontsize=FS_ANNOT,
                   rotation=angle, rotation_mode="anchor",
                   ha="center", va="bottom" if above else "top",
                   transform=offset, zorder=30, path_effects=halo(1.8))


import matplotlib.transforms as mtransforms

# ============================================================
# LOAD REFERENCE
# ============================================================

reference = load_D_scan(tE_reference)

print()
print("=" * 80)
print(f"REFERENCE tE = {tE_reference:g} d")
print(reference["directory"])
print("=" * 80)

u0 = reference["u0"]
P_over_tE = reference["P_over_tE"]
D_map = reference["D"]

contour_reference = reconstruct_D_contour(u0, P_over_tE, D_map, level=D_REF)

u0_edges = log_edges(u0)
P_edges = log_edges(P_over_tE)
ylim_a = (P_edges[0], P_edges[-1])

# Verify the physical family against the stored truth vector.
_ref_files = sorted(glob.glob(os.path.join(reference["directory"],
                                           "scan_kepler_u0_*.npz")))
with np.load(_ref_files[0]) as _d:
    _a_stored = _d["a_AU_of_P"].astype(float)
    _xi_stored = _d["xiE_of_P"].astype(float)
    _P_stored = _d["P_grid"].astype(float)
_a_kepler = (M_TOT_MSUN * (_P_stored / YEAR_D) ** 2) ** (1.0 / 3.0)
assert np.allclose(_a_stored, _a_kepler, rtol=1e-12, atol=0.0)
assert np.allclose(_xi_stored, _a_stored / REHAT_AU, rtol=1e-12, atol=0.0)


def draw_panel_a(alt=False):
    fig, ax, cax = new_panel()

    pcm = heatmap(ax, u0_edges, P_edges, np.ma.masked_invalid(D_map).T,
                  norm=d_norm(D_VMIN, D_VMAX), cmap=D_CMAP)

    plot_reconstructed_contour(ax, contour_reference, color="crimson",
                               linewidth=LW_MAIN)

    finish_axes(ax, ylim_a)
    cbar = top_colorbar(fig, cax, pcm, D_LABEL)
    cbar.ax.xaxis.set_major_locator(LogLocator(base=10, numticks=7))

    # Label both branches of D = 1e-2
    cu = contour_reference["u0"]
    label_on_curve(ax, cu, contour_reference["upper"], 0.03,
                   r"$10^{-2}$", "crimson", dy_pts=1.5, above=True)
    label_on_curve(ax, cu, contour_reference["lower"], 4.0,
                   r"$10^{-2}$", "crimson", dy_pts=1.5, above=False)

    panel_label(ax, "a", color="white")

    overlays = {}
    if alt:
        u = np.logspace(-2, 1, 400)
        # (1) xi_E / u0 = 1, xi_E = q/(1+q) a_rel / rEhat (Eqs. 2-5)
        a_needed = u * REHAT_AU * (1.0 + Q_MASS) / Q_MASS
        P_xi = YEAR_D * np.sqrt(a_needed ** 3 / M_TOT_MSUN) / tE_reference
        # (2) P = u0 tE
        P_cross = u.copy()
        # (3) P = 1 yr
        P_year = YEAR_D / tE_reference

        kw = dict(color="white", lw=LW_THIN, zorder=25)
        ax.plot(u, P_xi, ls=(0, (4, 2)), **kw)
        ax.plot(u, P_cross, ls=(0, (1, 1.5)), **kw)
        ax.axhline(P_year, ls=(0, (6, 2, 1, 2)), **kw)

        label_on_curve(ax, u, P_xi, 0.25, r"$\xi_E=u_0$", "white",
                       dy_pts=1.0, above=True).set_path_effects([])
        label_on_curve(ax, u, P_cross, 1.6, r"$P=u_0t_E$", "white",
                       dy_pts=1.0, above=True).set_path_effects([])
        ax.text(1.1e-2, P_year * 1.15, r"$P=1$ yr", color="white",
                fontsize=FS_ANNOT, ha="left", va="bottom", zorder=30)
        overlays = {
            "xiE_over_u0_eq_1": {"u0": u.tolist(), "P_over_tE": P_xi.tolist()},
            "P_eq_u0_tE": {"u0": u.tolist(), "P_over_tE": P_cross.tolist()},
            "P_eq_1yr_P_over_tE": P_year,
        }

    stem = os.path.join(
        output_directory,
        f"D_heatmap_tE{int(tE_reference)}_complete_contour"
        + ("_alt" if alt else ""))
    out = save_figure(fig, stem)
    plt.close(fig)
    return out, overlays


out_a, _ = draw_panel_a(alt=False)
out_a_alt, overlays_alt = draw_panel_a(alt=True)
print("FIGURE 2a:", *out_a)
print("FIGURE 2a alt:", *out_a_alt)

# Values actually plotted (for before/after verification)
np.savez(
    os.path.join(verify_directory,
                 f"D_heatmap_tE{int(tE_reference)}_plotted_values.npz"),
    u0=u0, P_over_tE=P_over_tE, D=D_map,
    contour_u0=contour_reference["u0"],
    contour_upper=contour_reference["upper"],
    contour_lower=contour_reference["lower"],
)
with open(os.path.join(verify_directory,
                       f"D_heatmap_tE{int(tE_reference)}_alt_overlays.json"),
          "w") as fh:
    json.dump({
        "definition": {
            "xiE_over_u0_eq_1": "xi_E = q_M/(1+q_M) * a_rel/rEhat, "
                                "a_rel = (Mtot P_yr^2)^(1/3) AU, "
                                "Mtot=3, q_M=0.5, rEhat=5 AU, yr=365.25 d",
            "P_eq_u0_tE": "P/tE = u0",
            "P_eq_1yr": "P/tE = 365.25/tE",
        },
        **overlays_alt,
    }, fh)

# ============================================================
# FIGURE 2b: D = 1e-2 CONTOURS FOR MANY tE
# ============================================================

all_scans = {}
for tE in tE_values:
    try:
        scan = load_D_scan(tE)
        all_scans[float(tE)] = scan
        print(f"Loaded tE={tE:g} d -> {scan['D'].shape}")
    except Exception as error:
        print(f"[SKIP] tE={tE:g}: {error}")

if len(all_scans) == 0:
    raise RuntimeError("No scans available.")

tE_loaded = np.array(sorted(all_scans.keys()), dtype=float)
tE_norm = mcolors.LogNorm(vmin=tE_loaded.min(), vmax=tE_loaded.max())
# plasma truncated before its pale-yellow end so every line stays visible
tE_cmap = mcolors.LinearSegmentedColormap.from_list(
    "plasma_trunc", plt.get_cmap("plasma")(np.linspace(0.0, 0.88, 256)))

fig2, ax, cax = new_panel()

plotted_b = {}
P_min_all, P_max_all = np.inf, 0.0
for tE in tE_loaded:
    scan = all_scans[float(tE)]
    contour = reconstruct_D_contour(scan["u0"], scan["P_over_tE"], scan["D"],
                                    level=D_REF)
    plot_reconstructed_contour(ax, contour, color=tE_cmap(tE_norm(tE)),
                               linewidth=1.0)
    plotted_b[f"tE{int(tE)}_u0"] = contour["u0"]
    plotted_b[f"tE{int(tE)}_upper"] = contour["upper"]
    plotted_b[f"tE{int(tE)}_lower"] = contour["lower"]
    e = log_edges(scan["P_over_tE"])
    P_min_all, P_max_all = min(P_min_all, e[0]), max(P_max_all, e[-1])

finish_axes(ax, (6e-3, 2e2))
ax.text(0.04, 0.05, r"$D_{\rm BSPL-PSPL}=10^{-2}$", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=FS_ANNOT)
panel_label(ax, "b")

sm = cm.ScalarMappable(norm=tE_norm, cmap=tE_cmap)
sm.set_array([])
top_colorbar(fig2, cax, sm, r"$t_E$ [d]")

out_b = save_figure(fig2, os.path.join(output_directory,
                                       "D_contours_many_tE_complete"))
plt.close(fig2)
np.savez(os.path.join(verify_directory,
                      "D_contours_many_tE_plotted_values.npz"), **plotted_b)
print("FIGURE 2b:", *out_b)
