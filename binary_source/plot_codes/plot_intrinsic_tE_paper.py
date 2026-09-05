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
# ============================================================

plt.rcParams.update({

    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",

    "font.size": 10,

    "axes.labelsize": 12,
    "axes.titlesize": 11,

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "legend.fontsize": 9,

    "axes.linewidth": 0.9,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 5,
    "ytick.major.size": 5,

    "xtick.minor.size": 3,
    "ytick.minor.size": 3,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,

    "savefig.dpi": 600,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


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
# ============================================================
#
# LOAD REFERENCE
#
# ============================================================
# ============================================================

reference = load_D_scan(

    tE_reference

)


print()

print("=" * 80)

print(

    f"REFERENCE tE = {tE_reference:g} d"

)

print(

    reference["directory"]

)

print("=" * 80)


# ============================================================
# ============================================================
#
# FIGURE 1
#
# D HEATMAP + COMPLETE RED D=1e-2 CONTOUR
#
# ============================================================
# ============================================================

u0 = reference[

    "u0"

]


P_over_tE = reference[

    "P_over_tE"

]


D_map = reference[

    "D"

]


# ============================================================
# Reconstruct contour
# ============================================================

contour_reference = reconstruct_D_contour(

    u0,

    P_over_tE,

    D_map,

    level=D_REF,

)


# ============================================================
# Heatmap edges
# ============================================================

u0_edges = log_edges(

    u0

)


P_edges = log_edges(

    P_over_tE

)


# ============================================================
# Figure
# ============================================================

fig1, ax = plt.subplots(

    figsize=(

        4.6,

        4.0,

    )

)


fig1.subplots_adjust(

    left=0.16,

    right=0.96,

    bottom=0.14,

    top=0.78,

)


# ============================================================
# Original simulated heatmap
# ============================================================

D_norm = mcolors.LogNorm(

    vmin=D_VMIN,

    vmax=D_VMAX,

)


pcm = ax.pcolormesh(

    u0_edges,

    P_edges,

    np.ma.masked_invalid(

        D_map

    ).T,

    norm=D_norm,

    cmap="viridis",

    shading="auto",

    rasterized=True,

)


# ============================================================
# ONLY RED D=1e-2 CONTOUR
# ============================================================

plot_reconstructed_contour(

    ax,

    contour_reference,

    color="crimson",

    linewidth=1.7,

)


# ============================================================
# Contour label
#
# We put text separately so clabel does not remove part
# of the red line.
# ============================================================

ax.text(

    0.018,

    15,

    r"$D=10^{-2}$",

    color="crimson",

    fontsize=9,

    rotation=22,

    ha="center",

    va="center",

    path_effects=[

        pe.Stroke(

            linewidth=2.4,

            foreground="white",

        ),

        pe.Normal(),

    ],

)


# ============================================================
# FORMAT
# ============================================================

format_log_axes(

    ax

)


ax.set_xlabel(

    r"$u_0$"

)


ax.set_ylabel(

    r"$P/t_E$"

)


# ax.text(

#     0.05,

#     0.95,

#     "(a)",

#     transform=ax.transAxes,

#     ha="left",

#     va="top",

#     fontsize=11,

# )


ax.text(

    0.96,

    0.9,

    rf"$t_E={tE_reference:g}\,\mathrm{{d}}$",

    transform=ax.transAxes,

    ha="right",

    va="bottom",

    fontsize=9,

)


# ============================================================
# TOP COLORBAR
# ============================================================

pos = ax.get_position()


cax = fig1.add_axes([

    pos.x0,

    0.84,

    pos.width,

    0.027,

])


cbar = fig1.colorbar(

    pcm,

    cax=cax,

    orientation="horizontal",

)


cbar.ax.xaxis.set_ticks_position(

    "top"

)


cbar.ax.xaxis.set_label_position(

    "top"

)


cbar.ax.tick_params(

    direction="in",

    labelsize=9,

    pad=2,

)


cbar.set_label(

    r"$D_{\rm BSPL-PSPL}$",

    fontsize=11,

    labelpad=4,

)


# ============================================================
# SAVE FIGURE 1
# ============================================================

figure1_png = os.path.join(

    output_directory,

    f"D_heatmap_tE{int(tE_reference)}_complete_contour.png",

)


figure1_pdf = os.path.join(

    output_directory,

    f"D_heatmap_tE{int(tE_reference)}_complete_contour.pdf",

)


fig1.savefig(

    figure1_png,

    dpi=600,

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

print(

    figure1_png

)

print(

    figure1_pdf

)

print("=" * 80)


plt.show()


# ============================================================
# ============================================================
#
# FIGURE 2
#
# COMPLETE D=1e-2 CONTOURS FOR MANY tE
#
# ============================================================
# ============================================================


# ============================================================
# Load scans
# ============================================================

all_scans = {}


for tE in tE_values:


    try:


        scan = load_D_scan(

            tE

        )


        all_scans[

            float(tE)

        ] = scan


        print(

            f"Loaded tE={tE:g} d -> "
            f"{scan['D'].shape}"

        )


    except Exception as error:


        print(

            f"[SKIP] tE={tE:g}: "
            f"{error}"

        )


if len(all_scans) == 0:

    raise RuntimeError(

        "No scans available."

    )


# ============================================================
# Color normalization
# ============================================================

tE_loaded = np.array(

    sorted(

        all_scans.keys()

    ),

    dtype=float,

)


tE_norm = mcolors.LogNorm(

    vmin=tE_loaded.min(),

    vmax=tE_loaded.max(),

)


tE_cmap = plt.get_cmap(

    "plasma"

)


# ============================================================
# Figure 2
# ============================================================

fig2, ax = plt.subplots(

    figsize=(

        4.6,

        4.0,

    )

)


fig2.subplots_adjust(

    left=0.16,

    right=0.96,

    bottom=0.14,

    top=0.78,

)


# ============================================================
# Reconstruct each contour
# ============================================================

for tE in tE_loaded:


    scan = all_scans[

        float(tE)

    ]


    contour = reconstruct_D_contour(

        scan[

            "u0"

        ],

        scan[

            "P_over_tE"

        ],

        scan[

            "D"

        ],

        level=D_REF,

    )


    color = tE_cmap(

        tE_norm(

            tE

        )

    )


    plot_reconstructed_contour(

        ax,

        contour,

        color=color,

        linewidth=1.4,

    )


# ============================================================
# FORMAT
# ============================================================

format_log_axes(

    ax

)


ax.set_xlabel(

    r"$u_0$"

)


ax.set_ylabel(

    r"$P/t_E$"

)


ax.text(

    0.05,

    0.95,

    "(b)",

    transform=ax.transAxes,

    ha="left",

    va="top",

    fontsize=11,

)


ax.text(

    0.06,

    0.06,

    r"$D_{\rm BSPL-PSPL}=10^{-2}$",

    transform=ax.transAxes,

    ha="left",

    va="bottom",

    fontsize=9,

)


# ============================================================
# TOP tE COLORBAR
# ============================================================

pos = ax.get_position()


cax = fig2.add_axes([

    pos.x0,

    0.84,

    pos.width,

    0.027,

])


sm = cm.ScalarMappable(

    norm=tE_norm,

    cmap=tE_cmap,

)


sm.set_array([])


cbar = fig2.colorbar(

    sm,

    cax=cax,

    orientation="horizontal",

)


cbar.ax.xaxis.set_ticks_position(

    "top"

)


cbar.ax.xaxis.set_label_position(

    "top"

)


cbar.ax.tick_params(

    direction="in",

    labelsize=9,

    pad=2,

)


cbar.set_label(

    r"$t_E\,[\mathrm{d}]$",

    fontsize=11,

    labelpad=4,

)


# ============================================================
# SAVE FIGURE 2
# ============================================================

figure2_png = os.path.join(

    output_directory,

    "D_contours_many_tE_complete.png",

)


figure2_pdf = os.path.join(

    output_directory,

    "D_contours_many_tE_complete.pdf",

)


fig2.savefig(

    figure2_png,

    dpi=600,

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

print(

    figure2_png

)

print(

    figure2_pdf

)

print("=" * 80)


