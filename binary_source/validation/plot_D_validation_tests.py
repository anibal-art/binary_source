# ============================================================
# plot_D_validation_tests.py
#
# PAPER FIGURE:
#
# (a) convergence with N_points
# (b) convergence with time window
# (c) normalization test
# (d) physical limit q_M -> 0
#
# SECOND FIGURE:
#
# Two events with approximately the same RMS but very
# different D.
#
# ============================================================


import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.ticker import LogLocator


# ============================================================
# PATHS
# ============================================================

home = os.path.expanduser("~")

root = os.path.join(
    home,
    "binary_source",
    "results",
    "D_validation",
)


N_file = os.path.join(
    root,
    "N_convergence.csv",
)

W_file = os.path.join(
    root,
    "window_convergence.csv",
)

q_file = os.path.join(
    root,
    "q_limit.csv",
)

u0_file = os.path.join(
    root,
    "u0_normalization_test.csv",
)

pair_file = os.path.join(
    root,
    "same_RMS_different_D_pair.npz",
)


# ============================================================
# PAPER STYLE
# ============================================================

plt.rcParams.update({

    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",

    "font.size": 9,

    "axes.labelsize": 10,

    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,

    "legend.fontsize": 7.5,

    "axes.linewidth": 0.8,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 4,
    "ytick.major.size": 4,

    "xtick.minor.size": 2.3,
    "ytick.minor.size": 2.3,

    "savefig.dpi": 600,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

})


# ============================================================
# LOAD
# ============================================================

df_N = pd.read_csv(
    N_file
)

df_W = pd.read_csv(
    W_file
)

df_q = pd.read_csv(
    q_file
)

df_u0 = pd.read_csv(
    u0_file
)


# ============================================================
# CALCULATE CONVERGENCE ERRORS
# ============================================================

df_N[
    "relative_error"
] = np.nan


for case, group in df_N.groupby(
    "case"
):

    group = group.sort_values(
        "N_points"
    )

    D_ref = group.iloc[
        -1
    ][
        "D"
    ]


    indices = group.index


    df_N.loc[
        indices,
        "relative_error"
    ] = (

        np.abs(
            group[
                "D"
            ]
            -
            D_ref
        )

        /

        np.abs(
            D_ref
        )

    )


df_W[
    "relative_error"
] = np.nan


for case, group in df_W.groupby(
    "case"
):

    group = group.sort_values(
        "W"
    )

    D_ref = group.iloc[
        -1
    ][
        "D"
    ]


    indices = group.index


    df_W.loc[
        indices,
        "relative_error"
    ] = (

        np.abs(
            group[
                "D"
            ]
            -
            D_ref
        )

        /

        np.abs(
            D_ref
        )

    )


# ============================================================
# MAIN VALIDATION FIGURE
# ============================================================

fig, axes = plt.subplots(

    2,
    2,

    figsize=(
        7.2,
        6.0,
    ),

)


fig.subplots_adjust(

    left=0.10,

    right=0.97,

    bottom=0.09,

    top=0.97,

    wspace=0.30,

    hspace=0.30,

)


# ============================================================
# PANEL (a)
#
# N convergence
# ============================================================

ax = axes[
    0,
    0
]


for case, group in df_N.groupby(
    "case"
):

    group = group.sort_values(
        "N_points"
    )


    u0 = group.iloc[
        0
    ][
        "u0"
    ]


    P_tE = group.iloc[
        0
    ][
        "P_over_tE"
    ]


    ax.plot(

        group[
            "N_points"
        ],

        group[
            "relative_error"
        ],

        marker="o",

        ms=3.5,

        lw=1.2,

        label=(
            rf"$u_0={u0:g},\ "
            rf"P/t_E={P_tE:g}$"
        ),

    )


ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)


ax.set_xlabel(
    r"$N_{\rm points}$"
)

ax.set_ylabel(
    r"$|D-D_{\rm ref}|/D_{\rm ref}$"
)


ax.legend(
    frameon=False
)


ax.text(

    0.04,
    0.95,

    "(a)",

    transform=ax.transAxes,

    ha="left",
    va="top",

    fontsize=10,

)


# ============================================================
# PANEL (b)
#
# Window convergence
# ============================================================

ax = axes[
    0,
    1
]


for case, group in df_W.groupby(
    "case"
):

    group = group.sort_values(
        "W"
    )


    u0 = group.iloc[
        0
    ][
        "u0"
    ]


    P_tE = group.iloc[
        0
    ][
        "P_over_tE"
    ]


    ax.plot(

        group[
            "W"
        ],

        group[
            "relative_error"
        ],

        marker="o",

        ms=3.5,

        lw=1.2,

        label=(
            rf"$u_0={u0:g},\ "
            rf"P/t_E={P_tE:g}$"
        ),

    )


ax.set_yscale(
    "log"
)


ax.set_xlabel(
    r"Half-window $W$ in $[t_0-Wt_E,t_0+Wt_E]$"
)

ax.set_ylabel(
    r"$|D-D_{\rm ref}|/D_{\rm ref}$"
)


ax.text(

    0.04,
    0.95,

    "(b)",

    transform=ax.transAxes,

    ha="left",
    va="top",

    fontsize=10,

)


# ============================================================
# PANEL (c)
#
# Normalization identity
#
# RMS(residual)/D  vs RMS(signal)
# ============================================================

ax = axes[
    1,
    0
]


valid = (

    np.isfinite(
        df_u0[
            "D"
        ]
    )

    &

    np.isfinite(
        df_u0[
            "RMS_residual"
        ]
    )

    &

    np.isfinite(
        df_u0[
            "RMS_signal"
        ]
    )

    &

    (
        df_u0[
            "D"
        ] > 0
    )

    &

    (
        df_u0[
            "RMS_signal"
        ] > 0
    )

)


sub = df_u0.loc[
    valid
].copy()


x = sub[
    "RMS_signal"
].values


y = (

    sub[
        "RMS_residual"
    ].values

    /

    sub[
        "D"
    ].values

)


sc = ax.scatter(

    x,

    y,

    c=np.log10(
        sub[
            "P_over_tE"
        ].values
    ),

    s=5,

    alpha=0.65,

    rasterized=True,

)


lims = [

    min(
        np.nanmin(x),
        np.nanmin(y),
    ),

    max(
        np.nanmax(x),
        np.nanmax(y),
    ),

]


ax.plot(

    lims,
    lims,

    color="black",

    linestyle="--",

    lw=1.0,

)


ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)


ax.set_xlim(
    lims
)

ax.set_ylim(
    lims
)


ax.set_xlabel(
    r"$R_{\rm RMS}(A_{\rm BSPL}-1)$"
)

ax.set_ylabel(
    r"$R_{\rm RMS}(\Delta A)/D_{\rm BSPL-PSPL}$"
)


cbar = fig.colorbar(

    sc,

    ax=ax,

    pad=0.02,

)


cbar.set_label(
    r"$\log_{10}(P/t_E)$"
)


# ------------------------------------------------------------
# quantify the agreement
# ------------------------------------------------------------

frac_diff = (

    np.abs(
        y - x
    )

    /
    x

)


median_difference = np.nanmedian(
    frac_diff
)


p95_difference = np.nanpercentile(
    frac_diff,
    95,
)


ax.text(

    0.04,
    0.95,

    "(c)",

    transform=ax.transAxes,

    ha="left",
    va="top",

    fontsize=10,

)


ax.text(

    0.96,
    0.06,

    (
        rf"median $|\Delta|={median_difference:.1e}$"
        "\n"
        rf"95\% $={p95_difference:.1e}$"
    ),

    transform=ax.transAxes,

    ha="right",
    va="bottom",

    fontsize=7.5,

)


# ============================================================
# PANEL (d)
#
# q_M -> 0
# ============================================================

ax = axes[
    1,
    1
]


for P_tE, group in df_q.groupby(
    "P_over_tE"
):

    group = group.sort_values(
        "q_M"
    )


    ax.plot(

        group[
            "q_M"
        ],

        group[
            "D"
        ],

        marker=".",

        ms=3,

        lw=1.1,

        label=(
            rf"$P/t_E={P_tE:g}$"
        ),

    )


ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)


ax.set_xlabel(
    r"$q_M=M_2/M_1$"
)

ax.set_ylabel(
    r"$D_{\rm BSPL-PSPL}$"
)


ax.legend(

    frameon=False,

    ncol=2,

)


ax.text(

    0.04,
    0.95,

    "(d)",

    transform=ax.transAxes,

    ha="left",
    va="top",

    fontsize=10,

)


# ============================================================
# GENERAL AXES STYLE
# ============================================================

for ax in axes.flat:

    ax.tick_params(

        which="both",

        direction="in",

        top=True,

        right=True,

    )

    ax.grid(
        False
    )


# ============================================================
# SAVE
# ============================================================

png = os.path.join(
    root,
    "paper_D_validation.png",
)

pdf = os.path.join(
    root,
    "paper_D_validation.pdf",
)


fig.savefig(

    png,

    dpi=600,

    bbox_inches="tight",

    facecolor="white",

)


fig.savefig(

    pdf,

    bbox_inches="tight",

    facecolor="white",

)


print(
    "Saved:",
    png,
)

print(
    "Saved:",
    pdf,
)


plt.show()


# ============================================================
# ============================================================
#
# SECOND FIGURE
#
# SAME RMS, DIFFERENT D
#
# ============================================================
# ============================================================

with np.load(
    pair_file,
    allow_pickle=False,
) as d:

    t_A = d[
        "t_A"
    ]

    t_B = d[
        "t_B"
    ]

    A_truth_A = d[
        "A_truth_A"
    ]

    A_fit_A = d[
        "A_fit_A"
    ]

    A_truth_B = d[
        "A_truth_B"
    ]

    A_fit_B = d[
        "A_fit_B"
    ]

    u0_A = float(
        d["u0_A"]
    )

    u0_B = float(
        d["u0_B"]
    )

    P_tE_A = float(
        d["P_over_tE_A"]
    )

    P_tE_B = float(
        d["P_over_tE_B"]
    )

    RMS_A = float(
        d["RMS_A"]
    )

    RMS_B = float(
        d["RMS_B"]
    )

    D_A = float(
        d["D_A"]
    )

    D_B = float(
        d["D_B"]
    )


t0 = 50.0
tE = 150.0


tau_A = (
    t_A
    -
    t0
) / tE


tau_B = (
    t_B
    -
    t0
) / tE


res_A = (
    A_truth_A
    -
    A_fit_A
)


res_B = (
    A_truth_B
    -
    A_fit_B
)


signal_A = (
    A_truth_A
    -
    1.0
)


signal_B = (
    A_truth_B
    -
    1.0
)


# ============================================================
# FIGURE
# ============================================================

fig2, axes2 = plt.subplots(

    2,
    1,

    figsize=(
        4.0,
        4.8,
    ),

    sharex=True,

)


fig2.subplots_adjust(

    left=0.17,

    right=0.97,

    bottom=0.11,

    top=0.97,

    hspace=0.08,

)


# ============================================================
# RESIDUALS
# ============================================================

ax = axes2[
    0
]


ax.plot(

    tau_A,

    res_A,

    lw=1.2,

    label=(
        rf"Case A: "
        rf"$R_{{\rm RMS}}={RMS_A:.1e}$, "
        rf"$D={D_A:.1e}$"
    ),

)


ax.plot(

    tau_B,

    res_B,

    lw=1.2,

    label=(
        rf"Case B: "
        rf"$R_{{\rm RMS}}={RMS_B:.1e}$, "
        rf"$D={D_B:.1e}$"
    ),

)


ax.axhline(

    0,

    color="0.5",

    linestyle=":",

    lw=0.7,

)


ax.set_ylabel(
    r"$\Delta A$"
)


ax.legend(

    frameon=False,

    fontsize=7,

)


ax.text(

    0.03,
    0.94,

    "(a)",

    transform=ax.transAxes,

    ha="left",
    va="top",

)


# ============================================================
# EVENT SIGNAL
# ============================================================

ax = axes2[
    1
]


ax.plot(

    tau_A,

    signal_A,

    lw=1.2,

    label=(
        rf"$u_0={u0_A:.3g},\ "
        rf"P/t_E={P_tE_A:.3g}$"
    ),

)


ax.plot(

    tau_B,

    signal_B,

    lw=1.2,

    label=(
        rf"$u_0={u0_B:.3g},\ "
        rf"P/t_E={P_tE_B:.3g}$"
    ),

)


ax.set_yscale(
    "log"
)


ax.set_xlabel(
    r"$(t-t_0)/t_E$"
)


ax.set_ylabel(
    r"$A_{\rm BSPL}-1$"
)


ax.legend(

    frameon=False,

    fontsize=7,

)


ax.text(

    0.03,
    0.94,

    "(b)",

    transform=ax.transAxes,

    ha="left",
    va="top",

)


for ax in axes2:

    ax.tick_params(

        which="both",

        direction="in",

        top=True,

        right=True,

    )

    ax.grid(
        False
    )


# ============================================================
# SAVE
# ============================================================

png2 = os.path.join(
    root,
    "paper_same_RMS_different_D.png",
)

pdf2 = os.path.join(
    root,
    "paper_same_RMS_different_D.pdf",
)


fig2.savefig(

    png2,

    dpi=600,

    bbox_inches="tight",

    facecolor="white",

)


fig2.savefig(

    pdf2,

    bbox_inches="tight",

    facecolor="white",

)


print(
    "Saved:",
    png2,
)

print(
    "Saved:",
    pdf2,
)


plt.show()
