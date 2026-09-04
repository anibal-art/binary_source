import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# Style
# ============================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",

        "font.size": 12,

        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.top": True,
        "ytick.right": True,
    }
)


# ============================================================
# Load
# ============================================================

home = Path.home()

filename = (
    home
    / "binary_source"
    / "results"
    / "scan_physical_mass_luminosity"
    / "summary_mass_luminosity.npz"
)

d = np.load(
    filename,
    allow_pickle=True,
)

tracks = [
    str(x)
    for x in d["track_names"]
]

qM = d["qM_grid"].astype(float)

qf = d["qf"].astype(float)

P_over_tE = d[
    "P_over_tE_grid"
].astype(float)

D = d["D"].astype(float)


# ============================================================
# Track indices
# ============================================================

i_dark = tracks.index(
    "dark"
)

i_qM4 = tracks.index(
    "qM4"
)

i_piece = tracks.index(
    "piecewise_MS"
)

i_equal = tracks.index(
    "qf_eq_qM"
)


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    len(P_over_tE),
    figsize=(
        12,
        3.8,
    ),
    sharex=True,
    sharey=True,
)


for k, ax in enumerate(
    axes
):

    ax.plot(
        qM,
        D[
            i_dark,
            :,
            k,
        ],
        linestyle="--",
        linewidth=1.8,
        label=r"$q_f=0$",
    )

    ax.plot(
        qM,
        D[
            i_qM4,
            :,
            k,
        ],
        linewidth=1.8,
        label=r"$q_f=q_M^4$",
    )

    ax.plot(
        qM,
        D[
            i_piece,
            :,
            k,
        ],
        linewidth=2.2,
        label="piecewise MS",
    )

    ax.plot(
        qM,
        D[
            i_equal,
            :,
            k,
        ],
        linestyle=":",
        linewidth=1.8,
        label=r"$q_f=q_M$",
    )


    ax.axhline(
        1e-2,
        linestyle="--",
        linewidth=1.0,
        color="black",
    )


    # --------------------------------------------------------
    # Physical companion boundaries
    # --------------------------------------------------------

    Mjup_to_Msun = 9.5458e-4

    q_jupiter = (
        1.0
        * Mjup_to_Msun
    )

    q_bd = (
        13.0
        * Mjup_to_Msun
    )

    q_hburn = 0.08


    for q_boundary in [
        q_jupiter,
        q_bd,
        q_hburn,
    ]:

        ax.axvline(
            q_boundary,
            linewidth=0.8,
            linestyle=":",
            color="black",
            alpha=0.5,
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

    ax.set_title(
        rf"$P/t_E={P_over_tE[k]:g}$"
    )

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )


axes[0].set_ylabel(
    r"$D_{\rm BSPL-PSPL}$"
)

axes[-1].legend(
    frameon=False,
    fontsize=9,
)


fig.tight_layout()

plt.show()


# ============================================================
# SECOND FIGURE:
# actual q_f relations
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        5,
        4,
    )
)

ax.plot(
    qM,
    qf[
        i_qM4,
        :
    ],
    label=r"$q_f=q_M^4$",
)

ax.plot(
    qM,
    qf[
        i_piece,
        :
    ],
    label="piecewise MS",
)

ax.plot(
    qM,
    qM,
    linestyle="--",
    label=r"$q_f=q_M$",
)

ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)

ax.set_xlabel(
    r"$q_M$"
)

ax.set_ylabel(
    r"$q_f$"
)

ax.legend(
    frameon=False,
)

ax.tick_params(
    which="both",
    direction="in",
    top=True,
    right=True,
)

fig.tight_layout()

plt.show()
