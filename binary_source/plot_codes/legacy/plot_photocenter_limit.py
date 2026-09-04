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
    / "scan_photocenter_small_xi"
    / "summary_photocenter_small_xi.npz"
)

d = np.load(
    filename,
    allow_pickle=True,
)

xi_over_u0 = d[
    "xi_over_u0_grid"
].astype(float)

qM = d[
    "qM_grid"
].astype(float)

families = [
    str(x)
    for x in d["families"]
]

D = d[
    "D"
].astype(float)


# ============================================================
# Indices
# ============================================================

i_dark = families.index(
    "dark"
)

i_cancel = families.index(
    "photocenter_cancel"
)


# ============================================================
# Ratio
# ============================================================

D_dark = D[
    :,
    :,
    i_dark,
]

D_cancel = D[
    :,
    :,
    i_cancel,
]

ratio = (
    D_cancel
    / D_dark
)


# ============================================================
# FIGURE 1:
# ratio vs q_M
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        5.5,
        4.2,
    )
)


for i, xi in enumerate(
    xi_over_u0
):

    ax.plot(
        qM,
        ratio[
            i,
            :
        ],
        linewidth=2,
        label=(
            rf"$\xi_{{\rm rel}}/u_0={xi:g}$"
        ),
    )


ax.axhline(
    1.0,
    color="black",
    linestyle="--",
    linewidth=1.2,
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
    r"$D(q_f=q_M)/D(q_f=0)$"
)

ax.legend(
    frameon=False,
    fontsize=9,
)

ax.tick_params(
    which="both",
    direction="in",
    top=True,
    right=True,
)

fig.tight_layout()

plt.show()


# ============================================================
# FIGURE 2:
# ratio vs xi/u0 for representative q_M
# ============================================================

qM_targets = [
    1e-3,
    1e-2,
    1e-1,
    0.5,
    1.0,
]


fig, ax = plt.subplots(
    figsize=(
        5.5,
        4.2,
    )
)


for q_target in qM_targets:

    iq = np.argmin(
        np.abs(
            np.log10(qM)
            -
            np.log10(q_target)
        )
    )

    ax.plot(
        xi_over_u0,
        ratio[
            :,
            iq,
        ],
        marker="o",
        linewidth=2,
        label=(
            rf"$q_M={qM[iq]:.3g}$"
        ),
    )


ax.axhline(
    1.0,
    color="black",
    linestyle="--",
    linewidth=1.2,
)

ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)

ax.set_xlabel(
    r"$\xi_{\rm rel}/u_0$"
)

ax.set_ylabel(
    r"$D(q_f=q_M)/D(q_f=0)$"
)

ax.legend(
    frameon=False,
    fontsize=9,
)

ax.tick_params(
    which="both",
    direction="in",
    top=True,
    right=True,
)

fig.tight_layout()

plt.show()


# ============================================================
# FIGURE 3:
# actual D for one representative q_M
# ============================================================

q_target = 0.1

iq = np.argmin(
    np.abs(
        np.log10(qM)
        -
        np.log10(q_target)
    )
)


fig, ax = plt.subplots(
    figsize=(
        5.2,
        4.0,
    )
)


ax.plot(
    xi_over_u0,
    D_dark[
        :,
        iq,
    ],
    marker="o",
    linewidth=2,
    label=r"$q_f=0$",
)

ax.plot(
    xi_over_u0,
    D_cancel[
        :,
        iq,
    ],
    marker="o",
    linewidth=2,
    label=r"$q_f=q_M$",
)


ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)

ax.set_xlabel(
    r"$\xi_{\rm rel}/u_0$"
)

ax.set_ylabel(
    r"$D_{\rm BSPL-PSPL}$"
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
