from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator


# ============================================================
# PATHS
# ============================================================

home = Path.home()

repo = (
    home
    / "binary_source"
)

isochrone_file = (
    repo
    / "data"
    / "parsec"
    / "isochrone_output717432377933.dat"
)

output_directory = (
    repo
    / "figures"
    / "current"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# READ PARSEC HEADER
# ============================================================

with open(
    isochrone_file,
    "r",
) as f:

    lines = f.readlines()


header = None

for line in lines:

    if line.startswith(
        "# Zini"
    ):

        header = (
            line[1:]
            .strip()
            .split()
        )

        break


if header is None:

    raise RuntimeError(
        "PARSEC column header not found."
    )


# ============================================================
# READ TABLE
# ============================================================

df = pd.read_csv(

    isochrone_file,

    comment="#",

    sep=r"\s+",

    names=header,

)


print()
print("=" * 80)
print("PARSEC ISOCHRONE")
print("=" * 80)

print(
    "Rows       =",
    len(df),
)

print(
    "logAge     =",
    np.unique(
        df["logAge"]
    ),
)

print(
    "[M/H]      =",
    np.unique(
        df["MH"]
    ),
)

print(
    "Labels     =",
    sorted(
        df["label"].unique()
    ),
)


# ============================================================
# MAIN SEQUENCE ONLY
# ============================================================

ms = (
    df[
        df["label"] == 1
    ]
    .copy()
    .sort_values(
        "Mass"
    )
)


print()
print("MAIN SEQUENCE")

print(
    "N points   =",
    len(ms),
)

print(
    "Mass range =",
    ms["Mass"].min(),
    "...",
    ms["Mass"].max(),
    "Msun",
)


# ============================================================
# F146 MASS -> MAGNITUDE RELATION
# ============================================================

mass_grid = (
    ms["Mass"]
    .to_numpy(
        dtype=float
    )
)

F146_grid = (
    ms["F146mag"]
    .to_numpy(
        dtype=float
    )
)


F146_of_mass = PchipInterpolator(

    mass_grid,

    F146_grid,

    extrapolate=False,

)


# ============================================================
# PHOTOCENTER QUANTITIES
# ============================================================

def flux_ratio_F146(
    M1,
    M2,
):

    mag1 = float(
        F146_of_mass(
            M1
        )
    )

    mag2 = np.asarray(
        F146_of_mass(
            M2
        ),
        dtype=float,
    )

    qf = 10.0 ** (

        -0.4
        * (
            mag2
            - mag1
        )

    )

    return qf


def photocenter_coefficient(
    qM,
    qf,
):

    return (

        (qM - qf)

        /

        (
            (1.0 + qM)
            *
            (1.0 + qf)
        )

    )


def suppression_relative_to_dark(
    qM,
    qf,
):

    return (

        np.abs(
            qM - qf
        )

        /

        (
            qM
            *
            (1.0 + qf)
        )

    )


# ============================================================
# PRIMARY MASSES
# ============================================================

M1_values = [

    0.5,
    0.7,
    0.8,

]


tracks = {}


for M1 in M1_values:

    if not (
        mass_grid.min()
        <= M1
        <= mass_grid.max()
    ):

        raise ValueError(
            f"M1={M1} Msun outside MS grid."
        )


    qM_min = (
        mass_grid.min()
        /
        M1
    )


    qM = np.geomspace(

        qM_min,

        1.0,

        300,

    )


    M2 = (
        qM
        *
        M1
    )


    qf = flux_ratio_F146(

        M1,
        M2,

    )


    Cph = photocenter_coefficient(

        qM,
        qf,

    )


    Sph = suppression_relative_to_dark(

        qM,
        qf,

    )


    tracks[M1] = {

        "qM":
            qM,

        "M2":
            M2,

        "qf":
            qf,

        "Cph":
            Cph,

        "Sph":
            Sph,

    }


# ============================================================
# SOME PHYSICAL NUMBERS
# ============================================================

print()
print("=" * 80)
print("F146 PHOTOCENTER SUPPRESSION")
print("=" * 80)


for M1 in M1_values:

    print()
    print(
        f"M1 = {M1:.1f} Msun"
    )

    for qM_test in [

        0.2,
        0.5,
        0.8,
        0.95,

    ]:

        M2_test = (
            qM_test
            *
            M1
        )


        if (
            M2_test
            <
            mass_grid.min()
        ):

            continue


        qf_test = float(

            flux_ratio_F146(

                M1,
                M2_test,

            )

        )


        S_test = float(

            suppression_relative_to_dark(

                qM_test,
                qf_test,

            )

        )


        print(

            f"  qM={qM_test:4.2f}  "
            f"qf(F146)={qf_test:8.5f}  "
            f"Sph={S_test:8.5f}"

        )


# ============================================================
# FIGURE 1:
# PHYSICAL qf(qM)
# ============================================================

fig, ax = plt.subplots(

    figsize=(
        6.5,
        5.0,
    )

)


q_ref = np.logspace(

    -1.1,

    0.0,

    500,

)


ax.plot(

    q_ref,
    q_ref,

    linestyle="--",

    linewidth=1.5,

    label=r"$q_f=q_M$ (exact cancellation)",

)


ax.plot(

    q_ref,
    q_ref**4,

    linestyle=":",

    linewidth=1.5,

    label=r"$q_f=q_M^4$ (toy model)",

)


for M1 in M1_values:

    track = tracks[
        M1
    ]

    ax.plot(

        track["qM"],
        track["qf"],

        linewidth=2,

        label=(
            rf"PARSEC F146, "
            rf"$M_1={M1:.1f}\,M_\odot$"
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
    r"$q_f^{\rm F146}=F_2/F_1$"
)


ax.grid(
    alpha=0.25
)

ax.legend(
    fontsize=8
)


fig.tight_layout()


fig.savefig(

    output_directory
    /
    "physical_qM_qf_PARSECF146.pdf"

)

fig.savefig(

    output_directory
    /
    "physical_qM_qf_PARSECF146.png",

    dpi=300,

)


# ============================================================
# FIGURE 2:
# PHOTOCENTER SUPPRESSION
# ============================================================

fig, ax = plt.subplots(

    figsize=(
        6.5,
        5.0,
    )

)


for M1 in M1_values:

    track = tracks[
        M1
    ]

    ax.plot(

        track["qM"],
        track["Sph"],

        linewidth=2,

        label=(
            rf"$M_1={M1:.1f}\,M_\odot$"
        ),

    )


ax.axhline(

    1.0,

    linestyle=":",

    linewidth=1,

)


ax.axhline(

    0.0,

    linestyle="--",

    linewidth=1,

)


ax.set_xscale(
    "log"
)


ax.set_xlabel(
    r"$q_M=M_2/M_1$"
)

ax.set_ylabel(
    r"$S_{\rm ph}$"
)


ax.set_ylim(
    -0.03,
    1.05,
)


ax.grid(
    alpha=0.25
)

ax.legend(
    fontsize=9
)


fig.tight_layout()


fig.savefig(

    output_directory
    /
    "physical_photocenter_suppression_PARSECF146.pdf"

)

fig.savefig(

    output_directory
    /
    "physical_photocenter_suppression_PARSECF146.png",

    dpi=300,

)


plt.show()
