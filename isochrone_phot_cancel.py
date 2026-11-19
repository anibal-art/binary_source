from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator


# ============================================================
# PATHS
# ============================================================

home = Path.home()

repo = Path(__file__).resolve().parent          # repository root

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
# TRACKS FOR run_paper_figures.py AND FIGURES IN THE PAPER STYLE
# ============================================================

import json
import sys

sys.path.insert(0, str(repo / "binary_source" / "plot_codes"))
from paper_style import (apply_paper_style, fig_width, panel_label, save_figure,  # noqa: E402
                         M1_STYLE, LW_MAIN, LW_THIN, FS_ANNOT)

json_path = output_directory / "physical_qM_qf_PARSECF146.json"
json_path.write_text(json.dumps({
    "isochrone": isochrone_file.name,
    "band": "Roman F146",
    "tracks": {f"{M1:g}": {"qM": np.asarray(tracks[M1]["qM"]).tolist(),
                           "qf": np.asarray(tracks[M1]["qf"]).tolist()}
               for M1 in M1_values},
}, indent=1))
print("saved", json_path)

apply_paper_style()
w = fig_width(0.49)


def track_lines(ax, key):
    for M1 in M1_values:
        st = M1_STYLE[M1]
        ax.plot(tracks[M1]["qM"], tracks[M1][key], color=st["color"], lw=LW_MAIN,
                marker=st["marker"], markevery=0.2, ms=3.0, mec="white", mew=0.5,
                label=rf"$M_{{S,1}}={M1:g}\,M_\odot$")


# (a) F146 flux ratio against mass ratio
fig, ax = plt.subplots(figsize=(w, 0.85 * w))
q_ref = np.logspace(-1.1, 0.0, 500)
ax.plot(q_ref, q_ref, color="k", lw=LW_THIN, ls="--")
ax.plot(q_ref, q_ref ** 4, color="0.45", lw=LW_THIN, ls=":")
ax.text(0.22, 0.42, r"$q_f=q_M$", fontsize=FS_ANNOT)
ax.text(0.33, 2.5e-3, r"$q_f=q_M^4$", fontsize=FS_ANNOT, color="0.45")
track_lines(ax, "qf")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.08, 1.05)
ax.set_ylim(3e-5, 1.5)
ax.set_xlabel(r"$q_M=M_{S,2}/M_{S,1}$")
ax.set_ylabel(r"$q_f=F_{S,2}/F_{S,1}$ (F146)")
ax.legend(loc="lower right")
panel_label(ax, "a")
for f in save_figure(fig, output_directory / "physical_qM_qf_PARSECF146"):
    print("saved", f)
plt.close(fig)

# (b) surviving first-order photocenter amplitude
fig, ax = plt.subplots(figsize=(w, 0.85 * w))
ax.axhline(1.0, color="0.6", lw=LW_THIN, ls=":")
ax.axhline(0.0, color="0.6", lw=LW_THIN, ls="--")
track_lines(ax, "Sph")
ax.set_xlim(0.1, 1.0)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel(r"$q_M=M_{S,2}/M_{S,1}$")
ax.set_ylabel(r"$S_{\rm ph}$")
ax.legend(loc="lower left")
panel_label(ax, "b")
for f in save_figure(fig, output_directory / "physical_photocenter_suppression_PARSECF146"):
    print("saved", f)
plt.close(fig)
