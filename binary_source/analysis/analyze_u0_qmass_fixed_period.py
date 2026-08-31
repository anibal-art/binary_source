#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import numpy as np


SCRIPT = Path(__file__).resolve()
SOURCE_DIR = SCRIPT.parents[1]
REPO_ROOT = SOURCE_DIR.parent
RESULTS_ROOT = REPO_ROOT / "results"


FEATURE_QM_MIN = 3e-2
FEATURE_QM_MAX = 3e-1


def find_summary():

    candidates = list(
        RESULTS_ROOT.glob(
            "diagnostic_u0_qmass_PoverTE1_qf0_*/"
            "summary_u0_qmass_fixed_period.npz"
        )
    )

    if not candidates:

        raise FileNotFoundError(
            "No u0-qM diagnostic summary found."
        )

    return max(
        candidates,
        key=lambda p:
            p.stat().st_mtime,
    )


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=Path,
    default=None,
)

args = parser.parse_args()


filename = (
    args.input.expanduser().resolve()
    if args.input is not None
    else find_summary().resolve()
)


with np.load(
    filename,
    allow_pickle=False,
) as d:

    u0 = np.asarray(
        d["u0_grid"],
        dtype=float,
    )

    qM = np.asarray(
        d["qM_grid"],
        dtype=float,
    )

    D = np.asarray(
        d["D"],
        dtype=float,
    )

    U1MIN = np.asarray(
        d["U1MIN"],
        dtype=float,
    )

    U2MIN = np.asarray(
        d["U2MIN"],
        dtype=float,
    )

    DT_U1MIN = np.asarray(
        d["DT_U1MIN_OVER_TE"],
        dtype=float,
    )

    XI1_U0 = np.asarray(
        d["xi1_over_u0"],
        dtype=float,
    )

    XI2_U0 = np.asarray(
        d["xi2_over_u0"],
        dtype=float,
    )

    DU0 = np.asarray(
        d["DU0"],
        dtype=float,
    )

    SUCCESS = np.asarray(
        d["SUCCESS"],
        dtype=bool,
    )

    tE = float(
        d["tE_true"].item()
    )

    P_over_tE = float(
        d["P_over_tE"].item()
    )

    qf = float(
        d["qf"].item()
    )

    commit = str(
        d["code_commit"].item()
    )


feature_mask_q = (
    (qM >= FEATURE_QM_MIN)
    & (qM <= FEATURE_QM_MAX)
)


qM_Dmax = np.full(
    len(u0),
    np.nan,
)

Dmax = np.full(
    len(u0),
    np.nan,
)

qM_U1min = np.full(
    len(u0),
    np.nan,
)

U1min_feature = np.full(
    len(u0),
    np.nan,
)


for i_u in range(
    len(u0)
):

    valid = (
        feature_mask_q
        & SUCCESS[i_u]
        & np.isfinite(
            D[i_u]
        )
        & np.isfinite(
            U1MIN[i_u]
        )
    )

    idx = np.where(
        valid
    )[0]

    if len(idx) == 0:
        continue


    j_D = idx[
        np.argmax(
            D[
                i_u,
                idx,
            ]
        )
    ]


    j_u1 = idx[
        np.argmin(
            U1MIN[
                i_u,
                idx,
            ]
        )
    ]


    qM_Dmax[i_u] = (
        qM[j_D]
    )

    Dmax[i_u] = (
        D[
            i_u,
            j_D,
        ]
    )

    qM_U1min[i_u] = (
        qM[j_u1]
    )

    U1min_feature[i_u] = (
        U1MIN[
            i_u,
            j_u1,
        ]
    )


valid_alignment = (
    np.isfinite(
        qM_Dmax
    )
    & np.isfinite(
        qM_U1min
    )
)


delta_log_q = (
    np.log10(
        qM_Dmax[
            valid_alignment
        ]
    )
    - np.log10(
        qM_U1min[
            valid_alignment
        ]
    )
)


print()
print("=" * 88)
print("u0 x qM FIXED-PERIOD DIAGNOSTIC")
print("=" * 88)

print(
    "file        =",
    filename,
)

print(
    "commit      =",
    commit,
)

print(
    "P/tE        =",
    P_over_tE,
)

print(
    "qf          =",
    qf,
)

print(
    "shape D     =",
    D.shape,
)

print(
    "success     =",
    np.count_nonzero(
        SUCCESS
    ),
    "/",
    SUCCESS.size,
)

print(
    "feature qM  =",
    FEATURE_QM_MIN,
    "->",
    FEATURE_QM_MAX,
)


if len(
    delta_log_q
) > 0:

    print()
    print(
        "median log10(qM_Dmax/qM_U1min) =",
        np.nanmedian(
            delta_log_q
        ),
    )

    print(
        "median |delta log10 qM|        =",
        np.nanmedian(
            np.abs(
                delta_log_q
            )
        ),
    )


print()
print("=" * 88)
print("REPRESENTATIVE u0 VALUES")
print("=" * 88)

print(
    "u0        "
    "qM(D max)   "
    "Dmax        "
    "qM(U1 min)  "
    "U1min       "
    "xi1/u0@Dmax "
    "dt(U1min)/tE"
)


targets = [
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
]


for target in targets:

    i_u = int(
        np.argmin(
            np.abs(
                np.log10(u0)
                - np.log10(target)
            )
        )
    )


    if not np.isfinite(
        qM_Dmax[i_u]
    ):
        continue


    j_D = int(
        np.argmin(
            np.abs(
                np.log10(qM)
                - np.log10(
                    qM_Dmax[i_u]
                )
            )
        )
    )


    print(
        f"{u0[i_u]:8.3e} "
        f"{qM_Dmax[i_u]:11.4e} "
        f"{Dmax[i_u]:11.4e} "
        f"{qM_U1min[i_u]:11.4e} "
        f"{U1min_feature[i_u]:11.4e} "
        f"{XI1_U0[i_u, j_D]:11.4e} "
        f"{DT_U1MIN[i_u, j_D]:11.4e}"
    )


outfile = (
    filename.parent
    / "feature_diagnostics.npz"
)


np.savez_compressed(

    outfile,

    u0_grid=u0,

    qM_Dmax=qM_Dmax,
    Dmax=Dmax,

    qM_U1min=qM_U1min,
    U1min_feature=U1min_feature,

    feature_qM_min=np.float64(
        FEATURE_QM_MIN
    ),

    feature_qM_max=np.float64(
        FEATURE_QM_MAX
    ),

    source_summary=np.array(
        str(filename)
    ),

    code_commit=np.array(
        commit
    ),
)


print()
print(
    "Saved:",
    outfile,
)
