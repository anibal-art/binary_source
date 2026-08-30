#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-figure pipeline for the BSPL--PSPL project.

The pipeline NEVER runs microlensing simulations or fits.
It only reads existing final NPZ products and generates figures.

Usage
-----
Generate every available figure:

    python binary_source/plot_codes/run_paper_figures.py

Use a particular numerical production:

    python binary_source/plot_codes/run_paper_figures.py \
        --dataset-root results/final_ba8003a1fdc8

Generate only selected figures:

    python binary_source/plot_codes/run_paper_figures.py \
        --only photocenter qmass_qflux

Strict mode:

    python binary_source/plot_codes/run_paper_figures.py \
        --strict
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import (
    LogNorm,
    TwoSlopeNorm,
)


# ============================================================
# Paths
# ============================================================

SCRIPT = Path(__file__).resolve()

PLOT_DIR = SCRIPT.parent
SOURCE_DIR = PLOT_DIR.parent
REPO_ROOT = SOURCE_DIR.parent

RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"

if str(PLOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PLOT_DIR),
    )

from paper_style import apply_paper_style


# ============================================================
# Expected final numerical products
# ============================================================

SUMMARY_PATHS = {

    "photocenter":
        Path(
            "photocenter_small_xi_tE150"
        )
        / "summary_photocenter_small_xi.npz",

    "qmass_qflux":
        Path(
            "qmass_qflux_tE150"
        )
        / "summary_qM_qf.npz",

    "mass_luminosity":
        Path(
            "mass_luminosity_tE150"
        )
        / "summary_mass_luminosity.npz",
}


# ============================================================
# Helpers
# ============================================================

def git_state():
    """
    Git provenance of the plotting code itself.
    """

    try:

        commit = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--short=12",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        status = subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )

        dirty = bool(
            status.strip()
        )

    except Exception:

        commit = "unknown"
        dirty = True

    return commit, dirty


def find_dataset_root():
    """
    Find the most complete final_<commit>/ numerical production.

    Priority:
        1. largest number of required summaries
        2. most recently modified
    """

    candidates = [
        path
        for path in RESULTS_ROOT.glob(
            "final_*"
        )
        if path.is_dir()
        and "_dirty" not in path.name
    ]

    if not candidates:

        raise FileNotFoundError(
            f"No final_* datasets found below {RESULTS_ROOT}"
        )

    ranked = []

    for root in candidates:

        score = sum(
            (
                root
                / relpath
            ).exists()
            for relpath
            in SUMMARY_PATHS.values()
        )

        ranked.append(
            (
                score,
                root.stat().st_mtime,
                root,
            )
        )

    ranked.sort(
        reverse=True,
        key=lambda item:
            (
                item[0],
                item[1],
            ),
    )

    return ranked[0][2]


def load_dataset_commit(
    filename,
):

    with np.load(
        filename,
        allow_pickle=False,
    ) as d:

        if "code_commit" not in d.files:

            raise KeyError(
                f"{filename} does not contain code_commit"
            )

        return str(
            d["code_commit"].item()
        )


def get_dataset_commit(
    dataset_root,
):
    """
    Read commit from all available summaries and ensure consistency.
    """

    commits = {}

    for name, relative in SUMMARY_PATHS.items():

        filename = (
            dataset_root
            / relative
        )

        if not filename.exists():
            continue

        commits[name] = (
            load_dataset_commit(
                filename
            )
        )

    if not commits:

        raise RuntimeError(
            f"No known summary files found in {dataset_root}"
        )

    unique = set(
        commits.values()
    )

    if len(unique) != 1:

        raise RuntimeError(
            "Inconsistent numerical provenance:\n"
            + "\n".join(
                f"  {key}: {value}"
                for key, value
                in commits.items()
            )
        )

    return next(
        iter(unique)
    )


def output_directory(
    dataset_commit,
):

    directory = (
        FIGURES_ROOT
        / f"draft_{dataset_commit}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def save_figure(
    fig,
    output_dir,
    stem,
    metadata,
):

    png = (
        output_dir
        / f"{stem}.png"
    )

    pdf = (
        output_dir
        / f"{stem}.pdf"
    )

    sidecar = (
        output_dir
        / f"{stem}.json"
    )

    fig.savefig(
        png,
        dpi=600,
    )

    fig.savefig(
        pdf,
    )

    plt.close(
        fig
    )

    sidecar.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"PNG  : {png}"
    )

    print(
        f"PDF  : {pdf}"
    )

    print(
        f"META : {sidecar}"
    )

    return {
        "png": str(png),
        "pdf": str(pdf),
        "metadata": str(sidecar),
    }


def lognorm_limits(
    values,
    lower_percentile=1.0,
    upper_percentile=99.5,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = values[
        np.isfinite(values)
        & (values > 0.0)
    ]

    if len(valid) == 0:

        raise ValueError(
            "No positive finite values for LogNorm."
        )

    low = np.percentile(
        valid,
        lower_percentile,
    )

    high = np.percentile(
        valid,
        upper_percentile,
    )

    if low <= 0.0:
        low = np.min(valid)

    if high <= low:
        high = np.max(valid)

    # Round to useful powers of ten.
    vmin = 10.0 ** np.floor(
        np.log10(low)
    )

    vmax = 10.0 ** np.ceil(
        np.log10(high)
    )

    if vmax <= vmin:
        vmax = 10.0 * vmin

    return (
        float(vmin),
        float(vmax),
    )


def symmetric_limit(
    values,
    percentile=99.0,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = np.abs(
        values[
            np.isfinite(values)
        ]
    )

    if len(valid) == 0:
        return 1.0

    vmax = np.percentile(
        valid,
        percentile,
    )

    if (
        not np.isfinite(vmax)
        or vmax <= 0.0
    ):
        vmax = np.max(
            valid
        )

    if vmax <= 0.0:
        vmax = 1.0

    return float(
        vmax
    )


def configure_log_map_axis(
    ax,
    qM,
    qf_positive,
):

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        qM.min(),
        qM.max(),
    )

    ax.set_ylim(
        qf_positive.min(),
        qf_positive.max(),
    )

    ax.set_xlabel(
        r"$q_M$"
    )

    ax.set_ylabel(
        r"$q_f$"
    )

    # First-order photocenter cancellation.
    q_diag_min = max(
        qM.min(),
        qf_positive.min(),
    )

    q_diag_max = min(
        qM.max(),
        qf_positive.max(),
    )

    diagonal = np.logspace(
        np.log10(q_diag_min),
        np.log10(q_diag_max),
        300,
    )

    ax.plot(
        diagonal,
        diagonal,
        linestyle="--",
        linewidth=1.4,
    )


# ============================================================
# Figure 1:
# qM-qf D maps
# ============================================================

def plot_qmass_qflux(
    filename,
    output_dir,
):

    print()
    print("=" * 80)
    print("FIGURE: qM-qf D map")
    print("=" * 80)

    with np.load(
        filename,
        allow_pickle=False,
    ) as d:

        qM = np.asarray(
            d["qM_grid"],
            dtype=float,
        )

        qf = np.asarray(
            d["qf_grid"],
            dtype=float,
        )

        P_over_tE = np.asarray(
            d["P_over_tE_grid"],
            dtype=float,
        )

        D = np.asarray(
            d["D"],
            dtype=float,
        )

        success = np.asarray(
            d["SUCCESS"],
            dtype=bool,
        )

        dataset_commit = str(
            d["code_commit"].item()
        )


    if not np.all(
        success
    ):

        print(
            "WARNING: qM-qf dataset contains failed fits."
        )


    # qf=0 is stored in the numerical dataset but cannot be
    # displayed on a logarithmic y axis.
    positive_qf = (
        qf > 0.0
    )

    qf_plot = qf[
        positive_qf
    ]

    D_plot = D[
        :,
        positive_qf,
        :,
    ]


    vmin, vmax = lognorm_limits(
        D_plot
    )


    apply_paper_style()


    n_period = len(
        P_over_tE
    )


    fig, axes = plt.subplots(
        1,
        n_period,
        figsize=(
            4.4 * n_period,
            4.4,
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )


    axes = np.atleast_1d(
        axes
    )


    mappable = None


    for i_P, ax in enumerate(
        axes
    ):


        Z = (
            D_plot[
                :,
                :,
                i_P,
            ].T
        )


        mappable = ax.pcolormesh(
            qM,
            qf_plot,
            Z,
            shading="auto",
            cmap="viridis",
            norm=LogNorm(
                vmin=vmin,
                vmax=vmax,
            ),
        )


        configure_log_map_axis(
            ax,
            qM,
            qf_plot,
        )


        ax.set_title(
            rf"$P/t_E={P_over_tE[i_P]:g}$"
        )


        ax.text(
            0.05,
            0.94,
            f"({chr(97 + i_P)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )


    cbar = fig.colorbar(
        mappable,
        ax=axes.tolist(),
        pad=0.02,
    )

    cbar.set_label(
        r"$D_{\rm BSPL-PSPL}$"
    )


    metadata = {
        "figure":
            "qmass_qflux_D_map",

        "source":
            str(filename),

        "dataset_commit":
            dataset_commit,

        "qf_zero_displayed":
            False,

        "qf_zero_reason":
            "logarithmic qf axis",

        "photocenter_cancellation_line":
            "qf=qM",

        "D_vmin":
            vmin,

        "D_vmax":
            vmax,
    }


    return save_figure(
        fig,
        output_dir,
        "qmass_qflux_D_map",
        metadata,
    )


# ============================================================
# Figure 2:
# mass-luminosity tracks
# ============================================================

def plot_mass_luminosity(
    filename,
    output_dir,
):

    print()
    print("=" * 80)
    print("FIGURE: mass-luminosity tracks")
    print("=" * 80)


    with np.load(
        filename,
        allow_pickle=False,
    ) as d:


        tracks = np.asarray(
            d["track_names"]
        ).astype(str)


        qM = np.asarray(
            d["qM_grid"],
            dtype=float,
        )


        P_over_tE = np.asarray(
            d["P_over_tE_grid"],
            dtype=float,
        )


        D = np.asarray(
            d["D"],
            dtype=float,
        )


        success = np.asarray(
            d["SUCCESS"],
            dtype=bool,
        )


        dataset_commit = str(
            d["code_commit"].item()
        )


    track_labels = {

        "dark":
            r"$q_f=0$",

        "power4_toy":
            r"$q_f=q_M^4$",

        "photocenter_cancel":
            r"$q_f=q_M$",
    }


    line_styles = {
        "dark":
            "-",

        "power4_toy":
            "--",

        "photocenter_cancel":
            ":",
    }


    apply_paper_style()


    fig, axes = plt.subplots(
        1,
        len(P_over_tE),
        figsize=(
            4.4 * len(P_over_tE),
            4.4,
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )


    axes = np.atleast_1d(
        axes
    )


    for i_P, ax in enumerate(
        axes
    ):


        for i_track, track in enumerate(
            tracks
        ):


            y = np.asarray(
                D[
                    i_track,
                    :,
                    i_P,
                ],
                dtype=float,
            )


            good = (
                success[
                    i_track,
                    :,
                    i_P,
                ]
                & np.isfinite(y)
                & (y > 0.0)
            )


            y_plot = np.full_like(
                y,
                np.nan,
            )

            y_plot[
                good
            ] = y[
                good
            ]


            ax.plot(
                qM,
                y_plot,
                linestyle=line_styles.get(
                    track,
                    "-",
                ),
                linewidth=2.0,
                label=track_labels.get(
                    track,
                    track,
                ),
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

        ax.set_title(
            rf"$P/t_E={P_over_tE[i_P]:g}$"
        )


        ax.text(
            0.05,
            0.94,
            f"({chr(97 + i_P)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )


    axes[0].set_ylabel(
        r"$D_{\rm BSPL-PSPL}$"
    )


    axes[0].legend(
        frameon=False,
    )


    metadata = {

        "figure":
            "mass_luminosity_D_tracks",

        "source":
            str(filename),

        "dataset_commit":
            dataset_commit,

        "tracks":
            tracks.tolist(),

        "note":
            (
                "power4_toy is an illustrative qf=qM^4 relation, "
                "not a bandpass-specific stellar isochrone"
            ),
    }


    return save_figure(
        fig,
        output_dir,
        "mass_luminosity_D_tracks",
        metadata,
    )


# ============================================================
# Figure 3:
# D and fitted-parameter biases in qM-qf space
# ============================================================

def plot_qmass_qflux_biases(
    filename,
    output_dir,
    target_P_over_tE=1.0,
):

    print()
    print("=" * 80)
    print("FIGURE: qM-qf biases")
    print("=" * 80)


    with np.load(
        filename,
        allow_pickle=False,
    ) as d:


        qM = np.asarray(
            d["qM_grid"],
            dtype=float,
        )


        qf = np.asarray(
            d["qf_grid"],
            dtype=float,
        )


        P_over_tE = np.asarray(
            d["P_over_tE_grid"],
            dtype=float,
        )


        D = np.asarray(
            d["D"],
            dtype=float,
        )


        DT0 = np.asarray(
            d["DT0"],
            dtype=float,
        )


        DU0 = np.asarray(
            d["DU0"],
            dtype=float,
        )


        DTE = np.asarray(
            d["DTE"],
            dtype=float,
        )


        u0_true = float(
            d["u0_true"].item()
        )


        tE_true = float(
            d["tE_true"].item()
        )


        dataset_commit = str(
            d["code_commit"].item()
        )


    i_P = int(
        np.argmin(
            np.abs(
                P_over_tE
                - target_P_over_tE
            )
        )
    )


    selected_P = float(
        P_over_tE[
            i_P
        ]
    )


    positive_qf = (
        qf > 0.0
    )

    qf_plot = qf[
        positive_qf
    ]


    D_map = (
        D[
            :,
            positive_qf,
            i_P,
        ].T
    )


    dt0_map = (
        DT0[
            :,
            positive_qf,
            i_P,
        ].T
        / tE_true
    )


    du0_map = (
        DU0[
            :,
            positive_qf,
            i_P,
        ].T
        / u0_true
    )


    dte_map = (
        DTE[
            :,
            positive_qf,
            i_P,
        ].T
        / tE_true
    )


    D_vmin, D_vmax = (
        lognorm_limits(
            D_map
        )
    )


    bias_maps = [
        dt0_map,
        du0_map,
        dte_map,
    ]


    bias_limits = [
        symmetric_limit(
            values
        )
        for values
        in bias_maps
    ]


    apply_paper_style()


    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            10.0,
            8.6,
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )


    axes = axes.ravel()


    # --------------------------------------------------------
    # D
    # --------------------------------------------------------

    pcm = axes[0].pcolormesh(
        qM,
        qf_plot,
        D_map,
        shading="auto",
        cmap="viridis",
        norm=LogNorm(
            vmin=D_vmin,
            vmax=D_vmax,
        ),
    )


    configure_log_map_axis(
        axes[0],
        qM,
        qf_plot,
    )


    axes[0].set_title(
        r"$D_{\rm BSPL-PSPL}$"
    )


    cb = fig.colorbar(
        pcm,
        ax=axes[0],
        pad=0.02,
    )

    cb.set_label(
        r"$D$"
    )


    # --------------------------------------------------------
    # Bias panels
    # --------------------------------------------------------

    bias_specs = [

        (
            dt0_map,
            bias_limits[0],
            r"$\Delta t_0/t_E$",
        ),

        (
            du0_map,
            bias_limits[1],
            r"$\Delta u_0/u_0$",
        ),

        (
            dte_map,
            bias_limits[2],
            r"$\Delta t_E/t_E$",
        ),
    ]


    for ax, (
        values,
        limit,
        label,
    ) in zip(
        axes[1:],
        bias_specs,
    ):


        pcm = ax.pcolormesh(
            qM,
            qf_plot,
            values,
            shading="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(
                vmin=-limit,
                vcenter=0.0,
                vmax=limit,
            ),
        )


        configure_log_map_axis(
            ax,
            qM,
            qf_plot,
        )


        ax.set_title(
            label
        )


        cb = fig.colorbar(
            pcm,
            ax=ax,
            pad=0.02,
        )

        cb.set_label(
            label
        )


    for i, ax in enumerate(
        axes
    ):

        ax.text(
            0.05,
            0.94,
            f"({chr(97 + i)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )


    fig.suptitle(
        rf"$P/t_E={selected_P:g}$",
        fontsize=16,
    )


    metadata = {

        "figure":
            "qmass_qflux_biases",

        "source":
            str(filename),

        "dataset_commit":
            dataset_commit,

        "P_over_tE":
            selected_P,

        "u0_true":
            u0_true,

        "tE_true":
            tE_true,

        "normalizations": {
            "DT0":
                "DT0/tE_true",

            "DU0":
                "DU0/u0_true",

            "DTE":
                "DTE/tE_true",
        },

        "photocenter_cancellation_line":
            "qf=qM",
    }


    return save_figure(
        fig,
        output_dir,
        "qmass_qflux_biases",
        metadata,
    )


# ============================================================
# Figure 4:
# existing photocenter scaling plot
# ============================================================

def plot_photocenter(
    filename,
):

    script = (
        PLOT_DIR
        / "plot_photocenter_scaling.py"
    )

    if not script.exists():

        raise FileNotFoundError(
            script
        )


    print()
    print("=" * 80)
    print("FIGURE: photocenter scaling")
    print("=" * 80)


    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(filename),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


# ============================================================
# Manifest
# ============================================================

def write_manifest(
    output_dir,
    dataset_root,
    dataset_commit,
    available,
    generated,
):

    plotting_commit, plotting_dirty = (
        git_state()
    )

    filename = (
        output_dir
        / "figure_manifest.json"
    )


    # ========================================================
    # Load existing manifest, if present.
    #
    # This allows --only runs to add/update figures without
    # erasing provenance from figures generated previously.
    # ========================================================

    if filename.exists():

        try:

            existing = json.loads(
                filename.read_text()
            )

        except Exception:

            existing = {}

    else:

        existing = {}


    previous_figures = existing.get(
        "generated_figures",
        {},
    )


    # Update only the figures generated in this invocation.
    previous_figures.update(
        generated
    )


    # ========================================================
    # Explicit numerical-source provenance
    # ========================================================

    numerical_sources = {}


    for name, source_file in available.items():

        source_file = Path(
            source_file
        ).resolve()

        try:

            source_commit = (
                load_dataset_commit(
                    source_file
                )
            )

        except Exception:

            source_commit = "UNKNOWN"


        numerical_sources[
            name
        ] = {
            "file":
                str(source_file),

            "code_commit":
                source_commit,
        }


    payload = {

        "last_updated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "dataset_root":
            str(
                Path(
                    dataset_root
                ).resolve()
            ),

        "primary_dataset_commit":
            dataset_commit,

        "plotting_code_commit":
            plotting_commit,

        "plotting_code_dirty":
            plotting_dirty,

        "numerical_sources":
            numerical_sources,

        "generated_figures":
            previous_figures,
    }


    filename.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    print()
    print(
        "Manifest:",
        filename,
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Explicit results/final_<commit>/ directory. "
            "If omitted, choose the most complete final dataset."
        ),
    )


    parser.add_argument(
        "--only",
        nargs="+",
        choices=[
            "photocenter",
            "qmass_qflux",
            "mass_luminosity",
            "biases",
        ],
        default=None,
    )


    parser.add_argument(
        "--qmass-qflux-summary",
        type=Path,
        default=None,
        help=(
            "Explicit summary_qM_qf.npz from a compatible "
            "numerical production. Its original code_commit "
            "is preserved in the figure metadata."
        ),
    )


    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if a requested numerical summary is missing."
        ),
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
    )


    args = parser.parse_args()


    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
        if args.dataset_root
        is not None
        else find_dataset_root()
    )


    if not dataset_root.exists():

        raise FileNotFoundError(
            dataset_root
        )


    dataset_commit = (
        get_dataset_commit(
            dataset_root
        )
    )


    output_dir = (
        output_directory(
            dataset_commit
        )
    )


    available = {

        name:
            dataset_root
            / relative

        for name, relative
        in SUMMARY_PATHS.items()

        if (
            dataset_root
            / relative
        ).exists()
    }


    # ========================================================
    # Optional compatible source from another final production
    # ========================================================

    if args.qmass_qflux_summary is not None:

        qmqf_file = (
            args.qmass_qflux_summary
            .expanduser()
            .resolve()
        )

        if not qmqf_file.exists():

            raise FileNotFoundError(
                qmqf_file
            )

        available[
            "qmass_qflux"
        ] = qmqf_file


    selected = (

        args.only

        if args.only
        is not None

        else [
            "photocenter",
            "qmass_qflux",
            "mass_luminosity",
            "biases",
        ]
    )


    print()
    print("=" * 80)
    print("BSPL--PSPL PAPER FIGURE PIPELINE")
    print("=" * 80)

    print(
        "dataset root  =",
        dataset_root,
    )

    print(
        "dataset commit=",
        dataset_commit,
    )

    print(
        "output        =",
        output_dir,
    )

    print(
        "available     =",
        list(
            available
        ),
    )

    print(
        "requested     =",
        selected,
    )

    print()
    print("numerical sources:")

    for source_name, source_file in available.items():

        try:

            source_commit = load_dataset_commit(
                source_file
            )

        except Exception:

            source_commit = "UNKNOWN"

        print(
            f"  {source_name:16s} "
            f"{source_commit:14s} "
            f"{source_file}"
        )

    print("=" * 80)


    if args.dry_run:
        return


    generated = {}


    # ========================================================
    # Photocenter
    # ========================================================

    if "photocenter" in selected:

        if "photocenter" not in available:

            message = (
                "Missing photocenter summary."
            )

            if args.strict:
                raise FileNotFoundError(
                    message
                )

            print(
                "SKIP:",
                message,
            )

        else:

            plot_photocenter(
                available[
                    "photocenter"
                ]
            )

            generated[
                "photocenter"
            ] = {
                "status":
                    "generated"
            }


    # ========================================================
    # qM-qf D map
    # ========================================================

    if "qmass_qflux" in selected:

        if "qmass_qflux" not in available:

            message = (
                "Missing qmass_qflux summary."
            )

            if args.strict:
                raise FileNotFoundError(
                    message
                )

            print(
                "SKIP:",
                message,
            )

        else:

            generated[
                "qmass_qflux"
            ] = plot_qmass_qflux(
                available[
                    "qmass_qflux"
                ],
                output_dir,
            )


    # ========================================================
    # mass-luminosity
    # ========================================================

    if "mass_luminosity" in selected:

        if (
            "mass_luminosity"
            not in available
        ):

            message = (
                "Missing mass_luminosity summary."
            )

            if args.strict:
                raise FileNotFoundError(
                    message
                )

            print(
                "SKIP:",
                message,
            )

        else:

            generated[
                "mass_luminosity"
            ] = plot_mass_luminosity(
                available[
                    "mass_luminosity"
                ],
                output_dir,
            )


    # ========================================================
    # biases
    # ========================================================

    if "biases" in selected:

        if "qmass_qflux" not in available:

            message = (
                "Bias figure requires qmass_qflux summary."
            )

            if args.strict:
                raise FileNotFoundError(
                    message
                )

            print(
                "SKIP:",
                message,
            )

        else:

            generated[
                "biases"
            ] = plot_qmass_qflux_biases(
                available[
                    "qmass_qflux"
                ],
                output_dir,
            )


    write_manifest(
        output_dir,
        dataset_root,
        dataset_commit,
        available,
        generated,
    )


    print()
    print("=" * 80)
    print("FIGURE PIPELINE FINISHED")
    print("=" * 80)

    print(
        "Figures:",
        output_dir,
    )


if __name__ == "__main__":
    main()
