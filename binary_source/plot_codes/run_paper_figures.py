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

from paper_style import (
    apply_paper_style,
    fig_width,
    heatmap,
    panel_label,
    pow10_label,
    halo,
    d_norm,
    save_figure as style_save_figure,
    D_CMAP,
    D_VMIN,
    D_VMAX,
    D_LABEL,
    BIAS_CMAP,
    FS_ANNOT,
    FS_LEGEND,
    LW_MAIN,
    LW_THIN,
    LS_CANCEL,
    M1_STYLE,
)
from matplotlib.ticker import LogLocator

VERIFY_ROOT = REPO_ROOT / "figures" / "_verify"
VERIFY_ROOT.mkdir(parents=True, exist_ok=True)


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
        / "current"
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
    """Save <stem>.pdf/.png at the designed (printed) size + JSON sidecar."""

    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    sidecar = output_dir / f"{stem}.json"

    style_save_figure(fig, output_dir / stem)
    plt.close(fig)

    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(f"PNG : {png}")
    print(f"PDF : {pdf}")
    print(f"META : {sidecar}")

    return {"png": str(png), "pdf": str(pdf), "metadata": str(sidecar)}


def lognorm_limits(
    values,
    lower_percentile=1.0,
    upper_percentile=99.5,
):
    """Kept for reference: the restyled figures use the paper-wide D range."""
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values) & (values > 0.0)]
    if len(valid) == 0:
        raise ValueError("No positive finite values for LogNorm.")
    low = np.percentile(valid, lower_percentile)
    high = np.percentile(valid, upper_percentile)
    if low <= 0.0:
        low = np.min(valid)
    if high <= low:
        high = np.max(valid)
    vmin = 10.0 ** np.floor(np.log10(low))
    vmax = 10.0 ** np.ceil(np.log10(high))
    if vmax <= vmin:
        vmax = 10.0 * vmin
    return float(vmin), float(vmax)


def symmetric_limit(
    values,
    percentile=99.0,
):
    values = np.asarray(values, dtype=float)
    valid = np.abs(values[np.isfinite(values)])
    if len(valid) == 0:
        return 1.0
    vmax = np.percentile(valid, percentile)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = np.max(valid)
    if vmax <= 0.0:
        vmax = 1.0
    return float(vmax)


def configure_log_map_axis(
    ax,
    qM,
    qf_positive,
    *,
    show_xlabel=True,
    show_ylabel=True,
    label_diagonal=False,
    diag_color="white",
    hide_last_xlabel=False,
):
    """Log-log (q_M, q_f) axes with the q_f = q_M cancellation locus (dashed)."""

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(qM.min(), qM.max())
    ax.set_ylim(qf_positive.min(), qf_positive.max())
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.set_xlabel(r"$q_M$" if show_xlabel else "")
    ax.set_ylabel(r"$q_f$" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    if not show_xlabel:
        ax.tick_params(labelbottom=False)

    q_diag_min = max(qM.min(), qf_positive.min())
    q_diag_max = min(qM.max(), qf_positive.max())
    diagonal = np.logspace(np.log10(q_diag_min), np.log10(q_diag_max), 400)
    ax.plot(diagonal, diagonal, color=diag_color, ls=LS_CANCEL, lw=LW_MAIN,
            zorder=7)

    if hide_last_xlabel:
        # avoid the "10^0" label colliding with the next panel's "10^-4"
        from matplotlib.ticker import FuncFormatter
        xmax = qM.max()
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: "" if np.isclose(v, xmax)
            else rf"$10^{{{int(round(np.log10(v)))}}}$"))

    if label_diagonal:
        q_label = 10.0 ** (np.log10(q_diag_min)
                           + 0.20 * (np.log10(q_diag_max) - np.log10(q_diag_min)))
        ax.text(q_label, q_label * 1.25, r"$q_f=q_M$", color="white",
                fontsize=FS_ANNOT, rotation=45, rotation_mode="anchor",
                ha="left", va="bottom", zorder=8)


def label_contours(ax, cs, x_axes_frac, color="white", y_axes_frac=0.18):
    """Inline labels for every contour segment, placed near a fixed x position.

    Each segment receives one label at the point closest (in axes coordinates)
    to x = x_axes_frac on that segment; labels of nested L-shaped contours in
    the (q_M, q_f) maps therefore sit on their horizontal arms.
    """
    to_axes = ax.transData + ax.transAxes.inverted()
    manual = []
    for i, _lv in enumerate(cs.levels):
        for seg in cs.allsegs[i]:
            if len(seg) < 8:
                continue
            xy = to_axes.transform(seg)
            inside = (xy[:, 0] > 0.06) & (xy[:, 0] < 0.94) & \
                     (xy[:, 1] > 0.06) & (xy[:, 1] < 0.90)
            if not np.any(inside):
                continue
            idx = np.where(inside)[0]
            # one label on the horizontal arm (closest to x = x_axes_frac)
            # and one on the vertical arm (closest to y = y_axes_frac) when
            # the L-shaped contour has two well-separated arms
            j_h = idx[np.argmin(np.abs(xy[idx, 0] - x_axes_frac))]
            j_v = idx[np.argmin(np.abs(xy[idx, 1] - y_axes_frac))]
            h_ok = abs(xy[j_h, 0] - x_axes_frac) < 0.1
            v_ok = abs(xy[j_v, 1] - y_axes_frac) < 0.1
            if h_ok:
                manual.append(tuple(seg[j_h]))
            if v_ok and (not h_ok or np.hypot(*(xy[j_h] - xy[j_v])) > 0.25):
                manual.append(tuple(seg[j_v]))
    texts = ax.clabel(cs, fmt={lv: pow10_label(lv) for lv in cs.levels},
                      fontsize=FS_ANNOT, inline=True, inline_spacing=2,
                      manual=manual, colors=color)
    return texts


def load_parsec_tracks():
    """F146 flux-ratio tracks written by isochrone_phot_cancel.py."""
    path = FIGURES_ROOT / "current" / "physical_qM_qf_PARSECF146.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {float(k): (np.asarray(v["qM"]), np.asarray(v["qf"]))
            for k, v in data["tracks"].items()}


# ============================================================
# Figure 5: qM-qf D maps
# ============================================================

def plot_qmass_qflux(
    filename,
    output_dir,
    alt=False,
):
    print()
    print("=" * 80)
    print("FIGURE: qM-qf D map" + (" (alt)" if alt else ""))
    print("=" * 80)

    with np.load(filename, allow_pickle=False) as d:
        qM = np.asarray(d["qM_grid"], dtype=float)
        qf = np.asarray(d["qf_grid"], dtype=float)
        P_over_tE = np.asarray(d["P_over_tE_grid"], dtype=float)
        D = np.asarray(d["D"], dtype=float)
        success = np.asarray(d["SUCCESS"], dtype=bool)
        dataset_commit = str(d["code_commit"].item())

    if not np.all(success):
        print("WARNING: qM-qf dataset contains failed fits.")

    # qf=0 exists in the grid but cannot be shown on a logarithmic axis.
    positive_qf = qf > 0.0
    qf_plot = qf[positive_qf]
    D_plot = D[:, positive_qf, :]

    vmin, vmax = D_VMIN, D_VMAX
    contour_levels = np.array([1e-3, 1e-2, 1e-1], dtype=float)

    tracks = load_parsec_tracks() if alt else None
    if alt and tracks is None:
        raise FileNotFoundError("Run isochrone_phot_cancel.py first (PARSEC tracks).")

    apply_paper_style()
    n_period = len(P_over_tE)
    fig = plt.figure(figsize=(fig_width(1.0), 2.45 if not alt else 2.75))
    left, right_cb, cb_w, gap = 0.075, 0.885, 0.018, 0.02
    w = (right_cb - 0.025 - left - (n_period - 1) * gap) / n_period
    bottom = 0.15 if not alt else 0.135
    top = 0.89 if not alt else 0.79
    axes = [fig.add_axes([left + k * (w + gap), bottom, w, top - bottom])
            for k in range(n_period)]
    cax = fig.add_axes([right_cb, bottom, cb_w, top - bottom])

    mappable = None
    plotted = {}
    for i_P, ax in enumerate(axes):
        Z = D_plot[:, :, i_P].T
        mappable = heatmap(ax, qM, qf_plot, Z, cmap=D_CMAP,
                           norm=d_norm(vmin, vmax))
        configure_log_map_axis(ax, qM, qf_plot, show_ylabel=(i_P == 0),
                               label_diagonal=(i_P == 0),
                               hide_last_xlabel=(i_P < n_period - 1))

        finite_Z = Z[np.isfinite(Z) & (Z > 0.0)]
        levels_here = contour_levels[(contour_levels >= finite_Z.min())
                                     & (contour_levels <= finite_Z.max())]
        if len(levels_here) > 0:
            cs = ax.contour(qM, qf_plot, Z, levels=levels_here, colors="white",
                            linewidths=LW_THIN, zorder=5)
            label_contours(ax, cs, x_axes_frac=0.30)

        if alt:
            for M1, (tq, tf) in sorted(tracks.items()):
                st = M1_STYLE[M1]
                ax.plot(tq, tf, color=st["color"], lw=LW_MAIN, ls="-",
                        marker=st["marker"], markevery=0.2, ms=3.0,
                        mec="white", mew=0.5, zorder=9,
                        path_effects=halo(2.0, "white"))

        ax.set_title(rf"$P/t_E={P_over_tE[i_P]:g}$")
        panel_label(ax, "abc"[i_P], color="white", x=0.05, y=0.95)
        plotted[f"D_P{i_P}"] = Z

    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(D_LABEL)
    cbar.ax.tick_params(which="both", direction="in")

    if alt:
        handles = [plt.Line2D([], [], color=M1_STYLE[m]["color"],
                              marker=M1_STYLE[m]["marker"], ms=3.0, lw=LW_MAIN,
                              label=rf"$M_1={m:g}\,M_\odot$")
                   for m in sorted(tracks)]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   bbox_to_anchor=(0.48, 1.0), frameon=False,
                   title=r"PARSEC F146 main-sequence tracks", title_fontsize=FS_LEGEND)

    np.savez(VERIFY_ROOT / ("qmass_qflux_D_map" + ("_alt" if alt else "")
                            + "_plotted_values.npz"),
             qM=qM, qf=qf_plot, P_over_tE=P_over_tE, **plotted)

    metadata = {
        "figure": "qmass_qflux_D_map" + ("_alt" if alt else ""),
        "source": str(filename),
        "dataset_commit": dataset_commit,
        "qf_zero_displayed": False,
        "qf_zero_reason": "logarithmic qf axis",
        "photocenter_cancellation_line": "qf=qM",
        "contour_levels_D": contour_levels.tolist(),
        "D_vmin": vmin,
        "D_vmax": vmax,
    }
    if alt:
        metadata["parsec_tracks"] = "figures/current/physical_qM_qf_PARSECF146.json"

    return save_figure(fig, output_dir,
                       "qmass_qflux_D_map" + ("_alt" if alt else ""), metadata)
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


def _percent_formatter():
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _pos=None):
        s = f"{round(100.0 * v, 6):g}"
        return "$0$" if s in ("-0", "0") else f"${s}$"

    return FuncFormatter(fmt)


def plot_qmass_qflux_biases(
    filename,
    output_dir,
    target_P_over_tE=1.0,
    alt=False,
):
    print()
    print("=" * 80)
    print("FIGURE: qM-qf biases" + (" (alt)" if alt else ""))
    print("=" * 80)

    with np.load(filename, allow_pickle=False) as d:
        qM = np.asarray(d["qM_grid"], dtype=float)
        qf = np.asarray(d["qf_grid"], dtype=float)
        P_over_tE = np.asarray(d["P_over_tE_grid"], dtype=float)
        D = np.asarray(d["D"], dtype=float)
        DT0 = np.asarray(d["DT0"], dtype=float)
        DU0 = np.asarray(d["DU0"], dtype=float)
        DTE = np.asarray(d["DTE"], dtype=float)
        u0_true = float(d["u0_true"].item())
        tE_true = float(d["tE_true"].item())
        dataset_commit = str(d["code_commit"].item())

    i_P = int(np.argmin(np.abs(P_over_tE - target_P_over_tE)))
    selected_P = float(P_over_tE[i_P])

    positive_qf = qf > 0.0
    qf_plot = qf[positive_qf]

    D_map = D[:, positive_qf, i_P].T
    dt0_map = DT0[:, positive_qf, i_P].T / tE_true
    du0_map = DU0[:, positive_qf, i_P].T / u0_true
    dte_map = DTE[:, positive_qf, i_P].T / tE_true

    bias_maps = [dt0_map, du0_map, dte_map]
    bias_limits = [symmetric_limit(values) for values in bias_maps]
    bias_labels = [r"$\Delta t_0/t_E$ (\%)", r"$\Delta u_0/u_0$ (\%)",
                   r"$\Delta t_E/t_E$ (\%)"]

    apply_paper_style()
    pfmt = _percent_formatter()

    def bias_colorbar(fig, cax, pcm, label, limit, orientation):
        cb = fig.colorbar(pcm, cax=cax, orientation=orientation, extend="both")
        n_max = 4 if orientation == "vertical" else 2
        for step in (0.001, 0.002, 0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1):
            n = int(np.floor(limit / step + 1e-9))
            if n <= n_max:
                break
        cb.set_ticks(np.arange(-n, n + 1) * step)
        axis = cb.ax.yaxis if orientation == "vertical" else cb.ax.xaxis
        axis.set_major_formatter(pfmt)
        axis.set_minor_locator(plt.NullLocator())
        cb.set_label(label)
        cb.ax.tick_params(which="both", direction="in")
        return cb

    if not alt:
        # 2x2, same panels and content as the manuscript version
        fig = plt.figure(figsize=(fig_width(1.0), 5.0))
        pw, ph = 0.335, 0.405
        xs = [0.085, 0.565]
        ys = [0.555, 0.09]
        cbw, cbpad = 0.017, 0.012
        axes = [fig.add_axes([xs[c], ys[r], pw, ph])
                for r in range(2) for c in range(2)]
        caxes = [fig.add_axes([xs[c] + pw + cbpad, ys[r], cbw, ph])
                 for r in range(2) for c in range(2)]

        pcm = heatmap(axes[0], qM, qf_plot, D_map, cmap=D_CMAP,
                      norm=d_norm(D_VMIN, D_VMAX))
        configure_log_map_axis(axes[0], qM, qf_plot, show_xlabel=False)
        cb = fig.colorbar(pcm, cax=caxes[0])
        cb.set_label(D_LABEL)
        cb.ax.tick_params(which="both", direction="in")

        for k, (ax, cax, values, limit, label) in enumerate(
                zip(axes[1:], caxes[1:], bias_maps, bias_limits, bias_labels)):
            pcm = heatmap(ax, qM, qf_plot, values, cmap=BIAS_CMAP,
                          norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
            configure_log_map_axis(ax, qM, qf_plot,
                                   show_xlabel=(k >= 1),
                                   show_ylabel=(k == 1),
                                   diag_color="0.15")
            bias_colorbar(fig, cax, pcm, label, limit, "vertical")

        for i, ax in enumerate(axes):
            panel_label(ax, "abcd"[i], color="white" if i == 0 else "k")
    else:
        # 1x3 bias panels; panel (a) of the manuscript version duplicates Fig. 5b
        fig = plt.figure(figsize=(fig_width(1.0), 2.55))
        left, gap = 0.075, 0.045
        w = (0.985 - left - 2 * gap) / 3
        bottom, top = 0.155, 0.74
        axes = [fig.add_axes([left + k * (w + gap), bottom, w, top - bottom])
                for k in range(3)]
        caxes = [fig.add_axes([left + k * (w + gap), 0.80, w, 0.03])
                 for k in range(3)]
        for k, (ax, cax, values, limit, label) in enumerate(
                zip(axes, caxes, bias_maps, bias_limits, bias_labels)):
            pcm = heatmap(ax, qM, qf_plot, values, cmap=BIAS_CMAP,
                          norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
            configure_log_map_axis(ax, qM, qf_plot, show_ylabel=(k == 0),
                                   diag_color="0.15",
                                   hide_last_xlabel=(k < 2))
            cb = bias_colorbar(fig, cax, pcm, label, limit, "horizontal")
            cb.ax.xaxis.set_ticks_position("top")
            cb.ax.xaxis.set_label_position("top")
            panel_label(ax, "abc"[k])

    np.savez(VERIFY_ROOT / ("qmass_qflux_biases" + ("_alt" if alt else "")
                            + "_plotted_values.npz"),
             qM=qM, qf=qf_plot, D=D_map, dt0=dt0_map, du0=du0_map, dte=dte_map,
             bias_limits=np.array(bias_limits), P_over_tE=selected_P)

    metadata = {
        "figure": "qmass_qflux_biases" + ("_alt" if alt else ""),
        "source": str(filename),
        "dataset_commit": dataset_commit,
        "P_over_tE": selected_P,
        "u0_true": u0_true,
        "tE_true": tE_true,
        "normalizations": {"DT0": "DT0/tE_true", "DU0": "DU0/u0_true",
                           "DTE": "DTE/tE_true"},
        "bias_color_limits_99pct": bias_limits,
        "photocenter_cancellation_line": "qf=qM",
    }

    return save_figure(fig, output_dir,
                       "qmass_qflux_biases" + ("_alt" if alt else ""), metadata)



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
            generated[
                "qmass_qflux_alt"
            ] = plot_qmass_qflux(
                available[
                    "qmass_qflux"
                ],
                output_dir,
                alt=True,
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
            generated[
                "biases_alt"
            ] = plot_qmass_qflux_biases(
                available[
                    "qmass_qflux"
                ],
                output_dir,
                alt=True,
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
