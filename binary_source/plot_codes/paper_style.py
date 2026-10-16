#!/usr/bin/env python3
"""Shared matplotlib style for every manuscript figure.

Figures are designed at their final printed width (``latex/manuscript.tex``:
article class, letter paper, 3 cm side margins -> \\textwidth = 6.14 in), so
the font sizes below are the sizes on paper.  Do not save with
``bbox_inches="tight"``: it changes the figure size.

Fixed visual semantics used throughout the paper
------------------------------------------------
D_BSPL-PSPL        viridis, LogNorm(D_VMIN, D_VMAX)
parameter biases   RdBu_r, TwoSlopeNorm centred on zero
log10 Delta chi^2  cividis
q_f = 0            solid line
q_f = q_M          dashed line
F146 = 19/21/23    F146_STYLE
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm, TwoSlopeNorm


# ------------------------------------------------------------------
# Page geometry
# ------------------------------------------------------------------

TEXTWIDTH_IN = (21.59 - 6.0) / 2.54  # 6.14 in


def fig_width(fraction=1.0):
    """Printed width in inches of an \\includegraphics[width=fraction\\linewidth]."""
    return fraction * TEXTWIDTH_IN


# ------------------------------------------------------------------
# Typography (points at printed size)
# ------------------------------------------------------------------

FS_LABEL = 11.0
FS_TICK = 9.5
FS_LEGEND = 9.5
FS_PANEL = 11.0
FS_ANNOT = 9.5

LW_MAIN = 1.2
LW_THIN = 0.8

PAPER_RC = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",

    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "legend.title_fontsize": FS_LEGEND,
    "figure.titlesize": FS_LABEL,

    "axes.linewidth": 0.6,
    "axes.labelpad": 2.5,
    "axes.titlepad": 3.0,
    "axes.grid": False,
    "axes.facecolor": "white",
    "axes.formatter.use_mathtext": True,
    "axes.formatter.useoffset": False,
    "axes.formatter.limits": (-3, 4),
    "axes.prop_cycle": mpl.cycler(
        color=["#000000", "#0072B2", "#D55E00", "#009E73",
               "#CC79A7", "#E69F00"]
    ),

    "lines.linewidth": LW_MAIN,
    "lines.markersize": 3.5,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,

    "grid.color": "0.88",
    "grid.linewidth": 0.4,

    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4,
    "legend.labelspacing": 0.3,

    "figure.facecolor": "white",
    "figure.dpi": 150,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,

    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "image.interpolation": "none",
}


def apply_paper_style(usetex=True):
    rc = dict(PAPER_RC)
    if not usetex:
        rc.update({"text.usetex": False, "font.serif": ["cmr10", "DejaVu Serif"]})
    mpl.rcParams.update(rc)


def write_mplstyle(path):
    """Write the same settings as a .mplstyle file (for reference/reuse)."""
    lines = ["# Generated from binary_source/plot_codes/paper_style.py"]
    for key, value in PAPER_RC.items():
        if key == "axes.prop_cycle":
            colors = ", ".join(f"'{c[1:]}'" for c in value.by_key()["color"])
            value = f"cycler('color', [{colors}])"
        elif isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        elif value is None:
            value = "standard" if key == "savefig.bbox" else "None"
        elif isinstance(value, str) and value.startswith("#"):
            value = value[1:]
        lines.append(f"{key}: {value}")
    Path(path).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------
# Colour semantics
# ------------------------------------------------------------------

D_CMAP = "viridis"
D_VMIN = 1e-5
D_VMAX = 1e0
D_LABEL = r"$D_{\rm BSPL-PSPL}$"

BIAS_CMAP = "RdBu_r"
CHI2_CMAP = "cividis"

# Okabe-Ito qualitative colours, reused with the same meaning per figure
C_BLACK = "#000000"
C_BLUE = "#0072B2"
C_VERMILION = "#D55E00"
C_GREEN = "#009E73"
C_PURPLE = "#CC79A7"
C_ORANGE = "#E69F00"
C_SKY = "#56B4E9"

# Dark companion vs photocenter cancellation
LS_DARK = "-"          # q_f = 0
LS_CANCEL = "--"       # q_f = q_M

# Roman F146 magnitudes: one fixed colour + line style per magnitude.
F146_STYLE = {
    19: dict(color="#FFFFFF", ls="-"),
    21: dict(color=C_ORANGE, ls="--"),
    23: dict(color="#000000", ls=":"),
}

# PARSEC primary masses (Fig. 7 and the Fig. 5 alternative)
M1_STYLE = {
    0.5: dict(color=C_BLUE, marker="o"),
    0.7: dict(color=C_VERMILION, marker="s"),
    0.8: dict(color=C_PURPLE, marker="^"),
}


def d_norm(vmin=D_VMIN, vmax=D_VMAX):
    return LogNorm(vmin=vmin, vmax=vmax)


def bias_norm(values, percentile=100.0):
    """Symmetric TwoSlopeNorm around zero from finite values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lim = float(np.nanpercentile(np.abs(v), percentile)) if v.size else 1.0
    lim = lim if lim > 0 else 1.0
    return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def heatmap(ax, x, y, z, **kwargs):
    """pcolormesh without cell edges or checkerboard artifacts."""
    kw = dict(shading="auto", edgecolors="none", linewidth=0,
              antialiased=False, rasterized=True)
    kw.update(kwargs)
    return ax.pcolormesh(x, y, z, **kw)


def panel_label(ax, letter, color="k", x=0.04, y=0.95, **kwargs):
    """'(a)' in the upper-left corner of the axes."""
    return ax.text(x, y, f"({letter})", transform=ax.transAxes,
                   ha="left", va="top", fontsize=FS_PANEL, color=color,
                   zorder=20, **kwargs)


def sci_tex(value, sig=1):
    """Compact LaTeX scientific notation: 2.3e-3 -> $2.3\\times10^{-3}$."""
    value = float(value)
    if value == 0 or not np.isfinite(value):
        return f"${value:g}$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10 ** exponent
    m = f"{mantissa:.{max(sig - 1, 0)}f}"
    if float(m) == 10.0:
        exponent += 1
        m = f"{1.0:.{max(sig - 1, 0)}f}"
    if exponent == 0:
        return f"${m}$"
    if m in ("1", "1.0", "1.00"):
        return rf"$10^{{{exponent}}}$"
    return rf"${m}\times10^{{{exponent}}}$"


def pow10_label(value):
    """$10^{-2}$ for exact powers of ten."""
    return rf"$10^{{{int(round(np.log10(value)))}}}$"


def halo(width=2.0, color="white"):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=width, foreground=color)]


def frame_on_top(fig, zorder=100):
    """Draw spines and ticks above every data artist.

    Curves and contours with zorder > 2.5 are otherwise drawn over the axes
    frame, so a line running along an axis hides it.
    """
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_zorder(zorder)
        ax.xaxis.set_zorder(zorder)
        ax.yaxis.set_zorder(zorder)


def save_figure(fig, path_stem, png=True, pdf=True):
    """Save <stem>.pdf (vector, rasterized heatmaps at 300 dpi) and <stem>.png."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    frame_on_top(fig)
    out = []
    if pdf:
        fig.savefig(path_stem.with_suffix(".pdf"), dpi=300)
        out.append(path_stem.with_suffix(".pdf"))
    if png:
        fig.savefig(path_stem.with_suffix(".png"), dpi=300)
        out.append(path_stem.with_suffix(".png"))
    return out


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "figures" / "paper.mplstyle"
    write_mplstyle(target)
    print(target)


def place_contour_labels(ax, contour_sets, fmts, avoid_lines=(), box=(0.08, 0.92, 0.10, 0.90),
                         n_candidates=60, label_halo=None, colors=None):
    """Greedy, collision-aware inline labels for one or more contour sets.

    For every contour segment a label position is chosen among points inside
    `box` (axes fraction: x0, x1, y0, y1; or one box per contour set)
    that maximises the distance to
    (i) all other contour segments, (ii) extra polylines `avoid_lines`
    given in data coordinates and (iii) labels already placed.
    Returns the list of label Text objects.
    """
    to_ax = ax.transData + ax.transAxes.inverted()

    def axes_xy(seg):
        xy = to_ax.transform(seg)
        good = np.all(np.isfinite(xy), axis=1)
        return xy[good], seg[good]

    all_polys = []
    for cs in contour_sets:
        for level_segs in cs.allsegs:
            for seg in level_segs:
                if len(seg) > 1:
                    all_polys.append(axes_xy(np.asarray(seg))[0])
    for line in avoid_lines:
        all_polys.append(axes_xy(np.asarray(line))[0])

    placed = []
    texts = []
    boxes = box if isinstance(box[0], (tuple, list)) else [box] * len(contour_sets)
    for k, cs in enumerate(contour_sets):
        x0, x1, y0, y1 = boxes[k]
        manual = []
        for i, level_segs in enumerate(cs.allsegs):
            for seg in level_segs:
                if len(seg) < 6:
                    continue
                xy, data = axes_xy(np.asarray(seg))
                inside = (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
                idx = np.where(inside)[0]
                if idx.size == 0:
                    continue
                cand = idx[np.linspace(0, idx.size - 1, min(n_candidates, idx.size)).astype(int)]
                best, best_score = None, -np.inf
                for j in cand:
                    p = xy[j]
                    d_other = np.inf
                    for poly in all_polys:
                        if poly is xy or poly.size == 0 or (poly.shape == xy.shape and np.allclose(poly, xy)):
                            continue
                        d_other = min(d_other, np.min(np.hypot(poly[:, 0] - p[0], poly[:, 1] - p[1])))
                    d_lab = min([np.hypot(*(p - q)) for q in placed], default=np.inf)
                    score = min(d_other, 1.5 * d_lab)
                    if score > best_score:
                        best, best_score = j, score
                manual.append(tuple(data[best]))
                placed.append(xy[best])
        if manual:
            kw = dict(fmt=fmts[k], fontsize=FS_ANNOT, inline=True, inline_spacing=3, manual=manual)
            if colors is not None and colors[k] is not None:
                kw["colors"] = colors[k]
            t = ax.clabel(cs, **kw)
            if label_halo is not None and label_halo[k] is not None:
                for tt in t:
                    tt.set_path_effects(label_halo[k])
            texts.extend(t)
    return texts
