#!/usr/bin/env python3

import matplotlib as mpl


def apply_paper_style():

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",

            "font.size": 14,

            "axes.labelsize": 16,
            "axes.titlesize": 16,

            "xtick.labelsize": 13,
            "ytick.labelsize": 13,

            "legend.fontsize": 12,

            "axes.linewidth": 1.2,

            "xtick.direction": "in",
            "ytick.direction": "in",

            "xtick.top": True,
            "ytick.right": True,

            "xtick.major.size": 6,
            "ytick.major.size": 6,

            "xtick.minor.size": 3,
            "ytick.minor.size": 3,

            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,

            "savefig.bbox": "tight",
            "savefig.dpi": 600,
        }
    )
