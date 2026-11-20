#!/usr/bin/env python3
"""
Section 6: examples of simulated Roman light curves of the binary of
Figure 9 (M1 = 2 Msun, dark M2 = 1 Msun, rhat_E = 5 AU, tE = 30 d) that the
test detects, that sit at the detection limit, and that it does not detect.

The light curves are noisy realizations of the H1 validation run
(results/roman_lrt/sec6_h1_validation_v1: runner JSON, the .npz written with
SAVE_ARRAYS=1, and the omega profile), so nothing is simulated or fitted here.
For each chosen cell the realization whose T is closest to the median T of
the cell is shown, so the example is typical rather than selected.

One column per example, three panels per column, as Figure 1:
  top     source-plane trajectories of the luminous source and the dark
          companion (truth, rebuilt with the runner's roman_rubin sim_event)
          and of the best-fitting PSPL model; lens at the origin
  middle  Roman F146 data in bins, noiseless truth, best PSPL and best BSPL
          (the BSPL of the maximum of the omega profile, i.e. of T), with T,
          T_3sigma, lambda, the 3 sigma efficiency and Delta tE/tE
  bottom  data - PSPL in bins and BSPL - PSPL

Inputs
  --cells    validation cells (roman_sec6_power.py --validate output dir):
             cells.json, cells_result.csv (thresholds used) and cells_k_h1.json
  --run      runner output dir with json/<event>.json, json/<event>.npz and
             omega_profile/<event>_omega_profile.json
  --examples cell ids, one per column (default: detected, limit, undetected)

Usage (repository root, ROMAN_RUBIN_REPO set, PYTHONPATH=$PWD):
    python binary_source/plot_codes/plot_sec6_lightcurve_examples.py \
        --cells results/roman_sec6_final/validation/cells.json \
        --run results/roman_lrt/sec6_h1_validation_v1 \
        --out results/roman_sec6_final/figures/sec6_lightcurve_examples [--no-usetex]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import (apply_paper_style, fig_width, save_figure,  # noqa: E402
                         C_BLUE, C_VERMILION, FS_ANNOT, FS_PANEL, LW_MAIN, LW_THIN)
from plot_lrt_event_fits import binned  # noqa: E402
from plot_lightcurve_examples import add_direction_arrows  # noqa: E402
from binary_source.analysis import roman_bspl_lrt_catalog_runner as runner  # noqa: E402

# detected (efficiency ~1), detection limit (~0.5), undetected (~0.05) at 3 sigma
DEFAULT_EXAMPLES = ("sec6_validation:c04", "sec6_validation:c03", "sec6_validation:c00")
STATUS = ("detected", "detection limit", "not detected")


def load_cells(cells_path):
    """Cells with the thresholds and k_local used by --validate, and k_H1."""
    spec = json.loads(Path(cells_path).read_text())
    cells = {c["config_id"]: c for c in spec["cells"]}
    res = Path(cells_path).with_name(Path(cells_path).stem + "_result.csv")
    for r in csv.DictReader(open(res)):
        for k in ("c_alpha05", "c_3sigma", "k_local"):
            cells[r["config_id"]][k] = float(r[k])
    kh1 = json.loads(Path(cells_path).with_name(Path(cells_path).stem + "_k_h1.json").read_text())
    return spec, cells, float(kh1["k_h1"])


def typical_event(run, cid):
    """(event json, arrays, profile) of the realization with T closest to the cell median."""
    prof = {}
    for f in sorted((Path(run) / "omega_profile").glob("*_omega_profile.json")):
        d = json.loads(f.read_text())
        if d.get("config_id") == cid:
            prof[f] = d
    if not prof:
        raise SystemExit(f"no omega profiles for {cid} in {run}/omega_profile")
    T = np.array([d["T_profile"] for d in prof.values()])
    f = list(prof)[int(np.argmin(np.abs(T - np.median(T))))]
    name = f.name.replace("_omega_profile.json", "")
    js = Path(run) / "json" / f"{name}.json"
    return json.loads(js.read_text()), np.load(js.with_suffix(".npz")), prof[f], float(np.median(T)), len(T)


def best_bspl_vector(profile):
    """BSPL parameters at the maximum of the omega profile (T_profile)."""
    ref = profile.get("refined") or {}
    nodes = [p for p in profile["profile"] if p.get("delta_chi2") is not None]
    node = max(nodes, key=lambda p: p["delta_chi2"])
    if ref.get("best_model") is not None and ref["delta_chi2"] >= node["delta_chi2"]:
        return ref["best_model"]
    return node["best_model"]


def truth_trajectories(d):
    """Source-plane trajectories of the true BSPL at the Roman epochs (runner's sim_event)."""
    event = runner.load_event_from_catalog(catalog_path=d["catalog_path"],
                                           catalog_event_index=d["catalog_event_index"])
    tp, tds = runner.truth_switches(d["truth_kind"])
    model, pyp, _ = runner.frr.sim_event(
        i=int(d["simulation_seed"]), data=dict(event), path_ephemerides=str(runner.EPHEMERIDES),
        model="PSPL", time_window=None, use_roman=True, use_rubin=False, truth_parallax=tp,
        truth_double_source=tds, add_photometric_noise=False, apply_detection_criteria=False,
        apply_photometric_filter=True)
    tel = model.event.telescopes[0]
    t = np.asarray(getattr(tel.lightcurve["time"], "value", tel.lightcurve["time"]), float)
    s1x, s1y, s2x, s2y = (np.asarray(getattr(v, "value", v), float)
                          for v in model.sources_trajectory(tel, pyp, data_type="photometry")[:4])
    return t, s1x, s1y, s2x, s2y


def pspl_trajectory(model, vector):
    pyp = model.compute_pyLIMA_parameters(np.asarray(vector, float))
    tr = model.sources_trajectory(model.event.telescopes[0], pyp, data_type="photometry")
    return (np.asarray(getattr(v, "value", v), float) for v in tr[:2])


def xi_E(P_days, spec):
    """xi_E of the luminous source for the binary of Figure 9 (Kepler, AU, yr, Msun)."""
    a_rel = (spec["Mtot_Msun"] * (P_days / 365.25) ** 2) ** (1.0 / 3.0)
    return spec["q_mass"] / (1.0 + spec["q_mass"]) * a_rel / spec["rEhat_AU"]


def column(fig, gs_col, d, a, profile, cell, spec, kh1, status, letter, args):
    lc = np.column_stack([a["time"], a["mag"], a["err_mag"]])
    t, m, e = lc.T
    w = 1.0 / e ** 2
    ra, dec, ref = d["event_ra_deg"], d["event_dec_deg"], d["fit_reference"]
    pspl = d["fits"]["PSPL"]["best"]
    names = pspl["fit_parameter_names"]
    t0, tE = spec["t0"], spec["tE"]
    tE_fit = abs(pspl["best_model"][names.index("tE")])

    m_pspl_model = runner.build_fit_model(lc, ra, dec, ref, False)
    m_bspl_model = runner.build_fit_model(lc, ra, dec, ref, True)
    v_bspl = best_bspl_vector(profile)
    m_pspl = runner.model_magnitude(m_pspl_model, pspl["best_model"])
    fx, fy = pspl_trajectory(m_pspl_model, pspl["best_model"])
    # fitted models on a dense time grid, so that gaps in the data (saturated
    # points removed at the peak, season edges) are not drawn as straight segments
    td = np.linspace(t0 - args.zoom_tE * tE, t0 + args.zoom_tE * tE, 4001)
    lcd = np.column_stack([td, np.full_like(td, np.median(m)), np.full_like(td, np.median(e))])
    md_pspl = runner.model_magnitude(runner.build_fit_model(lcd, ra, dec, ref, False),
                                     pspl["best_model"])
    md_bspl = runner.model_magnitude(runner.build_fit_model(lcd, ra, dec, ref, True), v_bspl)
    taud = (td - t0) / tE
    ts = np.sort(t)                                # dense times with Roman data nearby
    i = np.clip(np.searchsorted(ts, td), 1, len(ts) - 1)
    cadence = np.median(np.diff(ts))
    tz = ts[np.abs((ts - t0) / tE) < args.zoom_tE]         # intervals without data (saturation)
    k = np.flatnonzero(np.diff(tz) > 3 * cadence)
    gaps = [((tz[j] - t0) / tE, (tz[j + 1] - t0) / tE) for j in k]

    tau = (t - t0) / tE
    z = np.abs(tau) < args.zoom_tE
    o = np.argsort(t)
    oz = o[z[o]]

    axT = fig.add_subplot(gs_col[0])
    axA = fig.add_subplot(gs_col[1])
    axR = fig.add_subplot(gs_col[2], sharex=axA)

    # trajectories (truth: luminous source 1 and dark companion 2; best PSPL)
    tt, s1x, s1y, s2x, s2y = truth_trajectories(d)
    ot = np.argsort(tt)
    loc = np.abs((tt[ot] - t0) / tE) < args.traj_tE
    lf = np.abs(tau[o]) < args.traj_tE
    axT.axhline(0.0, color="0.85", lw=0.5, zorder=0)
    axT.axvline(0.0, color="0.85", lw=0.5, zorder=0)
    axT.plot(s2x[ot][loc], s2y[ot][loc], color="0.55", lw=LW_THIN, zorder=2)
    axT.plot(s1x[ot][loc], s1y[ot][loc], color="k", lw=LW_MAIN, zorder=3)
    axT.plot(fx[o][lf], fy[o][lf], color=C_VERMILION, lw=LW_MAIN, ls="--", zorder=4)
    axT.plot([0.0], [0.0], marker="+", ms=6, mew=1.0, color="k", ls="none", zorder=5)
    xs = np.concatenate([s1x[ot][loc], fx[o][lf], [0.0]])
    ys = np.concatenate([s1y[ot][loc], s2y[ot][loc], fy[o][lf], [0.0]])
    hx = 1.05 * np.nanmax(np.abs(xs))
    yc, hy = 0.5 * (np.nanmax(ys) + np.nanmin(ys)), 0.6 * (np.nanmax(ys) - np.nanmin(ys))
    hy = max(hy, 0.35 * hx)
    axT.set_xlim(-hx, hx)
    axT.set_ylim(yc - hy, yc + hy)
    add_direction_arrows(axT, s1x[ot][loc], s1y[ot][loc], "k", n_arrows=1,
                         scale=0.12 * hx, lw=LW_THIN)
    axT.set_xlabel(r"$\tau$", labelpad=1.0)
    axT.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    axT.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axT.xaxis.set_minor_locator(AutoMinorLocator())
    axT.yaxis.set_minor_locator(AutoMinorLocator())
    xiE = xi_E(cell["P_days"], spec)
    axT.set_title(rf"({letter}) {status}" "\n"
                  rf"$F146={cell['F146']:g}$, $g={cell['blend_ratio']:g}$, "
                  rf"$u_0={cell['u0']:.2g}$" "\n"
                  rf"$P/t_E={cell['P_days'] / tE:.3g}$, $\xi_E={xiE:.2g}$",
                  fontsize=FS_ANNOT, loc="left", linespacing=1.25)

    # light curve
    tb, mb, eb = binned(t[z], m[z], w[z], args.bin_days)
    axA.errorbar((tb - t0) / tE, mb, eb, fmt=".", ms=2.5, color="0.45", lw=0.5, zorder=1)
    if "truth_mag" in a:                           # stored at the data epochs: break at gaps
        otr = np.argsort(a["truth_time"])
        xt = (np.asarray(a["truth_time"], float)[otr] - t0) / tE
        yt = np.asarray(a["truth_mag"], float)[otr]
        keep = np.abs(xt) < args.zoom_tE
        xt, yt = xt[keep], yt[keep]
        cut = np.flatnonzero(np.diff(xt) > 5 * np.median(np.diff(xt))) + 1
        axA.plot(np.insert(xt, cut, np.nan), np.insert(yt, cut, np.nan), color="k",
                 lw=LW_MAIN, zorder=3)
    axA.plot(taud, md_pspl, color=C_VERMILION, lw=LW_MAIN, ls="--", zorder=4)
    axA.plot(taud, md_bspl, color=C_BLUE, lw=LW_THIN, ls=":", zorder=5)
    axA.invert_yaxis()
    faint, bright = axA.get_ylim()                 # headroom above the peak for the labels
    axA.set_ylim(faint, bright - 0.55 * (faint - bright))
    axA.tick_params(labelbottom=False)
    T, c3s, lam = float(profile["T_profile"]), cell["c_3sigma"], cell["lambda"]
    eff = float(stats.ncx2.sf(c3s, kh1, lam))
    axA.text(0.97, 0.96,
             rf"$T={T:.1f}$, $T_{{3\sigma}}={c3s:.1f}$" "\n"
             rf"$\lambda={lam:.1f}$, $\varepsilon_{{3\sigma}}={eff:.2f}$" "\n"
             rf"$\Delta t_E/t_E={100 * (tE_fit - tE) / tE:+.1f}\%$",
             transform=axA.transAxes, ha="right", va="top", fontsize=FS_ANNOT,
             linespacing=1.3)

    # residuals relative to the best PSPL, in mmag
    r = m - m_pspl
    tr_, rb, rbe = binned(t[z], r[z], w[z], args.resid_bin_days)
    axR.axhline(0.0, color="0.6", lw=LW_THIN, ls=":")
    axR.errorbar((tr_ - t0) / tE, 1e3 * rb, 1e3 * rbe, fmt=".", ms=2.5, color="0.45", lw=0.5)
    axR.plot(taud, 1e3 * (md_bspl - md_pspl), color=C_BLUE, lw=LW_MAIN)
    for g0, g1 in gaps:
        for ax in (axA, axR):
            ax.axvspan(g0, g1, color="0.9", lw=0, zorder=0)
    lim = 1.2 * np.nanmax(np.abs(1e3 * rb) + 1e3 * rbe)
    axR.set_ylim(lim, -lim)                         # magnitudes: brighter up
    axR.set_xlabel(r"$(t-t_0)/t_E$")
    axR.set_xlim(-args.zoom_tE, args.zoom_tE)
    axR.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
    return axT, axA, axR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--examples", nargs="+", default=list(DEFAULT_EXAMPLES))
    ap.add_argument("--status", nargs="+", default=list(STATUS))
    ap.add_argument("--zoom-tE", type=float, default=1.3)
    ap.add_argument("--traj-tE", type=float, default=1.0)
    ap.add_argument("--bin-days", type=float, default=0.5)
    ap.add_argument("--resid-bin-days", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()
    if len(args.status) != len(args.examples):
        raise SystemExit("--status needs one label per example")

    apply_paper_style(usetex=not args.no_usetex)
    spec, cells, kh1 = load_cells(args.cells)
    n = len(args.examples)
    fig = plt.figure(figsize=(fig_width(1.0), 6.8))
    gs = fig.add_gridspec(4, n, height_ratios=[0.9, 0.2, 1.5, 0.8], hspace=0.12, wspace=0.32,
                          left=0.08, right=0.96, bottom=0.07, top=0.86)
    axes = []
    for j, (cid, status) in enumerate(zip(args.examples, args.status)):
        d, a, profile, Tmed, nreal = typical_event(args.run, cid)
        c = cells[cid]
        print(f"{cid}: g {c['blend_ratio']:g} F146 {c['F146']:g} u0 {c['u0']:.4g} "
              f"P {c['P_days']:.1f} d lambda {c['lambda']:.2f} T_3s {c['c_3sigma']:.2f} | "
              f"event seed {d['simulation_seed']} T {profile['T_profile']:.2f} "
              f"(median of {nreal}: {Tmed:.2f})")
        sub = [gs[0, j], gs[2, j], gs[3, j]]        # row 1: room for the x label of the top row
        axes.append(column(fig, sub, d, a, profile, c, spec, kh1, status, "abc"[j], args))
    axes[0][0].set_ylabel(r"$\beta$")
    axes[0][1].set_ylabel("F146 [mag]")
    axes[0][2].set_ylabel(r"$\Delta m$ [mmag]")
    fig.align_ylabels([ax for ax in axes[0]])
    handles = [Line2D([], [], color="0.45", marker=".", ls="none"),
               Line2D([], [], color="k", lw=LW_MAIN),
               Line2D([], [], color=C_VERMILION, lw=LW_MAIN, ls="--"),
               Line2D([], [], color=C_BLUE, lw=LW_THIN, ls=":"),
               Line2D([], [], color="0.55", lw=LW_THIN),
               Line2D([], [], color="k", marker="+", ms=6, mew=1.0, ls="none")]
    fig.legend(handles, ["Roman data (binned)", "BSPL truth", "best-fitting PSPL",
                         "best-fitting BSPL", "dark companion", "lens"],
               loc="upper center", ncol=3, frameon=False, fontsize=FS_ANNOT,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.6)
    for f in save_figure(fig, args.out):
        print("saved", f)


if __name__ == "__main__":
    main()
