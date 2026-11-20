#!/usr/bin/env python3
"""
Check of Figure 4(b): are the per-orientation PSPL offsets the first-order
prediction of the projected orbital velocity at t0?

To first order (Section 4), with v_par, v_perp the projected orbital velocity of
the luminous star at t0 along and across the lens-source motion, in units of
v_orb = 2 pi xi_E tE / P,

    Delta tE / (tE v_orb)      = -v_par  / v_orb,
    Delta t0 / (tE u0 v_orb)   = -v_perp / v_orb.

For isotropic orbits both ratios are uniform in [-1, 1]. The figure shows the
per-orientation medians over the first-order cells, ranked; the Delta t0
squares lie to the left of the uniform line. This script tells apart two
explanations:

  (1) sampling: the 24 orientations of the scan happen to give a skewed set of
      v_perp values. Then measured and predicted values agree orientation by
      orientation, and the predicted sample is skewed in the same way.
  (2) physics or convention: the measured Delta t0 does not follow -v_perp.

v_par and v_perp are computed from Equations 8-9 of the paper,
    s(Omega) = (cos Omega, sin i sin Omega),  s_proj = R(-alpha) s,
at Omega = phi (t = t_xi = t0). Because the pyLIMA convention could differ by a
sign or by the orientation of the axes, the measured medians are compared with
the four candidates (+-v_x, +-v_y) and the best one is reported. A clean match
(slope ~ 1, small scatter) for the expected candidate confirms both the paper
formalism and the first-order relations.

The cells and the per-orientation medians are those of the figure
(plot_xiE_ptE_maps.bias_cells), so the selection is identical.

Usage (repository root):
    python binary_source/analysis/check_bias_orientations.py \
        --summary results/final_52468b04987a/xiE_ptE_maps_u0_0p01_0p3_1_refine2/summary_xiE_ptE.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "binary_source" / "plot_codes"))
from plot_xiE_ptE_maps import bias_cells  # noqa: E402


def unit_velocity(geometry):
    """(v_x, v_y)/v_orb at Omega = phi from Equations 8-9; geometry rows (alpha, phi, i)."""
    alpha, phi, inc = geometry[:, 0], geometry[:, 1], geometry[:, 2]
    wx = -np.sin(phi)                       # d s_x / d Omega
    wy = np.sin(inc) * np.cos(phi)          # d s_y / d Omega
    ca, sa = np.cos(alpha), np.sin(alpha)   # R(-alpha) = [[ca, sa], [-sa, ca]]
    return ca * wx + sa * wy, -sa * wx + ca * wy


def per_orientation_median(r, ok):
    """Median of r over the valid cells of each orientation (unranked)."""
    n_geo = r.shape[2]
    med, lo, hi, n = (np.full(n_geo, np.nan) for _ in range(4))
    for g in range(n_geo):
        v = r[:, :, g, :][ok[:, :, g, :]]
        if v.size:
            lo[g], med[g], hi[g] = np.percentile(v, [16, 50, 84])
            n[g] = v.size
    return med, lo, hi, n


def best_candidate(measured, vx, vy):
    """Least-squares slope and rms residual of measured against the four candidates."""
    rows = []
    for name, pred in (("+v_x", vx), ("-v_x", -vx), ("+v_y", vy), ("-v_y", -vy)):
        ok = np.isfinite(measured)
        slope = np.sum(measured[ok] * pred[ok]) / np.sum(pred[ok] ** 2)
        rms = np.sqrt(np.mean((measured[ok] - pred[ok]) ** 2))
        rows.append((rms, name, slope))
    return sorted(rows)


def ks_uniform(x):
    x = x[np.isfinite(x)]
    return stats.kstest(x, "uniform", args=(-1.0, 2.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--figure", action="store_true",
                    help="write figures/_verify/check_bias_orientations.png")
    args = ap.parse_args()

    S = np.load(args.summary, allow_pickle=False)
    geo = np.asarray(S["geometries"], float)[1:]            # random orientations only
    B = bias_cells(S, S["xi_E"], S["P_over_tE"])
    assert B["r_tE"].shape[2] == len(geo), (B["r_tE"].shape, geo.shape)

    vx, vy = unit_velocity(geo)
    m_tE, lo_tE, hi_tE, n_tE = per_orientation_median(B["r_tE"], B["ok"])
    m_t0, lo_t0, hi_t0, n_t0 = per_orientation_median(B["r_t0"], B["ok"])

    print(f"summary: {args.summary}")
    print(f"orientations: {len(geo)}; first-order cells per orientation: "
          f"{int(np.nanmin(n_tE))}-{int(np.nanmax(n_tE))}")
    print("\nlargest 16-84 spread over cells of one orientation "
          f"(the error bars of Fig. 4b): tE {np.nanmax(hi_tE - lo_tE):.3f}, "
          f"t0 {np.nanmax(hi_t0 - lo_t0):.3f}")

    print("\n1. Which projected velocity component each offset follows")
    for label, m in (("Delta tE/(tE v_orb)   ", m_tE), ("Delta t0/(tE u0 v_orb)", m_t0)):
        rows = best_candidate(m, vx, vy)
        print(f"   {label}: " + "   ".join(f"{nm}: rms {r:.3f}, slope {s:+.2f}"
                                         for r, nm, s in rows))
    print("   expected (paper convention): Delta tE <-> -v_x (v_par), Delta t0 <-> -v_y (v_perp)")

    print("\n2. Per orientation (alpha, phi, i in rad)")
    print("   geo  alpha   phi     i     | -v_x    tE med | -v_y    t0 med")
    for g in range(len(geo)):
        a, p, i = geo[g]
        print(f"   {g + 1:3d}  {a:5.2f}  {p:5.2f}  {i:+5.2f}  | {-vx[g]:+.3f}  {m_tE[g]:+.3f} "
              f"| {-vy[g]:+.3f}  {m_t0[g]:+.3f}")

    print("\n3. Kolmogorov-Smirnov test against uniform in [-1, 1] (24 values each)")
    for label, x in (("measured Delta tE/(tE v_orb)   ", m_tE),
                     ("predicted -v_par/v_orb         ", -vx),
                     ("measured Delta t0/(tE u0 v_orb)", m_t0),
                     ("predicted -v_perp/v_orb        ", -vy)):
        res = ks_uniform(x)
        print(f"   {label}: D = {res.statistic:.3f}, p = {res.pvalue:.3f}, "
              f"mean = {np.nanmean(x):+.3f}")

    print("\n4. Delta t0 medians by u0 (same cells, one u0 at a time)")
    for iu, u0 in enumerate(S["u0"]):
        ok_u = np.zeros_like(B["ok"])
        ok_u[iu] = B["ok"][iu]
        m_u = per_orientation_median(B["r_t0"], ok_u)[0]
        d = m_u - (-vy)
        print(f"   u0 = {u0:<5g}: median of (measured + v_perp) over orientations = "
              f"{np.nanmedian(d):+.3f}, max |.| = {np.nanmax(np.abs(d)):.3f}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
        for ax, m, pred, lab in ((axes[0], m_tE, -vx, r"$\Delta t_E/(t_E v_{\rm orb})$"),
                                 (axes[1], m_t0, -vy, r"$\Delta t_0/(t_E u_0 v_{\rm orb})$")):
            ax.plot([-1, 1], [-1, 1], color="0.6", lw=0.8, ls="--")
            ax.plot(pred, m, "o", ms=4)
            ax.set_xlabel("first-order prediction")
            ax.set_ylabel(f"measured median {lab}")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
        out = REPO / "figures" / "_verify" / "check_bias_orientations.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
