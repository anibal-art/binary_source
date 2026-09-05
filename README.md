# Binary Sources: BSPL–PSPL Degeneracy

This repository studies when a **binary-source point-lens (BSPL)** microlensing event can be confused with a standard **point-source point-lens (PSPL)** event.

The main goal is to quantify the geometric degeneracy between BSPL and PSPL models, first intrinsically and then after including the expected sampling and photometric precision of the **Nancy Grace Roman Space Telescope**.

This README is intended both for collaborators and as a practical **"future me" guide** for returning to the project after several months.

---


## Repository scope

This repository contains the **source code required to regenerate the
numerical analyses and figures used in the paper**.

It intentionally does **not** contain:

- numerical production outputs under `results/`;
- generated figures;
- the LaTeX manuscript and bibliography.

The manuscript is maintained separately in Overleaf. The definitive list of
paper figures expected from this repository is stored in
`tools/paper_figure_manifest.txt`.

## 1. Scientific idea

A binary source can produce a light curve that differs from a PSPL event because the luminous source(s) move around the source-system center of mass.

The central question is:

> When can the resulting BSPL perturbation be absorbed by changes in the best-fitting PSPL parameters?

The repository contains three related analyses:

1. **Intrinsic BSPL–PSPL degeneracy**
   - Compare noiseless BSPL light curves with their best PSPL fits.
   - Main mismatch metric:
     \[
     D \equiv
     \frac{\int |A_{\rm BSPL}(t)-A_{\rm PSPL}(t)|\,dt}
          {\int A_{\rm BSPL}(t)\,dt}
     \]
     or the equivalent implementation used by the production scripts.

2. **Two-luminous-source / photocenter analysis**
   - Explore dependence on mass ratio \(q_M\) and flux ratio \(q_f\).
   - The first-order photocenter contribution is proportional to
     \[
     C_{\rm ph}
     =
     \frac{q_M-q_f}{(1+q_M)(1+q_f)}.
     \]
   - The locus \(q_f=q_M\) cancels the first-order photocenter term, but it is **not an exact BSPL–PSPL degeneracy**.
   - In the small-separation limit, the remaining deviation scales as
     \[
     D_{\rm BSPL-PSPL}\propto \xi_{\rm rel}^2.
     \]
     For \(\xi_{\rm rel}\gtrsim1\), higher-order terms can still produce strongly distinguishable light curves.

3. **Roman detectability**
   - Compute the Asimov separation
     \[
     \Delta\chi^2
     =
     \chi^2_{\rm PSPL}
     -
     \chi^2_{\rm BSPL},
     \]
     where the BSPL truth is noiseless and the PSPL model is fitted using Roman sampling and photometric uncertainties.

---

## 2. Current fiducial setup

The main analysis now uses

```text
tE = 30 d
M1 = 2 Msun
M2 = 1 Msun
qM = 0.5
Mtot = 3 Msun
rE_hat = 5 AU
qf = 0 for the single-luminous-source limit
fit window = t0 +/- 3.5 tE
```

Important:

- Older runs with `tE = 150 d` exist and are useful only as long-timescale / validation comparisons.
- Do **not** accidentally use the old `tE=150 d` results as the primary manuscript results.

---

## 3. Repository layout

The most relevant directories are:

```text
binary_source/
├── analysis/
│   └── roman_bspl_pspl_asimov.py
│
├── plot_codes/
│   ├── plot_intrinsic_tE_paper.py
│   ├── plot_intrinsic_biases_paper.py
│   ├── plot_roman_intrinsic_comparison.py
│   └── plot_lightcurve_examples.py
│
├── functions_aux.py
├── degeneracy_fit.py
└── ...

results/
├── scan_many_tE_200x200/
│   └── scan_u0_tE30/
│
├── final_483d90a0fd07/
│   └── qmass_qflux_tE30/
│       └── summary_qM_qf.npz
│
├── final_6b888737a3c3/
│   └── qmass_fixed_mtot_tE30/
│
├── roman_asimov_tE30/                 # legacy W149-based Roman result
└── roman_asimov_f146_tE30/            # current Roman F146 result

figures/
├── current/
├── appendix/
└── figures_lightcurves/

├── manuscript.tex

```

---

## 4. Environment

The local environment used during development is:

```bash
conda activate rubin-sim
```

The code depends mainly on:

```text
numpy
scipy
pandas
matplotlib
astropy
pyLIMA
```

Before running anything after a long break, check:

```bash
cd ~/binary_source

git status
git branch
git log -5 --oneline

conda activate rubin-sim

python -c "import numpy, scipy, pandas, matplotlib, astropy, pyLIMA; print('imports OK')"
```

It is also useful to compile the key scripts:

```bash
python -m py_compile \
    binary_source/functions_aux.py \
    binary_source/analysis/roman_bspl_pspl_asimov.py \
    binary_source/plot_codes/plot_intrinsic_tE_paper.py \
    binary_source/plot_codes/plot_intrinsic_biases_paper.py \
    binary_source/plot_codes/plot_roman_intrinsic_comparison.py \
    binary_source/plot_codes/plot_lightcurve_examples.py
```

---

## 5. Main intrinsic grid

The primary intrinsic one-luminous-source grid is:

```text
results/scan_many_tE_200x200/scan_u0_tE30/
```

Properties:

```text
tE = 30 d
Nu0 = 200
NP = 60
P = 1e1 ... 1e5 d
N_TIME = 10000
time window = +/- 3.5 tE
```

The main paper figure is generated with:

```bash
python binary_source/plot_codes/plot_intrinsic_tE_paper.py
```

Relevant outputs include:

```text
figures/current/D_heatmap_tE30_complete_contour.pdf
figures/current/D_contours_many_tE_complete.pdf
```

The intrinsic PSPL-parameter biases are generated with:

```bash
python binary_source/plot_codes/plot_intrinsic_biases_paper.py
```

Relevant outputs:

```text
figures/current/paper_tE_bias_D_contours_tE30.pdf
figures/current/paper_u0_tE_correlated_bias_tE30.pdf
```

---

## 6. Two-luminous-source analysis

### Main qM-qf grid

Primary result:

```text
results/final_483d90a0fd07/qmass_qflux_tE30/summary_qM_qf.npz
```

Successful fits:

```text
40200 / 40200
```

Figures:

```text
figures/current/qmass_qflux_D_map.pdf
figures/current/qmass_qflux_biases.pdf
figures/current/photocenter_scaling.pdf
```

### Fixed-total-mass qM scan

Primary result:

```text
results/final_6b888737a3c3/qmass_fixed_mtot_tE30/
```

Successful fits:

```text
90000 / 90000
```

### Important notation

Use:

- \(\xi_E\): orbital amplitude of source 1 in the **single-luminous-source limit**.
- \(\xi_{\rm rel}\): relative binary-source separation in the **general two-luminous-source case**.

Do not use \(\xi_E\) in the Section 5 two-luminous-source expansion.

At \(q_f=q_M\), the first-order photocenter term cancels, but the system is not exactly equivalent to a PSPL event.

---

## 7. Roman analysis: current F146 implementation

The current Roman analysis is:

```text
binary_source/analysis/roman_bspl_pspl_asimov.py
```

### Current Roman prescription

The repository was updated from the legacy W149 prescription to **F146**.

Current settings:

```text
filter = F146
high-cadence sampling = 12.1 min
nominal high-cadence epochs = 8390
fit window = +/- 3.5 tE
anchor season = index 2
off-season sparse sampling = included
```

The photometric model is defined in:

```text
binary_source/functions_aux.py
```

as:

```python
sigma_F146_func(...)
```

The simplified survey-level precision model is normalized to:

```text
S/N = 100 at F146_AB = 21.2
```

with

```text
F146 zero point = 27.648 AB
exposure = 66 s
bright-source floor = 0.001 mag
```

Sanity-check values:

```text
F146=19.0  sigma ~ 0.002727 mag
F146=21.0  sigma ~ 0.009388 mag
F146=21.2  sigma ~ 0.010857 mag  -> S/N ~ 100
F146=23.0  sigma ~ 0.046877 mag
```

Important:

> This is a simplified survey-level noise prescription, not a complete Roman detector/crowding simulation.

The old W149 results are retained only as a legacy comparison.

---

## 8. Roman F146 smoke test

Before a large Roman production, always run:

```bash
python binary_source/analysis/roman_bspl_pspl_asimov.py \
    --mode smoke \
    --tE 30 \
    --source-mag 21
```

The expected sampling header should contain:

```text
N epochs = 8390
tE = 30 d
F146 baseline = 21
off seasons = True
```

The fit also prints a numerical chi-square validation:

```text
chi2 validation:
reported = ...
recomputed = ...
reldiff ~ 1e-16
```

If the reported and recomputed chi-square values do not agree, do not launch the full grid.

---

## 9. Roman F146 full grid

The current production command is:

```bash
cd ~/binary_source

mkdir -p results/roman_asimov_f146_tE30

time python binary_source/analysis/roman_bspl_pspl_asimov.py \
    --mode grid \
    --tE 30 \
    --magnitudes 19 21 23 \
    --anchor-season 2 \
    --fit-window-tE 3.5 \
    --intrinsic-grid-dir results/scan_many_tE_200x200/scan_u0_tE30 \
    --intrinsic-u0-max 1.0 \
    --output results/roman_asimov_f146_tE30/roman_intrinsic_grid_f146_tE30.npz \
    |& tee results/roman_asimov_f146_tE30/run_f146_tE30.log
```

This should produce:

```text
133 u0 nodes
x 60 period nodes
x 3 F146 magnitudes
= 23940 Roman fits
```

Monitor with:

```bash
tail -f results/roman_asimov_f146_tE30/run_f146_tE30.log
```

After completion:

```bash
ls -lh \
results/roman_asimov_f146_tE30/roman_intrinsic_grid_f146_tE30.npz

tail -50 \
results/roman_asimov_f146_tE30/run_f146_tE30.log
```

---

## 10. Roman figure

The plotting script is:

```text
binary_source/plot_codes/plot_roman_intrinsic_comparison.py
```

It currently expects the F146 production file:

```text
results/roman_asimov_f146_tE30/roman_intrinsic_grid_f146_tE30.npz
```

Run:

```bash
python binary_source/plot_codes/plot_roman_intrinsic_comparison.py
```

Expected output:

```text
figures/current/roman_intrinsic_comparison_f146.pdf
figures/current/roman_intrinsic_comparison_f146.png
```

Figure convention:

- background: \(\log_{10}\Delta\chi^2_{\rm Roman}\)
- intrinsic \(D\) contours: \(10^{-3}\), \(10^{-2}\)
- Roman detectability contour: \(\Delta\chi^2=100\)
- panels: F146 = 19, 21, 23

The old file

```text
results/roman_asimov_tE30/roman_intrinsic_grid_tE30.npz
```

is the legacy W149 calculation and should not be used as the primary Roman forecast.

---

## 11. Light-curve examples

The illustrative light-curve producer is:

```text
binary_source/plot_codes/plot_lightcurve_examples.py
```

Outputs:

```text
figures/figures_lightcurves/example_1.png
figures/figures_lightcurves/example_1.pdf
figures/figures_lightcurves/example_2_confused.png
figures/figures_lightcurves/example_2_confused.pdf
```

The goal is to show:

1. a clearly distinguishable BSPL morphology;
2. a BSPL event whose perturbation is largely absorbed by a PSPL fit.

Important lesson from the exploratory search:

- maximizing \(D\) or \(R_{\max}\) alone can select a single sharp residual near the main peak;
- for a pedagogical example, morphology matters: secondary features and visibly non-rectilinear source motion are more informative.

If a final good example has already been selected, freeze its parameters rather than re-running a large automatic morphology search every time the figure is regenerated.

---

## 12. PARSEC / physical-source interpretation

The physical two-luminous-source interpretation uses PARSEC/OBC photometry in Roman F146.

Current physical intuition:

\[
C_{\rm ph}
=
\frac{q_M-q_f}
{(1+q_M)(1+q_f)}.
\]

Relative to a dark companion,

\[
S_{\rm ph}
=
\frac{|q_M-q_f|}
{q_M(1+q_f)}.
\]

The line \(q_f=q_M\) is an exact **first-order photocenter cancellation locus**, not a generic main-sequence relation.

For realistic main-sequence binaries, PARSEC F146 mass-luminosity relations can approach this cancellation region for nearly equal masses, producing strong partial suppression.

Do not quantitatively interpolate physical PARSEC systems directly on a qM-qf grid if that grid was computed at a different fixed total source mass. For a quantitative physical prediction, run the actual physical masses.

---


## 14. Scientific conventions that are easy to forget

### Primary timescale

Use:

```text
tE = 30 d
```

for the main manuscript results.

`tE = 150 d` is legacy / auxiliary.

### Single vs two luminous sources

Use:

```text
qf = 0
```

for the one-luminous-source / dark-companion limit.

For two luminous sources, both source magnifications contribute:

\[
A_{\rm BSPL}
=
\frac{
F_1 A_1 + F_2 A_2
}{
F_1+F_2
}.
\]

### Photocenter cancellation

Do not write:

> \(q_f=q_M\) implies exact BSPL–PSPL degeneracy.

Correct statement:

> \(q_f=q_M\) cancels the first-order photocenter term. In the small-separation limit the leading deviation becomes second order in \(\xi_{\rm rel}\), but higher-order structure can remain large for \(\xi_{\rm rel}\gtrsim1\).

### Roman filter

Primary forecast:

```text
F146
```

Legacy comparison:

```text
W149
```

Do not relabel a W149 result as F146 without changing the photometric precision model.

---

## 15. Useful diagnostics

### Inspect NPZ contents

```bash
python - <<'PY'
import numpy as np

path = "PATH/TO/result.npz"

d = np.load(path, allow_pickle=True)

print(d.files)

for k in d.files:
    x = d[k]

    try:
        print(k, x.shape, x.dtype)
    except Exception:
        print(k, type(x))
PY
```

### Search for stale W149 references

```bash
grep -Rni \
'W149\|w149\|sigma_W149' \
binary_source \
--exclude='*.ipynb' \
--exclude-dir='__pycache__'
```

Note that legacy W149 functions/files may intentionally remain in the repository. The current F146 Roman analysis and current F146 plotting script should not accidentally depend on them.

### Check current F146 precision

```bash
python - <<'PY'
from binary_source.functions_aux import sigma_F146_func

for mag in [19, 21, 21.2, 23]:
    sigma = float(sigma_F146_func(mag))
    snr = 1.0857362047581294 / sigma

    print(
        mag,
        sigma,
        snr,
    )
PY
```

---

## 16. Results that should not be mixed

### Current / primary

```text
Intrinsic:
results/scan_many_tE_200x200/scan_u0_tE30/

Two luminous sources:
results/final_483d90a0fd07/qmass_qflux_tE30/
results/final_6b888737a3c3/qmass_fixed_mtot_tE30/

Roman:
results/roman_asimov_f146_tE30/
```

### Legacy / comparison

```text
tE = 150 d intrinsic experiments
Roman W149 results
old window/boundary/optimizer validation experiments
```

Never quote legacy numerical fractions in the manuscript as if they were the final F146/tE=30 results.

---

## 17. Returning to the project after several months

Recommended sequence:

```text
1. Read this README.
2. git status / git log.
3. Activate rubin-sim.
4. Compile the key Python scripts.
5. Check that the tE=30 intrinsic NPZ files still exist.
6. Check whether the F146 Roman production completed.
7. Run the F146 smoke test.
8. Regenerate the main figures.
```

In commands:

```bash
cd ~/binary_source

git status
git log -5 --oneline

conda activate rubin-sim

python -m py_compile \
    binary_source/functions_aux.py \
    binary_source/analysis/roman_bspl_pspl_asimov.py

ls -lh \
    results/scan_many_tE_200x200/scan_u0_tE30 \
    results/roman_asimov_f146_tE30

python binary_source/analysis/roman_bspl_pspl_asimov.py \
    --mode smoke \
    --tE 30 \
    --source-mag 21

python binary_source/plot_codes/plot_intrinsic_tE_paper.py
python binary_source/plot_codes/plot_intrinsic_biases_paper.py
python binary_source/plot_codes/plot_roman_intrinsic_comparison.py

```

---

## 18. Current open tasks

As of the F146 update, the main remaining tasks are:

- finish / verify the full F146 Roman grid;
- extract the final F146 Roman summary metrics;
- update the manuscript text and Roman figure from W149 to F146;
- finalize the illustrative distinguishable/confused light-curve examples;
- optionally test PARSEC F146 robustness against metallicity/age;
- final consistency check of notation:
  - \(\xi_E\) for the single-luminous limit;
  - \(\xi_{\rm rel}\) for the general two-luminous-source case.

---

## 19. General rule before changing the analysis

Before modifying a production script:

1. preserve the previous result;
2. create a separate output directory;
3. run a smoke test;
4. verify numerical identities/checks;
5. only then launch the expensive grid.

This repository contains several scientifically useful legacy runs. Avoid overwriting them.

---

## 20. One-sentence project summary

> We quantify when orbital motion in binary-source microlensing produces an intrinsically distinguishable signal and when that signal can instead be absorbed by a standard PSPL fit, with particular attention to photocenter cancellation and Roman F146 detectability.

<!-- PAPER_REPRO_START -->

## Reproducing every paper figure

GitHub contains the code required to regenerate the numerical analyses and all paper figures. Numerical products in `results/`, generated figures, legacy material, and the manuscript itself are not versioned.

The authoritative figure list is `tools/paper_figure_manifest.txt`.

All commands assume:

```bash
cd ~/binary_source
conda activate rubin-sim
```

The primary manuscript analysis uses `tE = 30 d`.

| Paper asset | Numerical production | Figure generation |
|---|---|---|
| `figures/figures_lightcurves/example_1.png` | `none` | `python binary_source/plot_codes/plot_lightcurve_examples.py` |
| `figures/figures_lightcurves/example_2_confused.png` | `none` | `python binary_source/plot_codes/plot_lightcurve_examples.py` |
| `figures/current/D_heatmap_tE30_complete_contour.pdf` | `python binary_source/scan_codes/run_final_scans.py` | `python binary_source/plot_codes/plot_intrinsic_tE_paper.py` |
| `figures/current/D_contours_many_tE_complete.pdf` | `python binary_source/scan_codes/run_final_scans.py` | `python binary_source/plot_codes/plot_intrinsic_tE_paper.py` |
| `figures/current/paper_tE_bias_D_contours_tE30.pdf` | `python binary_source/scan_codes/run_final_scans.py` | `python binary_source/plot_codes/plot_intrinsic_biases_paper.py` |
| `figures/current/paper_u0_tE_correlated_bias_tE30.pdf` | `python binary_source/scan_codes/run_final_scans.py` | `python binary_source/plot_codes/plot_intrinsic_biases_paper.py` |
| `figures/current/qmass_qflux_D_map.pdf` | `python binary_source/scan_codes/scan_qmass_qflux.py` | `python binary_source/plot_codes/run_paper_figures.py` |
| `figures/current/photocenter_scaling.pdf` | `python binary_source/scan_codes/scan_photocenter_limit.py && python binary_source/analysis/analyze_photocenter_scaling.py` | `python binary_source/plot_codes/plot_photocenter_scaling.py` |
| `figures/current/physical_qM_qf_PARSECF146.pdf` | `none` | `python isochrone_phot_cancel.py` |
| `figures/current/physical_photocenter_suppression_PARSECF146.pdf` | `none` | `python isochrone_phot_cancel.py` |
| `figures/current/qmass_qflux_biases.pdf` | `python binary_source/scan_codes/scan_qmass_qflux.py` | `python binary_source/plot_codes/run_paper_figures.py` |
| `figures/current/roman_intrinsic_comparison.pdf` | `Roman F146 grid; see command below` | `python binary_source/plot_codes/plot_roman_intrinsic_comparison.py` |
| `figures/appendix/optimizer_stratified_refit_audit.pdf` | `python binary_source/validation/validate_optimizer_and_grid_audit.py` | `python binary_source/plot_codes/plot_stratified_refit_audit.py` |
| `figures/appendix/window_sensitivity.pdf` | `python binary_source/validation/validate_window_and_blending.py` | `python binary_source/plot_codes/plot_final_validation.py` |
| `figures/appendix/roman_te_bound_validation.pdf` | `python binary_source/validation/validate_roman_te_bound.py` | `python binary_source/plot_codes/plot_roman_te_bound_validation.py` |
| `figures/appendix/geometry_robustness_appendix.pdf` | `python binary_source/validation/validate_geometry_robustness.py` | `python binary_source/plot_codes/plot_geometry_robustness_appendix.py` |

### Roman F146 production

Before the full Roman run, use the smoke test:

```bash
python binary_source/analysis/roman_bspl_pspl_asimov.py --mode smoke --tE 30 --source-mag 21
```

The expected sampling is 8390 F146 epochs.

Generate the final Roman grid with:

```bash
mkdir -p results/roman_asimov_f146_tE30

python binary_source/analysis/roman_bspl_pspl_asimov.py \
    --mode grid \
    --tE 30 \
    --magnitudes 19 21 23 \
    --anchor-season 2 \
    --fit-window-tE 3.5 \
    --intrinsic-grid-dir results/scan_many_tE_200x200/scan_u0_tE30 \
    --intrinsic-u0-max 1.0 \
    --output results/roman_asimov_f146_tE30/roman_intrinsic_grid_f146_tE30.npz
```

Then run:

```bash
python binary_source/plot_codes/plot_roman_intrinsic_comparison.py
```

### Reproducibility audit

```bash
python tools/audit_paper_reproducibility.py
```

Expected result:

```text
Total manuscript figure assets: 16
Potential Spanish documentation lines: 0
```

<!-- PAPER_REPRO_END -->
