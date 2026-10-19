# Intrinsic BSPL/xallarap vs PSPL+parallax grid

Noise-free intrinsic comparison used to study how much a PSPL+annual-parallax
model can reproduce a BSPL/xallarap magnification curve.

Configuration:

- tE = 30 d
- t0 = 2461849.0 JD
- q_mass = 0.5
- q_flux = 0
- true parallax = 0
- intrinsic window = t0 +/- 3.5 tE
- 10000 uniformly sampled epochs
- u0 grid = 133 nodes with u0 <= 1 retained from the 200-node log grid
- P grid = 60 logarithmic nodes from 10 to 1e5 d
- fitted parallax bounds: piEN, piEE in [-40, 40]

Stored quantities include:

- D0: best BSPL-vs-PSPL intrinsic mismatch
- Dpi: best BSPL-vs-PSPL+parallax intrinsic mismatch
- Dpi/D0
- fitted piEN, piEE and |piE|
- fitted PSPL shape parameters
- numerical diagnostics

Full production grid: 7980/7980 successful nodes, no parallax-bound hits.
