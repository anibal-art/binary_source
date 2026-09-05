#!/usr/bin/env python3

"""
Audit of the remaining referee questions.

NO fits are performed.

Checks
------
1. Locate the largest adjacent jump in log10(D) in:
   a) the one-luminous intrinsic (u0, P) grid;
   b) the two-luminous (qM, qf) grid.

2. Inspect the optimizer-validation code to determine:
   - which bounds were used by the TRF same-objective cross-check;
   - how the 60-point stratified re-fit audit was performed.

3. Inspect the two-luminous production code to recover the mass convention
   used when varying qM.

4. Locate finite-source code/results already present in the repository.

Outputs
-------
Printed report plus:
    results/validation_optimizer_grid_audit/
        referee_remaining_audit.txt
"""

from __future__ import annotations

import re
import json
from pathlib import Path

import numpy as np


ROOT = Path.cwd()

OUTDIR = ROOT / "results" / "validation_optimizer_grid_audit"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTDIR / "referee_remaining_audit.txt"


# ======================================================================
# Utilities
# ======================================================================

REPORT = []


def emit(*args):
    s = " ".join(str(x) for x in args)
    print(s)
    REPORT.append(s)


def section(title):
    emit()
    emit("=" * 88)
    emit(title)
    emit("=" * 88)


def pick_key(data, candidates):
    keys = list(data.keys())

    # exact
    for c in candidates:
        if c in keys:
            return c

    # case-insensitive
    low = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]

    return None


def source_context(path, patterns, radius=4):
    """
    Return source-code contexts around lines matching any regex pattern.
    """
    path = Path(path)

    if not path.exists():
        return []

    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return []

    regexes = [re.compile(p, re.I) for p in patterns]
    hits = []

    matched_lines = set()

    for i, line in enumerate(lines):
        if any(r.search(line) for r in regexes):
            matched_lines.add(i)

    # merge overlapping contexts
    intervals = []
    for i in sorted(matched_lines):
        a = max(0, i - radius)
        b = min(len(lines), i + radius + 1)

        if intervals and a <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], b))
        else:
            intervals.append((a, b))

    for a, b in intervals:
        block = []
        for j in range(a, b):
            marker = ">>" if j in matched_lines else "  "
            block.append(f"{marker} {j+1:5d}: {lines[j]}")
        hits.append("\n".join(block))

    return hits


def search_repo(patterns, include_dirs=("binary_source",), max_files=100):
    """
    Search Python code recursively and return files containing relevant terms.
    """
    results = []

    for dirname in include_dirs:
        base = ROOT / dirname
        if not base.exists():
            continue

        for path in base.rglob("*.py"):
            contexts = source_context(path, patterns, radius=3)
            if contexts:
                results.append((path, contexts))

    return results[:max_files]


def finite_logD(D):
    D = np.asarray(D, dtype=float)

    out = np.full_like(D, np.nan, dtype=float)

    m = np.isfinite(D) & (D > 0)
    out[m] = np.log10(D[m])

    return out


# ======================================================================
# 1. One-luminous intrinsic grid: locate 0.114 dex jump
# ======================================================================

section("1. ONE-LUMINOUS GRID: LOCATION OF LARGEST ADJACENT log10(D) JUMP")

pattern = (
    ROOT
    / "results"
    / "scan_many_tE_200x200"
    / "scan_u0_tE150"
)

files = sorted(pattern.glob("scan_kepler_u0_*.npz"))

emit("Grid directory:", pattern)
emit("N files:", len(files))

one_luminous_rows = []

for fn in files:
    try:
        z = np.load(fn, allow_pickle=True)
    except Exception as exc:
        emit("Could not load:", fn, exc)
        continue

    D_key = pick_key(
        z,
        [
            "D",
            "RMS",
            "D_BSPL_PSPL",
            "DISTANCE",
        ],
    )

    P_key = pick_key(
        z,
        [
            "P_grid",
            "P",
            "period_grid",
            "periods",
        ],
    )

    u0_key = pick_key(
        z,
        [
            "u0",
            "u0_true",
            "u0_grid",
        ],
    )

    if D_key is None:
        emit("No D key in", fn.name, "keys =", list(z.keys()))
        continue

    D = np.asarray(z[D_key]).squeeze()

    if D.ndim != 1:
        emit(
            "Unexpected D shape in",
            fn.name,
            D.shape,
            "keys =",
            list(z.keys()),
        )
        continue

    if P_key is not None:
        P = np.asarray(z[P_key]).squeeze()
        if P.ndim == 0:
            P = np.full(len(D), float(P))
    else:
        P = np.arange(len(D), dtype=float)

    u0 = np.nan

    if u0_key is not None:
        arr = np.asarray(z[u0_key]).squeeze()

        if arr.ndim == 0:
            u0 = float(arr)

        elif arr.size == 1:
            u0 = float(arr.ravel()[0])

        elif arr.size == len(D):
            # if every value is equal, this is effectively scalar
            if np.allclose(arr, arr.flat[0]):
                u0 = float(arr.flat[0])

    one_luminous_rows.append(
        {
            "file": fn,
            "D": D,
            "P": P,
            "u0": u0,
            "D_key": D_key,
            "P_key": P_key,
            "u0_key": u0_key,
        }
    )


if one_luminous_rows:

    # If u0 was recoverable, sort physically.
    if all(np.isfinite(r["u0"]) for r in one_luminous_rows):
        one_luminous_rows.sort(key=lambda r: r["u0"])

    lengths = {len(r["D"]) for r in one_luminous_rows}

    if len(lengths) == 1:
        D2 = np.vstack([r["D"] for r in one_luminous_rows])
        L = finite_logD(D2)

        best = None

        # Adjacent in period direction
        jump_P = np.abs(np.diff(L, axis=1))

        if np.any(np.isfinite(jump_P)):
            idx = np.unravel_index(np.nanargmax(jump_P), jump_P.shape)
            i, j = idx
            val = float(jump_P[i, j])

            best = {
                "jump": val,
                "axis": "P",
                "i1": (i, j),
                "i2": (i, j + 1),
            }

        # Adjacent in u0 direction
        jump_u = np.abs(np.diff(L, axis=0))

        if np.any(np.isfinite(jump_u)):
            idx = np.unravel_index(np.nanargmax(jump_u), jump_u.shape)
            i, j = idx
            val = float(jump_u[i, j])

            if best is None or val > best["jump"]:
                best = {
                    "jump": val,
                    "axis": "u0",
                    "i1": (i, j),
                    "i2": (i + 1, j),
                }

        if best is not None:
            emit()
            emit("Largest adjacent jump:")
            emit("  Delta log10(D) =", best["jump"], "dex")
            emit("  direction      =", best["axis"])

            i1, j1 = best["i1"]
            i2, j2 = best["i2"]

            r1 = one_luminous_rows[i1]
            r2 = one_luminous_rows[i2]

            emit()
            emit("Cell 1:")
            emit("  file =", r1["file"].name)
            emit("  index =", (i1, j1))
            emit("  u0 =", r1["u0"])
            emit("  P =", float(np.asarray(r1["P"]).ravel()[j1]))
            emit("  P/tE =", float(np.asarray(r1["P"]).ravel()[j1]) / 150.0)
            emit("  D =", D2[i1, j1])
            emit("  log10(D) =", L[i1, j1])

            emit()
            emit("Cell 2:")
            emit("  file =", r2["file"].name)
            emit("  index =", (i2, j2))
            emit("  u0 =", r2["u0"])
            emit("  P =", float(np.asarray(r2["P"]).ravel()[j2]))
            emit("  P/tE =", float(np.asarray(r2["P"]).ravel()[j2]) / 150.0)
            emit("  D =", D2[i2, j2])
            emit("  log10(D) =", L[i2, j2])

            emit()
            emit(
                "NOTE: this 0.114-dex audit belongs to the ONE-LUMINOUS "
                "(u0, P) production grid."
            )

    else:
        emit("Cannot build rectangular grid; row lengths =", lengths)

else:
    emit("No one-luminous files found.")


# ======================================================================
# 2. Two-luminous qM-qf grid: locate its own largest jumps
# ======================================================================

section("2. TWO-LUMINOUS GRID: ADJACENT-JUMP LOCATION")

two_path = (
    ROOT
    / "results"
    / "final_a1b8a9b31002"
    / "qmass_qflux_tE150"
    / "summary_qM_qf.npz"
)

emit("File:", two_path)

if two_path.exists():

    z = np.load(two_path, allow_pickle=True)

    emit("Keys:")
    for k in z.keys():
        emit("  ", k, np.asarray(z[k]).shape)

    D_key = pick_key(z, ["D", "RMS", "DISTANCE"])
    qM_key = pick_key(
        z,
        ["qM_grid", "qM", "q_mass", "qmass_grid", "qmass"],
    )
    qf_key = pick_key(
        z,
        ["qf_grid", "qf", "q_flux", "qflux_grid", "qflux"],
    )
    P_key = pick_key(
        z,
        ["P_over_tE", "P_tE_grid", "P_grid", "periods", "P"],
    )

    emit()
    emit("Detected keys:")
    emit("  D  =", D_key)
    emit("  qM =", qM_key)
    emit("  qf =", qf_key)
    emit("  P  =", P_key)

    if D_key is not None:

        D = np.asarray(z[D_key], dtype=float)
        L = finite_logD(D)

        emit("D shape =", D.shape)

        for axis in range(D.ndim):

            if D.shape[axis] <= 1:
                continue

            jumps = np.abs(np.diff(L, axis=axis))

            if not np.any(np.isfinite(jumps)):
                continue

            idx = list(np.unravel_index(np.nanargmax(jumps), jumps.shape))

            val = float(jumps[tuple(idx)])

            idx2 = idx.copy()
            idx2[axis] += 1

            emit()
            emit(f"Axis {axis}:")
            emit("  largest Delta log10(D) =", val, "dex")
            emit("  cell 1 index =", tuple(idx))
            emit("  cell 2 index =", tuple(idx2))
            emit("  D1 =", D[tuple(idx)])
            emit("  D2 =", D[tuple(idx2)])

            # Try to provide physical coordinate.
            coord_keys = [qM_key, qf_key, P_key]

            if axis < len(coord_keys):
                ck = coord_keys[axis]

                if ck is not None:
                    coord = np.asarray(z[ck]).squeeze()

                    emit("  coordinate key =", ck)

                    try:
                        if coord.ndim == 1:
                            emit("  coord1 =", coord[idx[axis]])
                            emit("  coord2 =", coord[idx2[axis]])
                    except Exception:
                        pass

        emit()
        emit(
            "This allows direct comparison with the reported qM ~ 0.1, "
            "P/tE ~ 1 cusp-like structure."
        )

else:
    emit("Two-luminous summary file not found.")


# ======================================================================
# 3. TRF same-objective bounds
# ======================================================================

section("3. TRF SAME-OBJECTIVE CROSS-CHECK: WHICH BOUNDS WERE USED?")

validation_script = (
    ROOT
    / "binary_source"
    / "validation"
    / "validate_optimizer_and_grid_audit.py"
)

emit("Inspecting:", validation_script)

patterns = [
    r"least_squares",
    r"bounds\s*=",
    r"\blower\b",
    r"\bupper\b",
    r"\btrf\b",
    r"x_scale",
    r"same.objective",
    r"optimizer.crosscheck",
]

contexts = source_context(validation_script, patterns, radius=7)

if contexts:
    for block in contexts:
        emit()
        emit(block)
else:
    emit("No matching code found in expected validation script.")


# ======================================================================
# 4. 60-point stratified re-fit audit: exact method
# ======================================================================

section("4. 60-POINT STRATIFIED RE-FIT AUDIT: EXACT METHOD")

patterns = [
    r"stratif",
    r"60",
    r"refit",
    r"re-fit",
    r"quantile",
    r"transition",
    r"1e-2",
    r"10\*\*-2",
    r"least_squares",
    r"nelder",
    r"trf",
]

contexts = source_context(validation_script, patterns, radius=9)

if contexts:
    for block in contexts:
        emit()
        emit(block)
else:
    emit("No stratified-refit logic found in expected script.")

emit()
emit("Searching other validation scripts for stratified/refit logic...")

for path, blocks in search_repo(
    [
        r"stratif",
        r"refit",
        r"60.point",
        r"60_point",
    ],
    include_dirs=("binary_source/validation",),
):
    emit()
    emit("---", path.relative_to(ROOT), "---")
    for block in blocks:
        emit(block)


# ======================================================================
# 5. Two-luminous mass convention
# ======================================================================

section("5. TWO-LUMINOUS qM/qf GRID: MASS CONVENTION")

emit(
    "Searching for the code that generates qM/qf grids and for definitions "
    "of M1, M2, total mass, and qM..."
)

mass_patterns = [
    r"qM_grid",
    r"qmass",
    r"q_mass",
    r"Mtot",
    r"M_total",
    r"total_mass",
    r"M1\s*=",
    r"M2\s*=",
    r"1\s*\+\s*qM",
    r"qM\s*\*\s*M1",
    r"qmass_qflux",
    r"summary_qM_qf",
]

mass_hits = search_repo(
    mass_patterns,
    include_dirs=("binary_source",),
    max_files=50,
)

if mass_hits:
    for path, blocks in mass_hits:
        emit()
        emit("---", path.relative_to(ROOT), "---")
        for block in blocks[:8]:
            emit(block)
else:
    emit("No matching mass-convention code found.")


# ======================================================================
# 6. Existing finite-source tests/results
# ======================================================================

section("6. FINITE-SOURCE: EXISTING CODE AND RESULTS")

finite_patterns = [
    r"finite.source",
    r"finite_source",
    r"\brho1\b",
    r"\brho2\b",
    r"\brho_1\b",
    r"\brho_2\b",
    r"0\.569",
    r"0\.086",
    r"close.approach",
]

finite_hits = search_repo(
    finite_patterns,
    include_dirs=("binary_source",),
    max_files=50,
)

if finite_hits:
    for path, blocks in finite_hits:
        emit()
        emit("---", path.relative_to(ROOT), "---")
        for block in blocks[:6]:
            emit(block)
else:
    emit("No finite-source source-code hits found.")


emit()
emit("Matching result files:")

result_matches = []

for pat in [
    "*finite*",
    "*rho*",
    "*close*approach*",
]:
    result_matches.extend((ROOT / "results").rglob(pat))

result_matches = sorted(set(result_matches))

for p in result_matches[:100]:
    emit("  ", p.relative_to(ROOT))

if len(result_matches) > 100:
    emit("  ...", len(result_matches) - 100, "additional matches")


# ======================================================================
# Finish
# ======================================================================

section("DONE")

emit("Report saved to:")
emit(OUTFILE)

OUTFILE.write_text("\n".join(REPORT) + "\n")
