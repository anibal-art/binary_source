#!/usr/bin/env python3
"""
Make roman_bspl_lrt_catalog_runner.py read the LRT catalogues through
binary_source.analysis.lrt_catalog_io:

  1. replace roman_rubin's catalog.event_reader.load_event_from_catalog
     (it overwrites simulation_seed with catalog_event_index, so H0 row k and
     H1 row k would share the same noise) by the LRT loader, which validates
     the row and preserves the catalogue seed;
  2. check that --truth-kind is compatible with the row
     (h0a -> H0 row with pi_E = 0, h0b -> H0 row, h1 -> H1 row);
  3. store the row identity in the output JSON.

The script edits the file only if every required anchor is found exactly once,
keeps a timestamped backup, prints the diff and byte-compiles the result.
"""

import difflib
import py_compile
import re
import shutil
import sys
import time
from pathlib import Path

RUNNER = Path("binary_source/analysis/roman_bspl_lrt_catalog_runner.py")

src = RUNNER.read_text()
new = src

# 1. loader import ------------------------------------------------------------
pat_import = re.compile(
    r"^from catalog\.event_reader import load_event_from_catalog.*$", re.M)
if "lrt_catalog_io" in new:
    print("runner already uses lrt_catalog_io; nothing to do for the import")
else:
    hits = pat_import.findall(new)
    if len(hits) != 1:
        sys.exit(f"ABORT: expected one 'from catalog.event_reader import "
                 f"load_event_from_catalog' line, found {len(hits)}")
    new = pat_import.sub(
        "from binary_source.analysis.lrt_catalog_io import (  # noqa: E402\n"
        "    load_event_from_catalog,\n"
        "    check_truth_kind,\n"
        "    row_identity,\n"
        ")",
        new)

# 2. truth-kind compatibility -------------------------------------------------
pat_switch = re.compile(
    r"^(?P<ind>[ \t]+)truth_parallax,\s*truth_double_source\s*=\s*truth_switches\(\s*"
    r"args\.truth_kind\s*\)[ \t]*$", re.M)
if "check_truth_kind(event, args.truth_kind)" not in new:
    m = list(pat_switch.finditer(new))
    if len(m) != 1:
        sys.exit(f"ABORT: expected one truth_switches(args.truth_kind) call, "
                 f"found {len(m)}")
    ind = m[0].group("ind")
    insert = (f"\n\n{ind}# Reject e.g. --truth-kind h0a on a pi_E != 0 row, or h1 on H0.\n"
              f"{ind}check_truth_kind(event, args.truth_kind)")
    new = new[:m[0].end()] + insert + new[m[0].end():]

# 3. row identity in the output (optional anchor) ------------------------------
anchor = '"catalog_event_index": int(args.catalog_event_index),'
if '"catalog_row": row_identity(event),' not in new:
    n = new.count(anchor)
    if n == 1:
        i = new.index(anchor)
        line_start = new.rfind("\n", 0, i) + 1
        ind = new[line_start:i]
        new = new.replace(anchor, anchor + f"\n{ind}\"catalog_row\": row_identity(event),")
    else:
        print(f"WARNING: payload anchor found {n} times; catalog_row not added")

if new == src:
    print("No changes needed.")
    sys.exit(0)

backup = RUNNER.with_suffix(f".py.bak_{time.strftime('%Y%m%d_%H%M%S')}")
shutil.copy2(RUNNER, backup)
RUNNER.write_text(new)
try:
    py_compile.compile(str(RUNNER), doraise=True)
except py_compile.PyCompileError as exc:
    shutil.copy2(backup, RUNNER)
    sys.exit(f"ABORT: patched runner does not compile, restored backup.\n{exc}")

sys.stdout.writelines(difflib.unified_diff(
    src.splitlines(True), new.splitlines(True),
    fromfile=str(backup), tofile=str(RUNNER)))
print(f"\nbackup = {backup}")
print("RUNNER PATCH: PASS")
