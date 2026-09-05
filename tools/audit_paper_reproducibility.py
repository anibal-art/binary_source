#!/usr/bin/env python3

"""
Audit the reproducibility documentation of the BSPL--PSPL paper.

This script performs two checks:

1. Read every final paper figure from tools/paper_figure_manifest.txt.
2. Search paper-facing Python source files for comments/docstrings that
   still appear to contain Spanish documentation.

The manuscript itself is maintained separately in Overleaf. This script does not modify any scientific code or numerical result.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
FIGURE_MANIFEST = ROOT / "tools" / "paper_figure_manifest.txt"
PAPER_SOURCE_DIRS = [
    ROOT / "binary_source" / "analysis",
    ROOT / "binary_source" / "plot_codes",
    ROOT / "binary_source" / "scan_codes",
    ROOT / "binary_source" / "validation",
]

PAPER_SOURCE_FILES = [
    ROOT / "binary_source" / "functions_aux.py",
    ROOT / "binary_source" / "validate_optimizer_and_grid_audit.py",
    ROOT / "isochrone_phot_cancel.py",
]


SPANISH_WORDS = {
    "ajuste",
    "ajustes",
    "archivo",
    "archivos",
    "calcula",
    "calcular",
    "cargar",
    "caso",
    "casos",
    "curva",
    "curvas",
    "datos",
    "devuelve",
    "directorio",
    "figura",
    "figuras",
    "fuente",
    "fuentes",
    "magnitud",
    "muestra",
    "observaciones",
    "periodo",
    "parámetros",
    "parametros",
    "queremos",
    "resultado",
    "resultados",
    "simulación",
    "simulacion",
    "temporada",
    "ventana",
    "verdadero",
    "verdaderos",
}


def extract_figures():
    """
    Return the final paper figure assets listed in the repository manifest.

    The manuscript itself is maintained separately in Overleaf and is not
    part of this source-code repository.
    """

    if not FIGURE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing paper figure manifest: {FIGURE_MANIFEST}"
        )

    figures = []

    for line in FIGURE_MANIFEST.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line or line.startswith("#"):
            continue

        figures.append(line)

    return figures


def iter_python_files():
    files = []

    for directory in PAPER_SOURCE_DIRS:
        if not directory.exists():
            continue

        for path in directory.rglob("*.py"):
            if "legacy" in path.parts:
                continue

            files.append(path)

    for path in PAPER_SOURCE_FILES:
        if path.exists():
            files.append(path)

    return sorted(set(files))


def looks_like_documentation(line):
    stripped = line.strip()

    return (
        stripped.startswith("#")
        or stripped.startswith('"""')
        or stripped.startswith("'''")
        or stripped.endswith('"""')
        or stripped.endswith("'''")
    )


def find_spanish_documentation(path):
    hits = []

    try:
        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()
    except UnicodeDecodeError:
        return hits

    for lineno, line in enumerate(lines, start=1):

        if not looks_like_documentation(line):
            continue

        words = set(
            re.findall(
                r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+",
                line.lower(),
            )
        )

        if words & SPANISH_WORDS:
            hits.append(
                (
                    lineno,
                    line.rstrip(),
                )
            )

    return hits


def main():
    print()
    print("=" * 80)
    print("PAPER FIGURE INVENTORY")
    print("=" * 80)

    figures = extract_figures()

    for i, figure in enumerate(figures, start=1):
        print(
            f"{i:02d}. {figure}"
        )

    print()
    print(
        f"Total manuscript figure assets: "
        f"{len(figures)}"
    )

    print()
    print("=" * 80)
    print("POSSIBLE NON-ENGLISH CODE DOCUMENTATION")
    print("=" * 80)

    n_hits = 0

    for path in iter_python_files():

        hits = find_spanish_documentation(
            path
        )

        if not hits:
            continue

        rel = path.relative_to(ROOT)

        print()
        print(rel)

        for lineno, line in hits:
            print(
                f"  {lineno:5d}: {line}"
            )

        n_hits += len(hits)

    print()
    print("=" * 80)
    print(
        f"Potential Spanish documentation lines: "
        f"{n_hits}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
