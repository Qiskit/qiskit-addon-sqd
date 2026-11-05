# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check or renumber the execution counts of the guide notebooks.

The docs build does not execute notebooks (``nbsphinx_execute = "never"``), so the
hosted guides render whatever outputs are stored in the committed ``.ipynb``. A
notebook saved without executing it, or edited by hand, therefore publishes cells whose
output area is empty -- which is how qiskit-addon-sqd#427 shipped a guide with three
cells missing their output.

Consecutive execution counts are the invariant worth enforcing. Requiring every code
cell to carry output would be wrong: a cell that only imports a module prints nothing,
and that empty output is correct. But a cell that *ran* always has a non-null
``execution_count``, whether or not it printed anything, so a notebook whose counts read
1, 2, 3, ... from top to bottom is one that was executed in order from a fresh kernel.

Consecutive counts also normalize the file. The kernel hands out one ever-increasing
number per execution, so a notebook saved mid-session carries whatever the author's
clicking order produced -- re-run the second of three cells and its count becomes 4, not
2. Renumbering makes the stored counts depend only on cell order, so two people who
executed the same notebook in different orders commit the same file.

What this does not catch: outputs deleted from a notebook whose counts were left alone.
Nothing in the stored format distinguishes that from a cell that genuinely printed
nothing, so it needs a reader's eye on the rendered page.

``--check`` reports violations and exits non-zero, for CI. Without it, the counts are
renumbered in place.

Usage (from the repo root):

    python tools/notebook_execution_counts.py --check [PATH ...]
    python tools/notebook_execution_counts.py [PATH ...]

With no PATH, every notebook under ``docs/`` is processed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import nbformat

# One guide declares nbformat 4.5 but predates its required per-cell "id". That is not
# this check's business, and the warning would be noise on every CI run.
warnings.filterwarnings("ignore", category=nbformat.warnings.MissingIDFieldWarning)

DEFAULT_ROOT = "docs"


def _code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def _renumber(notebook: nbformat.NotebookNode) -> int:
    """Renumber code cells consecutively from 1. Return the number of cells changed."""
    changed = 0
    for number, cell in enumerate(_code_cells(notebook), start=1):
        if cell.execution_count != number:
            cell.execution_count = number
            changed += 1
        # An execute_result output carries its own copy of the count, which Jupyter
        # renders in the output prompt. Leaving it stale would contradict the cell.
        for output in cell.outputs:
            if "execution_count" in output:
                output["execution_count"] = number
    return changed


def _describe(notebook: nbformat.NotebookNode) -> str | None:
    """Return a complaint about this notebook's execution counts, or None if it is fine."""
    counts = [cell.execution_count for cell in _code_cells(notebook)]
    expected = list(range(1, len(counts) + 1))
    if counts == expected:
        return None
    if all(count is None for count in counts):
        return f"never executed: all {len(counts)} code cell(s) have a null execution count"
    unexecuted = [i for i, count in enumerate(counts, start=1) if count is None]
    if unexecuted:
        positions = ", ".join(str(i) for i in unexecuted)
        return (
            f"code cell(s) at position {positions} have a null execution count; counts are {counts}"
        )
    return f"execution counts are {counts}, expected {expected}"


def main() -> int:
    """Check or renumber notebook execution counts; return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help=f"Notebooks or directories to process (default: {DEFAULT_ROOT}/).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report notebooks whose counts are not consecutive and exit non-zero.",
    )
    args = parser.parse_args()

    notebooks: list[pathlib.Path] = []
    for raw in args.paths or [DEFAULT_ROOT]:
        path = pathlib.Path(raw)
        notebooks.extend(sorted(path.rglob("*.ipynb")) if path.is_dir() else [path])
    # A Jupyter checkpoint is a stale copy of a notebook, not a notebook of its own.
    notebooks = [p for p in notebooks if ".ipynb_checkpoints" not in p.parts]

    if not notebooks:
        print("ERROR: no notebooks found.", file=sys.stderr)
        return 1

    bad = []
    renumbered = []
    for path in notebooks:
        notebook = nbformat.read(str(path), as_version=4)
        if args.check:
            complaint = _describe(notebook)
            if complaint is not None:
                bad.append((path, complaint))
            continue
        if _renumber(notebook):
            nbformat.write(notebook, str(path))
            renumbered.append(path)

    if args.check:
        for path, complaint in bad:
            print(f"ERROR: {path}: {complaint}", file=sys.stderr)
        if bad:
            print(
                f"\n{len(bad)} of {len(notebooks)} notebook(s) are not numbered consecutively. "
                "Re-execute from a fresh kernel, or run "
                "`python tools/notebook_execution_counts.py` to renumber.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {len(notebooks)} notebook(s) numbered consecutively.")
        return 0

    for path in renumbered:
        print(f"renumbered {path}")
    print(f"{len(renumbered)} of {len(notebooks)} notebook(s) renumbered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
