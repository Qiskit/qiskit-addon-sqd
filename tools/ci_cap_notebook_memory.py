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

"""Cap the SQD subspace dimension in the Fulqrum guide, for CI notebook testing only.

``docs/guides/integrate_fulqrum.ipynb`` intentionally runs its SQD workflow with an
unbounded subspace (no ``max_dim``): that is the honest, general example we want on the
documentation site, and it is what a reader copies. But executing it that way on a
memory-constrained CI runner exhausts memory when Fulqrum materializes the projected
Hamiltonian as a sparse (CSR) matrix.

This script injects a ``max_dim`` argument into the notebook's
``diagonalize_fermionic_hamiltonian`` calls **in the working-tree copy only**, so the
``notebook`` tox environment can execute it within the runner's memory. The change is
never committed and never reaches the docs build (which does not execute notebooks --
``nbsphinx_execute = "never"``), so the published notebook and its outputs stay
unbounded.

The injection is deliberately strict: it asserts it edited exactly the expected number
of calls and exits non-zero otherwise, so a future refactor of the notebook cannot
silently reintroduce the out-of-memory failure.

Usage (from the repo root):

    python tools/ci_cap_notebook_memory.py [--max-dim N] [--check]

``--check`` verifies the edit would apply (the expected calls are present and not
already capped) without writing, for a fast fail in CI if the notebook drifts.
"""

from __future__ import annotations

import argparse
import json
import sys

NOTEBOOK = "docs/guides/integrate_fulqrum.ipynb"
# Each diagonalize_fermionic_hamiltonian call in this notebook passes this argument;
# we insert max_dim immediately before it. Matching a stable, meaningful line keeps the
# injection anchored to the actual calls rather than to line numbers.
ANCHOR = "symmetrize_spin=True,"
EXPECTED_CALLS = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-dim", type=int, default=300, help="Per-spin-sector cap (default: 300).")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the edit would apply without writing; exit non-zero if not.",
    )
    parser.add_argument("--notebook", default=NOTEBOOK, help=f"Notebook path (default: {NOTEBOOK}).")
    args = parser.parse_args()

    with open(args.notebook, encoding="utf-8") as f:
        nb = json.load(f)

    injected = 0
    already = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell["source"]
        joined = "".join(src)
        if "diagonalize_fermionic_hamiltonian" not in joined or ANCHOR not in joined:
            continue
        if "max_dim=" in joined:
            already += 1
            continue
        for i, line in enumerate(src):
            if ANCHOR in line:
                indent = line[: len(line) - len(line.lstrip())]
                src.insert(i, f"{indent}max_dim={args.max_dim},\n")
                injected += 1
                break

    total = injected + already
    if total != EXPECTED_CALLS:
        print(
            f"ERROR: expected {EXPECTED_CALLS} diagonalize_fermionic_hamiltonian call(s) "
            f"with the '{ANCHOR}' anchor in {args.notebook}, found {total}. "
            "The notebook may have changed; update tools/ci_cap_notebook_memory.py.",
            file=sys.stderr,
        )
        return 1

    if args.check:
        print(f"OK: {injected} call(s) would be capped, {already} already capped.")
        return 0

    if injected:
        with open(args.notebook, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")
    print(f"Injected max_dim={args.max_dim} into {injected} call(s) ({already} already capped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
