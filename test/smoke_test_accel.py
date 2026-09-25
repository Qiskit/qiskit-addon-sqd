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

"""Standalone smoke test run inside each cibuildwheel-built wheel.

Confirms that a binary wheel ships the compiled ``sqd-hpc`` engine and that,
once activated through coheriq, accelerated configuration recovery produces
valid output.  Kept free of the heavy test dependencies (pyscf, jax) so it runs
on every wheel target.
"""

import coheriq
import numpy as np

# Importing the package registers (and materializes) the coheriq domain.
import qiskit_addon_sqd  # noqa: F401
from qiskit_addon_sqd.configuration_recovery import recover_configurations

# A binary wheel must ship the compiled extension and so must advertise the
# engine.  If the extension is missing, the build backend declares no entry point
# for it and this raises CoheriqEngineNotFoundError; if the entry point is there
# but the extension is not, the module import raises ImportError.  Either way the
# smoke test fails, which is the point.
coheriq.enable_engine("qiskit_addon_sqd", "sqd-hpc")

# Deterministic all-flip case: zeros with full occupancy and half-filling
# targets must become all ones, regardless of the RNG stream.
mat, probs = recover_configurations(
    np.array([[False, False, False, False]]),
    np.array([1.0]),
    (np.array([1.0, 1.0]), np.array([1.0, 1.0])),
    2,
    2,
    rand_seed=4224,
)
assert mat.tolist() == [[True, True, True, True]], mat.tolist()
assert probs.tolist() == [1.0], probs.tolist()

# A larger random case: check the Hamming-weight invariants hold.
rng = np.random.default_rng(0)
norb = 6
bs = rng.integers(2, size=(50, 2 * norb)).astype(bool)
p = rng.random(50)
p /= p.sum()
occs = (rng.random(norb), rng.random(norb))
mat, probs = recover_configurations(bs, p, occs, 3, 2, rand_seed=1)
assert np.all(mat[:, norb:].sum(axis=1) == 3)
assert np.all(mat[:, :norb].sum(axis=1) == 2)
assert abs(probs.sum() - 1.0) < 1e-12

print("OK: accelerated configuration recovery smoke test passed")
