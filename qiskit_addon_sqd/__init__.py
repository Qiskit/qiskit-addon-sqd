# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Warning: this module is not documented and it does not have an RST file.
# If we ever publicly expose interfaces users can import from this module,
# we should set up its RST file.
"""Primary SQD functionality."""

# Importing a module that has ``@_acceleration_candidate`` decorators is what
# registers its candidates with the domain, so ``configuration_recovery`` is
# imported for that side effect and not for a name.  Such a module must not
# eagerly pull in heavyweight dependencies.  (The domain itself lives in
# ``_coheriq_domain`` so that these modules can reach it without importing the
# package back, which would make the import graph cyclic.)
from . import configuration_recovery  # noqa: F401
from ._coheriq_domain import _domain

# Registration is only possible while the domain is under construction, so this
# has to come after every import above.
_domain.materialize()
