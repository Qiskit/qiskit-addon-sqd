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

from coheriq import AccelerationDomain

# This package is a coheriq *domain*: it marks certain functions as candidates
# for acceleration and ships a pure-Python default implementation.  An engine
# (the ``sqd-hpc`` engine defined in this same package) can replace them with a
# compiled implementation, selected via ``coheriq.enable_engine`` or the
# ``QISKIT_ADDON_SQD_ENGINE`` environment variable.
_domain = AccelerationDomain("qiskit_addon_sqd", env_prefix="QISKIT_ADDON_SQD")
_acceleration_candidate = _domain.acceleration_candidate

# Import every module that has an ``@_acceleration_candidate`` decorator so that
# its candidates are registered before we materialize the domain.  These imports
# must not eagerly pull in heavyweight dependencies.
from . import configuration_recovery  # noqa: E402,F401  pylint: disable=wrong-import-position

_domain.materialize()
