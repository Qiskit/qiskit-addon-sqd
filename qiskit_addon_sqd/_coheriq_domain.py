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

"""The coheriq acceleration domain for this package.

This package is a coheriq *domain*: it marks certain functions as candidates for
acceleration and ships a pure-Python default implementation.  An engine (the
``sqd-hpc`` engine defined in this same package) can replace them with a compiled
implementation, selected via ``coheriq.enable_engine`` or the
``QISKIT_ADDON_SQD_ENGINE`` environment variable.

The domain lives here rather than in ``__init__.py`` so that the modules defining
candidates can reach it without importing the package they belong to.  This
module imports nothing from the package, which keeps the import graph acyclic.

The domain is deliberately *not* materialized here.  Candidates can only be
registered while it is still under construction, so ``materialize()`` must run
after every module carrying an ``@_acceleration_candidate`` has been imported;
``__init__.py`` owns that call.
"""

from coheriq import AccelerationDomain

_domain = AccelerationDomain("qiskit_addon_sqd", env_prefix="QISKIT_ADDON_SQD")
_acceleration_candidate = _domain.acceleration_candidate
