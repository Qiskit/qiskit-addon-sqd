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

"""Tests that the installed metadata agrees with what the installation ships.

The ``sqd-hpc`` engine needs the compiled ``qiskit_addon_sqd._accel`` extension.
Its entry point is added by the build backend (``tools/_sqd_build.py``) only to
distributions that ship the extension, so that a distribution never advertises an
engine it cannot run.  These tests check that invariant against the *installed*
distribution, whichever kind it is.
"""

import json
import unittest
from importlib.metadata import PackageNotFoundError, distribution, entry_points

DOMAIN = "qiskit_addon_sqd"
ENGINE = "sqd-hpc"
GROUP = f"coheriq.engines.{DOMAIN}"
EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")


def _distribution():
    """The installed qiskit-addon-sqd distribution, or None if there is none."""
    try:
        return distribution("qiskit-addon-sqd")
    except PackageNotFoundError:
        return None


def _accel_shipped(dist):
    """Whether ``dist`` ships the compiled ``_accel`` extension.

    This asks the *distribution's* file list rather than importing or using
    ``find_spec``.  Both of those resolve against ``sys.path``, and pytest runs
    with the source tree first on it, so the uncompiled ``qiskit_addon_sqd/``
    directory in the repository shadows an installed compiled copy -- which would
    compare this installation's metadata against a different tree entirely.
    """
    return any(
        f.name.split(".")[0] == "_accel" and f.name.endswith(EXTENSION_SUFFIXES)
        for f in (dist.files or [])
    )


def _engine_advertised():
    return ENGINE in {ep.name for ep in entry_points(group=GROUP)}


class TestEnginePackaging(unittest.TestCase):
    """The advertised engine must match the shipped extension."""

    def test_entry_point_present_exactly_when_extension_is(self):
        """An engine is advertised if and only if its extension is available."""
        dist = _distribution()
        if dist is None:
            self.skipTest("qiskit-addon-sqd is not installed as a distribution")
        if dist.files is None:
            self.skipTest("this installation does not record a file list")
        self.assertEqual(
            _engine_advertised(),
            _accel_shipped(dist),
            "the sqd-hpc entry point and the _accel extension must be present or "
            "absent together; a distribution must not advertise an engine it "
            "cannot run, nor hide one it can",
        )

    def test_entry_point_targets_the_engine_module(self):
        """When advertised, the entry point names the engine module."""
        if not _engine_advertised():
            self.skipTest("this installation does not provide the sqd-hpc engine")
        (ep,) = (ep for ep in entry_points(group=GROUP) if ep.name == ENGINE)
        self.assertEqual(ep.value, "qiskit_addon_sqd._sqd_hpc_engine")

    def test_build_info_agrees_with_the_entry_points(self):
        """``sqd_engine_build_info.json`` records the same availability."""
        dist = _distribution()
        if dist is None:
            self.skipTest("qiskit-addon-sqd is not installed as a distribution")
        raw = dist.read_text("sqd_engine_build_info.json")
        if raw is None:
            # An editable install made by an older backend, or a tree on sys.path
            # that was never built, has no build info to check.
            self.skipTest("this installation ships no sqd_engine_build_info.json")

        info = json.loads(raw)
        self.assertEqual(info["schema"], 1)
        self.assertEqual(info["domain"], DOMAIN)
        self.assertIn(ENGINE, info["engines"])
        engine = info["engines"][ENGINE]
        self.assertEqual(engine["requires_extension"], "_accel")
        self.assertEqual(engine["provided"], _engine_advertised())


if __name__ == "__main__":
    unittest.main()
