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

"""Build backend: declare coheriq engine entry points per built distribution.

A thin wrapper around scikit-build-core.  The engines this package *can* provide
are declared in ``[tool.qiskit-addon-sqd]``, and an engine's entry point is written
into a
distribution only when that distribution actually ships the compiled extension
the engine needs.  The universal (``py3-none-any``) wheel, built with
``-C wheel.cmake=false``, therefore advertises no engine at all, and
``coheriq.enable_engine`` reports it as not found rather than failing on a
missing extension at import time.

Why a wrapper backend, rather than scikit-build-core's own facilities:

* ``[project.entry-points]`` is static metadata.  ``-C wheel.cmake=false``
  changes the build, not the metadata, so a statically declared entry point
  lands in every wheel including the one that cannot honor it.
* A dynamic-metadata provider cannot help: it runs before CMake, with no
  settings and no ``SKBUILD_*`` environment to inspect, so it cannot tell which
  kind of wheel is being built.  It also requires ``experimental = true``.

We therefore key off the *artifact*.  Once the real backend has produced a
wheel, its contents are ground truth, and the decision holds however the build
was invoked -- locally, in CI, or from an sdist.

PEP 621 treats ``entry-points`` as an *extendable* field: a backend may add
entries but must not remove statically declared ones.  So engines are declared
in ``[tool.qiskit-addon-sqd]`` and *added* here, rather than declared in
``[project.entry-points]`` and stripped; that is the direction a future
standardised mechanism could express.

Alongside the entry points, the outcome is recorded in the installed
``.dist-info`` as ``sqd_engine_build_info.json``, readable with
``importlib.metadata.distribution(...).read_text("sqd_engine_build_info.json")``.
It names every engine this package can provide and whether this particular
distribution provides it, so a user who finds no engine entry point can see what
a compiled wheel would offer and why theirs does not.

That file is a build record for this distribution, and not a discovery mechanism:
coheriq finds engines through entry points, and an engine shipped by a separate
distribution would leave no trace here.  It is useful only because these engines
are bundled with the domain that owns them, so one build decides both whether the
extension exists and whether the engine can be advertised.  Hence the
``sqd_``-prefixed name and the ``comment`` field in the payload, which keep a
reader who meets the file first from taking it for something general.

The ``[tool.qiskit-addon-sqd]`` table is named on the same reasoning.  The schema
it holds is ours: coheriq neither defines nor reads it, only this backend does.
Calling it ``[tool.coheriq]`` would imply coheriq owns the schema, and would take
a name coheriq should keep for one of its own if this is ever generalized
upstream.
"""

from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

# tomllib is stdlib from Python 3.11; tomli is its backport, requested in
# build-system.requires for older interpreters.  Drop both this block and that
# requirement once the Python floor reaches 3.11.
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

# Re-export every hook scikit-build-core provides, so this module is a complete
# PEP 517 backend; only the wheel-producing hooks (below) get different behavior.
#
# NB: if scikit-build-core is missing (e.g. --no-build-isolation without the
# build requires installed), pip reports "Cannot import '_sqd_build'" and
# discards the underlying ModuleNotFoundError, so a nicer message here would not
# reach the user.  Not worth the dead code.
from scikit_build_core.build import *  # noqa: F403
from scikit_build_core.build import build_editable as _build_editable
from scikit_build_core.build import build_wheel as _build_wheel

#: Suffixes a compiled extension module may carry.  A ``STABLE_ABI`` nanobind
#: module installs as e.g. ``_accel.abi3.so``, and a non-stable one as
#: ``_accel.cpython-312-x86_64-linux-gnu.so``, so the stem is matched up to the
#: first dot rather than by a prefix test (which would also match ``_accelfoo``).
EXTENSION_SUFFIXES = (".so", ".pyd", ".dylib")

#: Bump when the sqd_engine_build_info.json layout changes in a way a reader must
#: notice.  A reader should accept a schema it knows and ignore anything newer.
BUILD_INFO_SCHEMA = 1


def _declaration(source_dir="."):
    """Read ``[tool.qiskit-addon-sqd]`` from pyproject.toml.

    PEP 517 requires the frontend to call build hooks with the source tree as
    the current directory, which is the same contract scikit-build-core itself
    relies on to find ``pyproject.toml``.
    """
    with open(Path(source_dir) / "pyproject.toml", "rb") as f:
        return tomllib.load(f).get("tool", {}).get("qiskit-addon-sqd", {})


def _check(decl):
    """Reject a structurally malformed ``[tool.qiskit-addon-sqd]`` declaration.

    Only the shape is checked, not the meaning.  Whether ``domain`` names the
    domain this package actually registers cannot be settled here: the domain is
    constructed at runtime, under whatever private name and in whatever module
    the package chooses, and an engine package may target a domain owned by a
    different distribution entirely.  A check that guessed at that would pass on
    a wrong-but-plausible value and so earn more trust than it deserves.
    """
    if not decl.get("domain") or not isinstance(decl["domain"], str):
        raise ValueError(
            "[tool.qiskit-addon-sqd] declares engines but no 'domain'; it must name "
            "the coheriq acceleration domain whose engines these are"
        )
    for engine, spec in decl["engines"].items():
        if not isinstance(spec.get("target"), str):
            raise ValueError(
                f"[tool.qiskit-addon-sqd.engines.{engine}] has no 'target' module to import"
            )


def _ships_extension(names, stem):
    """Whether the wheel contains a compiled extension module named ``stem``."""
    return any(Path(n).name.split(".")[0] == stem and n.endswith(EXTENSION_SUFFIXES) for n in names)


def _resolve(decl, names):
    """Which declared engines this distribution provides, and which it does not."""
    provided, absent = {}, {}
    for engine, spec in sorted(decl.get("engines", {}).items()):
        stem = spec.get("requires-extension")
        if stem is None or _ships_extension(names, stem):
            provided[engine] = spec
        else:
            absent[engine] = spec
    return provided, absent


def _entry_points_txt(existing, group, engines):
    """``existing`` entry_points.txt content with ``engines`` added under ``group``."""
    parser = configparser.ConfigParser()
    parser.optionxform = str  # entry-point names are case-sensitive
    if existing:
        parser.read_string(existing)
    if engines:
        if not parser.has_section(group):
            parser.add_section(group)
        for engine, spec in engines.items():
            parser.set(group, engine, spec["target"])
    out = io.StringIO()
    parser.write(out)
    return out.getvalue()


def _build_info(decl, provided, absent):
    """The sqd_engine_build_info.json payload recorded in the distribution's .dist-info.

    The ``comment`` travels with the file so that it explains itself wherever it
    is found, without the reader having to locate this backend first.
    """
    return {
        "comment": (
            "Build-time record of the coheriq engines bundled with this "
            "distribution, and whether this build can provide each one.  coheriq "
            "discovers engines through entry points; this file is not that "
            "mechanism and is specific to qiskit-addon-sqd."
        ),
        "schema": BUILD_INFO_SCHEMA,
        "domain": decl["domain"],
        "engines": {
            engine: {
                "provided": engine in provided,
                "target": spec.get("target"),
                "requires_extension": spec.get("requires-extension"),
                "description": spec.get("description"),
            }
            for engine, spec in sorted({**provided, **absent}.items())
        },
    }


def _record_row(path, data):
    """A RECORD row: path, urlsafe-unpadded sha256, and byte count."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return [path, f"sha256={digest}", str(len(data))]


def _rewrite(wheel, updates, drop):
    """Rewrite ``wheel``, applying ``updates`` and removing ``drop``, fixing RECORD."""
    with zipfile.ZipFile(wheel) as zf:
        items = [(i, zf.read(i.filename)) for i in zf.infolist()]
    record_path = next(i.filename for i, _ in items if i.filename.endswith(".dist-info/RECORD"))

    rows, buf = [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        seen = set()
        for info, data in items:
            if info.filename == record_path or info.filename in drop:
                continue
            if info.filename in updates:
                data = updates[info.filename]
                seen.add(info.filename)
            new = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new.external_attr = info.external_attr
            new.compress_type = info.compress_type
            out.writestr(new, data)
            rows.append(_record_row(info.filename, data))
        for path, data in updates.items():
            if path not in seen:
                out.writestr(zipfile.ZipInfo(path), data)
                rows.append(_record_row(path, data))

        # RECORD lists itself with no hash or size.
        rows.append([record_path, "", ""])
        csv_buf = io.StringIO()
        csv.writer(csv_buf, lineterminator="\n").writerows(rows)
        out.writestr(zipfile.ZipInfo(record_path), csv_buf.getvalue())

    wheel.write_bytes(buf.getvalue())


def _declare_engines(wheel_directory, name):
    """Add entry points and the manifest to the wheel the real backend built."""
    decl = _declaration()
    if not decl.get("engines"):
        return name
    _check(decl)

    wheel = Path(wheel_directory) / name
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
    dist_info = next(n.split("/")[0] for n in names if n.endswith(".dist-info/RECORD"))

    provided, absent = _resolve(decl, names)
    group = f"coheriq.engines.{decl['domain']}"
    ep_path = f"{dist_info}/entry_points.txt"
    with zipfile.ZipFile(wheel) as zf:
        existing = zf.read(ep_path).decode() if ep_path in names else ""

    updates = {
        f"{dist_info}/sqd_engine_build_info.json": (
            json.dumps(_build_info(decl, provided, absent), indent=2) + "\n"
        ).encode(),
    }
    drop = set()
    new_ep = _entry_points_txt(existing, group, provided)
    if new_ep.strip():
        updates[ep_path] = new_ep.encode()
    elif ep_path in names:
        drop.add(ep_path)

    _rewrite(wheel, updates, drop)
    if provided:
        print(f"*** {name}: provides coheriq engines {sorted(provided)}")
    if absent:
        print(f"*** {name}: no compiled extension for {sorted(absent)}; not advertised")
    return name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel, then declare the engines it can actually provide."""
    return _declare_engines(
        wheel_directory,
        _build_wheel(wheel_directory, config_settings, metadata_directory),
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable wheel, then declare the engines it can actually provide.

    An editable install compiles the extension in place, so ``pip install -e .``
    must get the entry point just as a regular build does.
    """
    return _declare_engines(
        wheel_directory,
        _build_editable(wheel_directory, config_settings, metadata_directory),
    )
