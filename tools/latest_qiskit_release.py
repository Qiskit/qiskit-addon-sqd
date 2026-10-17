"""Report the newest qiskit on PyPI and whether it is a pre-release.

The nightly pre-release workflow runs only while a pre-release is the newest
thing on PyPI; once its final release lands, `Latest version tests` covers the
same wheel and this would be redundant.
"""

import json
import os
import urllib.request

from packaging.version import InvalidVersion, Version

URL = "https://pypi.org/pypi/qiskit/json"


def main() -> None:
    with urllib.request.urlopen(URL) as response:
        releases = json.load(response)["releases"]

    newest = None
    newest_final = None
    for raw, files in releases.items():
        # Skip a version with nothing installable left -- all files yanked, or a
        # key present with no upload. It must not decide whether this run happens.
        if not any(not entry.get("yanked", False) for entry in files):
            continue
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        if newest is None or version > newest:
            newest = version
        if not version.is_prerelease and (newest_final is None or version > newest_final):
            newest_final = version

    if newest is None:
        raise SystemExit(f"found no installable qiskit release at {URL}")

    # PEP 440 ordering, not `sort -V`, which ranks 2.6.0rc1 above 2.6.0 and would
    # leave a stale pre-release winning over its own final release forever.
    prerelease = newest.is_prerelease

    for line in (
        f"newest={newest}",
        f"newest_final={newest_final}",
        f"prerelease={str(prerelease).lower()}",
    ):
        print(line)
        if output := os.environ.get("GITHUB_OUTPUT"):
            with open(output, "a") as handle:
                handle.write(line + "\n")

    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as handle:
            handle.write("### Newest qiskit on PyPI\n\n")
            handle.write(f"- newest: `{newest}`\n")
            handle.write(f"- newest final release: `{newest_final}`\n")
            if prerelease:
                handle.write(f"\nTesting against the `{newest}` wheel.\n")
            else:
                handle.write(
                    "\nNot a pre-release, so there is nothing here that "
                    "`Latest version tests` does not already cover. Skipped.\n"
                )


if __name__ == "__main__":
    main()
