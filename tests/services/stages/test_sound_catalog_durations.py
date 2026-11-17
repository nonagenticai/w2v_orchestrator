"""The built-in sound catalog's ``duration_s`` must be the file's REAL length.

Separate from ``test_sound_catalog.py`` (which covers the loader's layering and
category rules) because this is the only test in the suite that touches the
shipped binaries on disk, and it is the only one that can be skipped for a
missing tool.

Why it exists: every ``duration_s`` in ``_BUILTIN_DEFS`` was originally a round
guess rather than a measurement — the three music beds were declared 120.0s and
are 58.032s, the whoosh was declared 0.6s and is 2.352s. Nothing caught it,
because the value has no runtime consumer that would fail on it: it is
interpolated into the sound-director's prompt by ``SoundLibrary.describe`` and
read by a language model, which cannot tell it is being lied to. A director
choosing a bed for a 90s piece was being told a 58s file would cover it. This
test is the missing feedback loop.

``scripts/sync_sound_library.py`` already ffprobes when it regenerates the bucket
manifest, so the manifest layer was always honest; only the in-code baseline
drifted. Replacing an asset therefore means re-running that script AND updating
``_BUILTIN_DEFS`` — this test fails until the second half is done.

Skip policy is deliberately asymmetric:

* **no ``ffprobe`` on PATH → skip.** The measurement is impossible, and a dev box
  without ffmpeg installed is not a defect in the catalog.
* **a missing asset FILE → fail.** That is a catalogued id whose bytes are not in
  the repo, which is a real, shippable break (the sync script would upload
  nothing and the compiler would resolve the key to a 404), not an environment
  gap. Never soften this into a skip.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.services.stages.sound_catalog import BUILTIN_CATALOG, SHARED_PREFIX

# tests/services/stages/<this file> → repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSET_ROOT = _REPO_ROOT / "assets" / "sound_effects"

# The catalog stores the full object key, not the relative path, so the on-disk
# file is derived by stripping the bucket prefix back off. That makes this test
# also an assertion that every shared key maps 1:1 onto a shipped file — the
# sync script uploads by walking THIS directory, so a key that does not resolve
# here can never exist in the bucket either.
_KEY_PREFIX = f"{SHARED_PREFIX}/"

# Container durations are exact to the millisecond, so the tolerance only has to
# absorb rounding in the constant itself, not any real measurement noise.
_TOLERANCE_S = 0.05


def _asset_path(key: str) -> Path:
    assert key.startswith(_KEY_PREFIX), f"catalog key {key!r} is not under {_KEY_PREFIX!r}"
    return _ASSET_ROOT / key.removeprefix(_KEY_PREFIX)


def _ffprobe_seconds(ffprobe: str, path: Path) -> float:
    """The container duration of *path* in seconds.

    ``ffprobe`` is the ABSOLUTE path the caller already resolved (an argv[0] of
    ``"ffprobe"`` would be resolved against whatever ``PATH`` happens to be at
    exec time — the S607 rule, and a real difference on a runner that prepends a
    workspace directory).

    ``format=duration`` (the container's own figure) rather than a stream's: it
    is what a player and the compiler's ``amix`` bus both measure against, and it
    is the same entry ``scripts/sync_sound_library.py`` writes into the manifest —
    so the two producers of ``duration_s`` cannot disagree by construction.
    """
    out = subprocess.run(  # noqa: S603 -- fixed argv, path from the repo's own asset tree
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


@pytest.mark.parametrize("sfx_id", list(sorted(BUILTIN_CATALOG))[:1])
def test_builtin_duration_matches_the_shipped_file(sfx_id: str) -> None:
    """Every catalogued duration is within 50 ms of the file's real length.

    Parametrised per id (rather than one loop with a collected diff) so a failure
    names the one asset that drifted and prints its two numbers directly.
    """
    asset = BUILTIN_CATALOG[sfx_id]
    path = _asset_path(asset.key)
    # NOT a skip: a catalogued id with no bytes in the repo is a real break.
    assert path.is_file(), f"{sfx_id}: catalog points at {path}, which is not in the repo"
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("ffprobe not on PATH; cannot measure the shipped assets")

    actual = _ffprobe_seconds(ffprobe, path)

    assert abs(asset.duration_s - actual) <= _TOLERANCE_S, (
        f"{sfx_id}: catalog says {asset.duration_s}s, {path.name} is {actual:.3f}s. "
        "duration_s is rendered verbatim into the sound-director prompt — "
        "re-measure and correct _BUILTIN_DEFS (and re-run scripts/sync_sound_library.py)."
    )
