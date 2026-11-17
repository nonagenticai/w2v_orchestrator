"""Sync a directory of royalty-free sounds into the orchestrator's MinIO bucket.

This replaces the old bespoke ``seed_sound_effects.py``. It uses the normal
``StorageService`` (own ``w2v-orchestrator`` bucket, own creds — no foreign
bucket, no extra IAM, no raw ``Minio`` client) and writes the data-driven
``manifest.json`` the runtime catalog reads.

What it does
------------
For ``--src <dir>`` laid out as ``<category-dir>/<file>.<ext>`` (category dirs:
``transitions``/``accents``/``music``/``ambient``), it uploads each audio file to
``sound_library/<scope>/<category-dir>/<file>`` and then (re)writes
``sound_library/<scope>/manifest.json`` indexing every uploaded sound. After that,
the sounds are **instantly available** to every w2v app — adding hundreds more is
just "drop files in the dir, re-run this".

``--scope shared`` (the only scope) seeds the all-tenant library. A tenant's private
library is NOT seeded here any more: since P6 ORCH-4 it is owned by the
``tenant_sound_assets`` table and uploaded through ``POST /api/tenant/music`` (its
manifest is regenerated from the table on every change, so anything this script wrote
there would be overwritten — and its ids would lack the reserved ``tenant_`` prefix).

Descriptions come from an optional ``<src>/descriptions.json`` (``{id: text}``),
else a humanised filename. Durations come from ``ffprobe`` when available, else 0.

Usage
-----
::

    # From inside an orchestrator pod (or with STORAGE_* env pointing at MinIO):
    python -m scripts.sync_sound_library                       # shared, from assets/sound_effects
    python -m scripts.sync_sound_library --src /path/to/sounds

Idempotent — re-running overwrites objects + manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.services.stages.sound_library_sync import DEFAULT_SOUND_SRC, sync

__all__ = ["main", "sync"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync sounds into the orchestrator MinIO bucket")
    parser.add_argument(
        "--src", type=Path, default=DEFAULT_SOUND_SRC, help="source dir of category subdirs"
    )
    parser.add_argument("--scope", choices=("shared",), default="shared")
    args = parser.parse_args()
    sys.exit(asyncio.run(sync(args.src, args.scope, None)))


if __name__ == "__main__":
    main()
