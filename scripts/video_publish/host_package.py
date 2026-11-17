#!/usr/bin/env python3
"""Host a Playwright recording-package zip on MinIO and presign it.

NOTE: this in-pod script is now the BREAK-GLASS path. The preferred way to mint a
``brief.package_url`` is the HTTP route ``POST /api/uploads/package`` (multipart
zip → presigned URL), which needs no pod access — see
``src/api/routes/uploads.py::upload_package``. Keep this script for pod-side
debugging / when the API is unavailable.

Runs INSIDE a w2v-orchestrator pod (reuses the pod's own ``StorageService`` — no
external MinIO credential needed). Reads the zip as **base64 on stdin**, stores
it, and prints ``KEY=`` + ``PACKAGE_URL=`` (the presigned URL to drop into
``brief.package_url`` for a footage-driven / ``showcase`` render).

Configuration is via ENV (not argv) so the program can be piped to ``python -``
with the zip data on stdin without the two fighting over stdin — see
``run_in_pod.py`` for the base64-embed exec pattern:

    VIDEO_PKG_NAMESPACE   storage namespace/prefix   (default: video-packages)
    VIDEO_PKG_TYPE        artifact type              (default: recording-package)
    VIDEO_PKG_FILENAME    stored object filename     (default: recording-package.zip)
    VIDEO_PKG_EXPIRES     presign TTL seconds        (default: 604800 = 7 days)
"""

import asyncio
import base64
import io
import os
import sys
import zipfile

from src.services.storage_service import StorageService

NAMESPACE = os.environ.get("VIDEO_PKG_NAMESPACE", "video-packages")
ARTIFACT_TYPE = os.environ.get("VIDEO_PKG_TYPE", "recording-package")
FILENAME = os.environ.get("VIDEO_PKG_FILENAME", "recording-package.zip")
EXPIRES = int(os.environ.get("VIDEO_PKG_EXPIRES", str(7 * 24 * 3600)))

data: bytes = base64.b64decode(sys.stdin.buffer.read(), validate=False)
z: zipfile.ZipFile = zipfile.ZipFile(io.BytesIO(data))
if z.testzip() is not None:
    raise RuntimeError("corrupt zip")
print(f"decoded={len(data)} members={len(z.namelist())}", file=sys.stderr)


async def main() -> None:
    s = StorageService()
    await s.initialize()
    path = await s.store_stage_artifact(NAMESPACE, ARTIFACT_TYPE, data, FILENAME)
    url = await s.get_presigned_url(path, expires=EXPIRES)
    if not url:
        raise RuntimeError(f"presign failed: get_presigned_url returned no URL for {path}")
    print("KEY=" + path)
    print("PACKAGE_URL=" + url)


asyncio.run(main())
