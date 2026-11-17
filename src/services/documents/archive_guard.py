"""Pre-parse decompression-bomb guard for zip-container uploads (docx, epub).

python-docx and EbookLib inflate whole zip members into memory. A 1 MiB docx whose
``word/document.xml`` deflates from 1 GiB of whitespace took the orchestrator to
3.1 GB peak RSS before failing — and the pod limit is 4 GiB with ``replicas: 1``.
So every zip container a user uploads is checked from its central directory
BEFORE any parser touches it:

* the summed declared ``file_size`` of all members must not exceed
  :data:`MAX_ARCHIVE_UNCOMPRESSED_BYTES`;
* at most :data:`MAX_ARCHIVE_MEMBERS` members;
* no member over :data:`RATIO_CHECK_MIN_BYTES` (inflated) may compress better
  than :data:`MAX_MEMBER_COMPRESSION_RATIO`:1 — prose XML deflates ~5-20:1, a
  bomb ~1000:1.

Why the declared size is enough to bound ACTUAL inflated bytes: both parsers
read through :mod:`zipfile`, whose ``ZipExtFile`` truncates decompressed output
to the central directory's ``file_size`` (CPython ``zipfile._read1``:
``data = data[:self._left]``) and raises ``BadZipFile`` on the CRC mismatch a
lying header produces. A member cannot yield more than it declares.

Shared by ``documents.extract`` (source documents) and
``stages.audiobook.parsing`` (manuscripts); each maps :class:`UnsafeArchiveError`
onto its own error type.
"""

from __future__ import annotations

import io
import zipfile
from typing import Final

#: Summed uncompressed size of every member, at most. A real docx/epub of a 25 MiB
#: upload is a fraction of this; it bounds the parser's working set well under 4 GiB.
MAX_ARCHIVE_UNCOMPRESSED_BYTES: Final = 200 * 1024 * 1024
#: Members per archive at most (central-directory parsing alone is O(members)).
MAX_ARCHIVE_MEMBERS: Final = 10_000
#: A member inflating past RATIO_CHECK_MIN_BYTES must compress worse than this.
MAX_MEMBER_COMPRESSION_RATIO: Final = 100
RATIO_CHECK_MIN_BYTES: Final = 4 * 1024 * 1024

_MIB: Final = 1024 * 1024


class UnsafeArchiveError(ValueError):
    """The zip container is corrupt or shaped like a decompression bomb."""


def check_zip_archive(data: bytes, filename: str) -> None:
    """Raise :class:`UnsafeArchiveError` unless ``data`` is a sanely-sized zip.

    Reads only the central directory — nothing is decompressed.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        raise UnsafeArchiveError(f"{filename!r} is a corrupt zip container") from exc
    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise UnsafeArchiveError(
            f"{filename!r} has {len(infos)} zip members (limit {MAX_ARCHIVE_MEMBERS})"
        )
    total = 0
    for info in infos:
        size = info.file_size
        if size > RATIO_CHECK_MIN_BYTES and size > MAX_MEMBER_COMPRESSION_RATIO * max(
            info.compress_size, 1
        ):
            ratio = size / max(info.compress_size, 1)
            raise UnsafeArchiveError(
                f"{filename!r} member {info.filename!r} has a {ratio:.0f}:1 compression "
                f"ratio (limit {MAX_MEMBER_COMPRESSION_RATIO}:1) — refusing a likely "
                "decompression bomb"
            )
        total += size
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise UnsafeArchiveError(
                f"{filename!r} inflates to more than {MAX_ARCHIVE_UNCOMPRESSED_BYTES // _MIB} "
                "MiB uncompressed — refusing a likely decompression bomb"
            )
