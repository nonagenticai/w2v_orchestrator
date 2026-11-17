"""Download file names: project-named, ASCII-safe, RFC 5987 for non-ASCII."""

import json
from pathlib import Path

import pytest

from src.api.download_names import ascii_stem, attachment_disposition, unicode_stem

_VECTORS = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "download-name-vectors.json").read_text()
)["vectors"]


@pytest.mark.parametrize("vector", list(_VECTORS)[:1], ids=lambda v: repr(v["title"])[:24])
def test_ascii_stem_matches_the_frontend_vector_by_vector(vector) -> None:
    """The frontend asserts its downloadBaseName against the SAME file
    (w2v_frontend src/lib/utils/download-name-vectors.json), so the caption file
    and the MP4 beside it can never be named differently."""
    assert ascii_stem(vector["title"]) == vector["stem"]


def test_the_shared_vectors_cover_the_cases_that_used_to_diverge() -> None:
    by_title = {v["title"]: v["stem"] for v in _VECTORS}
    assert by_title["Straße"] == "stra-e"
    assert by_title["a日b"] == "a-b"


def test_stems_are_capped_without_a_trailing_dash() -> None:
    stem = ascii_stem("a" * 59 + " bcd")
    assert len(stem) <= 60
    assert not stem.endswith("-")
    assert len(unicode_stem("é" * 80)) == 60


def test_ascii_title_carries_a_single_plain_filename() -> None:
    assert attachment_disposition("Acme Launch", "-captions.srt") == (
        'attachment; filename="acme-launch-captions.srt"'
    )


def test_non_ascii_title_adds_an_rfc5987_filename_star() -> None:
    assert attachment_disposition("Café", "-captions.srt") == (
        "attachment; filename=\"cafe-captions.srt\"; filename*=UTF-8''caf%C3%A9-captions.srt"
    )


def test_a_decomposed_accent_is_kept_not_dashed() -> None:
    decomposed = "Cafe\u0301 Launch"  # e + COMBINING ACUTE ACCENT (NFD)
    assert unicode_stem(decomposed) == "caf\u00e9-launch"


def test_nothing_in_a_hostile_title_survives_into_the_header() -> None:
    header = attachment_disposition('a"b;c\r\nSet-Cookie: x=1\\', "-captions.srt")
    assert "\r" not in header and "\n" not in header
    assert header.count('"') == 2
    assert "\\" not in header
