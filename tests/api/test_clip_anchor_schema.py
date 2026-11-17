"""C4: the ``clip_anchor`` request-door contract admits the content-aware ``"auto"``.

``ClipEdit`` (the Studio per-clip edit door, ``PATCH`` body) and ``ShowcaseLine``
(the authored line door) both widen ``clip_anchor`` to accept ``"auto"``, which
``build_shots`` resolves to a motion-settled ``("start", lead)`` at compile time.
Every other value stays as it was, and an unknown value still 422s at the door.
Pure Pydantic — no DB, no fixtures.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import ShowcaseLine
from src.api.schemas.runs import ClipEdit


class TestClipEditAnchor:
    def test_auto_is_admitted(self) -> None:
        assert ClipEdit.model_validate({"clip_anchor": "auto"}).clip_anchor == "auto"

    def test_start_and_end_still_admitted(self) -> None:
        assert ClipEdit.model_validate({"clip_anchor": "start"}).clip_anchor == "start"
        assert ClipEdit.model_validate({"clip_anchor": "end"}).clip_anchor == "end"

    def test_unknown_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ClipEdit.model_validate({"clip_anchor": "middle"})


class TestShowcaseLineAnchor:
    def test_auto_is_admitted(self) -> None:
        assert (
            ShowcaseLine.model_validate({"text": "hi", "clip_anchor": "auto"}).clip_anchor == "auto"
        )

    def test_unknown_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ShowcaseLine.model_validate({"text": "hi", "clip_anchor": "middle"})
