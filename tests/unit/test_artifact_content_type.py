"""A stored artifact must be served as what it IS, not as what the writer assumed.

Regression cover for a product-visible bug measured live on 2026-08-09: run
019efbb0's page reported "8/8 visuals", the proxy returned 8 responses whose
bodies were valid PNGs (``\\x89PNG\\r\\n\\x1a\\n`` ... ``IHDR``), and every single
one rendered as nothing — because ``store_stage_artifact`` had declared
``application/json`` for all of them and browsers will not decode an ``<img>``
served with that type.

Two halves, tested separately because they fail independently:
  * the WRITER must stop mislabelling new artifacts, and
  * the PROXY must out-vote the mislabel already baked into MinIO's metadata for
    every artifact written before the fix. Without the second half the entire
    run history stays broken forever.
"""

from __future__ import annotations

import pytest
from w2v_common.storage import StorageService as _FacadeStorage

from src.services.storage_service import guess_artifact_content_type


class TestGuessArtifactContentType:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [("019efbb1-3961-72ea-96be-0d2f35e4f07f.png", "image/png")],
    )
    def test_known_extensions_get_a_renderable_type(self, filename: str, expected: str) -> None:
        assert guess_artifact_content_type(filename) == expected

    def test_the_default_filename_is_still_json(self) -> None:
        """`store_stage_artifact`'s default must not change behaviour."""
        assert guess_artifact_content_type("output.json") == "application/json"

    def test_unknown_extension_falls_back_to_octet_stream_not_json(self) -> None:
        """An unknown type must not claim to be something a consumer will parse.

        octet-stream reads as "download this"; a mislabelled JSON reads as
        "this is broken", which is precisely how the thumbnails died quietly.
        """
        assert guess_artifact_content_type("mystery.qqq") == "application/octet-stream"
        assert guess_artifact_content_type("no_extension_at_all") == "application/octet-stream"

    def test_a_full_object_key_works_not_just_a_bare_filename(self) -> None:
        """The proxy passes a whole key, the writer passes a bare filename."""
        key = "artifacts/019efbb0-2968-7072-b962-0599f732768b/promo_slides/019efbb1.png"
        assert guess_artifact_content_type(key) == "image/png"

    def test_a_dotted_directory_does_not_confuse_the_suffix(self) -> None:
        assert guess_artifact_content_type("artifacts/v1.2/stage/out.png") == "image/png"


class TestWriterNoLongerHardcodesJson:
    """The writer half: `store_stage_artifact` must derive, not assume."""

    @pytest.mark.asyncio
    async def test_png_artifact_is_stored_as_image_png(self) -> None:
        from src.services import storage_service as mod

        captured: dict[str, object] = {}

        class _Facade(_FacadeStorage):
            """Test double: only ``put_bytes`` is exercised, so subclass the real
            facade type instead of a structural stand-in that cannot be assigned."""

            def __init__(self) -> None:  # no settings needed for this path
                pass

            async def put_bytes(
                self,
                key: str,
                data: bytes,
                *,
                content_type: str = "application/octet-stream",
            ) -> str:
                captured.update(path=key, content_type=content_type)
                return key

        svc = mod.StorageService.__new__(mod.StorageService)
        svc._facade = _Facade()

        await svc.store_stage_artifact("run-1", "promo_slides", b"\x89PNG\r\n", "a.png")

        assert captured["content_type"] == "image/png", (
            "a PNG stored as application/json is unrenderable in a browser"
        )
        assert captured["path"] == "artifacts/run-1/promo_slides/a.png"

    @pytest.mark.asyncio
    async def test_default_json_artifact_is_unchanged(self) -> None:
        from src.services import storage_service as mod

        captured: dict[str, object] = {}

        class _Facade(_FacadeStorage):
            """Test double: see the sibling test above."""

            def __init__(self) -> None:  # no settings needed for this path
                pass

            async def put_bytes(
                self,
                key: str,
                data: bytes,
                *,
                content_type: str = "application/octet-stream",
            ) -> str:
                captured.update(content_type=content_type)
                return key

        svc = mod.StorageService.__new__(mod.StorageService)
        svc._facade = _Facade()

        await svc.store_stage_artifact("run-1", "promo_script", b"{}")

        assert captured["content_type"] == "application/json"
