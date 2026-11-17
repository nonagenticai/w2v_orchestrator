"""Unit coverage for the per-clip voiceover ownership guard.

`ClipEdit.audio_url` is the only client-settable asset URL, and the compile-time
re-sign pass re-mints any object in a bucket WE own with our own MinIO creds. So
an `audio_url` under another tenant's prefix is a cross-tenant read primitive.
`_is_own_voiceover_url` is the predicate that closes it; these tests run in the
default (non-integration) CI pass so the security decision is always exercised —
the route-level assertion in `tests/api/routes/test_runs.py` is integration-marked
and does NOT run in the default archetype pass.
"""

from __future__ import annotations

from src.api.routes.runs.lifecycle import _is_own_voiceover_url


class TestIsOwnVoiceoverUrl:
    tenant = "11111111-1111-1111-1111-111111111111"
    other = "22222222-2222-2222-2222-222222222222"

    def _url(self, tenant: str) -> str:
        return (
            f"https://minio.test/artifacts/voiceovers/{tenant}"
            f"/clip_voiceover_override/take.mp3?X-Amz-Signature=deadbeef"
        )

    def test_own_prefix_is_allowed(self) -> None:
        assert _is_own_voiceover_url(self._url(self.tenant), self.tenant) is True

    def test_foreign_prefix_is_rejected(self) -> None:
        # The attack: point at another tenant's object.
        assert _is_own_voiceover_url(self._url(self.other), self.tenant) is False

    def test_none_is_allowed(self) -> None:
        # Explicit null is the "clear back to TTS" delete.
        assert _is_own_voiceover_url(None, self.tenant) is True

    def test_non_string_is_rejected(self) -> None:
        assert _is_own_voiceover_url(12345, self.tenant) is False

    def test_arbitrary_host_without_our_prefix_is_rejected(self) -> None:
        # A URL to an internal/external host that isn't a voiceover we minted.
        assert (
            _is_own_voiceover_url("http://169.254.169.254/latest/meta-data", self.tenant) is False
        )

    def test_prefix_substring_in_query_does_not_bypass(self) -> None:
        # The prefix must be in the PATH, not smuggled via the query string.
        sneaky = (
            f"https://minio.test/artifacts/voiceovers/{self.other}"
            f"/clip_voiceover_override/take.mp3?x=/voiceovers/{self.tenant}/clip_voiceover_override/"
        )
        assert _is_own_voiceover_url(sneaky, self.tenant) is False

    def test_traversal_out_of_our_prefix_does_not_bypass(self) -> None:
        """THE substring bug. This URL contains our own prefix VERBATIM, so the old
        ``prefix in path`` check passed it — while the path resolves to the victim's
        object. RFC 3986 §5.2.4 makes dot-segment removal mandatory for URI
        resolution, so a downstream fetch is entitled to resolve it that way, and the
        compile-time re-sign would then mint OUR credentials for it."""
        attack = (
            f"https://minio.test/artifacts/voiceovers/{self.tenant}"
            f"/clip_voiceover_override/../../{self.other}/clip_voiceover_override/secret.mp3"
        )
        assert _is_own_voiceover_url(attack, self.tenant) is False

    def test_traversal_that_lands_back_inside_our_prefix_is_allowed(self) -> None:
        """The mirror case — normalisation must not be a blanket 'contains ..' reject.
        This one genuinely resolves to OUR object, so it must still pass."""
        winding = (
            f"https://minio.test/artifacts/voiceovers/{self.other}"
            f"/clip_voiceover_override/../../{self.tenant}/clip_voiceover_override/take.mp3"
        )
        assert _is_own_voiceover_url(winding, self.tenant) is True

    def test_a_sibling_tenant_prefix_sharing_our_string_is_rejected(self) -> None:
        """Segment-wise, not string-wise: ``{tenant}evil`` starts with ``{tenant}``."""
        sibling = (
            f"https://minio.test/artifacts/voiceovers/{self.tenant}evil"
            f"/clip_voiceover_override/take.mp3"
        )
        assert _is_own_voiceover_url(sibling, self.tenant) is False

    def test_url_encoded_traversal_does_not_bypass(self) -> None:
        """One decoding layer is stripped before normalisation, so %2e%2e is the
        same attack wearing a hat."""
        encoded = (
            f"https://minio.test/artifacts/voiceovers/{self.tenant}"
            f"/clip_voiceover_override/%2e%2e/%2e%2e/{self.other}/clip_voiceover_override/x.mp3"
        )
        assert _is_own_voiceover_url(encoded, self.tenant) is False

    def test_a_bucket_style_path_still_matches(self) -> None:
        """Positional tolerance is preserved: a path-style presigned URL leads with the
        BUCKET, so the triple is not at index 0. Anchoring the match would 422 a
        legitimate edit."""
        bucket_style = (
            f"http://minio.test:8389/w2v-orchestrator/voiceovers/{self.tenant}"
            f"/clip_voiceover_override/take.mp3?X-Amz-Signature=deadbeef"
        )
        assert _is_own_voiceover_url(bucket_style, self.tenant) is True

    def test_a_relative_ref_with_no_leading_segments_still_matches(self) -> None:
        relative = f"/voiceovers/{self.tenant}/clip_voiceover_override/take.mp3"
        assert _is_own_voiceover_url(relative, self.tenant) is True

    def test_empty_string_is_rejected(self) -> None:
        assert _is_own_voiceover_url("", self.tenant) is False
