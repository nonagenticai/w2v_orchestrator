"""Unit tests for the artifact URL rewriter.

The rewriter turns ``http://minio.test:8389/<bucket>/<key>?<sig>`` URLs
in stage-output payloads into the orchestrator's
``/api/runs/<run_id>/artifact/<bucket>/<key>?<sig>`` proxy path so the
browser can fetch what the in-cluster video-compile service stored.
"""

from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from src.api.artifact_url_rewriter import (
    collect_internal_minio_refs,
    proxy_path,
    rewrite_minio_url,
    rewrite_payload,
)

_RUN_ID = UUID("aad0b25b-f7d7-45c8-8de1-d26119fc4b69")
_SIG = (
    "X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=svc-video-compiler%2F20260530"
    "&X-Amz-Signature=abc"
)


def test_proxy_path_prepends_api_prefix() -> None:
    # This test used to assert the SigV4 query survived VERBATIM, and that
    # assertion was pinning a live defect: `artifact_proxy` ignores the inbound
    # query and re-fetches with its own credentials, so a proxy path can never
    # expire -- but the SPA's `isPresignExpired` regexes ANY url string for
    # `X-Amz-Date` / `X-Amz-Expires`. One hour after a stage ran, all 20
    # `cut_plan` shot thumbnails on the REVIEW GATE rendered as "source link
    # expired" (measured in a real browser, 2026-08-20). The params are now
    # stripped, so the URL no longer claims something about itself that is false.
    out = proxy_path(_RUN_ID, "w2v-compiled-videos/hash.mp4", _SIG)
    assert out == f"/api/runs/{_RUN_ID}/artifact/w2v-compiled-videos/hash.mp4"


def test_proxy_path_handles_empty_query() -> None:
    out = proxy_path(_RUN_ID, "w2v-orchestrator/artifacts/x.wav", "")
    assert out == f"/api/runs/{_RUN_ID}/artifact/w2v-orchestrator/artifacts/x.wav"


def test_proxy_path_strips_leading_slash() -> None:
    # urlsplit().path yields a leading "/", which must not double up in the
    # rewritten path or the SPA router emits a 308 redirect loop on first
    # navigation.
    out = proxy_path(_RUN_ID, "/w2v-compiled-videos/hash.mp4", "k=v")
    assert "/artifact//" not in out


def test_rewrite_minio_url_with_port() -> None:
    url = f"http://minio.test:8389/w2v-compiled-videos/{_RUN_ID}/hash.mp4?{_SIG}"
    out = rewrite_minio_url(_RUN_ID, url)
    # No `?{_SIG}` -- see `test_proxy_path_prepends_api_prefix`.
    assert out == f"/api/runs/{_RUN_ID}/artifact/w2v-compiled-videos/{_RUN_ID}/hash.mp4"


def test_rewrite_minio_url_without_port() -> None:
    # Production config has port 8389 but local/dev may omit it. Rewriter
    # must accept both.
    url = "http://minio.test/w2v-orchestrator/artifacts/x.wav"
    assert rewrite_minio_url(_RUN_ID, url).startswith(f"/api/runs/{_RUN_ID}/artifact/")


def test_rewrite_minio_url_https_scheme() -> None:
    # If TLS gets enabled on MinIO later, the rewriter must still match.
    url = "https://minio.test:8389/w2v-compiled-videos/y.mp4"
    assert (
        rewrite_minio_url(_RUN_ID, url) == f"/api/runs/{_RUN_ID}/artifact/w2v-compiled-videos/y.mp4"
    )


def test_rewrite_minio_url_leaves_external_urls_alone() -> None:
    # Already-proxied URLs, file:// fallbacks from the local backend, and any
    # other CDN-hosted asset must pass through untouched. A regression here
    # would double-proxy or 404 perfectly valid links.
    untouched = [
        f"/api/runs/{_RUN_ID}/artifact/foo",  # already proxied
        "file:///tmp/x.wav",  # local backend
        "https://cdn.example.com/img.png",  # external
        "http://minio-other.test:8389/x",  # wrong host
        "not a url",
    ]
    for u in untouched:
        assert rewrite_minio_url(_RUN_ID, u) == u


def test_rewrite_payload_walks_nested_dicts_and_lists() -> None:
    # Mirrors the real stage-output shapes seen in production:
    # promo_render → flat artifact_url; promo_narrate → clips[].artifact_url.
    payload = {
        "artifact_url": f"http://minio.test:8389/w2v-compiled-videos/x.mp4?{_SIG}",
        "duration_seconds": "4.864",
        "captions_burned": "false",
        "clips": [
            {
                "id": "abc",
                "line_id": "def",
                "seconds": 2.2,
                "artifact_url": ("http://minio.test:8389/w2v-orchestrator/artifacts/run/x.wav"),
            },
            {
                "id": "xyz",
                "line_id": "qrs",
                "seconds": 3.0,
                # Already proxied — must not be touched.
                "artifact_url": f"/api/runs/{_RUN_ID}/artifact/w2v-orchestrator/y.wav",
            },
        ],
        "nested": {"deeply": {"url": "http://minio.test/x"}},
    }

    rewrite_payload(_RUN_ID, payload)

    assert payload["artifact_url"].startswith(f"/api/runs/{_RUN_ID}/artifact/")
    assert "minio.test" not in payload["artifact_url"]
    assert payload["clips"][0]["artifact_url"].startswith(f"/api/runs/{_RUN_ID}/artifact/")
    assert payload["clips"][1]["artifact_url"] == (
        f"/api/runs/{_RUN_ID}/artifact/w2v-orchestrator/y.wav"
    )
    assert payload["nested"]["deeply"]["url"].startswith(f"/api/runs/{_RUN_ID}/artifact/")
    # Non-URL scalars must round-trip unchanged.
    assert payload["duration_seconds"] == "4.864"
    assert payload["clips"][0]["seconds"] == 2.2


def test_rewrite_payload_is_idempotent() -> None:
    # Calling the rewriter twice on the same payload must leave it unchanged
    # the second time — otherwise re-serialization in a downstream caller
    # could mutate the prefix.
    payload = {
        "artifact_url": f"http://minio.test:8389/w2v-compiled-videos/{_RUN_ID}/x.mp4?{_SIG}",
    }
    rewrite_payload(_RUN_ID, payload)
    first = payload["artifact_url"]
    rewrite_payload(_RUN_ID, payload)
    assert payload["artifact_url"] == first


def test_rewrite_payload_handles_none_and_scalars() -> None:
    # Defensive: stage outputs sometimes carry None / numeric primitives at
    # the top level; the rewriter must not crash on them.
    assert rewrite_payload(_RUN_ID, None) is None
    assert rewrite_payload(_RUN_ID, 42) == 42
    assert rewrite_payload(_RUN_ID, "no url here") == "no url here"


# ---------------------------------------------------------------------------
# Collector / rewriter equivalence — the load-bearing invariant
# ---------------------------------------------------------------------------
# ``collect_internal_minio_refs`` is what the artifact proxy uses to decide what a
# run is allowed to serve; ``rewrite_payload`` is what the API uses to decide what
# it advertises. The whole design rests on those two answering the same question
# about the same payload, and until now that was asserted only in a comment: a
# mutant collector that returns ``into`` unread passed every test in the suite
# while production 404'd every artifact. These tests are that mutant's executioner
# — each one fails if the collector walks a different shape than the rewriter, in
# either direction.


def _refs_the_rewriter_found(payload: object) -> set[str]:
    """The ``{bucket}/{key}`` set implied by what ``rewrite_payload`` REWROTE.

    Derived from the rewritten payload, not from the matcher — so it is genuinely
    independent evidence of the rewriter's walk rather than a second call to the
    same code.
    """
    rewritten = rewrite_payload(_RUN_ID, deepcopy(payload))
    found: set[str] = set()
    prefix = f"/api/runs/{_RUN_ID}/artifact/"

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)
        elif isinstance(node, str) and node.startswith(prefix):
            found.add(node[len(prefix) :].split("?", 1)[0])

    _walk(rewritten)
    return found


def _assert_agree(payload: object) -> set[str]:
    collected = collect_internal_minio_refs(payload, set())
    assert collected == _refs_the_rewriter_found(payload)
    return collected


def test_collector_agrees_with_rewriter_on_a_flat_payload() -> None:
    """Mutation killed: ``collect_internal_minio_refs`` returning ``into`` unread
    (or matching on a different host pattern) — the rewriter still finds the URL."""
    payload = {
        "artifact_url": f"http://minio.test:8389/w2v-compiled-videos/x.mp4?{_SIG}",
        "poster": "https://minio.test/w2v-orchestrator/cover.png",
        "external": "https://cdn.example.com/img.png",
        "seconds": 4.2,
    }
    assert _assert_agree(payload) == {
        "w2v-compiled-videos/x.mp4",
        "w2v-orchestrator/cover.png",
    }


def test_collector_agrees_on_nested_lists_inside_dicts() -> None:
    """Mutation killed: a collector that stops descending at the first list, or
    one that walks lists but not the dicts inside them."""
    payload = {
        "clips": [
            {"artifact_url": "http://minio.test/w2v-orchestrator/a.wav"},
            {"variants": [{"url": "http://minio.test:8389/w2v-orchestrator/b.wav"}]},
            ["http://minio.test/w2v-orchestrator/c.wav"],
        ]
    }
    assert _assert_agree(payload) == {
        "w2v-orchestrator/a.wav",
        "w2v-orchestrator/b.wav",
        "w2v-orchestrator/c.wav",
    }


def test_collector_agrees_on_a_deeply_nested_structure() -> None:
    """Mutation killed: any recursion-depth or single-level shortcut in either
    walk — the ref is eight containers down, alternating dict and list."""
    deep: object = "http://minio.test:8389/w2v-audiobooks/deep/book.mp3"
    for level in range(4):
        deep = [{"level": level, "child": deep}]
    payload = {"root": deep}

    assert _assert_agree(payload) == {"w2v-audiobooks/deep/book.mp3"}


def test_collector_and_rewriter_both_ignore_tuples_and_sets() -> None:
    """Agreement includes what they BOTH decline to walk.

    Neither descends into a tuple or a set (JSONB never yields one), and that is
    only safe while it is true of BOTH: a collector that walked tuples would
    authorise a ref the API never proxied; a rewriter that walked them would
    advertise a ref the proxy refuses. Mutation killed: adding ``tuple``/``set``
    to either walk's isinstance chain.
    """
    url = "http://minio.test/w2v-orchestrator/hidden.wav"
    payload = {"as_tuple": (url,), "as_set": {url}}

    assert _assert_agree(payload) == set()


def test_collector_and_rewriter_both_ignore_dict_keys() -> None:
    """A URL used as a KEY is not a value either side may touch.

    Mutation killed: iterating ``.items()``/``.keys()`` instead of ``.values()``
    in the collector — that would authorise a ref the rewriter never emitted, and
    a caller controls their own output's keys.
    """
    url = "http://minio.test/w2v-orchestrator/keyed.wav"
    payload = {url: "value", "nested": {url: {"also": url}}}

    # The URL appears once as a real value (the innermost ``"also"``); the two
    # key positions must contribute nothing.
    assert _assert_agree(payload) == {"w2v-orchestrator/keyed.wav"}


def test_collector_agrees_on_non_str_keys_and_scalars() -> None:
    """Non-``str`` dict keys must not crash or divert either walk.

    Stage outputs round-trip through JSONB as ``str``-keyed, but the walks are
    also fed pre-serialisation dicts. Mutation killed: a collector that assumes
    ``str`` keys (``key.startswith(...)``) and raises on an ``int`` one.
    """
    payload = {
        1: "http://minio.test/w2v-orchestrator/int-keyed.wav",
        None: {"url": "http://minio.test/w2v-orchestrator/none-keyed.wav"},
        "scalars": [None, 42, True, 3.5],
    }
    assert _assert_agree(payload) == {
        "w2v-orchestrator/int-keyed.wav",
        "w2v-orchestrator/none-keyed.wav",
    }


def test_collector_accumulates_into_the_caller_set_across_payloads() -> None:
    """The proxy folds every stage output into ONE set; that accumulation is the
    collector's contract. Mutation killed: allocating a fresh set per call and
    returning it, which would leave the proxy with only the last row's refs."""
    shared: set[str] = set()
    collect_internal_minio_refs({"u": "http://minio.test/w2v-courses/one.mp4"}, shared)
    returned = collect_internal_minio_refs({"u": "http://minio.test/w2v-courses/two.mp4"}, shared)

    assert returned is shared
    assert shared == {"w2v-courses/one.mp4", "w2v-courses/two.mp4"}


def test_a_proxy_path_never_carries_a_presign_the_spa_can_call_expired() -> None:
    """THE fence for the dead-thumbnail defect.

    ``X-Amz-Date`` and ``X-Amz-Expires`` are what the SPA parses, and a proxy
    path that carries them is lying: the proxy re-signs server-side and the
    path has no TTL at all. Asserted on the two parameters the SPA actually
    reads, not just on the whole query, so a partial strip cannot pass.
    """
    dated = (
        "X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Date=20260101T000000Z"
        "&X-Amz-Expires=3600"
        "&X-Amz-Signature=abc"
    )
    out = proxy_path(_RUN_ID, "w2v-compiled-videos/hash.mp4", dated)
    assert "X-Amz-Date" not in out
    assert "X-Amz-Expires" not in out
    assert "X-Amz-Signature" not in out
    assert out.endswith("/artifact/w2v-compiled-videos/hash.mp4")


def test_a_non_presign_query_parameter_is_kept() -> None:
    """Only the SigV4 family is dropped. Anything else a future reader puts on
    the URL is none of this function's business, and silently eating it would
    be a second, quieter version of the same mistake."""
    out = proxy_path(_RUN_ID, "w2v-compiled-videos/hash.mp4", "X-Amz-Expires=3600&page=2")
    assert out.endswith("?page=2")
