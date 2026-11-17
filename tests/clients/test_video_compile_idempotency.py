"""VideoCompileClient idempotency-key plumbing.

A resumable-timeout re-dispatch re-runs a render stage's whole handler, so the
compile submit MUST carry a key that is STABLE across that re-run — otherwise the
compile service (which dedupes on ``Idempotency-Key``) starts a second, paid,
non-idempotent render. These drive the client through an ``httpx.MockTransport``
that records the submit headers.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.clients.video_compile import _RESUBMIT_FLOOR_SECONDS, VideoCompileClient

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*_a: object, **_k: object) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop)


def _client(handler) -> VideoCompileClient:
    client = VideoCompileClient(base_url="http://compile.test", license_token="t")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        timeout=client.timeout,
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )
    return client


def _recording_handler(submit_keys: list[str | None]):
    """POST → record its Idempotency-Key + return job-N; GET → completed."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            submit_keys.append(request.headers.get("Idempotency-Key"))
            return httpx.Response(200, json={"job_id": f"job-{len(submit_keys)}"})
        return httpx.Response(
            200,
            json={"status": "completed", "artifact_url": "http://x/y.mp4", "duration_seconds": 1.0},
        )

    return handler


async def test_stable_key_is_forwarded_on_the_wire() -> None:
    keys: list[str | None] = []
    client = _client(_recording_handler(keys))
    await client.compile(shots=[], idempotency_key="run-abc:demo_video_compile")
    assert keys == ["run-abc:demo_video_compile"]
    await client.close()


async def test_concat_audio_forwards_stable_key() -> None:
    keys: list[str | None] = []
    client = _client(_recording_handler(keys))
    await client.concat_audio(clips=[], idempotency_key="run-abc:audiobook_stitch:combined")
    assert keys == ["run-abc:audiobook_stitch:combined"]
    await client.close()


async def test_an_unkeyed_submit_is_refused_outright() -> None:
    """INVERTED, deliberately. This test used to assert the opposite.

    It read ``test_none_key_falls_back_to_a_fresh_uuid_per_submit`` and pinned
    that two unkeyed submits got two DIFFERENT random uuids. That behaviour
    looked keyed and deduped nothing:

    * two random keys cannot dedupe against each other, so a job-lost resubmit
      or a resume re-dispatch paid for a second render; and
    * ``cancel_run_jobs`` finds a run's live compiles by the ``f"{run_id}:"``
      key PREFIX — the orchestrator stores no compile job_ids durably — so a
      uuid key matched nothing and the render was UNCANCELLABLE. A cancelled run
      kept rendering, and kept billing, while cancel reported zero jobs flipped.

    Every production caller already passed a stable key, so this was a landmine
    rather than a live hole. The key is now required: ``ty`` rejects the omission
    at the call site, and a falsy one raises here.
    """
    keys: list[str | None] = []
    client = _client(_recording_handler(keys))
    with pytest.raises(ValueError, match="idempotency_key is required"):
        await client.compile(shots=[], idempotency_key="")
    assert keys == [], "the submit must be refused BEFORE any request is sent"
    await client.close()


async def test_stable_key_reused_across_a_job_loss_resubmit() -> None:
    """A stable key must ride BOTH the first submit and the job-lost resubmit.

    Otherwise a resubmit after a pod loss would re-key and could double-render.
    """
    keys: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            keys.append(request.headers.get("Idempotency-Key"))
            return httpx.Response(200, json={"job_id": f"job-{len(keys)}"})
        # First job: always 404 (lost) → triggers exactly one resubmit.
        # Second job: completed.
        if len(keys) == 1:
            return httpx.Response(404, json={"detail": "job not found"})
        return httpx.Response(
            200,
            json={"status": "completed", "artifact_url": "http://x/y.mp4", "duration_seconds": 1.0},
        )

    client = _client(handler)
    # Long timeout so the resubmit floor is not hit.
    await client.compile(shots=[], idempotency_key="run-xyz:demo_video_compile", timeout=3600.0)
    assert len(keys) == 2, "expected a first submit + exactly one job-lost resubmit"
    assert keys[0] == keys[1] == "run-xyz:demo_video_compile"
    _ = _RESUBMIT_FLOOR_SECONDS  # referenced to document the budget guard
    await client.close()
