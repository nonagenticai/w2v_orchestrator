"""site_capture RESUMES after a pod death instead of re-recording.

A recording is a paid video_production job. Before the step ledger the handler
restarted from the top on re-entry, minted NEW part/clip ids and submitted NEW
recordings; the old jobs were orphaned. These tests pin the resume contract:

* a FINISHED part is neither submitted nor polled again;
* a SUBMITTED part is polled by its stored job id, never resubmitted;
* the ids in the stage output are identical to an uninterrupted run's;
* changed capture inputs miss the ledger; a FAILED job is submitted afresh.

The death is simulated with ``tests/support/reentry.py``. site_capture soft-fails
on ``Exception`` by design, so a ``SimulatedPodDeath`` raised inside a client call
would be absorbed as "that clip failed" and the stage would carry on — which is not
what a pod death is. ``_Recorder.attempt`` therefore turns the death into what a
real one looks like from inside the handler: the dying call never returns and the
whole stage task is cancelled.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx

from src.services.runs.run_context import RunContext
from src.services.stages.generic.site_capture import (
    SiteCaptureOutput,
    SiteCaptureStage,
)
from tests.support.reentry import (
    InMemoryStepStore,
    Outbound,
    SimulatedPodDeath,
    ctx_with_ledger,
    run_with_reentry,
)

_STORE = "src.services.stages.generic.site_capture.store_capture_clip"
_STORE_IMG = "src.services.stages.generic.site_capture.store_capture_image"

_BASE = "https://app.test/"  # .test → the page scrape is refused (no network)
_A, _B, _C = f"{_BASE}a", f"{_BASE}b", f"{_BASE}c"
_SECRET = "s3cr3t-token-value"  # pragma: allowlist secret
_OTHER_SECRET = "an0ther-t0ken-value"  # pragma: allowlist secret


def _screens_zip(screen_ids: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for sid in screen_ids:
            zf.writestr(f"{sid}.png", f"png-{sid}".encode())
    return buf.getvalue()


def _gone(job_id: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", f"https://vp.test/api/record/jobs/{job_id}")
    return httpx.HTTPStatusError(
        "404 job not found", request=request, response=httpx.Response(404, request=request)
    )


class _Recorder:
    """A fake video_production recorder built from counting :class:`Outbound` fakes.

    Job ids come from a counter, so a re-submit would be visible as a NEW job id as
    well as an extra ``record_url`` count.
    """

    def __init__(self) -> None:
        self.out = Outbound()
        self.submits: list[dict[str, object]] = []
        self.polled: list[str] = []
        self.job_url: dict[str, str] = {}
        #: urls whose job is still running: ``wait`` never returns for them.
        self.running: set[str] = set()
        #: urls whose job the recorder reports FAILED.
        self.failing: set[str] = set()
        #: job ids the recorder no longer knows (404).
        self.gone: set[str] = set()
        self._died = asyncio.Event()

        self.client = AsyncMock()
        self.client.record_url = self._lethal(self.out.fake("record_url", self._submit))
        self.client.capture_screens = self._lethal(self.out.fake("capture_screens", self._submit))
        self.client.wait_for_content_recording = self._lethal(self.out.fake("wait", self._wait))
        self.client.get_recording_video = self._lethal(self.out.fake("video", self._video))
        self.store_clip = self._lethal(self.out.fake("store", self._store))
        self.stage = SiteCaptureStage(client=self.client)

    def _lethal(self, fake):
        async def _call(*args: object, **kwargs: object) -> object:
            try:
                return await fake(*args, **kwargs)
            except SimulatedPodDeath:
                self._died.set()
                await asyncio.Event().wait()  # the pod is gone: this call never returns

        return _call

    def _submit(self, **kwargs: object) -> str:
        job_id = f"job-{len(self.submits) + 1}"
        self.submits.append(kwargs)
        self.job_url[job_id] = str(kwargs["url"])
        return job_id

    async def _wait(self, job_id: str, **_: object) -> dict[str, object]:
        self.polled.append(job_id)
        if job_id in self.gone:
            raise _gone(job_id)
        url = self.job_url[job_id]
        if url in self.running:
            await asyncio.Event().wait()
        if url in self.failing:
            raise RuntimeError(f"video_production job {job_id} failed: page returned 500")
        return {"status": "completed", "duration_seconds": 4.0}

    def _video(self, job_id: str) -> tuple[bytes, str, None]:
        if self.job_url[job_id] == _BASE:
            return (_screens_zip(["home", "gov"]), "application/zip", None)
        return (f"mp4-{job_id}".encode(), "video/mp4", None)

    @staticmethod
    def _store(**kwargs: object) -> str:
        return f"https://minio.test/cap/{kwargs.get('clip_id') or kwargs.get('image_id')}"

    def submitted_urls(self) -> list[str]:
        return [str(s["url"]) for s in self.submits]

    async def run(self, ctx: RunContext) -> SiteCaptureOutput:
        with patch(_STORE, new=self.store_clip), patch(_STORE_IMG, new=self.store_clip):
            return await self.stage.run({}, ctx=ctx)

    async def attempt(self, ctx: RunContext) -> SiteCaptureOutput:
        """One stage entry; a simulated death CANCELS it, as a real pod death would."""
        task = asyncio.ensure_future(self.run(ctx))
        died = asyncio.ensure_future(self._died.wait())
        await asyncio.wait({task, died}, return_when=asyncio.FIRST_COMPLETED)
        if not self._died.is_set():
            died.cancel()
            return task.result()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._died.clear()
        self.running.clear()  # the recorder finished those jobs while the pod was down
        raise SimulatedPodDeath("stage task cancelled")


def _ctx(site_capture: dict | None = None, **brief: object) -> RunContext:
    return RunContext(
        run_id=uuid4(),
        deliverable_id=uuid4(),
        config_snapshot={
            "brief": {"source_url": _BASE, "site_capture": site_capture or {}, **brief}
        },
    )


_THREE_PATHS = {"paths": ["/a", "/b", "/c"]}


async def test_reentry_polls_the_submitted_part_and_submits_only_the_unstarted_one() -> None:
    """Die with part 1 finished, part 2 submitted (still recording), part 3 not started."""
    ctx = _ctx(_THREE_PATHS)
    rec = _Recorder()
    rec.running = {_B}

    # submit a, wait a, video a, store a, submit b, wait b (hangs) — then submit c dies.
    result = await run_with_reentry(rec.attempt, ctx=ctx, outbound=rec.out, die_after=6)

    assert result.first_attempt_calls == {"record_url": 2, "wait": 2, "video": 1, "store": 1}
    # Re-entry: part 1 is not touched at all, part 2 is POLLED (no submit), part 3 is submitted.
    assert result.second_attempt_calls == {"record_url": 1, "wait": 2, "video": 2, "store": 2}
    assert rec.submitted_urls() == [_A, _B, _C]
    job_of = {url: job for job, url in rec.job_url.items()}
    assert rec.polled[2:] == [job_of[_B], job_of[_C]]
    assert rec.polled.count(job_of[_A]) == 1

    # The output — ids included — is what an uninterrupted run of the same run produces.
    uninterrupted = await _Recorder().run(ctx)
    assert result.output.model_dump() == uninterrupted.model_dump()
    assert [c.path for c in result.output.clips] == ["/a", "/b", "/c"]
    assert len({c.id for c in result.output.clips}) == 3


async def test_every_submit_carries_a_deterministic_idempotency_key() -> None:
    ctx = _ctx(_THREE_PATHS)
    first, second = _Recorder(), _Recorder()
    await first.run(ctx)
    await second.run(ctx)  # no ledger at all: the ids must still be stable

    keys = [s["idempotency_key"] for s in first.submits]
    assert keys == [f"{ctx.run_id}:site_capture:{s['part_id']}" for s in first.submits]
    assert len(set(keys)) == 3
    assert keys == [s["idempotency_key"] for s in second.submits]
    # A different run never shares a key (or a part id) with this one.
    other = _Recorder()
    await other.run(_ctx(_THREE_PATHS))
    assert not set(keys) & {s["idempotency_key"] for s in other.submits}


async def test_finished_capture_makes_no_outbound_call_on_reentry() -> None:
    ctx = ctx_with_ledger(_ctx(_THREE_PATHS), InMemoryStepStore())
    rec = _Recorder()
    first = await rec.run(ctx)
    calls = dict(rec.out.calls)

    again = await rec.run(ctx)

    assert dict(rec.out.calls) == calls
    assert again.model_dump() == first.model_dump()


async def test_changed_inputs_miss_the_ledger() -> None:
    store = InMemoryStepStore()
    base = _ctx({"paths": ["/a"], "viewport": {"width": 1280, "height": 720}})
    rec = _Recorder()
    await rec.run(ctx_with_ledger(base, store))
    assert rec.out.calls["record_url"] == 1

    changed = RunContext(
        run_id=base.run_id,
        deliverable_id=base.deliverable_id,
        config_snapshot={
            "brief": {
                "source_url": _BASE,
                "site_capture": {"paths": ["/a"], "viewport": {"width": 1920, "height": 1080}},
            }
        },
    )
    await rec.run(ctx_with_ledger(changed, store))

    assert rec.out.calls["record_url"] == 2
    assert rec.submits[1]["recording_config"] != rec.submits[0]["recording_config"]


async def test_failed_part_is_resubmitted_on_reentry_and_finished_parts_are_not() -> None:
    ctx = ctx_with_ledger(_ctx(_THREE_PATHS), InMemoryStepStore())
    rec = _Recorder()
    rec.failing = {_B}

    first = await rec.run(ctx)

    # Soft-fail exactly as before: the stage succeeds with the two good clips.
    assert [c.path for c in first.clips] == ["/a", "/c"]
    assert any("recording failed" in n and _B in n for n in first.notes)

    rec.failing.clear()
    second = await rec.run(ctx)

    assert rec.submitted_urls() == [_A, _B, _C, _B]
    assert rec.out.calls == {"record_url": 4, "wait": 4, "video": 3, "store": 3}
    assert [c.path for c in second.clips] == ["/a", "/b", "/c"]
    # A NEW Idempotency-Key: the recorder replays a key's ORIGINAL (failed) job.
    assert rec.submits[3]["idempotency_key"] != rec.submits[1]["idempotency_key"]
    assert rec.submits[3]["part_id"] == rec.submits[1]["part_id"]
    assert second.model_dump()["clips"] == (await _Recorder().run(ctx)).model_dump()["clips"]


async def test_reattached_job_that_vanished_is_resubmitted_in_the_same_entry() -> None:
    """The recorder rolled too: the stored job id answers 404. Re-record, do not stall."""
    ctx = _ctx({"paths": ["/a"]})
    rec = _Recorder()
    rec.running = {_A}
    store = InMemoryStepStore()
    ledger_ctx = ctx_with_ledger(ctx, store)

    task = asyncio.ensure_future(rec.run(ledger_ctx))  # submit a, wait a (hangs) …
    await asyncio.sleep(0.01)
    task.cancel()  # … and the pod dies while the job is recording
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert rec.polled == ["job-1"]
    rec.running.clear()
    rec.gone = {"job-1"}

    out = await rec.run(ledger_ctx)

    assert rec.polled == ["job-1", "job-1", "job-2"]
    assert rec.out.calls["record_url"] == 2
    assert len(out.clips) == 1 and out.capture_status == "ok"
    assert rec.submits[1]["idempotency_key"] != rec.submits[0]["idempotency_key"]


async def test_per_screen_pass_is_polled_not_resubmitted_after_a_death() -> None:
    ctx = _ctx({"screens": [{"capture_id": "home", "label": "Home"}, {"capture_id": "gov"}]})
    rec = _Recorder()

    # capture_screens returns, then the pod dies while waiting on the job.
    result = await run_with_reentry(rec.attempt, ctx=ctx, outbound=rec.out, die_after=1)

    assert result.first_attempt_calls == {"capture_screens": 1}
    assert result.second_attempt_calls == {"wait": 1, "video": 1, "store": 2}
    assert rec.polled == ["job-1"]
    assert (
        rec.submits[0]["idempotency_key"]
        == f"{ctx.run_id}:site_capture:{rec.submits[0]['part_id']}"
    )
    uninterrupted = await _Recorder().run(ctx)
    assert result.output.model_dump() == uninterrupted.model_dump()
    assert [s.id for s in result.output.screens] == ["home", "gov"]

    # A further re-entry of the finished stage records, polls and stores nothing.
    calls = dict(rec.out.calls)
    again = await rec.run(ctx_with_ledger(ctx, result.store))
    assert dict(rec.out.calls) == calls
    assert again.model_dump() == uninterrupted.model_dump()


def _persisted(store: InMemoryStepStore) -> str:
    """Every value the ledger would write for a run, minus the run id itself."""
    rows = sorted(
        ([stage, key, row.input_hash, row.result] for (_, stage, key), row in store.rows.items()),
        key=lambda r: (r[0], r[1]),
    )
    return json.dumps(rows, sort_keys=True)


def _digests(secret: str) -> list[str]:
    """Unsalted digests an offline guesser would try, over the common encodings."""
    forms = (secret.encode(), json.dumps(secret).encode())
    return [
        hashlib.new(algo, form).hexdigest()
        for algo in ("sha256", "sha1", "md5", "sha512")
        for form in forms
    ]


_LOGINS = [
    lambda secret: {"type": "bearer", "token": secret},
    lambda secret: {"type": "form", "username": "demo@example.com", "password": secret},
    lambda secret: {"type": "cookies", "cookies": [{"name": "sid", "value": secret}]},
]


# another user: recorded again


async def test_goto_userinfo_never_reaches_the_ledger_but_is_still_sent() -> None:
    """A ``goto`` may embed ``user:pass@`` basic auth, like ``source_url`` may.

    The hashed copy of the steps is redacted (two runs differing ONLY in the
    embedded password persist byte-identical rows); the steps SENT to the recorder
    keep the credentials, because its browser needs them to load the page.
    """
    run_id = uuid4()

    async def _capture(secret: str) -> tuple[_Recorder, InMemoryStepStore]:
        store = InMemoryStepStore()
        goto = f"https://demo:{secret}@app.test/private"
        screens = {"screens": [{"capture_id": "home"}, {"capture_id": "gov", "goto": goto}]}
        ctx = replace(_ctx(screens), run_id=run_id)
        rec = _Recorder()
        await rec.run(ctx_with_ledger(ctx, store))
        sent = rec.submits[0]["steps"]
        assert isinstance(sent, list)
        assert sent[1]["goto"] == goto  # the credential is still SENT
        assert store.rows, "nothing was ledgered — the assertions below would be vacuous"
        return rec, store

    _, first = await _capture(_SECRET)
    _, second = await _capture(_OTHER_SECRET)
    assert _persisted(first) == _persisted(second), (
        "a persisted ledger value depends on a goto's embedded password"
    )
    stored = _persisted(first)
    assert _SECRET not in stored
    for digest in _digests(_SECRET):
        assert digest not in stored
