"""Tests for the orchestrator StorageService — the domain adapter over the shared
``w2v_common.storage.StorageService`` facade.

The facade's object-store mechanics (put/get/list/delete, presign + external
rewrite, object_exists tri-state, resign %-decoding, streaming range reads) are
tested in ``w2v_common``. Here we test only the orchestrator's *domain* surface:
run/stage key shapes, delegation to the facade, ``file://`` URL construction, the
media-cache lifecycle GC, ``close``, and the streaming pass-through.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple, cast
from unittest.mock import AsyncMock

import pytest

from src.services import storage_service as storage_module
from src.services.storage_service import (
    MAX_COMPOSE_PARTS,
    MIN_COMPOSE_PART_BYTES,
    ComposeConstraintError,
    StorageService,
    is_readable_w2v_bucket,
)

if TYPE_CHECKING:
    # Type-only. The classes are imported lazily inside ``_lifecycle_types``
    # below, mirroring the adapter: the module under test tolerates
    # ``miniopy-async`` being absent, so this file must not make it a hard
    # module-level dependency either.
    from miniopy_async.lifecycleconfig import Expiration, Filter, LifecycleConfig, Rule


def _adapter(backend: str = "minio") -> tuple[StorageService, AsyncMock]:
    """A StorageService with __init__ bypassed and the facade swapped for an
    AsyncMock. Returns ``(service, facade_mock)`` — assert on the mock."""
    svc = StorageService.__new__(StorageService)
    svc._backend = backend
    svc._local_path = Path("/data")
    svc._minio_bucket = "w2v-orchestrator"
    svc._media_cache_cross_run = True
    svc._media_cache_ttl_days = 30
    facade = AsyncMock()
    svc._facade = facade
    return svc, facade


# ── key shapes + delegation ─────────────────────────────────────────


class TestKeyShapesAndDelegation:
    async def test_store_stage_artifact_key_and_content_type(self):
        svc, facade = _adapter()
        facade.put_bytes.return_value = "artifacts/j1/narrate/output.json"
        ref = await svc.store_stage_artifact("j1", "narrate", b"{}")
        assert ref == "artifacts/j1/narrate/output.json"
        facade.put_bytes.assert_awaited_once_with(
            "artifacts/j1/narrate/output.json", b"{}", content_type="application/json"
        )

    async def test_retrieve_stage_artifact_key(self):
        svc, facade = _adapter()
        facade.get_bytes.return_value = b"data"
        assert await svc.retrieve_stage_artifact("j1", "narrate", "clip.wav") == b"data"
        facade.get_bytes.assert_awaited_once_with("artifacts/j1/narrate/clip.wav")

    async def test_store_final_video_key_and_content_type(self):
        svc, facade = _adapter()
        await svc.store_final_video("j1", b"mp4bytes", fmt="mp4")
        facade.put_bytes.assert_awaited_once_with(
            "videos/j1/final.mp4", b"mp4bytes", content_type="video/mp4"
        )

    async def test_put_object_default_content_type(self):
        svc, facade = _adapter()
        await svc.put_object("sound_library/beep.wav", b"x")
        facade.put_bytes.assert_awaited_once_with(
            "sound_library/beep.wav", b"x", content_type="application/octet-stream"
        )

    async def test_delete_and_list_job_artifacts_prefix(self):
        svc, facade = _adapter()
        await svc.delete_job_artifacts("j1")
        facade.delete_prefix.assert_awaited_once_with("artifacts/j1/")
        await svc.list_job_artifacts("j1")
        facade.list_prefix.assert_awaited_once_with("artifacts/j1/")

    async def test_generic_delegation(self):
        svc, facade = _adapter()
        await svc.get_object_bytes("k")
        facade.get_bytes.assert_awaited_once_with("k")
        await svc.list_object_keys("pre/")
        facade.list_prefix.assert_awaited_once_with("pre/")
        await svc.object_exists("http://minio/w2v-orchestrator/k")
        facade.object_exists.assert_awaited_once_with("http://minio/w2v-orchestrator/k")
        await svc.resign_presigned_url("http://minio/w2v-orchestrator/k?sig")
        facade.resign_presigned_url.assert_awaited_once_with("http://minio/w2v-orchestrator/k?sig")
        await svc.check_health()
        facade.check_health.assert_awaited_once()

    async def test_get_presigned_url_delegates_expires_and_bucket(self):
        svc, facade = _adapter()
        facade.presign_get.return_value = "https://minio/x?sig"
        out = await svc.get_presigned_url("k", expires=120, bucket="w2v-compiled-videos")
        assert out == "https://minio/x?sig"
        facade.presign_get.assert_awaited_once_with("k", expires=120, bucket="w2v-compiled-videos")


# ── URL construction (backend-dependent) ────────────────────────────


class TestArtifactUrls:
    async def test_stage_artifact_url_local_is_file_uri(self):
        svc, _ = _adapter(backend="local")
        url = await svc.stage_artifact_url("j1", "narrate", "clip.wav")
        assert url == "file:///data/artifacts/j1/narrate/clip.wav"

    async def test_stage_artifact_url_minio_presigns(self):
        svc, facade = _adapter(backend="minio")
        facade.presign_get.return_value = "https://minio/x?sig"
        url = await svc.stage_artifact_url("j1", "narrate", "clip.wav")
        assert url == "https://minio/x?sig"
        facade.presign_get.assert_awaited_once_with("artifacts/j1/narrate/clip.wav")

    async def test_object_url_local_and_minio(self):
        svc_l, _ = _adapter(backend="local")
        assert await svc_l.object_url("media-cache/x.wav") == "file:///data/media-cache/x.wav"
        svc_m, facade = _adapter(backend="minio")
        facade.presign_get.return_value = "https://minio/y?sig"
        assert await svc_m.object_url("media-cache/x.wav") == "https://minio/y?sig"
        facade.presign_get.assert_awaited_once_with("media-cache/x.wav")


# ── media-cache lifecycle GC + close ────────────────────────────────
#
# ``set_bucket_lifecycle`` REPLACES the bucket's entire lifecycle document; it
# does not merge. The live ``w2v-orchestrator`` bucket carries an
# operator-applied whole-bucket retention rule that exists in no git repo, next
# to ``tenants/`` (user uploads) and ``sound_library/`` (music beds). So the
# assertions below are data-loss fences, not coverage: if the merge is removed
# and the function goes back to writing a single-rule document, the FOREIGN-rule
# tests go red.


class _AbsentLifecycleError(Exception):
    """An error carrying the S3 code for "this bucket has no lifecycle config".

    A stand-in for ``S3Error``: the branch under test keys off ``exc.code``, and
    building a real ``S3Error`` needs an aiohttp response (see
    ``_no_such_key_error`` below for how much ceremony that costs).
    """

    code = "NoSuchLifecycleConfiguration"


class _LifecycleTypes(NamedTuple):
    """The four lifecycle classes as ONE object, reached by attribute.

    Deliberately not a 4-tuple the call sites unpack: unpacking would bind
    CapWords class names as local variables (N806), and every call site needs a
    different subset, so the unused ones had to be spelled ``_Filter``/``_Rule``
    just to keep the positions lined up. Attribute access needs neither.

    Field names are lowercase for the same reason — they are values here, not
    the classes' own names.
    """

    expiration: type[Expiration]
    rule_filter: type[Filter]
    config: type[LifecycleConfig]
    rule: type[Rule]


def _lifecycle_types() -> _LifecycleTypes:
    """The real ``miniopy_async`` lifecycle classes the adapter builds with.

    Foreign rules are constructed from the REAL classes rather than mocks so the
    preservation assertion is about the objects the client actually returns —
    a hand-rolled stub could satisfy a merge that the real ``Rule`` would not.

    Imported inside the function, not at module scope, because the module under
    test imports them lazily too (a ``local``-backend consumer never installs
    ``miniopy-async``).
    """
    from miniopy_async.lifecycleconfig import Expiration, Filter, LifecycleConfig, Rule

    return _LifecycleTypes(Expiration, Filter, LifecycleConfig, Rule)


def _foreign_rule(rule_id: str = "d8129rcus7pm44a84qu0", days: int = 60):
    """The shape of the live operator rule: whole bucket, N-day expiry."""
    lifecycle = _lifecycle_types()
    return lifecycle.rule(
        rule_id=rule_id,
        status="Enabled",
        rule_filter=lifecycle.rule_filter(prefix=""),
        expiration=lifecycle.expiration(days=days),
    )


def _our_rule(days: int = 30, status: str = "Enabled", prefix: str = "media-cache/"):
    """A rule carrying the id this service owns."""
    lifecycle = _lifecycle_types()
    return lifecycle.rule(
        rule_id="w2v-media-cache-expiry",
        status=status,
        rule_filter=lifecycle.rule_filter(prefix=prefix),
        expiration=lifecycle.expiration(days=days),
    )


def _written_rules(client: AsyncMock) -> list:
    """The rules of the ``LifecycleConfig`` handed to ``set_bucket_lifecycle``."""
    return list(client.set_bucket_lifecycle.call_args.args[1].rules)


class TestMediaCacheLifecycleAndClose:
    async def test_lifecycle_sets_rule_when_enabled(self):
        svc, facade = _adapter()
        facade._minio_client = AsyncMock()
        facade._minio_client.get_bucket_lifecycle.return_value = None
        await svc._ensure_media_cache_lifecycle()
        facade._minio_client.set_bucket_lifecycle.assert_awaited_once()
        assert facade._minio_client.set_bucket_lifecycle.call_args.args[0] == "w2v-orchestrator"

    async def test_existing_foreign_rule_is_preserved(self):
        """THE FENCE: the operator's whole-bucket retention rule survives.

        Remove the merge (go back to writing a one-rule config) and this reds:
        the written document would contain only ``w2v-media-cache-expiry``, and
        in prod MinIO would drop a 60-day retention rule that exists in no repo
        and guards ``tenants/`` + ``sound_library/``.
        """
        lifecycle = _lifecycle_types()
        svc, facade = _adapter()
        client = AsyncMock()
        facade._minio_client = client
        foreign = _foreign_rule()
        client.get_bucket_lifecycle.return_value = lifecycle.config([foreign])

        await svc._ensure_media_cache_lifecycle()

        written = _written_rules(client)
        # The foreign rule is preserved as the SAME object, not a rebuilt copy.
        assert foreign in written
        assert [r.rule_id for r in written] == [
            "d8129rcus7pm44a84qu0",
            "w2v-media-cache-expiry",
        ]

    async def test_our_own_rule_is_replaced_not_duplicated(self):
        """Re-running against a stale copy of our rule updates it in place.

        Two rules with the same id is an invalid document; appending blindly
        would also leave the OLD ttl live alongside the new one.
        """
        lifecycle = _lifecycle_types()
        svc, facade = _adapter()
        svc._media_cache_ttl_days = 45
        client = AsyncMock()
        facade._minio_client = client
        foreign = _foreign_rule()
        client.get_bucket_lifecycle.return_value = lifecycle.config([foreign, _our_rule(days=30)])

        await svc._ensure_media_cache_lifecycle()

        written = _written_rules(client)
        assert [r.rule_id for r in written] == [
            "d8129rcus7pm44a84qu0",
            "w2v-media-cache-expiry",
        ]
        ours = [r for r in written if r.rule_id == "w2v-media-cache-expiry"]
        assert len(ours) == 1
        assert ours[0].expiration.days == 45
        assert foreign in written

    async def test_no_existing_config_writes_only_our_rule(self):
        """An empty bucket policy (client returns ``None``) is not an error."""
        svc, facade = _adapter()
        client = AsyncMock()
        facade._minio_client = client
        client.get_bucket_lifecycle.return_value = None

        await svc._ensure_media_cache_lifecycle()

        written = _written_rules(client)
        assert [r.rule_id for r in written] == ["w2v-media-cache-expiry"]
        assert written[0].expiration.days == 30
        assert written[0].rule_filter.prefix == "media-cache/"

    async def test_absent_config_raised_as_error_is_treated_as_empty(self):
        """Some S3 clients RAISE ``NoSuchLifecycleConfiguration`` instead of
        returning ``None``. That is still "no config", so we proceed."""
        svc, facade = _adapter()
        client = AsyncMock()
        facade._minio_client = client
        client.get_bucket_lifecycle.side_effect = _AbsentLifecycleError("no lifecycle")

        await svc._ensure_media_cache_lifecycle()

        assert [r.rule_id for r in _written_rules(client)] == ["w2v-media-cache-expiry"]

    async def test_unreadable_existing_config_abstains_from_writing(self):
        """A read failure we cannot classify must NOT trigger a write.

        Writing here would replace a configuration we never saw — precisely the
        data loss this function guards against. Never fatal either: no raise.
        """
        svc, facade = _adapter()
        client = AsyncMock()
        facade._minio_client = client
        client.get_bucket_lifecycle.side_effect = RuntimeError("500 from minio")

        await svc._ensure_media_cache_lifecycle()

        client.set_bucket_lifecycle.assert_not_awaited()

    async def test_identical_config_skips_the_write_entirely(self):
        """A no-op write is a needless chance to lose someone else's rule."""
        lifecycle = _lifecycle_types()
        svc, facade = _adapter()
        client = AsyncMock()
        facade._minio_client = client
        client.get_bucket_lifecycle.return_value = lifecycle.config(
            [_foreign_rule(), _our_rule(days=30)]
        )

        await svc._ensure_media_cache_lifecycle()

        client.set_bucket_lifecycle.assert_not_awaited()

    async def test_lifecycle_skipped_when_disabled(self):
        svc, facade = _adapter()
        svc._media_cache_cross_run = False
        facade._minio_client = AsyncMock()
        await svc._ensure_media_cache_lifecycle()
        facade._minio_client.set_bucket_lifecycle.assert_not_awaited()
        # Short-circuits BEFORE any read — a disabled cache touches nothing.
        facade._minio_client.get_bucket_lifecycle.assert_not_awaited()

    async def test_lifecycle_skipped_when_ttl_not_positive(self):
        svc, facade = _adapter()
        svc._media_cache_ttl_days = 0
        facade._minio_client = AsyncMock()
        await svc._ensure_media_cache_lifecycle()
        facade._minio_client.get_bucket_lifecycle.assert_not_awaited()
        facade._minio_client.set_bucket_lifecycle.assert_not_awaited()

    async def test_close_closes_session(self):
        svc, facade = _adapter()
        facade._minio_client = AsyncMock()
        await svc.close()
        facade._minio_client.close_session.assert_awaited_once()


# ── streaming pass-through (artifact proxy) ─────────────────────────


class TestStreaming:
    async def test_get_object_stream_delegates(self):
        svc, facade = _adapter()
        sentinel = object()
        facade.get_object_stream.return_value = sentinel
        out = await svc.get_object_stream(
            "w2v-audiobooks", "book/ch1.mp3", request_headers={"Range": "bytes=0-9"}
        )
        assert out is sentinel
        facade.get_object_stream.assert_awaited_once_with(
            "book/ch1.mp3", bucket="w2v-audiobooks", request_headers={"Range": "bytes=0-9"}
        )


# ── chunked-upload assembly: compose_object + get_file ──────────────
#
# Cloudflare 413s any body over 100 MB, so a 134 MB source clip is uploaded as
# 32 MiB parts (one object each) and stitched server-side. These two methods are
# that stitch; every test below names the product failure a revert reintroduces.


def _minio_adapter() -> tuple[StorageService, AsyncMock]:
    """A minio-backend adapter with the facade's private MinIO client mocked.

    compose/fget are the two operations the shared ``w2v_common`` facade does not
    model, so the adapter reaches ``_facade._minio_client`` — the same sanctioned
    hatch the media-cache GC and ``close`` use. Returns ``(service, client_mock)``.
    """
    svc, facade = _adapter(backend="minio")
    client = AsyncMock()
    facade._minio_client = client
    return svc, client


def _stat(size: int) -> SimpleNamespace:
    """A ``stat_object`` result carrying just the field compose validation reads."""
    return SimpleNamespace(size=size)


def _no_such_key_error() -> BaseException:
    """A real ``S3Error`` with ``code == "NoSuchKey"``.

    Constructed positionally against the documented signature, with a fallback for
    signature drift across ``miniopy-async`` releases — the assertion under test is
    "``get_file`` maps NoSuchKey to False", not "we can call this constructor".
    """
    from miniopy_async.error import S3Error

    # Built through ``__new__`` rather than the constructor: the final positional
    # parameter is annotated ``ClientResponse`` and the only thing this fixture
    # needs is ``code == "NoSuchKey"``. Passing ``None`` there is what the library
    # itself does at its raise sites, but it is a type error to write here, and the
    # repo bans ``ty: ignore``. Bypassing __init__ keeps the fixture honest about
    # what it actually is: the narrowest object that satisfies the code path under
    # test, not a faithful S3Error.
    exc = S3Error.__new__(S3Error)
    exc._code = "NoSuchKey"
    return exc


class TestComposeObject:
    async def test_compose_concatenates_parts_in_the_order_given_not_sorted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The composed object's bytes are the parts in the CALLER's order.

        Reverted (or "helpfully" sorted anywhere up the call chain) ⇒ silently
        corrupt video: a lexicographic sort puts ``part10`` before ``part2``, the
        object is still exactly the right size and uploads fine, ``ffprobe``
        returns ``{}``, and the DAG carries a permanently zeroed clip duration
        downstream with no error anywhere. The part names here are chosen so a
        sort would visibly reorder them.

        The 5 MiB floor is patched down so this stays a fast unit test — the floor
        itself is asserted at its real value in the next two tests.
        """
        monkeypatch.setattr(storage_module, "MIN_COMPOSE_PART_BYTES", 4)
        svc, _ = _adapter(backend="local")
        svc._local_path = tmp_path
        for name, body in (("part1", b"AAAA"), ("part2", b"BBBB"), ("part10", b"CCCC")):
            (tmp_path / f"up/{name}").parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / f"up/{name}").write_bytes(body)

        ref = await svc.compose_object(
            "up/final.mp4", ["up/part1", "up/part2", "up/part10"], content_type="video/mp4"
        )

        assert Path(ref).read_bytes() == b"AAAABBBBCCCC"
        assert ref == str(tmp_path / "up/final.mp4")

    async def test_undersized_non_final_part_rejected_before_the_compose_call(self):
        """A non-final part below 5 MiB raises BEFORE MinIO is asked to compose.

        Reverted ⇒ MinIO answers ``EntityTooSmall`` only when it tries to assemble
        the object — i.e. after all 134 MB have already been PUT. The upload is
        spent and the only recovery is to redo every part. The guard costs one
        metadata HEAD per part and turns that into an actionable error naming the
        offending key. ``stat_object`` (a HEAD) is expected; ``compose_object``
        (the expensive, failing call) must NOT be reached.
        """
        svc, client = _minio_adapter()
        client.stat_object.side_effect = [
            _stat(MIN_COMPOSE_PART_BYTES),
            _stat(MIN_COMPOSE_PART_BYTES - 1),  # non-final and short → illegal
            _stat(17),
        ]

        with pytest.raises(ComposeConstraintError) as excinfo:
            await svc.compose_object("final.mp4", ["p0", "p1", "p2"])

        assert "p1" in str(excinfo.value)
        client.compose_object.assert_not_awaited()

    async def test_final_part_may_be_smaller_than_the_floor(self):
        """The LAST part is exempt from the 5 MiB floor (S3 allows a short final
        part). Reverted ⇒ every real upload is rejected, since the tail chunk is
        almost never an exact multiple of the part size."""
        svc, client = _minio_adapter()
        client.stat_object.side_effect = [_stat(MIN_COMPOSE_PART_BYTES), _stat(11)]

        assert await svc.compose_object("final.mp4", ["p0", "p1"]) == "final.mp4"
        client.compose_object.assert_awaited_once()

    async def test_minio_compose_sources_are_the_callers_keys_in_the_callers_order(self):
        """The ``ComposeSource`` list handed to MinIO is the caller's keys, in the
        caller's ORDER — asserted key by key, not merely "a compose happened".

        Reverted (e.g. ``source_keys = sorted(source_keys)`` slipped into
        ``_compose_minio``) ⇒ a right-sized, shuffled, undecodable video: MinIO
        concatenates in the order it is given, a lexical sort puts ``00010``-style
        keys before ``00002`` once an upload passes nine parts, ``ffprobe`` returns
        ``{}``, and the clip carries a permanently zeroed duration downstream.

        This is the ONLY test that can catch that mutation on the production path.
        The byte-level ordering test above exercises ``_compose_local``, which prod
        never runs, and every other minio test here asserts ``assert_awaited_once``
        with no arguments — a sorted-keys mutation leaves all of them green.
        """
        svc, client = _minio_adapter()
        # Deliberately in an order a sort would visibly change, with a legal size
        # for every non-final part so the plan reaches the compose call at all.
        keys = ["up/00001", "up/00010", "up/00002"]
        client.stat_object.side_effect = [_stat(MIN_COMPOSE_PART_BYTES)] * 2 + [_stat(7)]

        await svc.compose_object("up/original.mp4", keys, content_type="video/mp4")

        bucket, key, sources = client.compose_object.await_args.args
        assert bucket == "w2v-orchestrator"
        assert key == "up/original.mp4"
        assert [source.object_name for source in sources] == keys
        assert [source.bucket_name for source in sources] == ["w2v-orchestrator"] * 3
        assert client.compose_object.await_args.kwargs["metadata"] == {"Content-Type": "video/mp4"}
        # The stat sweep must visit the same keys in the same order — it is what
        # pairs each size with its key in the part-size guard's error message.
        assert [call.args[1] for call in client.stat_object.await_args_list] == keys

    async def test_more_than_max_parts_rejected_without_touching_minio(self):
        """More than 10,000 sources raises before any MinIO call at all.

        Reverted ⇒ S3 rejects the compose (multipart caps at 10,000 parts) only
        after the parts are uploaded. The count check runs before the stat sweep,
        so a pathological plan costs zero round-trips.
        """
        svc, client = _minio_adapter()

        with pytest.raises(ComposeConstraintError):
            await svc.compose_object("final.mp4", [f"p{i}" for i in range(MAX_COMPOSE_PARTS + 1)])

        client.stat_object.assert_not_awaited()
        client.compose_object.assert_not_awaited()

    async def test_empty_source_list_rejected(self):
        """Zero parts is a caller bug, not an empty object. Reverted ⇒ the route
        happily "composes" a run's source clip out of nothing and hands the DAG a
        key that either 404s or points at a zero-byte file."""
        svc, client = _minio_adapter()
        with pytest.raises(ComposeConstraintError):
            await svc.compose_object("final.mp4", [])
        client.compose_object.assert_not_awaited()


class TestGetFile:
    async def test_get_file_writes_object_to_path_and_returns_true(self, tmp_path: Path):
        """The object lands on disk at the requested path. Reverted ⇒ the
        post-compose ``ffprobe`` has no file to probe and every chunked upload
        reports a zero duration."""
        svc, _ = _adapter(backend="local")
        svc._local_path = tmp_path
        (tmp_path / "src.mp4").write_bytes(b"video-bytes")
        dest = tmp_path / "nested/out.mp4"

        assert await svc.get_file("src.mp4", str(dest)) is True
        assert dest.read_bytes() == b"video-bytes"

    async def test_get_file_returns_false_for_missing_key_without_raising(self, tmp_path: Path):
        """A missing object is a return value, not an exception. Reverted ⇒ the
        caller's "did the parts survive?" probe unwinds the whole request instead
        of branching, turning a recoverable absence into a 500."""
        svc, _ = _adapter(backend="local")
        svc._local_path = tmp_path
        assert await svc.get_file("nope.mp4", str(tmp_path / "out.mp4")) is False

    async def test_get_file_returns_false_on_minio_no_such_key(self, tmp_path: Path):
        """Same contract on the minio backend: ``NoSuchKey`` → False, not a raise.
        Reverted ⇒ the probe 500s in prod while passing on the local backend."""
        svc, client = _minio_adapter()
        client.fget_object.side_effect = _no_such_key_error()
        assert await svc.get_file("k.mp4", str(tmp_path / "out.mp4")) is False

    async def test_get_file_streams_via_fget_object_and_never_reads_bytes(self, tmp_path: Path):
        """``get_file`` MUST use the streaming ``fget_object`` path, never
        ``get_bytes``/``get_object_bytes``.

        Reverted ⇒ a 134 MiB 4K drone clip is materialised in the orchestrator's
        heap. This pod is memory-sized for JSON stage artifacts, so one concurrent
        upload is an OOMKill — which reads as an unexplained crash loop, nowhere
        near the upload in the logs. Asserting the streaming API was used (and the
        buffering one was not) is the only way to catch a "simplification" back to
        ``write_bytes(await get_object_bytes(key))``, which passes every
        behavioural assertion.
        """
        svc, client = _minio_adapter()
        dest = tmp_path / "out.mp4"

        assert await svc.get_file("uploads/final.mp4", str(dest)) is True

        client.fget_object.assert_awaited_once_with(
            "w2v-orchestrator", "uploads/final.mp4", str(dest)
        )
        client.get_object.assert_not_awaited()
        # Reached through the local name rather than ``svc._facade.get_bytes``:
        # the attribute is declared as the real facade's bound method, so the
        # checker cannot see the AsyncMock the fixture installed. Asserting on the
        # mock we put there is the same assertion without lying to the type system.
        facade_get_bytes = cast(AsyncMock, svc._facade).get_bytes
        facade_get_bytes.assert_not_awaited()


class TestObjectSize:
    async def test_object_size_returns_the_size_and_none_when_absent_local(self, tmp_path: Path):
        """Present ⇒ size, absent ⇒ ``None``.

        Reverted (or made to raise on absence) ⇒ the chunked upload's ``complete``
        cannot ask "is the final object already composed?", so a retry after a
        SIGTERM between compose and commit recomposes from parts that may already
        be swept — a permanent 500 on footage that is in fact safely stored.
        """
        svc, _ = _adapter(backend="local")
        svc._local_path = tmp_path
        (tmp_path / "original.mp4").write_bytes(b"0123456789")

        assert await svc.object_size("original.mp4") == 10
        assert await svc.object_size("nope.mp4") is None

    async def test_object_size_is_a_stat_not_a_read(self):
        """The probe is a metadata HEAD, never a download.

        Reverted (``get_object_bytes(key) is not None``) ⇒ every ``complete`` pulls
        the whole 134 MiB composed clip into the orchestrator's heap just to learn
        that it exists — in a pod memory-sized for JSON stage artifacts, i.e. an
        OOMKill that reads as an unrelated crash loop.
        """
        svc, client = _minio_adapter()
        client.stat_object.return_value = _stat(134 * 1024 * 1024)

        assert await svc.object_size("uploads/original.mp4") == 134 * 1024 * 1024

        client.stat_object.assert_awaited_once_with("w2v-orchestrator", "uploads/original.mp4")
        client.fget_object.assert_not_awaited()
        cast(AsyncMock, svc._facade).get_bytes.assert_not_awaited()

    async def test_object_size_returns_none_on_no_such_key_but_raises_otherwise(self):
        """``NoSuchKey`` ⇒ ``None``; any other S3 error PROPAGATES.

        Reverted (swallow every error as "absent") ⇒ a transport blip during a
        retry is read as "not composed yet", so the handler recomposes over a good
        object — or over parts a previous attempt already swept, turning a
        recoverable outage into a dead upload.
        """
        svc, client = _minio_adapter()
        client.stat_object.side_effect = _no_such_key_error()
        assert await svc.object_size("gone.mp4") is None

        client.stat_object.side_effect = RuntimeError("connection reset")
        with pytest.raises(RuntimeError):
            await svc.object_size("gone.mp4")


# ── shared bucket predicate (anti-drift with the artifact proxy) ────


def test_is_readable_w2v_bucket_predicate():
    assert is_readable_w2v_bucket("w2v-audiobooks", "w2v-orchestrator") is True
    assert is_readable_w2v_bucket("w2v-orchestrator", "w2v-orchestrator") is True
    assert is_readable_w2v_bucket("legacy-own", "legacy-own") is True
    assert is_readable_w2v_bucket("secret-bucket", "w2v-orchestrator") is False
    assert is_readable_w2v_bucket("", "w2v-orchestrator") is False
