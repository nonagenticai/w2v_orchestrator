from src.services.blueprints.registry import get_registry
from src.services.blueprints.website_review import WEBSITE_REVIEW_BLUEPRINT
from src.services.stages.website_review import critique as _critique_mod
from src.services.stages.website_review import redesign as _redesign_mod


def test_poll_timeouts_within_baseserviceclient_cap() -> None:
    # poll_job rejects any deadline > 600s (BaseServiceClient._clamp_timeout).
    # The client is mocked in stage unit tests, so this guards the constants.
    assert _critique_mod._POLL_TIMEOUT_SECONDS <= 600.0
    assert _redesign_mod._POLL_TIMEOUT_SECONDS <= 600.0


def test_website_redesign_stage_timeout_covers_wait_budget() -> None:
    # The stage envelope must be >= start(2x60) + wait((1+retries)x~590) +
    # result(2x60), or the stage clock fires before the wait poll exhausts its
    # retries. Guards the blueprint timeout against the redesign.py retry bump.
    spec = WEBSITE_REVIEW_BLUEPRINT.stage("website_redesign")
    wait_attempts = 1 + _redesign_mod._WAIT_RETRIES
    worst_case = (
        2 * _redesign_mod._START_TIMEOUT_SECONDS
        + wait_attempts * _redesign_mod._POLL_TIMEOUT_SECONDS
        + 2 * _redesign_mod._RESULT_TIMEOUT_SECONDS
    )
    # This heavy stage pins its own explicit ceiling (never inherits None).
    assert spec.default_timeout_seconds is not None
    assert spec.default_timeout_seconds >= worst_case


def test_website_review_registered() -> None:
    assert get_registry().resolve("website_review") is WEBSITE_REVIEW_BLUEPRINT


def test_website_review_topology() -> None:
    order = WEBSITE_REVIEW_BLUEPRINT.topological_order()
    # The critique→redesign→script→record→compile→growth_bundle spine is preserved,
    # and the opt-in site_capture leaf is present before compile (which consumes it).
    spine = [
        "website_critique",
        "website_redesign",
        "website_review_script",
        "website_review_record",
        "website_review_compile",
        "growth_bundle",
    ]
    assert [s for s in order if s in spine] == spine
    assert "site_capture" in order
    assert order.index("site_capture") < order.index("website_review_compile")
    # The opt-in source_ingest leaf runs before the script (which grounds in it)
    # but never gates a plain review run.
    assert "source_ingest" in order
    assert order.index("source_ingest") < order.index("website_review_script")
    # The visual director maps after-act lines to redesign sections; it runs after
    # script/redesign/site_capture and before compile (which consumes its mapping).
    assert "website_review_visual_director" in order
    assert order.index("website_review_script") < order.index("website_review_visual_director")
    assert order.index("website_redesign") < order.index("website_review_visual_director")
    assert order.index("website_review_visual_director") < order.index("website_review_compile")


def test_compile_depends_on_site_capture() -> None:
    compile_spec = next(
        s for s in WEBSITE_REVIEW_BLUEPRINT.stages if s.name == "website_review_compile"
    )
    assert "site_capture" in compile_spec.depends_on
    assert "website_review_record" in compile_spec.depends_on
    assert "website_review_visual_director" in compile_spec.depends_on


def test_visual_director_depends_on_script_redesign_capture() -> None:
    spec = next(
        s for s in WEBSITE_REVIEW_BLUEPRINT.stages if s.name == "website_review_visual_director"
    )
    assert set(spec.depends_on) == {
        "website_review_script",
        "website_redesign",
        "site_capture",
    }


def test_website_review_script_grounds_in_source_ingest() -> None:
    # Opt-in: source_ingest skips cleanly without brief.sources, so this dep
    # never blocks a plain review run; the script appends grounding_prompt_block.
    spec = WEBSITE_REVIEW_BLUEPRINT.stage("website_review_script")
    assert "source_ingest" in spec.depends_on
    assert WEBSITE_REVIEW_BLUEPRINT.stage("source_ingest").depends_on == ()


def test_website_review_has_no_default_review_gates() -> None:
    """Owner decision 2026-09-15: website_review runs STRAIGHT THROUGH.

    It used to pause after ``website_redesign`` and after ``website_review_script``.
    The UI never announced either checkpoint, so the owner's runs "paused a couple of
    times" and read as stalled rather than as waiting for an answer.

    This asserts the two stages carry ``default_pause_after=False`` BY NAME, so it reds
    if either gate comes back — a bare ``paused == set()`` would also pass on a
    blueprint that had lost the stages entirely.
    """
    assert {s.name for s in WEBSITE_REVIEW_BLUEPRINT.stages if s.default_pause_after} == set()
    for name in ("website_redesign", "website_review_script"):
        assert WEBSITE_REVIEW_BLUEPRINT.stage(name).default_pause_after is False, name


def test_website_review_media_discovery_is_an_opt_in_leaf_consumed_by_compile() -> None:
    # A no-dep, skip-clean opt-in leaf now consumed by website_review_compile,
    # which overlays the re-published candidates on the low-value scroll-fallback
    # lines (never the settled cover screens / sections / reveal clips).
    assert WEBSITE_REVIEW_BLUEPRINT.stage("media_discovery").depends_on == ()
    consumers = [
        s.name for s in WEBSITE_REVIEW_BLUEPRINT.stages if "media_discovery" in s.depends_on
    ]
    assert consumers == ["website_review_compile"]


def test_website_review_compile_grounds_in_source_ingest_and_media_discovery() -> None:
    # Ordering-only deps on the two opt-in skip-clean leaves so their
    # re-published media is ready before compile overlays it. Both stay no-dep
    # leaves, so a plain review run is unaffected.
    compile_deps = WEBSITE_REVIEW_BLUEPRINT.stage("website_review_compile").depends_on
    assert "source_ingest" in compile_deps
    assert "media_discovery" in compile_deps
    assert WEBSITE_REVIEW_BLUEPRINT.stage("media_discovery").depends_on == ()


# --- brief.pause_for_review: accepted at the door, INERT in the engine ---------


def _wr_project(brief: dict):
    from src.database.models import Project

    return Project(type="website_review", brief=brief)


# DELETED 2026-09-15, and the reason is the point of the deletion:
# ``test_website_review_pause_for_review_false_autopilots`` and
# ``test_website_review_opt_out_survives_child_gate_restore`` ("the opt-out holds")
# were VACUOUS the moment the gates were removed. An audit deleted the ENTIRE
# ``pause_for_review`` opt-out branch from ``_seed_pause_after_stages`` and all 14
# tests in this file stayed green — the branch could only ever return ``[]`` from a
# seed that was already ``[]``. ``tests/test_vacuous_test_guard.py`` did NOT catch it.
#
# No assertion CAN red on deleting code with no observable effect, so the fix was to
# delete the dead branch (see ``_seed_pause_after_stages``) and these two tests with
# it, rather than leave a test whose NAME claims a guarantee it cannot provide.
# ``brief.pause_for_review`` is inert for website_review and is documented as a no-op
# on ``WebsiteReviewBrief``; what remains under test is that it is still ACCEPTED and
# TYPED at the door (below), and that no gate comes back (the seed test above, proven
# red by restoring either ``default_pause_after=True``).
#
# The gate to re-apply if the gates ever return: the opt-out comes back WITH a test
# that reds when the branch is removed — otherwise it is a lie again.


def test_website_review_brief_door_types_pause_for_review() -> None:
    """The field is INERT but still typed: it must be accepted and coerced at the door.

    This is the honest remainder — it pins the CONTRACT (the FE generates from this
    schema), not a behaviour the field no longer has.
    """
    import pytest
    from pydantic import ValidationError

    from src.api.schemas import ProjectCreate, WebsiteReviewBrief

    assert WebsiteReviewBrief.model_validate({}).pause_for_review is True
    body = {
        "type": "website_review",
        "title": "t",
        "deliverables": [{"name": "d", "format": "16x9"}],
    }
    ProjectCreate.model_validate({**body, "brief": {"pause_for_review": False}})
    with pytest.raises(ValidationError):
        ProjectCreate.model_validate({**body, "brief": {"pause_for_review": "later"}})
