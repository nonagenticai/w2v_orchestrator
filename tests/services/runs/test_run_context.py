from uuid import uuid4

from src.services.runs.run_context import RunContext


def test_get_prior_output_present_and_missing() -> None:
    rid = uuid4()
    did = uuid4()
    ctx = RunContext(
        run_id=rid,
        deliverable_id=did,
        config_snapshot={"voice": "crisp"},
        prior_run_outputs={"script_draft": {"lines": [{"id": "l1", "text": "hi"}]}},
        open_annotations_by_stage={"script_draft": [{"text": "tighter"}]},
        tenant_id=uuid4(),
        subject_id=uuid4(),
        correlation_id="abc",
    )
    assert ctx.prior_output("script_draft") == {"lines": [{"id": "l1", "text": "hi"}]}
    assert ctx.prior_output("storyboard_draft") is None
    assert ctx.annotations_for("script_draft")[0]["text"] == "tighter"
    assert ctx.annotations_for("nope") == []
