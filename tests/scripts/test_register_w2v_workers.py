"""Guard the w2v worker-registration list — the WM workers the fleet depends on."""

from __future__ import annotations

from scripts.register_w2v_workers import (
    SEED_MANAGED,
    W2V_WORKER_NAMES,
    WORKER_MAX_TOKENS,
)


def test_seo_micro_llm_workers_registered() -> None:
    # The three SEO micro-LLM workers must be registered here so include_llm /
    # the content-gap analyzer don't 500 on WM's unreachable vllm auto-create.
    for name in ("seo-intent-classify", "seo-eeat-depth", "seo-content-gap"):
        assert name in W2V_WORKER_NAMES


def test_seo_content_gap_needs_no_seed_or_max_tokens_override() -> None:
    # Tiny structured output (F2): no seed ownership, no max_tokens override —
    # the provider default is ample.
    assert "seo-content-gap" not in SEED_MANAGED
    assert "seo-content-gap" not in WORKER_MAX_TOKENS


def test_phantom_workers_are_not_declared() -> None:
    # No invoker ⇒ no registration. ``story-architect`` never had one —
    # registering an uncalled worker just recreates the phantom.
    assert "story-architect" not in W2V_WORKER_NAMES
    assert "story-architect" not in SEED_MANAGED


def test_reconcile_timeline_relief_workers_are_registered_not_seeded() -> None:
    # The A22 ``reconcile_timeline`` C13 relief loop invokes ``pacing-critic`` (rhythm
    # score), ``timing-director`` (directive proposer), and ``beat-rewriter`` (per-beat
    # shortening) with inline system prompts, so this script must PUT them with
    # ``model_provider=inference_router`` (declared, NOT seed-managed ⇒ they land in
    # ``to_register`` and are not clobbered by / dependent on the WM seed).
    for name in ("pacing-critic", "timing-director", "beat-rewriter"):
        assert name in W2V_WORKER_NAMES
        assert name not in SEED_MANAGED


def test_redesign_workers_are_seed_managed() -> None:
    # WM's seed owns their curated model + system_prompt — a blind PUT from this
    # script would clobber the seeded prompt.
    for name in ("redesign-html", "redesign-accessibility-scorer"):
        assert name in W2V_WORKER_NAMES
        assert name in SEED_MANAGED


def test_claim_finish_workers_are_seed_managed() -> None:
    # ORCH-4.4 claim_finish drives a bounded grounder -> softener loop, so both
    # workers ARE invoked and must be declared (in neither list ⇒ WM auto-creates
    # them on the wrong default model and they answer prose, not JSON). WM's seed
    # owns their curated model + system_prompt, so they must ALSO be skipped here
    # (declared but not seed-managed ⇒ a blind PUT clobbers the seeded prompt).
    for name in ("claim-grounder", "claim-softener"):
        assert name in W2V_WORKER_NAMES
        assert name in SEED_MANAGED


def test_audiobook_workers_are_seed_managed() -> None:
    # The audiobook great-output DAG invokes audiobook-director (direct.py),
    # pronunciation-proposer (ingest.py) and audiobook-listenback-judge (qa.py)
    # with inline authoritative prompts, so all three must be declared (in
    # neither list => WM auto-creates them on the wrong default model and they
    # answer prose, not JSON). WM's seed owns their curated model (local-llm) +
    # system_prompt, so they must ALSO be skipped here (declared but seed-managed
    # => a blind PUT clobbers the seeded prompt and re-tiers them to model=<name>).
    for name in ("audiobook-director", "pronunciation-proposer", "audiobook-listenback-judge"):
        assert name in W2V_WORKER_NAMES
        assert name in SEED_MANAGED


def test_critique_workers_are_seed_managed() -> None:
    # analysis_service's critique workers (ux_reviewer.py / report_generator.py) are
    # seeded by WM at temperature 0 for repeatable scores. A blind PUT from this
    # script resets them to 0.1, so they must be declared AND skipped; their
    # max_tokens now lives in the seed, not WORKER_MAX_TOKENS.
    for name in ("critique-ux-review", "critique-summary"):
        assert name in W2V_WORKER_NAMES
        assert name in SEED_MANAGED
        assert name not in WORKER_MAX_TOKENS


def test_worker_names_are_unique() -> None:
    assert len(W2V_WORKER_NAMES) == len(set(W2V_WORKER_NAMES))
