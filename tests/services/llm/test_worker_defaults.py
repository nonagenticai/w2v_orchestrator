"""Vendored WM worker defaults (seed prompt + temperature) for in-process LLM calls."""

from __future__ import annotations

import pytest

from src.services.llm import worker_defaults
from src.services.llm.worker_defaults import AUTO_CREATE_TEMPERATURE, get_worker_defaults


def test_seeded_worker_carries_wm_seed_prompt_and_temperature() -> None:
    d = get_worker_defaults("script-doctor-critique")
    assert d.seeded is True
    assert d.system_prompt is not None
    assert d.system_prompt.startswith("You are a ruthless script doctor.")
    assert d.temperature == 0.1


def test_unseeded_worker_mirrors_wm_auto_create() -> None:
    d = get_worker_defaults("promo-review")
    assert (d.seeded, d.system_prompt, d.temperature) == (False, None, AUTO_CREATE_TEMPERATURE)


def test_unknown_worker_gets_auto_create_defaults() -> None:
    d = get_worker_defaults("never-heard-of-it")
    assert (d.system_prompt, d.temperature) == (None, AUTO_CREATE_TEMPERATURE)


def test_every_no_prompt_caller_is_seeded() -> None:
    """Callers that send no system_prompt rely on the seed: each must be vendored WITH one."""
    for name in (
        "script-doctor-critique",
        "script-doctor-revise",
        "voice-director",
        "retention-critic",
        "hook-doctor",
        "claim-grounder",
        "claim-softener",
    ):
        assert get_worker_defaults(name).system_prompt, name


@pytest.mark.parametrize(
    "raw",
    [{}],
)
def test_malformed_defaults_file_is_rejected(raw: object) -> None:
    with pytest.raises(ValueError):
        worker_defaults._parse(raw)


def test_script_draft_carries_the_measured_anti_loop_penalty() -> None:
    d = get_worker_defaults("script-draft")
    assert (d.seeded, d.system_prompt, d.temperature) == (False, None, AUTO_CREATE_TEMPERATURE)
    assert d.frequency_penalty == 0.2


def test_penalty_is_absent_unless_set() -> None:
    assert get_worker_defaults("promo-review").frequency_penalty is None
    assert get_worker_defaults("never-heard-of-it").frequency_penalty is None


@pytest.mark.parametrize("name", ["script-draft"])
def test_live_caller_names_are_vendored(name: str, caplog: pytest.LogCaptureFixture) -> None:
    """These four logged "no vendored defaults" on every call. WM never seeded them
    (it auto-created them: temperature 0.1, no prompt), so the vendored entry is the
    same default — documented, not a behaviour change, and no misleading log line."""
    with caplog.at_level("INFO", logger=worker_defaults.__name__):
        d = get_worker_defaults(name)
    assert (d.seeded, d.system_prompt, d.temperature) == (False, None, AUTO_CREATE_TEMPERATURE)
    assert "no vendored defaults" not in caplog.text


@pytest.mark.parametrize("penalty", [True])
def test_malformed_frequency_penalty_is_rejected(penalty: object) -> None:
    with pytest.raises(ValueError, match="frequency_penalty"):
        worker_defaults._parse({"workers": [{"worker_name": "a", "frequency_penalty": penalty}]})


@pytest.mark.parametrize(
    "name",
    ["footage-critic"],
)
def test_remaining_caller_names_are_vendored_as_unseeded(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The last six names the orchestrator calls with no vendored entry. WM's seed
    never held any of them (``git log -S`` over worker_management finds none), so
    WM auto-created each — temperature 0.1, no prompt — and every caller passes its
    own ``system_prompt`` per call. Vendored at exactly that default."""
    with caplog.at_level("INFO", logger=worker_defaults.__name__):
        d = get_worker_defaults(name)
    assert (d.seeded, d.system_prompt, d.temperature) == (False, None, AUTO_CREATE_TEMPERATURE)
    assert d.frequency_penalty is None
    assert "no vendored defaults" not in caplog.text
