"""Unit tests for :func:`src.models.showcase_line.beat_fingerprint`.

The fingerprint is the position-free join key that lets the structural-relief loop name
an over-budget beat by CONTENT, so the hint survives the relief child re-running
``demo_video_gen`` (which rebuilds the bdd suite and renumbers every ``clip-<n>``).
Its determinism is load-bearing: parent and child are DIFFERENT PROCESSES, so anything
salted or ``hash()``-based would silently never match.
"""

from __future__ import annotations

from src.models import AuthoredLine, authored_line_fingerprint, beat_fingerprint

# GOLDEN digests. Hardcoded on purpose: recomputing the expectation with the function
# under test would pass for any implementation, including a PYTHONHASHSEED-dependent one.
# These are the bytes a parent process emits and a child process must reproduce.
_TEXT_BEAT_0 = "89d74771d6902b72"  # pragma: allowlist secret
_SCENARIO_USER_LOGS_IN = "7eeeb3b41aec7e1d"  # pragma: allowlist secret
_GHERKIN_GIVEN_A_USER = "0ebc38708986d69a"  # pragma: allowlist secret


def test_digests_are_stable_across_processes() -> None:
    # If this fails after a change to the normalisation or the domain separator, every
    # in-flight relief child's hint silently stops matching — treat it as a contract break.
    assert beat_fingerprint(text="Beat 0.") == _TEXT_BEAT_0
    assert beat_fingerprint(scenario_name="User logs in") == _SCENARIO_USER_LOGS_IN
    assert beat_fingerprint(gherkin="Given a user") == _GHERKIN_GIVEN_A_USER


def test_normalisation_collapses_whitespace_and_case() -> None:
    # A trivial re-wrap / re-casing of the SAME beat must not read as a different beat.
    assert beat_fingerprint(text="  BEAT   0.  ") == _TEXT_BEAT_0
    assert beat_fingerprint(text="Beat\n\t0.") == _TEXT_BEAT_0


def test_field_precedence_is_scenario_then_gherkin_then_text() -> None:
    # The most regeneration-stable field wins: an unchanged scenario is regenerated with
    # a verbatim scenario_name, while its narration text may be re-worded.
    assert (
        beat_fingerprint(scenario_name="User logs in", gherkin="Given a user", text="Beat 0.")
        == _SCENARIO_USER_LOGS_IN
    )
    assert beat_fingerprint(gherkin="Given a user", text="Beat 0.") == _GHERKIN_GIVEN_A_USER
    assert beat_fingerprint(text="Beat 0.") == _TEXT_BEAT_0


def test_fields_are_domain_separated() -> None:
    # The same string as a scenario name and as narration must not collide, or a hint
    # aimed at one beat would strip an unrelated beat's pin/lock protection.
    same = "User logs in"
    assert beat_fingerprint(scenario_name=same) != beat_fingerprint(text=same)
    assert beat_fingerprint(gherkin=same) != beat_fingerprint(text=same)


def test_contentless_beat_has_no_fingerprint() -> None:
    # "" must never match anything — a digest of the empty string would make every
    # contentless beat collide with every other one.
    assert beat_fingerprint() == ""
    assert beat_fingerprint(scenario_name="", gherkin=None, text="   ") == ""


def test_distinct_content_gets_distinct_digests() -> None:
    assert beat_fingerprint(text="Beat 0.") != beat_fingerprint(text="Beat 1.")


def test_authored_line_fingerprint_matches_the_dict_shaped_call() -> None:
    # The model-shaped and persisted-JSON-shaped readers must agree, since the parent
    # fingerprints dicts off a stage output and the child fingerprints AuthoredLines.
    line = AuthoredLine(id="clip-3", text="Beat 0.")
    assert authored_line_fingerprint(line) == beat_fingerprint(text="Beat 0.") == _TEXT_BEAT_0
