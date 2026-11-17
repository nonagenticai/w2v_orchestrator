"""demo_video_gen — generate a Playwright-BDD demo suite + per-scenario narration.

Drives bdd_playwright_service: inspect the target app URL, generate a demo
walkthrough feature suite (one ``@clip-<n>``-tagged scenario per beat), and a
per-scenario narration script. ``feed_to_video=False`` — the orchestrator owns
the TTS / record / compile tail, so bdd only produces the suite + script here.

Output ``lines`` are the pipeline spine: ``id`` is the scenario's ``clip-<n>``
join id (== ``NarrationPart.part_id``), consumed unchanged by demo_video_narrate
(NarrationStageBase), demo_video_record (the FTR clip join), and the captions
table. ``bdd_job_id`` lets demo_video_record hand the generated package straight
to the frontend_test_runner.

**Authored mode** (the ``showcase`` blueprint): when the brief supplies its own
``lines`` (a hand-authored narration/dialogue script) the bdd generation call is
skipped entirely and those lines become the spine verbatim. Paired with a brief
``package_url`` (a pre-built Playwright package hosted at an http(s) URL), this
lets a *proven* recording suite drive the footage instead of a freshly generated
one — demo_video_record forwards that ``package_url`` to the frontend_test_runner
in place of a ``bdd_job_id``. Independent of mode, when ``brief.dialogue.enabled``
is set each line is cast to a speaker (``a``/``b``) so demo_video_narrate can
synthesise a two-voice dialogue.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, ClassVar

from src.clients.bdd_playwright_client import BDDPlaywrightClient
from src.clients.schemas.bdd_playwright import BddStepCoverage
from src.config import get_settings
from src.models import AuthoredLine, authored_line_fingerprint, normalize_clip_key
from src.services.auth_secrets import auth_for_origin, pages_for_login, unseal_auth
from src.services.captions import line_word_counts
from src.services.stages.base import (
    SlotUse,
    StageHandler,
    StageInput,
    StageOutput,
    StageOutputWithNotes,
)
from src.services.stages.primitives.dialogue import assign_dialogue_speakers
from src.services.stages.shared.authored_storyboard import authored_lines
from src.services.stages.shared.degradation import (
    NARRATION_ESTIMATE_OFF_TARGET,
    ORDERED_DURATION_INFEASIBLE,
    Degradation,
    advisory,
    degradation,
)
from src.services.stages.shared.duration_conformance import (
    DURATION_TOLERANCE_FRACTION,
    demo_video_ordered_seconds,
)
from src.services.stages.shared.duration_plan import (
    MEASURED_SECONDS_PER_BEAT_KEY,
    beats_for_seconds,
    exported_seconds_per_beat,
    plan_beats,
)
from src.services.stages.shared.speakable_text import speakable_text
from src.services.stages.shared.speech_rate import (
    FallbackReason,
    RateBasis,
    ScriptSpeechRate,
    resolve_effective_wpm,
    speech_rate_fallback_degradation,
    spoken_text_digests,
    spoken_texts,
)
from src.services.stages.shared.time_budget import (
    DEFAULT_WPM_NATURAL,
    NARRATION_BUDGET_RATE,
    VO_RATE_DRAG_FLOOR,
    TimelineManifest,
)
from src.services.stages.shared.title_cards import title_card_seconds
from src.services.stages.website_review.script import floor_wpm

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.clients.tts_client import TTSClient
    from src.services.runs.run_context import RunContext

# Multi-digit runs (allow embedded separators): counts / codes worth checking —
# NOT single digits like "1 place" that appear naturally in prose.
_MULTI_DIGIT = re.compile(r"\d[\d,._]*\d")

log = logging.getLogger(__name__)


def _multi_digit_numbers(text: str) -> set[str]:
    """Normalized multi-digit tokens in ``text`` (separators stripped)."""
    return {re.sub(r"[,._]", "", m) for m in _MULTI_DIGIT.findall(text or "")}


def _numeral_consistency_notes(lines: list[DemoVideoLine]) -> list[str]:
    """Flag a spoken number that disagrees with the clip's AUTHORED steps.

    A deliberately LOW-NOISE cross-check (there is no OCR of the recorded
    footage, so on-screen numbers can't be read directly): for each line whose
    ``gherkin`` steps carry their OWN multi-digit number(s) — i.e. a
    number-bearing screen the author asserted — flag any multi-digit number the
    narration ``text`` speaks that is NOT among them. Lines whose gherkin has no
    numbers are skipped entirely, so a normal narrated beat never trips it. This
    catches the "narration says 312 votes, the screen/spec says 284" class of
    drift for a human to reconcile; it is advisory (surfaced as a degraded note).
    """
    notes: list[str] = []
    for line in lines:
        expected = _multi_digit_numbers(line.gherkin)
        if not expected:
            continue
        stray = _multi_digit_numbers(line.text) - expected
        if stray:
            notes.append(
                f"{line.id or 'clip'}: narration speaks {sorted(stray)} not in the "
                f"clip's authored steps {sorted(expected)} — verify against the "
                "on-screen numbers"
            )
    return notes


# Authored lines may name their speaker as a role label rather than the bare
# ``a``/``b`` cast slot. Slot ``a`` is the primary/guide voice (af_heart by
# default), slot ``b`` the secondary/newcomer voice (am_michael) — see
# ``dialogue.DEFAULT_CAST``. Map natural role names onto those slots so a brief
# can read as ``{"speaker": "guide"}`` / ``{"speaker": "new"}``. Unknown labels
# pass through lowercased and fall back to the single voice if not in the cast.
_SPEAKER_ALIASES = {
    "a": "a",
    "guide": "a",
    "narrator": "a",
    "host": "a",
    "expert": "a",
    "b": "b",
    "new": "b",
    "newcomer": "b",
    "learner": "b",
    "guest": "b",
}


def _normalize_speaker(value: object) -> str | None:
    """Map an authored ``speaker``/``role`` label onto a cast slot (``a``/``b``).

    Unknown labels pass through lowercased (``voice_for_speaker`` falls back to the
    single voice if the slot isn't in the cast), and blanks become ``None``.
    """
    if not value:
        return None
    key = str(value).strip().lower()
    return _SPEAKER_ALIASES.get(key) or key or None


def _cast_speakers(lines: list[DemoVideoLine], brief: dict[str, Any]) -> list[DemoVideoLine]:
    """Ensure every line carries a speaker when ``brief.dialogue.enabled``.

    Authored speakers are preserved; any unset line is filled from the question
    (``?`` → ``a``) heuristic shared with promo / website_review. A no-op for
    single-narrator runs (the default), so demo_video proper is unaffected.
    """
    if not (brief.get("dialogue") or {}).get("enabled"):
        return lines
    if all(line.speaker for line in lines):
        return lines
    if not any(line.speaker for line in lines):
        return assign_dialogue_speakers(lines)
    return [
        line
        if line.speaker
        else line.model_copy(update={"speaker": "a" if line.text.rstrip().endswith("?") else "b"})
        for line in lines
    ]


# Structural hook beats whose ``hook_role`` marks them as load-bearing spine
# structure (a cold open, the open/payoff of a retention loop, a deliberate
# pattern interrupt). The relief beat-reducer must never drop these — cutting one
# breaks the very structure the retention pass built.
_PROTECTED_HOOK_ROLES = frozenset({"cold_open", "loop_open", "loop_payoff", "pattern_interrupt"})


def _culprit_indexes(lines: list[DemoVideoLine], relief: Mapping[str, Any]) -> frozenset[int]:
    """Indexes of the beats reconcile_timeline flagged as over-budget culprits.

    Two signals ride the relief config, and the CONTENT one wins:

    * ``over_budget_fingerprints`` — :func:`~src.models.showcase_line.beat_fingerprint`
      digests computed from the PARENT run's own beats. This is the only signal that
      survives a regeneration: a relief child re-runs ``demo_video_gen``, so the bdd
      suite is rebuilt with a different scenario count/order and the parent's
      ``clip-<n>`` ordinals no longer name the same content.
    * ``over_budget_ids`` — the raw ``infeasible_beat_ids`` (positional ordinals), kept
      for BACKWARDS COMPATIBILITY only: a child run spawned before this change carries
      no fingerprints, and its ids must keep working. Consulted ONLY when no
      fingerprints were supplied — otherwise a STALE ordinal that happens to match an
      unrelated child beat would strip that beat's pin/lock protection (a false
      positive), which is exactly the bug the fingerprints exist to close.
    """
    fingerprints = {str(x) for x in (relief.get("over_budget_fingerprints") or []) if str(x)}
    if fingerprints:
        return frozenset(
            i for i, line in enumerate(lines) if authored_line_fingerprint(line) in fingerprints
        )
    over_budget_ids = {str(x) for x in (relief.get("over_budget_ids") or [])}
    if not over_budget_ids:
        return frozenset()
    return frozenset(i for i, line in enumerate(lines) if str(line.id) in over_budget_ids)


def _apply_target_beats(
    lines: list[DemoVideoLine],
    relief: Mapping[str, Any] | None,
    per_beat_seconds: float | None = None,
    deliverable: Mapping[str, Any] | None = None,
    floor_ratio: float = 0.0,
) -> list[DemoVideoLine]:
    """Deterministically shrink the spine to ``relief['target_beats']`` beats.

    The relief-loop lever (PR-C): when a render is over budget and footage is
    pinned, the only way to buy time is to cut whole beats. A relief re-run passes
    ``structural_relief={"target_beats": N}`` in ``config_snapshot``; this drops
    the lowest-value droppable beats until at most ``N`` remain, applied on BOTH
    the authored and generated paths (it operates on the already-built spine).

    Protected (NEVER dropped): a structural hook beat (``hook_role`` in
    ``_PROTECTED_HOOK_ROLES``) always; and a beat pinned to a target ordinal
    (``pin_position``) or hard duration-locked (``time_lock``) UNLESS that beat is an
    over-budget culprit (see :func:`_culprit_indexes`) — cutting a culprit is the whole
    point of a relief re-run, so a pin/lock does not shield it. The protected set is a
    floor: if it already exceeds ``target`` no droppable beat is cut.

    Drop order is fully deterministic: the over-budget culprits first (they cause the
    overrun), then undirected filler before an author-``locked`` / directed beat, then
    original index as a stable tiebreak. Survivors are returned IN ORIGINAL ORDER with
    their ``id`` UNCHANGED — demo_video_record's ``clip-<n>`` footage join keys on
    ``id``, so beats are removed, never renumbered.

    A falsy ``relief``, a missing/non-positive ``target_beats``, or a spine already
    at/under target returns ``lines`` unchanged.

    RE-SIZED, NOT REFUSED, when the requested cut goes too deep. The lever exists to
    relieve an OVERRUN, and nothing used to stop it cutting straight THROUGH the
    underfull floor on the way out: ``next_target_beats`` floors at 1, so a 30s order
    could arrive here asking for a single beat, which then records ~4s and dies at
    ``reconcile_timeline``'s 50% gate having paid for a full render. Eleven of the
    twelve production failures of that gate were relief children shaped exactly like
    that. Solving one gate by violating the other is not convergence.

    So ``target`` is clamped UP to the smallest count that can still reach the floor at
    the MEASURED per-beat yield (:func:`~src.services.stages.shared.duration_plan.plan_beats`),
    capped at ``len(lines)`` — i.e. at worst no beat is cut at all, which is strictly
    better than a cut that guarantees a rejected film. With no measurement, no ordered
    target, or ``floor_ratio <= 0`` the clamp makes no judgement and the requested
    target stands unchanged. The spawner applies the SAME clamp before it decides to
    bill a child at all; this is the second, in-process application of one rule, not a
    second rule.
    """
    if not relief:
        return lines
    target = relief.get("target_beats")
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        return lines
    ordered = _target_seconds(deliverable or {})
    if ordered is not None:
        plan = plan_beats(
            target,
            target_seconds=ordered,
            floor_ratio=floor_ratio,
            per_beat_seconds=per_beat_seconds,
            max_beats=len(lines),
        )
        if plan.verdict in ("RESIZED", "UNSATISFIABLE"):
            log.warning(
                "demo_video_gen: relief target_beats %s -> %s — %s",
                target,
                plan.beats,
                plan.explain(),
            )
        target = plan.beats
    if len(lines) <= target:
        return lines

    # The cut-order hint from reconcile_timeline, resolved CONTENT-first so it survives
    # the child's bdd regeneration (see _culprit_indexes). Empty when the parent flagged
    # nothing that maps onto this spine — the deterministic drop order below then governs
    # and the target COUNT still shrinks the spine.
    over_budget = _culprit_indexes(lines, relief)
    protected = {
        i
        for i, line in enumerate(lines)
        # The FIRST beat is the positional opener — NEVER cut, on ANY path. On the
        # generated path hook_role is stamped by a LATER stage, so at gen time index 0 is
        # the only reliable opener signal; without this the relief cut could drop the
        # opening beat on a generated re-run.
        if i == 0
        # An explicit structural anchor (loop hook) is never cut either.
        or line.hook_role in _PROTECTED_HOOK_ROLES
        # An author pin / hard time-lock is kept UNLESS the beat is the over-budget
        # culprit relief must cut.
        or ((line.pin_position is not None or bool(line.time_lock)) and i not in over_budget)
    }
    droppable = [i for i in range(len(lines)) if i not in protected]
    n_to_drop = min(len(lines) - target, len(droppable))
    if n_to_drop <= 0:
        return lines

    # Cut the over-budget culprits FIRST (they cause the overrun), then undirected
    # filler, then directed beats; original index is the final stable tiebreak.
    ranked = sorted(
        droppable,
        key=lambda i: (
            i not in over_budget,
            bool(lines[i].locked),
            lines[i].beat_role is not None,
            i,
        ),
    )
    to_drop = set(ranked[:n_to_drop])
    return [line for i, line in enumerate(lines) if i not in to_drop]


# Per-poll wall-clock deadline. BaseServiceClient.poll_job HARD-REJECTS any value
# > 600s (server-side timeouts make longer pointless), so this is capped at 600;
# the StageSpec outer envelope (the dispatcher's stage timeout) is the larger
# budget. bdd generation (inspect + a few LLM passes) completes well within it.
_WAIT_TIMEOUT_S = 600.0
_DEFAULT_MAX_SCENARIOS = 5

# Ceiling on the derived count. Every scenario is a real browser recording, so an
# absurd target must not translate into an unbounded suite; past this the honest
# answer is "that target is not achievable from this app", which
# `reconcile_timeline` already says out loud.
_MAX_DERIVED_SCENARIOS = 20

# THE PROVENANCE OF THE PER-SCENARIO YIELD, said out loud wherever a message is built
# on it. The two rungs of the yield ladder are not the same KIND of number, and the
# degradations channel may not blur them:
#
# * RUNG ONE is a GUARANTEE. bdd exports
#   ``narration_script.parts[].expected_recording_seconds`` for the suite it just
#   generated, so a message may say the length is guaranteed — but only while naming
#   bdd as the source of that guarantee.
# * RUNG TWO is an OBSERVATION. ``config_snapshot["structural_relief"]`` carries what
#   the PARENT run's footage actually measured. It is evidence about a run that has
#   already happened, not a floor this run is obliged to hit, so a message built on it
#   may say "expected"/"projected" and never "guaranteed" or "cannot".
#
# A None source means the caller did not say; the wording then falls back to the
# weaker claim, because an unattributed number cannot carry the stronger one.
_BDD_GUARANTEED_YIELD = (
    "guaranteed by bdd for this suite via narration_script.parts[].expected_recording_seconds"
)
_PARENT_MEASURED_YIELD = (
    "measured on the parent run's footage and carried on "
    'config_snapshot["structural_relief"] — an observation of a previous suite, not a '
    "floor this run is obliged to hit"
)
_UNATTRIBUTED_YIELD = "source not recorded by this run"


def _measured_seconds_per_beat(ctx: RunContext) -> float | None:
    """The MEASURED footage seconds per beat this run may plan against, or ``None``.

    RUNG TWO of the yield ladder. Rung ONE is the value bdd_playwright EXPORTS
    (``narration_script.parts[].expected_recording_seconds``, bdd #153) — read in
    ``_generated_output`` and preferred by ``handle`` whenever it is present, because
    bdd now GUARANTEES that pacing rather than merely having produced it once. This
    parent-measurement rung remains for the authored path (no bdd call) and for a bdd
    stream that publishes nothing.

    THE CONSTANT BOTH RUNGS REPLACED. ``_SECONDS_PER_SCENARIO = 4.0`` sized the scenario
    count for an ordered duration and was extrapolated from exactly two hand-observed
    samples on one 2026-08 E2E run (4.16s, 4.40s). It matched NEITHER mode the pipeline
    has since been measured in — 1.2-3.9s when the recorded suite runs unpaced, and
    16.7-36.9s when it runs paced. A plan built on it therefore could not be checked
    against the floor that judges it, because it was not a measurement of anything; the
    disagreement surfaced only at ``reconcile_timeline``, after record AND render had
    both been paid for.

    The replacement is not another number, it is a READ. A relief child carries its
    parent's measured yield on ``config_snapshot["structural_relief"]`` under
    :data:`MEASURED_SECONDS_PER_BEAT_KEY` — computed once, by the spawner, from the
    ffprobe'd footage-pinned rows of the parent's own ``timeline_manifest``. One owner,
    written in one place, read here, never re-derived and never defaulted.

    ``None`` (a first-pass run with no parent, or a parent whose manifest carried no
    measured footage) means UNMEASURED. Every caller must then make NO duration
    judgement rather than substitute an assumed figure — an absent measurement is not
    evidence about the plan, and swapping in a fresh constant would simply reinstate the
    defect under a new name — at EITHER rung.
    """
    relief = ctx.config_snapshot.get("structural_relief")
    if not isinstance(relief, dict):
        return None
    value = relief.get(MEASURED_SECONDS_PER_BEAT_KEY)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    return float(value)


def _target_seconds(deliverable: Mapping[str, Any]) -> float | None:
    """The ordered duration, or ``None`` when none was ordered."""
    target = deliverable.get("target_duration_seconds")
    if not isinstance(target, (int, float)) or isinstance(target, bool) or target <= 0:
        return None
    return float(target)


def _scenarios_for_target(
    brief: Mapping[str, Any],
    deliverable: Mapping[str, Any],
    per_beat_seconds: float | None,
) -> int:
    """How many scenarios to ask bdd for, sized to the ordered duration.

    An explicit ``brief.max_scenarios_per_feature`` always wins — an author who named a
    number meant it.

    Otherwise the count is sized so the plan's OWN arithmetic reaches the duration that
    was ordered: ``ceil(target_seconds / per_beat_seconds)``, clamped into
    ``[_DEFAULT_MAX_SCENARIOS, _MAX_DERIVED_SCENARIOS]``. ``per_beat_seconds`` is a
    MEASUREMENT (:func:`_measured_seconds_per_beat`); with none available the target
    cannot be converted into a count at all, so the default ceiling stands and the
    stage says so rather than dividing by a fiction.

    ⚠️ WHAT THIS DOES AND DOES NOT DO — measured 2026-08-23 on E2E run 01a03011
    (demo-video-full-chain-todomvc, 60s target): the derivation asked for 15, bdd
    generated 5, capture returned 5 clips, and the film totalled 27.26s of a 60s target.

    The number returned here is a CEILING (``max_scenarios_per_feature``), NOT a quota.
    Raising it permits more scenarios; it does not make the generator produce more. That
    is precisely why sizing alone cannot be the fix and why
    :func:`_assert_plan_can_reach_floor` runs on the spine bdd ACTUALLY returned — the
    count that matters is the one that came back, not the one that was asked for.
    """
    explicit = brief.get("max_scenarios_per_feature")
    if explicit:
        return int(explicit)
    target = _target_seconds(deliverable)
    if target is None:
        return _DEFAULT_MAX_SCENARIOS
    needed = beats_for_seconds(target, per_beat_seconds or 0.0)
    if needed <= 0:
        # Unmeasured yield: the ordered duration cannot be converted into a count.
        return _DEFAULT_MAX_SCENARIOS
    return max(_DEFAULT_MAX_SCENARIOS, min(_MAX_DERIVED_SCENARIOS, needed))


def _min_scenarios_for_target(
    brief: Mapping[str, Any],
    deliverable: Mapping[str, Any],
    per_beat_seconds: float | None,
) -> int | None:
    """The scenario MINIMUM to send bdd, or ``None`` when none can be derived.

    ⭐ THE OTHER HALF OF ``max_scenarios_per_feature`` (wave 13, O3). That key is a
    CEILING, NOT A QUOTA — :func:`_scenarios_for_target`'s own docstring says so, and
    names the measured consequence: on E2E run 01a03011 the derivation asked for 15, bdd
    generated 5, capture returned 5 clips, and the film totalled 27.26s of a 60s order.
    Raising a ceiling permits more scenarios; it does not ask for them. So the ordered
    duration reached the generator as a bound it was free to ignore, and it did.

    ``min_scenarios`` is the floor that makes the order actionable. Derived from EXACTLY
    the arithmetic :func:`_scenarios_for_target` already uses — ``beats_for_seconds``
    against the same measured per-beat yield — so the ceiling and the floor cannot drift
    apart and neither is a fresh constant.

    ``None``, and the key OMITTED, whenever no floor is honest:

    * nothing was ordered — there is no duration to reach;
    * the yield is UNMEASURED (``beats_for_seconds`` returns 0) — a minimum derived from
      a fiction is the ``_SECONDS_PER_SCENARIO = 4.0`` sin this module exists to delete;
    * an author pinned ``max_scenarios_per_feature`` — their number is both bounds, and
      manufacturing a floor under an explicitly authored ceiling would override an author
      who named a count and meant it.

    Never above the ceiling actually being sent: a minimum that exceeds its own maximum is
    an incoherent request, and bdd would be right to reject it.

    ⚠️ CONTRACT — the key name ``min_scenarios`` is agreed with a concurrent change in
    ``bdd_playwright_service``. Renaming it breaks a sibling service. Safe in both deploy
    orders for the same reason the other budget keys are: bdd's ``DemoVideoRequest``
    declares no ``extra=`` policy, so pydantic's ``extra="ignore"`` default means an older
    bdd drops it silently — no 422.
    """
    if brief.get("max_scenarios_per_feature"):
        return None
    target = _target_seconds(deliverable)
    if target is None:
        return None
    needed = beats_for_seconds(target, per_beat_seconds or 0.0)
    if needed <= 0:
        return None
    ceiling = _scenarios_for_target(brief, deliverable, per_beat_seconds)
    return min(needed, ceiling)


def narration_words_target(ordered_seconds: float, wpm_natural: float = DEFAULT_WPM_NATURAL) -> int:
    """Total narration words that fill ``ordered_seconds`` at the SLOWEST in-design rate.

    ``ordered_seconds * wpm_natural * NARRATION_BUDGET_RATE / 60``.

    ⛔ WAVE 14, N9 — THE PARAGRAPH THAT USED TO OPEN HERE ARGUED FOR THE BUG. It read
    "Why a FLOOR rate rather than the natural rate ... the asymmetric error is
    deliberate, and it is preserved: the budget still sits strictly BELOW a natural read
    and can therefore still not overshoot." That is the 0.80 rationale, and the body
    below has multiplied by :data:`NARRATION_BUDGET_RATE` (1.0) since wave 13 removed
    it — so the docstring was arguing, in the present tense, for an asymmetry the code
    no longer has and that the owner's complaint was caused by. It is deleted rather
    than softened: a rationale for a deleted defect is not history, it is an instruction
    to put the defect back.

    The budget is the NATURAL-rate count. Undershooting is not free insurance — the
    rate lever "absorbs" it by DRAGGING THE VOICE, which is the reported defect. The
    long side is guarded where it belongs (see below).

    ⭐ WHICH RATE, and why this is the ROOT CAUSE of "the narration weirdly slows
    down" (wave 13). This budget was taken at ``VO_RATE_NATURAL_FLOOR`` (0.80), i.e.
    it asked the generator for only as many words as the target could hold WITH THE
    VOICE ALREADY DRAGGED 20% — 60 words for a 30s order. The run then delivered
    exactly what it had been budgeted for, ``vo_rate`` landed around 0.87-0.89, and
    every detector downstream was armed at 0.80 and saw nothing. So the drag was not a
    downstream accident being missed; it was **ordered upstream, here, by arithmetic**,
    and then certified as a pass.

    It is now taken at :data:`NARRATION_BUDGET_RATE` (1.0) — 75 words for a 30s order.

    ⛔ AND IT IS NOT TAKEN AT THE DRAG FLOOR EITHER, though that was the obvious first
    move and it is wrong. The budget and the detector are different decisions, and
    sharing one constant puts a PERFECT delivery exactly on the detection boundary,
    where integer rounding casts the deciding vote: measured across every ordered
    duration 5-180s, a budget written at the 0.97 floor is flagged ``underspoken``
    on **48.9%** of durations even when the generator delivers it exactly. At 1.0 that
    is 0.6%. See :data:`NARRATION_BUDGET_RATE` for the full reasoning, including why 1.0
    is a measurement rather than an optimistic target and where the long side is
    actually guarded.

    The former "asymmetric error is deliberate" rationale is retired with the 0.80: it
    bought insurance against overshoot by pricing in a 20% drag, which is the defect.

    ⛔ ``wpm_natural`` IS A PARAMETER, AND THAT IS THE WHOLE POINT. The first cut of
    this function hardcoded ``DEFAULT_WPM_NATURAL`` while
    ``demo_video_reconcile_timeline`` solved the SAME timeline at the rate the run's
    voice was MEASURED to speak at (~151.6 wpm live, vs the 150.0 constant — and a
    dialogue cast measures further off still). The budget bdd was handed and the budget
    the solver later checked it against were therefore two different numbers, so every
    beat came back underfilled, ``vo_rate`` pinned at ``VO_RATE_NATURAL_FLOOR``, and the
    delivered film played uniformly slow — the exact symptom the word budget exists to
    remove. The caller resolves ONE rate (see :meth:`DemoVideoGenStage._budget_wpm`) and
    both halves are taken at it.

    The default is the same constant as before, so a caller that cannot resolve a rate —
    a probe timeout — is byte-identical to the pre-measurement behaviour.
    """
    return round(ordered_seconds * wpm_natural * NARRATION_BUDGET_RATE / 60)


def narration_words_per_scenario(total_words: int, beats: int) -> list[int] | None:
    """Split ``total_words`` across ``beats``, or ``None`` when the count is unknown.

    The remainder is spread one word at a time over the leading beats, so the split
    normally sums to ``total_words`` exactly — a producer that honours every per-beat
    number lands on the total, with no rounding drift to reconcile.

    ``None`` for a non-positive beat count: an absent split is an honest "no per-beat
    guidance", whereas a fabricated one would budget against a beat count nobody
    established. Callers must omit the key entirely in that case rather than send null.

    ⭐ WAVE 14, N3 — WHY "NORMALLY", AND WHY THE EXACT SUM IS NOT THE TOP PROPERTY.
    The beats of a demo_video spine are allotted the SAME number of seconds each (one
    ``measured_seconds_per_beat`` for the whole spine — see
    :func:`~src.services.stages.shared.duration_plan.measured_seconds_per_beat`), while
    ``divmod`` hands ``beats - remainder`` of them ``base`` words and the rest
    ``base + 1``. So the short beats are asked to fill an AVERAGE beat's seconds with a
    BELOW-AVERAGE word count, and their delivery rate is ``base / (total / beats)`` —
    which is ``1 - remainder / total``, and which drops under
    :data:`~src.services.stages.shared.time_budget.VO_RATE_DRAG_FLOOR` whenever the
    remainder exceeds 3% of the budget. A beat that honoured its budget to the word is
    then flagged ``underspoken``, and ``reconcile_timeline`` arms a paid LLM re-ask on
    ``any(a.underspoken)``.

    Measured over every ordered duration 5-180s x beat counts 2-6 (880 pairs) at
    ``DEFAULT_WPM_NATURAL``: a FLAWLESS delivery was flagged on **70 of 880 (7.95%)**.
    The owner's worked example (30s, 4 beats -> ``[19, 19, 19, 18]`` against 7.50s each,
    ``18 / 18.75 = 0.960``) is one of them — but 4 is NOT the only hit at 30s: 6 beats
    gives ``[13, 13, 13, 12, 12, 12]`` and ``12 / 12.5 = 0.960`` as well.

    So when the exact split would order a sub-floor beat, EVERY beat is raised to
    ``base + 1`` instead. The split then overshoots ``total_words`` by
    ``beats - remainder`` words, and that is the correct direction to be wrong: the
    long side is guarded downstream (``unsayable``, ``plan_beats``' ceiling term,
    ``_apply_duration_ceiling``) whereas the short side is the owner's own reported
    defect, and wave 13's whole finding is that biasing the budget DOWN is what ordered
    the drag. The overshoot is +0.96% on average and the exact sum is preserved on
    795 of the 880 pairs.

    ``+ 0.5`` inside the comparison is ``narration_words_target``'s OWN ``round()``:
    ``total_words`` is already up to half a word below the exact budget, and that
    half-word is enough to push four more pairs (29s/5, 33s/4, 33s/5, 57s/6) under the
    floor on their delivered rate while their ``base / mean`` still reads as clearing
    it. Without it the sweep leaves those four behind.

    After the fix: **0 of 880 flagged** for every ordered duration from 6s, and nothing
    becomes ``unsayable``. The 5s order remains the single degenerate case — and it is
    NOT this function's doing: its splits there are exact and even, and the flag comes
    from ``narration_words_target`` rounding 12.5 words down to 12. That is the same
    0.6% residual wave 13 measured and deliberately left.
    """
    if beats <= 0 or total_words <= 0:
        return None
    base, remainder = divmod(total_words, beats)
    if remainder and base < VO_RATE_DRAG_FLOOR * (total_words + 0.5) / beats:
        return [base + 1] * beats
    return [base + (1 if i < remainder else 0) for i in range(beats)]


#: Fraction of its word budget a beat may fall below before the shortfall is reported.
#: Not zero: ``narration_words_per_scenario`` splits an integer budget evenly while a
#: writer lands wherever the sentence ends, so a few words either way is normal authoring
#: noise, not a defect. 0.85 flags a beat that is missing more than ~1 word in 7 — enough
#: to push its ``vo_rate`` toward the natural floor, which is the thing that actually
#: hurts.
_WORD_BUDGET_TOLERANCE = 0.85


def authored_word_shortfalls(
    lines: list[DemoVideoLine],
    ordered_seconds: float | None,
    wpm_natural: float,
    tolerance: float = _WORD_BUDGET_TOLERANCE,
) -> dict[str, tuple[int, int]]:
    """Beats whose AUTHORED word count falls short of their budget: ``{id: (has, want)}``.

    This is the live-path wiring of :func:`~src.services.captions.line_word_counts`, which
    was written for exactly this ("reconcile authored word counts against a computed
    beat_word_budget") and then sat unreferenced — its own docstring said "nothing on the
    live path calls it yet". It is called here rather than reimplemented so the count this
    compares is the SAME whitespace word count the caption tracks are built from.

    The budget is this module's own :func:`narration_words_target` split by
    :func:`narration_words_per_scenario` over the beats that actually came back — not over
    the count that was ASKED for. That distinction is the whole reason this check earns
    its place: ``max_scenarios_per_feature`` is a CEILING, not a quota (a derivation once
    asked for 15 and bdd returned 5), so the budget bdd was sent and the budget its output
    should be graded against are different numbers whenever the generator under-delivers.

    WHY IT IS WORTH CHECKING AT ALL, when ``reconcile_timeline`` also flags underspoken
    beats downstream: by the time that stage runs, the footage has been RECORDED and paid
    for. Here nothing has been recorded, so the finding is still cheap — and a beat short
    of its budget at this point is a forecast of the exact defect the owner reports
    (dragged narration, a master under its order), made before any of it is spent.

    Empty when nothing was ordered (no order ⇒ no budget ⇒ nothing to be short of), when
    the rate is unusable, or when no recognised script stage is present.
    """
    if ordered_seconds is None or ordered_seconds <= 0 or wpm_natural <= 0:
        return {}
    counts = line_word_counts({"demo_video_gen": {"lines": [line.model_dump() for line in lines]}})
    if not counts:
        return {}
    budget = narration_words_per_scenario(
        narration_words_target(ordered_seconds, wpm_natural), len(lines)
    )
    if budget is None:
        return {}
    # Zipped in SPINE ORDER, because that is the order `narration_words_per_scenario`
    # spread its remainder in — pairing a beat with another beat's share would invent
    # shortfalls and hide real ones.
    short: dict[str, tuple[int, int]] = {}
    for line, want in zip(lines, budget, strict=True):
        has = counts.get(str(line.id))
        if has is None:
            continue
        if has < want * tolerance:
            short[str(line.id)] = (has, want)
    return short


def narration_seconds_for_order(
    config_snapshot: Mapping[str, Any], ordered_seconds: float | None
) -> float | None:
    """The seconds of the ORDER the narration may fill: the order minus the title cards.

    Owner ruling: intro/outro cards count INSIDE the order. The compile splices them in
    as free-standing shots, so a script budgeted against the whole order delivers
    ``order + cards`` — an overshoot no downstream trim can recover, because the
    narration ALONE already fills the order. The reservation is
    :func:`~src.services.stages.shared.title_cards.title_card_seconds` at this order,
    i.e. ``plan_title_cards`` capped at 20% of the order — the same helper website_review
    and promo reserve with, never a local re-derivation.

    ``None`` passes through (no order ⇒ nothing to reserve against). No cards configured
    ⇒ exactly ``ordered_seconds``, so an un-opted run is unchanged.
    """
    if ordered_seconds is None or ordered_seconds <= 0:
        return ordered_seconds
    cards = title_card_seconds(config_snapshot.get("title_card"), ordered_seconds=ordered_seconds)
    return max(ordered_seconds - cards, 0.0)


#: How far over its narration budget a script's TOTAL word count may run before it is
#: reported: the conformance band's own upper edge (+10%), so a script flagged here is a
#: script whose words alone, spoken at the content rate, already deliver outside the band.
_WORD_CEILING_RATIO = 1.0 + DURATION_TOLERANCE_FRACTION


def narration_word_ceiling(narration_seconds: float | None, wpm_content: float) -> int | None:
    """The most words the whole script may hold: ``seconds x wpm / 60 x 1.10``.

    ``wpm_content`` is the rate the DRAFT was measured at (Phase B, ``resolved_wpm``) —
    the rate ``reconcile_timeline`` solves at — never the calibration passage. ``None``
    when nothing was ordered or the rate is unusable.
    """
    if narration_seconds is None or narration_seconds <= 0 or wpm_content <= 0:
        return None
    return int(narration_seconds * wpm_content / 60.0 * _WORD_CEILING_RATIO)


def structural_bound_unavailable(
    lines: list[DemoVideoLine],
    deliverable: Mapping[str, Any] | None,
    per_beat_seconds: float | None,
) -> bool:
    """True when a duration WAS ordered but no structural bound could be applied at all.

    The predicate for :func:`_apply_feasibility_floor`'s unmeasured-yield early return —
    the branch the live bdd runs actually take. It exists as a NAMED function, used by
    both that early return and the stage's output field, for one reason: the state is
    reported in two places, and a condition written out twice is a condition that drifts.
    Here it cannot.

    MEASURED, 2026-09-11: runs ee67 and 5bf5 (44.00s order) both hit this — bdd exported
    no ``expected_recording_seconds`` and neither had a parent — and the early return was
    SILENT. ee67 went on to deliver 68.075s against its 44.00s order with no plan-time
    bound of any kind. Making the state readable off the stage output is what turns "the
    plan was never bounded" from a line in a pod log into something an auditor, an E2E
    assertion, or a later stage can actually test for.

    ⚠️ Deliberately NOT a degradation, and that is not timidity. Nothing here MEASURED
    anything, and an absent measurement is not evidence of a bad plan — the rule the rest
    of this module follows. A degradation on every yield-less run would fire across the
    whole population and therefore signal nothing about any individual run (the same trap
    a mis-calibrated gate fell into when it would have flagged 96.7% of masters). The
    honest channel for "this run could not be bounded" is a FACT on the output, which
    costs nothing when it is common and is still there when it matters.
    """
    if not lines:
        return False
    if _target_seconds(deliverable or {}) is None:
        return False
    return per_beat_seconds is None or per_beat_seconds <= 0.0


def _apply_feasibility_floor(
    lines: list[DemoVideoLine],
    deliverable: Mapping[str, Any] | None,
    per_beat_seconds: float | None,
    floor_ratio: float,
    yield_source: str | None = None,
) -> tuple[list[DemoVideoLine], list[Degradation]]:
    """Refit a spine whose STRUCTURAL FLOOR exceeds the ordered duration.

    THE ARITHMETIC, measured live on run ``01a0b3f1`` (2026-09-10). bdd guarantees a
    fixed recording dwell per scenario (``HERO_DWELL_MS``, surfaced on the wire as
    ``narration_script.parts[].expected_recording_seconds`` — see
    :class:`~src.clients.schemas.bdd_playwright.BddNarrationPart`), so an N-scenario demo
    has a HARD FLOOR of ``N x per_scenario_seconds``. At 22.00s/scenario a 2-scenario
    spine cannot deliver less than 44.00s, and a 30s order is therefore **not reachable
    by any choice of content** — the run delivered 47.045s (+56.8%) and every gate passed
    it, because the floor gate only ever asked whether the film was too SHORT.

    ⚠️ NOT the same bound as :func:`_apply_duration_ceiling`. That one uses
    ``plan_beats``'s ``ceiling = ceil(target / per_beat)`` — the smallest count that
    *covers* the order, which by construction sits AT OR ABOVE it (ceil(30/22) = 2, so 2
    beats x 22s = 44s was judged ``OK``). This one is the mirror written with ``floor``:
    the largest count that still FITS inside the order. Both are needed; only this one
    can see an overshoot that the ceiling's own rounding created.

    Three outcomes, in the order the arithmetic decides them:

    * ``ordered >= floor`` — FEASIBLE. The spine is returned untouched and nothing is
      degraded. This is the no-change path for every order that was realistic.
    * ``ordered < floor`` and one scenario still fits — REFIT. The count drops to
      ``floor(ordered / per_scenario)`` (never below 1) through the relief lever's own
      :func:`_apply_target_beats` machinery, so the protected set, the deterministic drop
      order and the ``clip-<n>`` ``id`` join are exactly the ones ``demo_video_record``
      already depends on. The order is now deliverable, so this is a plan correction, not
      a quality defect, and it degrades nothing.
    * ``per_scenario > ordered`` — INFEASIBLE AT ONE. A single scenario already overruns
      the order and mutilating it is not on the table (the dwell is deliberate
      readability, owned by bdd). The plan goes TO THE FLOOR — one scenario — and says so
      LOUDLY on the structured channel via
      :data:`~src.services.stages.shared.degradation.ORDERED_DURATION_INFEASIBLE`.

    ⛔ Never raises. This runs on a paid run that is already mid-flight; refusing here
    would destroy work over an order the *caller* got wrong. The honest answer is to
    deliver the achievable film and record the figures — the ordered seconds, the
    per-scenario length planned against AND ITS PROVENANCE, and the refit taken. The
    entry states those; it does not assert that the order was physically impossible,
    because nothing here established that (``yield_source``).

    Makes NO judgement without both an ordered target and a per-scenario measurement —
    an absent measurement is not evidence of a bad plan (the same rule the rest of this
    module follows).
    """
    ordered = _target_seconds(deliverable or {})
    if ordered is None or not lines:
        return lines, []
    if structural_bound_unavailable(lines, deliverable, per_beat_seconds):
        # MEASURED, 2026-09-11: this is the branch the live bdd runs take. Runs ee67 and
        # 5bf5 (demo_video_full_chain_todomvc, 44.00s order) both reached here — bdd
        # exported no ``expected_recording_seconds`` and neither run had a parent, so
        # `_BDD_GUARANTEED_YIELD` and `_PARENT_MEASURED_YIELD` were both absent — and the
        # early return was SILENT. ee67 then recorded 82.76s of footage and delivered
        # 68.075s against the 44.00s order.
        #
        # Still no degradation and still no refit, and deliberately: nothing here
        # measured anything, and an absent measurement is not evidence of a bad plan
        # (the rule the rest of this module follows). A degradation on every
        # yield-less run would fire on the whole population and signal nothing.
        # What changes is that the gap is now STATED, and names where the plan does get
        # bounded — ``reconcile_timeline`` runs after ``demo_video_record`` and pins each
        # footage beat to the ffprobe'd clip length, which is a measurement this stage
        # cannot have yet because nothing has been recorded.
        log.warning(
            "demo_video_gen: %.2fs was ordered but no per-scenario recording length is "
            "available for this run (no bdd export, no parent measurement), so NO "
            "structural bound was applied to the %d-beat spine here. The plan is bounded "
            "against REAL footage downstream, in reconcile_timeline, once "
            "demo_video_record has measured the clips.",
            ordered,
            len(lines),
        )
        return lines, []

    if per_beat_seconds is None or per_beat_seconds <= 0.0:
        # UNREACHABLE, and kept anyway. `structural_bound_unavailable` above already
        # returned for exactly this state (``ordered`` and ``lines`` are both known
        # non-empty by the time it is called), but the type checker cannot see a
        # narrowing that happens inside a helper — and the arithmetic below divides by
        # this number. An explicit return rather than an `assert`: if the two ever do
        # drift apart, the honest outcome is "no bound applied", the same as the branch
        # above, not a crash on a paid mid-flight run.
        return lines, []

    n_scenarios = len(lines)
    floor_seconds = n_scenarios * per_beat_seconds
    if ordered >= floor_seconds:
        return lines, []  # FEASIBLE — the order can hold this spine; change nothing.

    # The largest scenario count that still fits INSIDE the order (floor, not ceil).
    # Floored at 1: a zero-beat film is not a deliverable, and the single-scenario
    # overrun is reported rather than "solved" by shipping nothing.
    fitting = max(1, int(ordered // per_beat_seconds))
    degradations: list[Degradation] = []
    # WHAT THIS ARITHMETIC ACTUALLY ESTABLISHES. `per_beat_seconds` is one number from
    # one of two rungs (see _BDD_GUARANTEED_YIELD / _PARENT_MEASURED_YIELD). On the bdd
    # rung it is a stated guarantee for THIS suite; on the parent rung it is a past
    # observation. Neither rung makes the ordered duration physically unreachable, and
    # the guard below compares two floats — it proves an OVERRUN AGAINST THIS FIGURE,
    # nothing more. So the wording states the figure, its provenance, the order, and
    # the refit taken, and leaves the impossibility claims to code that can prove them.
    source = yield_source or _UNATTRIBUTED_YIELD
    # ⚠️ "owned by bdd" is a claim about PROVENANCE, and this function is reached from the
    # AUTHORED path too (`handle` calls it for both spines), where no bdd suite exists and
    # the dwell figure came from the parent run's measurement — or from nowhere recorded.
    # Only the bdd rung establishes that bdd owns the dwell, so the clause is conditional
    # on the rung this run actually has, exactly as the log line below already is.
    dwell_clause = (
        ", whose recording dwell is owned by bdd" if yield_source == _BDD_GUARANTEED_YIELD else ""
    )
    if per_beat_seconds > ordered:
        log.error(
            "demo_video_gen: ORDERED DURATION BELOW ONE SCENARIO — %.2fs was ordered; "
            "the per-scenario recording length this run is planning against is %.2fs "
            "(%s), so the %d scenario(s) generated project a %.2fs floor and one "
            "scenario alone already projects past the order. Refitting to the smallest "
            "plan available, 1 scenario (~%.2fs) — the refit floor is 1 because a "
            "zero-beat film is not a deliverable. If that figure holds, the delivered "
            "film will overrun the order by ~%.2fs; raise the order or take the "
            "per-scenario length up with %s.",
            ordered,
            per_beat_seconds,
            source,
            n_scenarios,
            floor_seconds,
            per_beat_seconds,
            per_beat_seconds - ordered,
            "bdd" if yield_source == _BDD_GUARANTEED_YIELD else "whoever sets the pacing",
        )
        degradations.append(
            degradation(
                ORDERED_DURATION_INFEASIBLE,
                f"ordered {ordered:.2f}s; the per-scenario recording length planned "
                f"against is {per_beat_seconds:.2f}s ({source}), so the {n_scenarios} "
                f"scenario(s) generated project a {floor_seconds:.2f}s floor and one "
                f"scenario alone already projects {per_beat_seconds - ordered:.2f}s past "
                f"the order; refitted to 1 scenario (~{per_beat_seconds:.2f}s) rather "
                f"than trimming the single scenario{dwell_clause}",
            )
        )
    else:
        log.warning(
            "demo_video_gen: ordered %.2fs is below the %.2fs floor projected for %d "
            "scenario(s) at %.2fs of recording each (%s) — refitting the plan to %d "
            "scenario(s) (~%.2fs) so the order is deliverable against that figure",
            ordered,
            floor_seconds,
            n_scenarios,
            per_beat_seconds,
            source,
            fitting,
            fitting * per_beat_seconds,
        )
    return (
        _apply_target_beats(
            lines,
            {"target_beats": fitting},
            per_beat_seconds,
            deliverable,
            floor_ratio,
        ),
        degradations,
    )


def _apply_duration_ceiling(
    lines: list[DemoVideoLine],
    deliverable: Mapping[str, Any] | None,
    per_beat_seconds: float | None,
    floor_ratio: float,
) -> list[DemoVideoLine]:
    """Trim a spine that OVERSHOOTS the ordered duration at the measured per-beat yield.

    The floor had a guard from the start; the ceiling never did. Measured on live run
    ``01a089c3`` (2026-09-10): a 30s order produced 5 beats, bdd guaranteed 22.00s per
    scenario, and :func:`_assert_plan_can_reach_floor` logged "duration plan OK — 5
    beat(s) x 22.00s" because 5 clears the 1 beat the 50% floor needs. The film shipped
    at 218.47s — 728% of the order — and every gate passed it, because
    ``reconcile_timeline`` diagnoses an overshoot (``cant_converge``) and then completes
    anyway.

    This is the mirror of the relief lever, and it deliberately reuses that lever's
    machinery rather than adding a second drop order: same protected set (opener, hook,
    pin/time_lock), same deterministic ranking, same ``id`` preservation for
    ``demo_video_record``'s ``clip-<n>`` join. Only the trigger differs — the relief
    lever fires when a PARENT asks for a shorter spine, this fires when the ORDER itself
    cannot hold the spine that was built.

    Makes NO judgement without a measured yield or an ordered target: with either absent
    the plan is ``UNMEASURED`` and the spine stands unchanged. An absent measurement is
    not evidence of a bad plan.
    """
    ordered = _target_seconds(deliverable or {})
    if ordered is None or not lines:
        return lines
    plan = plan_beats(
        len(lines),
        target_seconds=ordered,
        floor_ratio=floor_ratio,
        per_beat_seconds=per_beat_seconds,
        max_beats=len(lines),
    )
    if plan.verdict != "RESIZED_DOWN":
        return lines
    log.warning(
        "demo_video_gen: spine overshoots the order — %s; trimming %d beat(s) to %d",
        plan.explain(),
        len(lines) - plan.beats,
        plan.beats,
    )
    return _apply_target_beats(
        lines,
        {"target_beats": plan.beats},
        per_beat_seconds,
        deliverable,
        floor_ratio,
    )


def _assert_plan_can_reach_floor(
    lines: list[DemoVideoLine],
    deliverable: Mapping[str, Any],
    per_beat_seconds: float | None,
    floor_ratio: float,
) -> str | None:
    """FAIL EARLY: refuse a spine that provably cannot clear the underfull floor.

    ``demo_video_record`` already computes this exact prediction one stage later
    ("capture yielded 2.56s; the 30.00s order needs >= 15.00s") and discards it as
    note-only by explicit design, so the mismatch was KNOWN before rendering and acted
    on never — ``reconcile_timeline`` raised on it after the render was billed. Billing
    in this fleet is append-only with no refund path anywhere (negative amounts are
    rejected at ingest), so those seconds are permanently unrecoverable.

    This is the earliest point the same arithmetic is knowable: the spine is final (bdd
    has returned, relief has cut), the ordered target is in hand, and the per-beat yield
    is a measurement. Nothing has been recorded or rendered yet.

    RE-SIZING IS NOT AVAILABLE HERE and refusing is therefore correct: the count is a
    ceiling, not a quota (see :func:`_scenarios_for_target`) — bdd returned what it
    returned, and asking again for more scenarios does not produce more. Where a re-size
    IS available the pipeline takes it instead of refusing: that is the clamp in
    :func:`~src.services.runs.structural_relief_policy.decide_relief_shape`, which
    raises a relief child's ``target_beats`` back up to the floor rather than spawning a
    child that must fail. This function is the last position where refusing still saves
    the record and render, and it fires only when NO valid sizing exists.

    Returns the refusal message, or ``None`` when the plan is legal or unjudgeable.
    ⛔ It does NOT weaken, widen, or duplicate the 50% floor — it applies the SAME
    ``demo_timeline_underfull_fail_ratio`` earlier. ``reconcile_timeline``'s gate stays
    exactly where it is armed, as the last line of defence.
    """
    target = _target_seconds(deliverable)
    if target is None or not lines:
        return None
    plan = plan_beats(
        len(lines),
        target_seconds=target,
        floor_ratio=floor_ratio,
        per_beat_seconds=per_beat_seconds,
        # bdd's output is what it is: the planned spine cannot grow past the beats that
        # actually came back, so the largest available count IS the current one.
        max_beats=len(lines),
        # ⭐ Wave 13, O4. The ENUMERATED ceiling check. Without it `plan_beats` bounds
        # the overshoot with a rounded-UP quotient, which cannot detect an overshoot of
        # the ORDER — so a spine of 2 beats x 22.00s measured against a 30.00s order
        # came back `OK` at +46.7%. As of wave 14 (N4) this is also `plan_beats`' OWN
        # DEFAULT — it was opt-in, and three of the four call sites (including the paid
        # relief path) omitted it, so the bound they read was the rounded quotient.
        # Still passed EXPLICITLY here, because this guard refuses a paid render and the
        # band it refuses against should be readable at the call site.
        #
        # ⛔ This is `duration_conformance.DURATION_TOLERANCE_FRACTION` — the band the
        # PROBED FILE is graded by — and NOT `time_budget.DEFAULT_TOLERANCE_PCT` (2.0),
        # which is the solver's underfull band on its own allotment sum. Different
        # objects; never merge them.
        tolerance_fraction=DURATION_TOLERANCE_FRACTION,
    )
    if plan.satisfiable:
        if plan.verdict != "UNMEASURED":
            log.info("demo_video_gen: %s", plan.explain())
        return None
    return (
        f"demo_video_gen: PLAN CANNOT REACH THE DURATION FLOOR — {plan.explain()}. "
        f"Refusing to record and render a film that is arithmetically certain to be "
        f"rejected by reconcile_timeline's {floor_ratio:.0%} underfull gate. Either "
        f"lower the ordered duration ({target:.2f}s) or give the app more to demo "
        f"(more focus_areas / target_pages) so more scenarios can be generated."
    )


def _bdd_login(brief: dict) -> dict | None:
    """The ONE login bdd takes (it scrapes + scaffolds the target app).

    The brief may carry a list (one login per origin, for a multi-app recording); bdd
    gets the entry for ``target_url``'s origin, unsealed in memory only — the brief
    stores it sealed (``auth_secrets``). When several logins share that origin (told
    apart by ``label``), ``brief.login_label`` names the one bdd scaffolds with."""
    label = brief.get("login_label")
    return auth_for_origin(
        unseal_auth(brief.get("auth_config")),
        brief.get("target_url"),
        brief,
        label=label if isinstance(label, str) and label else None,
    )


class DemoVideoGenInput(StageInput):
    pass  # All inputs come from ctx.config_snapshot["brief"].


class DemoVideoLine(AuthoredLine, StageOutput):
    """One demo scenario: the narration text + its ``clip-<n>`` join id.

    Stage-output twin of the API request model ``src.api.schemas.ShowcaseLine``. Both
    share the ``AuthoredLine`` field set (single source of truth in
    ``src.models.showcase_line``), so they can no longer drift. This variant carries the
    ``StageOutput`` marker and stays permissive — the API door, not the stage, enforces
    non-empty narration / ``clip_anchor='end'``; the stage legitimately builds looser
    lines (a footage-only beat with empty text, a free-form ``clip_anchor``).
    """

    # Required here (the base defaults it) — a stage-output addressable element MUST
    # carry an id; PR-6 annotations wire against it.
    id: str
    # Set by reconcile_timeline's MEASURED line fit (`narration_fit`): the seconds of
    # the plain-text clip it synthesized at (text, voice, ``vo_rate``). Present ⇒ narrate
    # replays exactly that clip — no SSML, no emphasis, no fit re-synthesis. ``None`` on
    # every unfitted line (fit fell back / not armed) — today's narrate path unchanged.
    fitted_seconds: float | None = None
    # The seconds a voice_director leading pause may take on a fitted line without
    # changing the film: the footage window the measured clip leaves unspoken, minus
    # its hold. 0.0 on a non-footage beat (its shot follows the audio, so any pad
    # would lengthen the film). A pause that does not fit is dropped.
    fitted_slack_seconds: float | None = None


class DemoVideoGenOutput(StageOutputWithNotes):
    # `lines` IS the demo_video spine — narrate, record, slides, sound_director,
    # footage_direction and compile all fan out over it, and compile raises a bare
    # RuntimeError when it is missing. Shared by FOUR handlers (gen, script_doctor,
    # reconcile_timeline, retention_plan) and inherited by ClaimFinishOutput; all
    # of them either hard-fail or skip-clean before emitting empty lines, EXCEPT
    # `script_doctor`, which passes `gen["lines"]` straight through — so an empty
    # here is real upstream drift, which is exactly what this declaration is for.
    # Every other field is legitimately empty in authored/showcase mode
    # (`bdd_job_id=""`, `features_generated=0`, `package_url=""`,
    # `timeline_manifest=None` off the relief loop), so none of them is declarable.
    produces: ClassVar[tuple[str, ...]] = ("lines",)

    lines: list[DemoVideoLine]
    bdd_job_id: str
    features_generated: int = 0
    package_url: str = ""
    # bdd's grounding report — see ``BddStepCoverage`` for the ABSENT (``None``)
    # / EMPTY (reported, every ``*_flagged`` count 0) / NON-EMPTY contract.
    # ``None`` on the authored/showcase path (bdd never ran — that is a
    # legitimate ABSENT, not a defect: those paths produce no LLM-invented
    # Gherkin to ground) and on any pre-this-change stage output being replayed.
    step_coverage: BddStepCoverage | None = None
    # Structured C13 relief-loop provenance (spec 23 §7.7), set by reconcile_timeline
    # when the opt-in relief loop runs. None on every gen-produced / off-relief output —
    # the string ``notes[]`` manifest is kept in parallel, this is the machine-assertable
    # twin that scripts/e2e_validate.py can read by dotted path.
    timeline_manifest: TimelineManifest | None = None
    # THE SINGLE-OWNER wpm this run's word budget was resolved at, once the draft
    # existed to sample (see `_resolve_effective_wpm_for_draft` / `resolve_effective_wpm`).
    # `demo_video_reconcile_timeline` reads this back rather than independently
    # re-measuring, which is the whole point: the budget and the solver must never be
    # able to disagree about the rate for the same script. None only when gen has not
    # yet run this Phase-B step (a pre-this-change stage output being replayed) — every
    # gen output from here on sets all four fields together.
    resolved_wpm: float | None = None
    resolved_wpm_measured: bool = False
    resolved_wpm_reason: FallbackReason | None = None
    resolved_wpm_basis: RateBasis | None = None
    # WHICH TEXT `resolved_wpm` WAS SAMPLED OVER — `spoken_text_digests` (one per line) of
    # the exact line texts handed to the sampler. A rate is a property of the text, not of the
    # run: `script_doctor` rewrites every line after gen samples the draft, and the
    # draft's quoted URLs alone made run 01a0afab's sample read 111.7 wpm for lines
    # delivered at ~150. `reconcile_timeline` reuses the stored rate ONLY when every one of
    # its own lines digests to a member of this list, and re-samples them otherwise. None on a pre-this-change
    # output ⇒ "unknown text" ⇒ the consumer re-samples rather than trusting it.
    resolved_wpm_text_digests: list[str] | None = None
    # THE TWO-TERM DECOMPOSITION of the same sample `resolved_wpm` was pooled from —
    # how much of the measured audio was SPEECH, how much was PAUSE, over how many
    # approximate sentence pieces and how many lines reported it at all.
    #
    # ⛔ MEASUREMENT ENABLEMENT ONLY — nothing in this service branches on these, and
    # nothing may start. A clip's duration is approximately
    # `words / speech_rate + n_pieces x per_piece_padding`, and the second term scales
    # with SENTENCE COUNT, which the LLM picks AFTER the budget is written. That two-term
    # budget was costed against real clips and REJECTED for exactly that reason
    # (`speech_rate._log_two_term_decomposition` carries the numbers: `s*` 0.83-1.24 once
    # the piece count is off by +/-2). These fields ship the EVIDENCE a future fit needs
    # from production; they are not the fit.
    #
    # Until 2026-09-15 `speech_rate` computed all four and dropped them at a
    # DIAGNOSTIC-ONLY log line, so no fleet-wide sample accumulated anywhere a query
    # could reach. Persisting them here — beside the rate they decompose, on the output
    # `reconcile_timeline` already reads back — is the whole change.
    #
    # `lines_split == 0` means NO EVIDENCE (a tts backend that does not report the
    # split), NOT "zero pause". Read it before reading the seconds.
    speech_seconds_split: float = 0.0
    pause_seconds_split: float = 0.0
    approx_pieces_split: int = 0
    lines_split: int = 0
    # THE STATE `_apply_feasibility_floor` USED TO SWALLOW. True when a duration was
    # ordered but no per-scenario recording length existed to bound the spine against, so
    # NO structural bound was applied at plan time (see `structural_bound_unavailable` for
    # why this is a fact on the output rather than a degradation). It was previously a
    # `log.warning` and nothing else — invisible to the stage output, to E2E assertions,
    # and to every later stage. False on a bounded run, so a healthy output is unchanged.
    structural_bound_unavailable: bool = False
    # WHERE the per-scenario figure came from when there WAS one (`_BDD_GUARANTEED_YIELD`
    # / `_PARENT_MEASURED_YIELD`), None when there was none. Provenance travels with the
    # number everywhere else in this module; it had no way to leave the stage until now.
    structural_bound_source: str | None = None
    # THE WORD BUDGET RE-SPLIT OVER THE BEATS THAT SURVIVED THE PLAN, keyed by line id.
    # bdd wrote each beat to ``order / scenarios_generated`` words; the plan then drops
    # beats to fit the order (run 01a156a6: 5 -> 3), and the dropped beats' share of the
    # order was simply lost — the three survivors kept 1/5-of-the-order texts against
    # 1/3-of-the-order footage. This is the order's narration budget split over the
    # survivors, so ``reconcile_timeline``'s re-ask can ask each one for its real share.
    # ``None`` when nothing was ordered.
    narration_words_per_beat: dict[str, int] | None = None


def survivor_word_budgets(
    lines: list[DemoVideoLine], narration_seconds: float | None, wpm_natural: float
) -> dict[str, int] | None:
    """The order's narration word budget split over ``lines`` (the SURVIVING beats).

    Same arithmetic as the budget bdd was sent (:func:`narration_words_target` split
    by :func:`narration_words_per_scenario`), taken over the post-plan beat count rather
    than the count bdd generated. ``None`` without an order, a usable rate, or lines.
    """
    if narration_seconds is None or narration_seconds <= 0 or wpm_natural <= 0 or not lines:
        return None
    split = narration_words_per_scenario(
        narration_words_target(narration_seconds, wpm_natural), len(lines)
    )
    if split is None:
        return None
    return {str(line.id): words for line, words in zip(lines, split, strict=True)}


class DemoVideoGenStage(StageHandler[DemoVideoGenInput, DemoVideoGenOutput]):
    name = "demo_video_gen"
    input_model = DemoVideoGenInput
    output_model = DemoVideoGenOutput
    provider_slots: ClassVar[tuple[SlotUse, ...]] = (
        SlotUse("llm"),
        SlotUse("tts"),
    )

    def __init__(
        self, client: BDDPlaywrightClient | None = None, tts_client: TTSClient | None = None
    ) -> None:
        self._client = client or BDDPlaywrightClient()
        # Named `tts_client` to match `ReconcileTimelineStage.__init__` — the two stages
        # run the SAME calibration probe, and the demo_video conftest's no-network guard
        # names that keyword in the error it raises.
        self._tts = tts_client

    async def _budget_wpm(self, ctx: RunContext) -> float:
        """The natural wpm this run's word budget is taken at (Phase A, pre-draft) —
        the SAME single-owner ladder ``demo_video_reconcile_timeline`` re-resolves
        against. See :func:`resolve_effective_wpm`'s module banner in
        ``speech_rate.py`` for why there is exactly one such ladder.

        Called BEFORE bdd has generated anything, so ``texts=()`` — this is
        necessarily Phase A only (an authored rate, else the calibration-passage
        measurement held to at most :data:`DEFAULT_WPM_NATURAL` by
        :func:`~src.services.stages.website_review.script.floor_wpm`, else
        :data:`DEFAULT_WPM_NATURAL`): there is nothing yet to sample. :meth:`_resolve_effective_wpm_for_draft` re-resolves once the draft
        exists and its answer, not this one, is what gets stored for the solver.

        Never raises — :func:`resolve_effective_wpm` swallows its own failures and
        returns the default, so a sick tts_backend costs the budget its calibration,
        not the run.
        """
        effective = await resolve_effective_wpm(
            ctx, texts=(), client=self._tts, default=DEFAULT_WPM_NATURAL
        )
        if effective.rate.measured:
            # ⛔ NEVER the calibration-passage rate as such. Phase A measures a fixed
            # passage that reads FASTER than real narration (a showcase run measured the
            # passage at 197.6 wpm and the drafted content at 168.8), so a budget
            # taken at it asks bdd for ~17% more words than `reconcile_timeline` — which
            # solves at the CONTENT rate gen stores below — will find room for. The
            # same mismatch website_review fixed in #1076; same helper: a
            # calibration-basis rate is held to at most DEFAULT_WPM_NATURAL, an
            # authored (or any content-measured) rate passes through unchanged.
            return floor_wpm(effective.rate.wpm, effective.rate.basis)
        log.warning(
            "demo_video_gen: speech-rate probe did not answer (reason=%s) — budgeting "
            "narration words at the %.1f wpm default.",
            effective.rate.reason,
            effective.rate.wpm,
        )
        return DEFAULT_WPM_NATURAL

    async def _resolve_effective_wpm_for_draft(
        self, ctx: RunContext, lines: list[DemoVideoLine]
    ) -> ScriptSpeechRate:
        """PHASE B — measure the draft that just came back, before the word-budget
        verdict (beat-count / duration-ceiling / feasibility-floor) is finalised.

        Samples the ACTUAL drafted lines (never the calibration passage) via the
        SAME single-owner :func:`resolve_effective_wpm` ``_budget_wpm`` calls, so an
        authored rate or a Phase-A fallback that already won at draft-1 sizing wins
        again here (memoised, so this costs no extra probe on that path) rather than
        silently disagreeing with it. The result is stored on the stage output (see
        ``handle``) so ``demo_video_reconcile_timeline`` solves this exact script at
        the exact rate it was budgeted at, instead of independently re-measuring.
        """
        texts = spoken_texts(str(line.text or "") for line in lines)
        return await resolve_effective_wpm(
            ctx, texts=texts, client=self._tts, default=DEFAULT_WPM_NATURAL
        )

    async def handle(self, payload: DemoVideoGenInput, ctx: RunContext) -> DemoVideoGenOutput:
        brief = ctx.config_snapshot.get("brief") or {}

        # Authored mode: the caller supplied the narration/dialogue spine itself
        # (a persisted/approved storyboard WINS over legacy brief.lines), so skip bdd
        # generation. demo_video_record uses brief.package_url to feed the proven suite
        # to the frontend_test_runner instead of a bdd_job_id. With no storyboard,
        # authored_lines() rebuilds the spine from brief.lines field-for-field →
        # byte-identical to the previous inline mapping.
        deliverable = ctx.config_snapshot.get("deliverable") or {}
        # The measured per-beat yield this run may plan against — a READ of one owner's
        # value, never a constant. See `_measured_seconds_per_beat`.
        per_beat_seconds = _measured_seconds_per_beat(ctx)
        # PROVENANCE TRAVELS WITH THE NUMBER. Whichever rung wins below, the message
        # built on it must be able to name where it came from — see
        # _BDD_GUARANTEED_YIELD / _PARENT_MEASURED_YIELD.
        yield_source = _PARENT_MEASURED_YIELD if per_beat_seconds is not None else None

        ordered_seconds = demo_video_ordered_seconds(ctx.config_snapshot)
        # THE SECONDS THE NARRATION MAY FILL — the order minus the intro/outro cards
        # (owner ruling: cards count inside the order). Every word budget below — the
        # one sent to bdd, the shortfall grade, the ceiling — is taken over THIS, never
        # over the raw order. See `narration_seconds_for_order`.
        narration_seconds = narration_seconds_for_order(ctx.config_snapshot, ordered_seconds)

        authored = authored_lines(brief)
        if authored is not None:
            output = self._authored_output(authored, brief)
        else:
            output, exported = await self._generated_output(
                brief,
                deliverable,
                per_beat_seconds,
                # The ORDERED film duration, sent DOWN to the generator (see
                # `_generated_output`). Resolved on demo_video's own target ladder —
                # the same one `reconcile_timeline` and the conformance record use —
                # not the deliverable rung alone, so an explicit
                # `direction.timing.target_seconds` override reaches bdd too.
                ordered_seconds,
                # THE RATE THE BUDGET IS TAKEN AT. Resolved here (the only place with
                # `ctx`) on `reconcile_timeline`'s own ladder, so the words bdd is asked
                # for and the words the solver later checks against are the same
                # arithmetic at the same rate — see `_budget_wpm` and
                # `narration_words_target`.
                await self._budget_wpm(ctx),
                narration_seconds,
            )
            # RUNG ONE WINS. bdd GUARANTEES this pacing for the suite it just generated;
            # a parent run's measured footage is history by comparison, and on a relief
            # child it is history of a DIFFERENT suite. Sizing a regenerated child off
            # the parent's measurement is precisely how 11 of the 12 production failures
            # were shaped.
            if exported is not None:
                per_beat_seconds = exported
                yield_source = _BDD_GUARANTEED_YIELD

        # PHASE B — the draft now exists (bdd-generated or authored). Measure it
        # BEFORE the word-budget verdict below (beat-count / duration-ceiling /
        # feasibility-floor) is finalised, and store the result so
        # `demo_video_reconcile_timeline` solves this exact script at the exact rate
        # it was budgeted at rather than independently re-measuring. See
        # `resolve_effective_wpm`'s module banner in `speech_rate.py`.
        # A rate nobody MEASURED is recorded HERE, on the stage that budgets the
        # narration words from it. It used to be silent: `_budget_wpm` only logged, and
        # `reconcile_timeline` reports an unmeasured voice on its CAST path only, so a
        # single-voice run whose probe failed (tts 503 → `probe_error`) stored
        # `resolved_wpm_measured: false` and completed clean, budgeted at a constant
        # known to be wrong for the voice. `reconcile_timeline`'s single-voice path
        # reads this same stored rate back and adds nothing, so the fact is recorded
        # once per run.
        effective = await self._resolve_effective_wpm_for_draft(ctx, output.lines)
        rate_degradations = (
            [speech_rate_fallback_degradation(stage=self.name, rate=effective.rate)]
            if effective.rate.assumed
            else []
        )
        output = output.model_copy(
            update={
                "resolved_wpm": effective.rate.wpm,
                "resolved_wpm_measured": effective.rate.measured,
                "resolved_wpm_reason": effective.rate.reason,
                "resolved_wpm_basis": effective.rate.basis,
                # The text that rate belongs to. Same normalisation the sampler was
                # handed (`spoken_texts`), so reconcile can prove it holds the SAME text.
                "resolved_wpm_text_digests": spoken_text_digests(
                    str(line.text or "") for line in output.lines
                ),
                # Recorded in the SAME update as the rate they decompose, so a stage
                # output can never carry one without the other. Zeros when the sample
                # never ran or no line reported a split — see the field comments.
                "speech_seconds_split": effective.speech_seconds_split,
                "pause_seconds_split": effective.pause_seconds_split,
                "approx_pieces_split": effective.approx_pieces_split,
                "lines_split": effective.lines_split,
            }
        )

        # Relief-loop lever (PR-C): a re-run may request a SHORTER spine (fewer
        # beats) to claw back budget when footage is pinned. Applied AFTER the spine
        # is built (both authored and generated paths) and BEFORE speaker casting so
        # dropped beats never get a voice slot. Gated by the SAME kill-switch as the
        # spawner (structural_relief_autoconverge_enabled) so the whole feature is fully
        # off when the flag is off — a stray `structural_relief` in config_overrides can't
        # trigger the lever behind the kill-switch's back.
        pipeline = get_settings().pipeline
        relief = (
            ctx.config_snapshot.get("structural_relief")
            if pipeline.structural_relief_autoconverge_enabled
            else None
        )
        floor_ratio = pipeline.demo_timeline_underfull_fail_ratio
        lines = _apply_target_beats(
            output.lines, relief, per_beat_seconds, deliverable, floor_ratio
        )
        # The other half of the contract. The relief lever above only fires when a
        # PARENT asked for a shorter spine; nothing bounded a first-pass spine against
        # the duration actually ordered. See _apply_duration_ceiling.
        lines = _apply_duration_ceiling(lines, deliverable, per_beat_seconds, floor_ratio)
        # ...and the half NEITHER of those two covers. The ceiling above rounds UP
        # (``ceil(target / per_beat)``), so a spine whose per-scenario floor does not
        # divide the order is judged OK while sitting structurally above it — that is
        # exactly the 30s order / 2 x 22.00s / 47.045s delivered shape. This refits the
        # count to what the order can actually hold, and degrades loudly when even one
        # scenario cannot fit. See _apply_feasibility_floor.
        lines, feasibility_degradations = _apply_feasibility_floor(
            lines, deliverable, per_beat_seconds, floor_ratio, yield_source
        )

        # FAIL EARLY — the whole point. The spine is final here and nothing has been
        # recorded or rendered, so this is the earliest point at which "will this film
        # clear the underfull floor?" is answerable. It used to be answered three stages
        # later, after both of those were paid for.
        refusal = _assert_plan_can_reach_floor(
            lines,
            deliverable,
            per_beat_seconds,
            floor_ratio,
        )
        if refusal:
            raise RuntimeError(refusal)

        # THE WORD-BUDGET RECONCILIATION, on the live path at last. Advisory-tier by
        # design: this is a PRE-AUDIO estimate (`words / wpm`), and the only number that
        # can grade a delivery is the delivered one — which is exactly what
        # NARRATION_ESTIMATE_OFF_TARGET exists to carry without filing a second, worse
        # opinion on a question the compile stages answer from the real artifact.
        shortfalls = authored_word_shortfalls(
            lines,
            narration_seconds,
            # The single-owner rate Phase B just resolved, NOT the constant — the same
            # number the budget was sent to bdd at and the same one reconcile_timeline
            # will solve against. Grading a script against a rate nobody else used is how
            # the two halves disagreed before (see `narration_words_target`).
            output.resolved_wpm or DEFAULT_WPM_NATURAL,
        )
        advisories = [*output.advisories]
        if shortfalls:
            named = ", ".join(
                f"{bid} has {has} words, budgeted {want}"
                for bid, (has, want) in sorted(shortfalls.items())
            )
            advisories.append(
                advisory(
                    NARRATION_ESTIMATE_OFF_TARGET,
                    f"demo_video_gen: {len(shortfalls)} beat(s) came back SHORT of their "
                    f"narration word budget — {named}. A beat below its budget cannot fill "
                    f"its seconds at a natural rate, so its delivery rate is dragged toward "
                    f"the floor and the beat still ships short. Flagged before any footage "
                    f"is recorded; reconcile_timeline re-asks for the missing words once "
                    f"the real clip lengths are known.",
                )
            )
        # ...and the LONG side, which had no check at all. A script whose words alone,
        # spoken at the rate its own draft was measured at, run past the band's upper
        # edge over (order - cards) is an overshoot NO downstream trim recovers — a
        # replay over 31 runs found that shape in 15 of 16 overshoots. Graded on BOTH
        # paths (bdd-generated and authored), at the content rate, never calibration.
        ceiling = narration_word_ceiling(
            narration_seconds, output.resolved_wpm or DEFAULT_WPM_NATURAL
        )
        total_words = sum(len(str(line.text or "").split()) for line in lines)
        if ceiling is not None and total_words > ceiling:
            advisories.append(
                advisory(
                    NARRATION_ESTIMATE_OFF_TARGET,
                    f"demo_video_gen: the script holds {total_words} words, over its "
                    f"{ceiling}-word ceiling ({narration_seconds:.2f}s of narration "
                    f"after title cards x {output.resolved_wpm or DEFAULT_WPM_NATURAL:.1f} "
                    f"wpm content rate x {_WORD_CEILING_RATIO:.2f}). Its narration alone "
                    f"delivers past the order's upper band edge.",
                )
            )

        return output.model_copy(
            update={
                "lines": _cast_speakers(lines, brief),
                # APPENDED, never replaced: _generated_output / _authored_output may
                # already have raised their own entries, and a feasibility refit must not
                # silently drop somebody else's quality signal.
                "degradations": [
                    *output.degradations,
                    *rate_degradations,
                    *feasibility_degradations,
                ],
                "advisories": advisories,
                # The plan-time bound state, no longer log-only. Computed from the SAME
                # predicate `_apply_feasibility_floor` branches on, so the field and the
                # behaviour cannot disagree.
                "structural_bound_unavailable": structural_bound_unavailable(
                    lines, deliverable, per_beat_seconds
                ),
                "structural_bound_source": yield_source,
                "narration_words_per_beat": survivor_word_budgets(
                    lines, narration_seconds, output.resolved_wpm or DEFAULT_WPM_NATURAL
                ),
            }
        )

    def _authored_output(
        self, authored: list[AuthoredLine], brief: dict[str, Any]
    ) -> DemoVideoGenOutput:
        # authored lines arrive already normalized to the AuthoredLine spine (id via
        # normalize_clip_key, text/seconds/clip coerced). _normalize_speaker stays here
        # (stage policy: the a/b cast-slot alias map) and is applied to each line's
        # speaker exactly as the previous inline loop did.
        #
        # The twin is built by DUMPING the base — never field-by-field. DemoVideoLine
        # adds no fields of its own (only tightens `id` to required), so a whole-model
        # dump carries EVERY AuthoredLine slot, including the ones a hand-listed
        # constructor kept silently dropping (locked / beat_role / doctor_note /
        # hook_role / pin_position) and the A22 timeline slots (time_lock /
        # min_seconds / max_seconds / vo_rate). New base fields now reach the stage
        # spine for free.
        lines: list[DemoVideoLine] = [
            DemoVideoLine.model_validate(
                {**a.model_dump(), "speaker": _normalize_speaker(a.speaker)}
            )
            for a in authored
        ]
        if not any(line.text for line in lines):
            raise RuntimeError("demo_video_gen: authored brief.lines have no narration text")
        return DemoVideoGenOutput(
            lines=lines,
            bdd_job_id="",
            features_generated=0,
            # Pass the hand-authored package through for demo_video_record; a bare
            # /api/bdd path here would be meaningless without a bdd_job_id.
            package_url=str(brief.get("package_url") or "").strip(),
            notes=_numeral_consistency_notes(lines),
        )

    async def _generated_output(
        self,
        brief: dict[str, Any],
        deliverable: Mapping[str, Any],
        per_beat_seconds: float | None,
        ordered_seconds: float | None = None,
        wpm_natural: float = DEFAULT_WPM_NATURAL,
        narration_seconds: float | None = None,
    ) -> tuple[DemoVideoGenOutput, float | None]:
        """Generate via bdd, and hand back the per-scenario duration bdd EXPORTED.

        The second element is rung ONE of the yield ladder (see
        :func:`_measured_seconds_per_beat`): the mean of the parts'
        ``expected_recording_seconds``, or ``None`` when bdd published none. It is read
        off the wire and never restated in this repo — a copy of that number here would
        desync the planner the next time bdd's ``HERO_DWELL_MS`` changes, which is the
        drift this whole change exists to remove.
        """
        url = (brief.get("target_url") or brief.get("source_url") or brief.get("url") or "").strip()
        if not url:
            raise RuntimeError(
                "demo_video_gen: brief.target_url (the app/site URL to demo) is required"
            )

        tone = str(brief.get("narration_tone") or "tutorial")
        bdd_payload: dict[str, Any] = {
            "url": url,
            "max_scenarios_per_feature": _scenarios_for_target(
                brief, deliverable, per_beat_seconds
            ),
            "narration": {"enabled": True, "tone": tone, "llm_polish": True},
            # Orchestrator owns TTS + record + compile — bdd must not fire its own
            # fire-and-forget downstream.
            "feed_to_video": False,
        }
        # THE SCENARIO FLOOR, beside the ceiling (wave 13, O3). `max_scenarios_per_feature`
        # is a CEILING, NOT A QUOTA, so until this key existed the ordered duration reached
        # bdd only as a bound it was free to undershoot — and it did, by 55% on run
        # 01a03011. See `_min_scenarios_for_target` for the derivation and for every case
        # in which the key is deliberately OMITTED rather than defaulted.
        min_scenarios = _min_scenarios_for_target(brief, deliverable, per_beat_seconds)
        if min_scenarios is not None:
            bdd_payload["min_scenarios"] = min_scenarios
        # THE ORDERED DURATION, SENT DOWN TO THE GENERATOR. Until this key existed the
        # order reached bdd only as a scenario-COUNT ceiling
        # (`max_scenarios_per_feature`), and a count is an integer: measured live
        # 2026-09-10, a 30s order yielded clips of ~23.8s each, so 1 beat delivered 85%
        # and 2 beats delivered 159% and NO integer beat count could land inside the
        # +/-10% conformance band. The overrun was structural, not a fit problem — bdd
        # authors BOTH the narration text and the dwell that paces the recording, and
        # neither had ever heard the ordered length. This sends the constraint to the
        # producer instead of trimming its output downstream.
        #
        # OMITTED, never defaulted, when nothing was ordered: demo_video has no
        # blueprint fallback duration (`reconcile_timeline` skips clean rather than
        # invent one), and a fabricated number here would make bdd budget narration
        # against an order nobody placed.
        #
        # Forward/backward compatible in BOTH directions with no negotiation: bdd's
        # `DemoVideoRequest` declares no `extra=` policy, so pydantic's `extra='ignore'`
        # default means an older bdd silently drops the key — no 422, and the two halves
        # of this change may deploy in either order.
        # ``> 0``, not merely ``is not None``: a zero or negative order is not an order.
        # ``demo_video_ordered_seconds`` coerces (and rejects ``bool``) but does not
        # judge the sign — ``duration_conformance`` applies that test at ITS door, and
        # ``_target_seconds`` applies it at this stage's. Sending ``0`` would ask bdd to
        # write a zero-second film; it is the same "absent order" case, so it takes the
        # same branch.
        if ordered_seconds is not None and ordered_seconds > 0:
            bdd_payload["target_duration_seconds"] = ordered_seconds
            # THE WORD BUDGET, sent alongside the duration. A duration alone is not
            # actionable by a SCRIPT WRITER: bdd authors the narration text, and a film
            # is exactly its narration, so the only lever that controls length at the
            # point of authoring is WORD COUNT. Derived from this repo's own
            # `DEFAULT_WPM_NATURAL` / `NARRATION_BUDGET_RATE` (imported, never restated)
            # so the budget bdd writes to and the budget `reconcile_timeline` later
            # solves against are the same arithmetic.
            #
            # `narration_words_per_scenario` rides ONLY when the beat count is actually
            # known, i.e. the author pinned `max_scenarios_per_feature`. The DERIVED
            # count is a CEILING, not a quota (see `_scenarios_for_target`: a derivation
            # asked for 15 and bdd returned 5), so splitting the budget across it would
            # hand every beat ~1/3 of its real share. An absent key is honest; a split
            # against a count bdd will not hit is worse than no split at all.
            #
            # Extra keys are safe in both deploy orders for the same reason
            # `target_duration_seconds` is: bdd's `DemoVideoRequest` declares no
            # `extra=` policy, so pydantic's `extra="ignore"` default means an older bdd
            # drops them silently — no 422.
            # Priced over the seconds the narration may FILL (order minus title cards,
            # see `narration_seconds_for_order`), not the whole order — the cards are
            # spliced into the same film. `None` (a direct caller that did not resolve
            # it) means no cards were reserved, i.e. the whole order.
            words_target = narration_words_target(
                ordered_seconds if narration_seconds is None else narration_seconds,
                wpm_natural,
            )
            bdd_payload["narration_words_target"] = words_target
            explicit_beats = brief.get("max_scenarios_per_feature")
            per_scenario = (
                narration_words_per_scenario(words_target, int(explicit_beats))
                if explicit_beats
                else None
            )
            if per_scenario is not None:
                bdd_payload["narration_words_per_scenario"] = per_scenario

        # Optional brief fields — only sent when present so bdd applies its defaults.
        optional: dict[str, object] = {k: brief.get(k) for k in ("focus_areas", "target_pages")}
        login = _bdd_login(brief)
        optional["auth_config"] = login
        if login is not None and optional["target_pages"]:
            # target_pages is not part of the login's binding: beside a login, only
            # pages on its bound origins travel (bdd hands both to webscraper).
            kept, dropped = pages_for_login(
                optional["target_pages"], brief.get("target_url"), login
            )
            if dropped:
                log.warning(
                    "demo_video_gen: dropped %d target_page(s) outside the login's "
                    "bound origins (a saved password never leaves its site)",
                    dropped,
                )
            optional["target_pages"] = kept
        bdd_payload.update({k: v for k, v in optional.items() if v})

        job_id = await self._client.start_demo_video(bdd_payload)
        await self._client.wait_for_job(job_id, timeout=_WAIT_TIMEOUT_S)
        result = await self._client.get_job_result(job_id)
        if result.status != "completed":
            raise RuntimeError(
                f"demo_video_gen: bdd job {job_id} ended {result.status!r}: {result.error}"
            )

        lines = [
            DemoVideoLine(
                # Normalize the id join key like the authored path (authored_line_id)
                # so both derive the same FTR ``clip-<n>`` key — defensive against a
                # bdd part_id ever arriving ``@``-prefixed (record keys footage by it).
                id=normalize_clip_key(part.part_id),
                # bdd's text is MODEL-GENERATED: a URL in it is spelled out by the voice
                # and priced as one word by every budget downstream (run 01a0b007). Made
                # speakable HERE, before the Phase-B rate sample, so the rate, the solver
                # and narrate all see the same text. Authored brief.lines never pass here.
                text=speakable_text(part.text),
                seconds=part.estimated_duration_seconds,
                scenario_name=part.scenario_name,
                gherkin=part.original_gherkin,
                # O1b: pack-authored expected on-screen text, carried on the bdd
                # narration part (see bdd job_processor authored_expected_text).
                expected_text=(part.expected_text.strip() or None) if part.expected_text else None,
            )
            for part in result.narration_script.parts
            if part.part_id
        ]
        if not lines:
            raise RuntimeError(f"demo_video_gen: bdd job {job_id} produced no narration parts")

        exported = exported_seconds_per_beat(
            [part.expected_recording_seconds for part in result.narration_script.parts]
        )
        if exported is not None:
            log.info(
                "demo_video_gen: bdd exports %.2fs guaranteed recording per scenario "
                "over %d part(s) — planning against that, not a local constant",
                exported,
                len(result.narration_script.parts),
            )
        return (
            DemoVideoGenOutput(
                lines=lines,
                bdd_job_id=job_id,
                features_generated=result.features_generated,
                package_url=f"/api/bdd/{job_id}/download",
                step_coverage=result.step_coverage,
                notes=_numeral_consistency_notes(lines),
            ),
            exported,
        )
