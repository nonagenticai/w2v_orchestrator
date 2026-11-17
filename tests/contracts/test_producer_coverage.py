"""Inert-knob guard: every field the compiler DECLARES must have a producer.

This is the second half of the cross-service contract. Its sibling
``test_video_compile_contract.py`` answers "does the orchestrator invent fields
the compiler doesn't have?" — the direction ``extra="forbid"`` used to cover.
This module answers the OPPOSITE, and far more expensive, question:

    **Did the compiler grow a knob that NO orchestrator producer ever sets?**

That is this project's signature bug. A field is added to
``video_compiler_service``'s ``CompileRequest``, implemented in the renderer,
unit-tested there, deployed — and the orchestrator never emits it, so the
feature ships permanently inert while every test on both sides reads green.
Confirmed instances, each found by hand-grepping across two repos at the cost
of most of a session: ``duck_open_lu_below_vo``, ``mix_gain_targets``,
``broadcast_safe``, ``ratio``/``duck_db``.

The mechanism is plain set algebra over dotted leaf paths::

    schema_leaves(artifact) - collect_emitted_leaves() - JUSTIFIED - BACKLOG == {}

``collect_emitted_leaves()`` runs every real compile-calling stage scenario and
unions what each one actually put on the wire, so "covered" means *a producer
demonstrably emits it*, not *someone remembered to write it down*.

**Two registries, and they mean different things.** ``JUSTIFIED`` is a DECISION:
the shipped default is already the value a producer would send, so no producer
should override it and none ever will. ``UNCOVERED_BACKLOG`` is a KNOWN GAP: the
field may well be wired, but nothing in the scenario list observes it yet, so the
detector genuinely cannot tell — it is debt with a name, not a verdict.

Moving a path OUT of ``UNCOVERED_BACKLOG`` and into the emitted set — by wiring a
producer, or by adding a scenario that drives the stage which already emits it —
is the intended direction of travel; the ratchet below enforces that direction.
"""

from __future__ import annotations

from tests.contracts.schema_walk import load_schema, schema_leaves
from tests.contracts.test_video_compile_contract import collect_emitted_leaves

# --------------------------------------------------------------------------- #
# JUSTIFIED
# --------------------------------------------------------------------------- #
#: Declared compiler fields that NO producer sets, ON PURPOSE, with the reason.
#:
#: This registry exists for decisions somebody actually made and can defend. It
#: is **not** a dumping ground for knobs nobody looked at: adding a line here
#: silences the only automated check that a compiler feature is reachable at
#: all, so an entry without a real, specific reason is strictly worse than a
#: failing test. The honest remedies for an uncovered path are, in order of
#: preference:
#:
#:   1. **Wire a producer.** The knob exists because someone wanted the
#:      behaviour; make an orchestrator stage able to ask for it.
#:   2. **Justify it here** with a reason naming the default and why leaving the
#:      default in place is correct for this pipeline.
#:
#: "It's hard to test" is not a reason. "The scenario list doesn't cover that
#: stage yet" is not a reason either — that means the scenario list in
#: ``test_video_compile_contract.py`` is what needs extending.
#:
#: Seeded 2026-07-19 with the eight music/ambient sidechain-timing knobs. Each
#: was checked individually against the compiler's own schema description and
#: the project's -18 LUFS narration bus: in every case the shipped default is
#: already the value a producer would send, so wiring them buys per-run
#: TUNABILITY, not a bug fix. That is a genuine "no producer, and that's fine"
#: — unlike ``duck_open_lu_below_vo``, whose default (``None``) means the
#: measured VO-relative threshold never engages at all.
JUSTIFIED: dict[str, str] = {
    "music.duck_attack_ms": (
        "Default 30 ms. Fast enough that the bed is already down by the first "
        "syllable, slow enough not to chatter on plosives; no brief has ever "
        "asked for a different music duck onset, so the producer would only "
        "ever re-send 30.0."
    ),
    "music.duck_release_ms": (
        "Default 300 ms, chosen against the ~400 ms inter-sentence VO gap this "
        "pipeline's TTS produces: the bed audibly swells back inside a normal "
        "breath. Longer pins the bed flat (the old ambient 1000 ms bug), "
        "shorter pumps. Nothing per-run would move it."
    ),
    "music.duck_threshold": (
        "Default 0.03 linear (~ -30 dBFS) as an ABSOLUTE key level. Our "
        "narration bus is mastered to -18 LUFS, so ordinary speech crosses it "
        "reliably while room tone does not. A per-run value would have to be "
        "derived from the measured VO — which is what duck_open_lu_below_vo is "
        "for, and that one is NOT justified here."
    ),
    "music.sidechain_hpf_hz": (
        "Default 150 Hz high-pass on the sidechain KEY (not the audible "
        "narration), so the duck triggers on speech presence rather than "
        "rumble, plosives or room tone. 150 Hz sits below the vocal "
        "fundamental and above the noise floor for every voice this pipeline "
        "uses; it is a property of speech, not of a run."
    ),
    "ambient.duck_attack_ms": (
        "Default 30 ms, deliberately IDENTICAL to the music bed's so the "
        "renderer reads both buses uniformly. A divergent ambient onset would "
        "make the two beds duck out of step under the same narration."
    ),
    "ambient.duck_release_ms": (
        "Default 300 ms, matching music. This value is the fix for the dead "
        "ambient duck (hardcoded release=1000 pinned the bed flat through "
        "every VO gap); the ambient bed now breathes on the same ~400 ms gap. "
        "Re-sending the shipped default buys nothing."
    ),
    "ambient.duck_threshold": (
        "Default 0.03 linear (~ -30 dBFS), replacing the dead hardcoded 0.125 "
        "that a -18 LUFS narration bus never crossed (so the compressor barely "
        "opened). 0.03 is correct for our bus level; a run-specific absolute "
        "key level would be guesswork."
    ),
    "ambient.sidechain_hpf_hz": (
        "Default 150 Hz, same key high-pass as music. Ambient beds are exactly "
        "where an un-HPF'd key hurts most — low-frequency room/traffic tone in "
        "the bed's own content would otherwise self-trigger the duck."
    ),
    # -- request-level alias the pipeline deliberately routes around ---------- #
    "master_lufs": (
        "REDUNDANT ALIAS, deliberately not sent. The orchestrator drives final-mix "
        "loudness through the `mastering` block: a brief's `voice.master_lufs` is "
        "aliased onto `mastering.loudness_target` by `_alias_brief_master_lufs` "
        "(run_setup.py, orch #608), which rides the wire as `mastering.loudness_target` "
        'and the compiler folds into `tags["master_lufs"]`. Setting the top-level '
        "field TOO would give one target two wire spellings and a precedence question "
        "nobody wants to answer. NOTE: the earlier reason here claimed no producer "
        "COULD set it without a client signature change — that was FALSE; the path was "
        "always reachable via `mastering.loudness_target`, no client change needed."
    ),
    # -- whole-film tonal dials: per-shot ColorAdjust owns these -------------- #
    # Working AS DESIGNED, not backlog. `as_finish_payload` builds `finish.grade`
    # from the finishing-EXCLUSIVE knobs only and drops these four on purpose, so
    # the day the grade scenario lands they must NOT read as a coverage gap.
    "finish.grade.contrast": (
        "Quoting `as_finish_payload` (shared/finish.py): the tonal dials "
        "(contrast / saturation / gamma / temperature) are DELIBERATELY not sent — "
        "per-shot `ColorAdjust` owns those and a whole-film tonal layer would "
        "double-apply on top of a user's per-clip grade. Contrast is authored per "
        "shot via `shots.adjust.contrast`; a film-wide second application would "
        "compound it, not override it."
    ),
    "finish.grade.saturation": (
        "Quoting `as_finish_payload` (shared/finish.py): the tonal dials "
        "(contrast / saturation / gamma / temperature) are DELIBERATELY not sent — "
        "per-shot `ColorAdjust` owns those and a whole-film tonal layer would "
        "double-apply on top of a user's per-clip grade. Saturation is authored per "
        "shot via `shots.adjust.saturation`; stacking a film-wide pass over it "
        "oversaturates exactly the clips a user already pushed."
    ),
    "finish.grade.gamma": (
        "Quoting `as_finish_payload` (shared/finish.py): the tonal dials "
        "(contrast / saturation / gamma / temperature) are DELIBERATELY not sent — "
        "per-shot `ColorAdjust` owns those and a whole-film tonal layer would "
        "double-apply on top of a user's per-clip grade. Gamma is authored per shot "
        "via `shots.adjust.gamma`, and gamma composes multiplicatively, so a second "
        "film-wide pass is the worst of the four to double-apply."
    ),
    "finish.grade.temperature": (
        "Quoting `as_finish_payload` (shared/finish.py): the tonal dials "
        "(contrast / saturation / gamma / temperature) are DELIBERATELY not sent — "
        "per-shot `ColorAdjust` owns those and a whole-film tonal layer would "
        "double-apply on top of a user's per-clip grade. Whole-film warmth is "
        "instead expressed through the finishing-exclusive `look` / "
        "`highlight_tint` / `shadow_tint` knobs, which ARE sent."
    ),
}

# --------------------------------------------------------------------------- #
# UNCOVERED_BACKLOG
# --------------------------------------------------------------------------- #
# Group notes. Each says WHY the path is not observable yet — never "the default
# is fine" (that claim belongs in JUSTIFIED and nowhere else).
_CAPTIONS = (
    "no contract scenario exercises the word-timed captions surface yet: the "
    "cue build in CompileStageBase.handle is gated on the "
    "PIPELINE_CAPTIONS_DEFAULT_ON_SHOWCASE kill-switch AND a burned_in/overlay "
    "deliverable AND a stage overriding caption_clip_refs AND live storage + "
    "transcription_worker to time the words. Wiring exists (compiler #91 / orch "
    "#509); a scenario with a stubbed ASR + storage would observe it."
)
_GRADE = (
    "no contract scenario opts into the whole-film finishing grade: "
    "build_finish_spec only emits `finish.grade` when the run sets "
    "finish.enabled AND PIPELINE_GRADE_DEFAULT_ON_SHOWCASE is on (default off, "
    "so the feature is INERT live). A scenario that flips the setting would "
    "observe the GradeSpec the colorist already builds."
)
_SHOT_POLISH = (
    "per-shot polish knob with no producer in any EXERCISED stage. Several of "
    "these are wired in stages the scenario list does not drive (Studio "
    "per-clip edits, ORCH-3.2/3.3 directed footage, S14 freeze-pad); until a "
    "scenario drives those stages the detector cannot tell wired from inert."
)

#: Declared compiler fields no scenario OBSERVES yet — a known gap, not a decision.
#:
#: An entry here says "we cannot currently tell whether a producer sets this",
#: which is a materially weaker claim than JUSTIFIED's "no producer should".
#: Consequently this dict is a RATCHET: it may shrink, never grow (see
#: ``test_backlog_never_grows``). The fix for an entry is to drive the owning
#: stage from ``test_video_compile_contract.py`` — or, if the walk proves there
#: is no producer at all, to wire one.
UNCOVERED_BACKLOG: dict[str, str] = {
    # -- captions surface ---------------------------------------------------- #
    "caption_mood": _CAPTIONS,
    "captions.cues.end": _CAPTIONS,
    "captions.cues.start": _CAPTIONS,
    "captions.cues.text": _CAPTIONS,
    "captions.cues.words.end": _CAPTIONS,
    "captions.cues.words.start": _CAPTIONS,
    "captions.cues.words.text": _CAPTIONS,
    "captions.shot_index": _CAPTIONS,
    "captions_karaoke": _CAPTIONS,
    "captions_render": _CAPTIONS,
    # -- alpha overlay-clip surface: PRODUCER LANDED 2026-08-10, entries removed
    # (action_reel/graphics_design renders the card, action_reel/compile joins it
    # onto Shot.overlay_clips; observed by _emit_action_reel_overlays). ------- #
    # -- finish / grade surface ---------------------------------------------- #
    "finish.grade.broadcast_safe": (
        "same scenario gap as the rest of `finish.grade` — and additionally the "
        "one knob the pipeline deliberately never sends: a pre-encode luma clamp "
        "cannot survive libx264 ringing (measured, then CUT). If the grade "
        "scenario lands, this path moves to JUSTIFIED, not to emitted."
    ),
    "finish.grade.enabled": _GRADE,
    "finish.grade.glow": _GRADE,
    "finish.grade.grade_intensity": _GRADE,
    "finish.grade.grain": _GRADE,
    "finish.grade.highlight_tint": _GRADE,
    "finish.grade.look": _GRADE,
    "finish.grade.lut_url": _GRADE,
    "finish.grade.match_sources": (
        "producer WIRED 2026-07-20: `as_finish_payload` had declared "
        "`match_sources` on FinishSpec but omitted it from the emitted grade "
        "dict, so unlike its siblings it could not have been observed even with "
        "a scenario — the shared `_GRADE` reason was factually wrong for this "
        "key. It is now emitted (and lifted off the author's `finish.grade` "
        "block), and stays uncovered only for the ordinary reason: no contract "
        "scenario flips PIPELINE_GRADE_DEFAULT_ON_SHOWCASE, so no `finish.grade` "
        "block reaches the wire at all."
    ),
    "finish.grade.shadow_tint": _GRADE,
    "finish.grade.sharpen": _GRADE,
    "finish.grade.vignette": _GRADE,
    # -- request-level odds and ends ----------------------------------------- #
    "emit_shot_timeline": (
        "only DemoVideoCompileStage overrides `_emit_shot_timeline` to True and "
        "no scenario drives that stage; the base returns False, and the client "
        "drops the kwarg when false so it never rides the wire. A demo_video "
        "scenario would observe it."
    ),
    "tags": (
        "WALKER ASYMMETRY, not an inert knob. `tags` is a free-form "
        "dict[str, str], so the schema side stops at the leaf `tags` while the "
        "payload side descends into the real keys (`tags.run_id`, `tags.crf`, "
        "...). Every scenario emits tags. Closing this means teaching the "
        "comparison that a descended free-form mapping covers its own leaf — a "
        "schema_walk change, not a producer change."
    ),
    # -- per-shot polish ------------------------------------------------------ #
    "shots.adjust.brightness": _SHOT_POLISH,
    "shots.adjust.contrast": _SHOT_POLISH,
    "shots.adjust.gamma": _SHOT_POLISH,
    "shots.adjust.preset": _SHOT_POLISH,
    "shots.adjust.saturation": _SHOT_POLISH,
    "shots.audio.fade_in_s": _SHOT_POLISH,
    "shots.audio.fade_out_s": _SHOT_POLISH,
    "shots.audio.gain_db": _SHOT_POLISH,
    "shots.audio.mute": _SHOT_POLISH,
    "shots.callouts.at": _SHOT_POLISH,
    "shots.callouts.bbox": _SHOT_POLISH,
    "shots.callouts.label": _SHOT_POLISH,
    "shots.callouts.type": _SHOT_POLISH,
    "shots.cursor_track.click": _SHOT_POLISH,
    "shots.cursor_track.t": _SHOT_POLISH,
    "shots.cursor_track.x": _SHOT_POLISH,
    "shots.cursor_track.y": _SHOT_POLISH,
    "shots.focus_track.bbox": _SHOT_POLISH,
    "shots.focus_track.t": _SHOT_POLISH,
    "shots.footage_viewport": _SHOT_POLISH,
    "shots.max_seconds": _SHOT_POLISH,
    "shots.overlays.color": _SHOT_POLISH,
    "shots.overlays.end": _SHOT_POLISH,
    "shots.overlays.font_size": _SHOT_POLISH,
    "shots.overlays.position": _SHOT_POLISH,
    "shots.overlays.start": _SHOT_POLISH,
    "shots.overlays.text": _SHOT_POLISH,
    "shots.overlays.x": _SHOT_POLISH,
    "shots.overlays.y": _SHOT_POLISH,
    "shots.retime": _SHOT_POLISH,
    "shots.speed": _SHOT_POLISH,
    "shots.split.layout": _SHOT_POLISH,
    "shots.split.second_source_url": _SHOT_POLISH,
    "shots.target_fps": _SHOT_POLISH,
    "shots.transition.duration": _SHOT_POLISH,
    "shots.transition.type": _SHOT_POLISH,
}

# --------------------------------------------------------------------------- #
# THE DETECTOR'S FIRST REAL CATCH — resolved by WIRING (do not re-silence)
# --------------------------------------------------------------------------- #
# `music.duck_open_lu_below_vo` and `ambient.duck_open_lu_below_vo` are
# deliberately in NEITHER registry, and they are the reason to trust this module.
#
# Run against pre-#584 main, `test_every_declared_compiler_field_has_a_producer`
# failed on EXACTLY those two paths and nothing else — no noise, no false
# positives. Their default is None, so the measured VO-relative duck threshold
# never engaged at all: a live inert knob, shipped and unreachable, precisely the
# bug class this file exists to surface.
#
# It went green when orchestrator PR #584 (`12f0ece`, 2026-07-19, merged and
# live) WIRED the field: `sound_design.py:153` sets it on the music dict and
# `:180` on each ambient cue, so both paths now ride the wire and the
# sound-design scenario observes them. Resolved by wiring a producer — never by
# adding a JUSTIFIED entry.
#
# Standing instruction: neither path may EVER be moved into JUSTIFIED. If either
# goes red again it means a producer regressed, and the fix is to restore the
# producer in `sound_design.py`, not to write a justification for it.

#: Current size of the backlog. May only ever be LOWERED.
#:
#: 70 → 68 on 2026-08-09: the `mastering` block (8) and `shots.clip_anchor` /
#: `shots.clip_lead_seconds` (2) came off — the first by adding a scenario that
#: authors a mastering block, the last two by adding a FOOTAGE scenario (the
#: framing knobs are footage-guarded, so no image-backed scenario could ever have
#: observed them). `shots.clip_out_seconds`, which the refreshed schema introduced
#: alongside them, never entered the backlog at all: a producer was wired for it in
#: the same change and the footage scenario observes it. Against that, the eight
#: `shots.overlay_clips.*` leaves went ON — a compiler surface with no orchestrator
#: producer yet. Net -2.
#:
#: 68 → 60 on 2026-08-10: those same eight `shots.overlay_clips.*` leaves came
#: straight back OFF. `action_reel/graphics_design` now designs and RENDERS a
#: transparent (VP9-alpha WebM) lower-third on the opening shot, and
#: `action_reel/compile` joins the cards onto `Shot.overlay_clips` — so the
#: reason those entries carried ("NO PRODUCER EXISTS YET") stopped being true.
#: They left via a SCENARIO (`_emit_action_reel_overlays`), which is exactly what
#: the entry itself demanded, not via a justification.
_BACKLOG_CEILING = 60

_MIN_REASON_CHARS = 20


def _uncovered() -> frozenset[str]:
    """Declared compiler leaves that no producer emits and nobody accounted for."""
    return (
        schema_leaves(load_schema())
        - collect_emitted_leaves()
        - frozenset(JUSTIFIED)
        - frozenset(UNCOVERED_BACKLOG)
    )


def test_every_declared_compiler_field_has_a_producer() -> None:
    """No compiler field may ship inert.

    Each path below is declared by ``CompileRequest`` but is never set by any
    orchestrator stage, so the feature behind it is unreachable in production.
    """
    uncovered = _uncovered()
    detail = "\n".join(f"  - {path}" for path in sorted(uncovered))
    assert not uncovered, (
        "the compiler declares fields NO orchestrator producer ever sets — each "
        "of these ships SILENTLY INERT (the knob exists, the renderer honours "
        f"it, nothing ever asks for it):\n{detail}\n\n"
        "Two legitimate remedies, in this order:\n"
        "  (a) WIRE A PRODUCER — make a compile stage emit the field (and, if "
        "the stage already can, add/extend a scenario in "
        "test_video_compile_contract.py so the emission is proven), or\n"
        "  (b) ADD A JUSTIFIED ENTRY in test_producer_coverage.py giving a "
        "real reason the shipped default is correct and no producer should "
        "override it.\n"
        "Silencing a path you have not actually looked at re-creates the exact "
        "bug class this test exists to kill."
    )


def test_every_justification_has_a_real_reason() -> None:
    """Every registry entry — JUSTIFIED *and* backlog — must carry an argument.

    An empty or one-word value would let anyone silence the coverage check
    without thinking, which is worse than no check at all.
    """
    bad = {
        path: reason
        for registry in (JUSTIFIED, UNCOVERED_BACKLOG)
        for path, reason in registry.items()
        if len(reason.strip()) < _MIN_REASON_CHARS
    }
    assert not bad, (
        "JUSTIFIED entries need a real reason naming the default and why it is "
        "right; UNCOVERED_BACKLOG entries need a real reason why the path is "
        f"not observable yet (>= {_MIN_REASON_CHARS} chars): {sorted(bad)}"
    )


def test_backlog_never_grows() -> None:
    """The backlog is a RATCHET: ``_BACKLOG_CEILING`` may only ever be LOWERED.

    Raising the ceiling to make this file green is the failure mode the whole
    module exists to prevent — it is the same move as writing a JUSTIFIED entry
    for a knob nobody looked at, only with an extra step. The project's standing
    convention for every ratchet (the Any-ratchet, the coverage budgets) is
    identical: narrow the code to fit the budget, never raise the budget to fit
    the code. A new uncovered path is a scenario to write or a producer to wire.
    """
    assert len(UNCOVERED_BACKLOG) <= _BACKLOG_CEILING, (
        f"UNCOVERED_BACKLOG grew to {len(UNCOVERED_BACKLOG)} entries, ceiling is "
        f"{_BACKLOG_CEILING}. Do NOT raise the ceiling: add a scenario in "
        "test_video_compile_contract.py that observes the new path, or wire a "
        "producer for it."
    )


def test_backlog_and_justified_are_disjoint() -> None:
    """A path is either a decision or a gap — never both.

    Overlap would mean the two registries disagree about whether anyone should
    ever set the field, and the softer of the two claims would silently win.
    """
    both = frozenset(JUSTIFIED) & frozenset(UNCOVERED_BACKLOG)
    assert not both, (
        "these paths are in BOTH registries — JUSTIFIED says 'the default is "
        "right, no producer should set it', UNCOVERED_BACKLOG says 'we cannot "
        f"tell yet'. Pick one: {sorted(both)}"
    )


def test_backlog_entries_are_still_declared() -> None:
    """A backlog path the compiler no longer declares is a stale silencer.

    Same hazard as a stale JUSTIFIED entry: after a rename the dead line keeps
    excusing a path that no longer exists while the REAL successor field goes
    unnoticed.
    """
    declared = schema_leaves(load_schema())
    stale = frozenset(UNCOVERED_BACKLOG) - declared
    assert not stale, (
        "UNCOVERED_BACKLOG lists paths the compiler no longer declares (renamed "
        f"or removed upstream) — drop them: {sorted(stale)}"
    )


def test_justified_paths_are_actually_declared_by_the_schema() -> None:
    """A JUSTIFIED path the compiler no longer declares is dead paperwork.

    Without this, a renamed/removed compiler field leaves a stale silencer
    behind that would later mask the REAL successor field.
    """
    declared = schema_leaves(load_schema())
    stale = frozenset(JUSTIFIED) - declared
    assert not stale, (
        "JUSTIFIED lists paths the compiler no longer declares (renamed or "
        f"removed upstream) — drop them: {sorted(stale)}"
    )


def test_schema_walk_finds_the_known_audio_knobs() -> None:
    """Anchor: the walker must actually reach the sidechain knobs.

    Both ``music`` and ``ambient`` sit behind optional/list wrappers
    (``anyOf[$ref MusicBed, null]`` and ``items: $ref AmbientCue``). A walker
    bug that stopped at those wrappers would return a leaf set with no
    ``music.*``/``ambient.*`` in it, and the coverage test above would pass
    VACUOUSLY — reading green precisely because it found nothing. Pin the two
    ends of that path so an empty-ish walk fails loudly here instead.
    """
    declared = schema_leaves(load_schema())
    assert "music.duck_attack_ms" in declared
    assert "ambient.sidechain_hpf_hz" in declared
    # Every seeded justification must be a path the walker really produced —
    # this is the anchor that ties the registry to the walk, not just to a name.
    assert frozenset(JUSTIFIED) <= declared
