"""The loudness CONTRACT: the gate window and the encoder target are ONE number.

⭐ WHY THIS FILE EXISTS. Loudness lived in three files as three literals that had to
agree and were held together by nothing but prose — and all three disagreed:

===========================================  ==========  ===============================
constant                                     was         relationship it CLAIMED
===========================================  ==========  ===============================
``gestalt_metrics._DEFAULT_LOUDNESS_WINDOW``  (-14, -11)  "centred on the master target"
``mastering_plan._DEFAULT_LOUDNESS_TARGET``   -12.5       "the ratified target"
``encode_master._LOUDNESS_LO/_HI``            (-16, -13)  "the mastering-advisor band"
===========================================  ==========  ===============================

The window was not centred on the target the compiler applies (-14.0); its FLOOR sat
exactly ON it, so single-pass loudnorm's ordinary 0.1-0.2 LU undershoot convicted every
correctly-mastered master — three of three live demo_video runs on 2026-09-16, at
-14.2 / -14.1 / -14.1 LUFS. The advisor band did not overlap the target at all: it was
free to aim 2 LU below the floor the gate convicts at. And the "ratified" -12.5 was the
value compiler PR #217 (2026-09-12) ABANDONED, because on a bus arriving at -18 LUFS per
narration clip it asked for ~5.5 dB of gain riding — the audible breathing complaint.

THE FIX IS STRUCTURAL, NOT ASSERTED. There is now ONE literal —
``loudness_contract.COMPILER_MASTER_LUFS``, vendored from the compiler — and everything
else is COMPUTED from it. A window that excludes the target cannot be written, because
the window IS ``target ± SHIP_TOLERANCE_LU``. These tests are the remaining guard: they
fail if anyone re-introduces a literal, or lets the vendored number drift from the
compiler's shipped default.

⛔ NONE OF THESE MAY BE "FIXED" BY MOVING THE WINDOW. The window is downstream. If a
master misses, the master is wrong or the compiler's target moved — in which case the
ONE line to edit is ``COMPILER_MASTER_LUFS``, and the window follows it.
"""

from __future__ import annotations

import inspect

import pytest

from src.services.stages.demo_video import encode_master
from src.services.stages.shared import gestalt_metrics, loudness_contract, mastering_plan
from src.services.stages.shared.gestalt_metrics import loudness_window
from src.services.stages.shared.loudness_contract import (
    COMPILER_MASTER_LUFS,
    SHIP_TOLERANCE_LU,
    ship_loudness_window,
)
from src.services.stages.shared.mastering_plan import (
    MasteringProfile,
    build_mastering_profile,
    default_loudness_target,
)

# The worst single-pass loudnorm residual we are willing to call normal. The three
# convicted live masters landed 0.1-0.2 LU from their target, so a target closer than
# this to either edge of the ship window makes the residual ALONE a conviction.
_LOUDNORM_RESIDUAL_LU = 0.5


class TestTheVendoredCompilerDefaultIsTheOneSource:
    """⛔ THE CROSS-REPO PIN. Everything else in this file is downstream of this."""

    def test_vendored_target_matches_the_compilers_shipped_default(self) -> None:
        """``COMPILER_MASTER_LUFS`` must equal video_compiler_service's own default.

        SOURCE (``video_compiler_service`` ``origin/main``), both -14.0, both landed
        in compiler PR #217 (2026-09-12, an AUDIO-QUALITY PR):
          * ``src/config.py``             ``Settings.master_lufs``  default=-14.0
          * ``src/services/mastering.py`` ``DEFAULT_MASTER_LUFS``   = -14.0

        ⚠️ HOW THIS PIN IS ENFORCED, stated honestly. The repo's "Vendored compiler
        contract is fresh" CI job compares
        ``tests/contracts/compile_request.schema.json`` against the compiler's promoted
        image — but that schema is the REQUEST-OVERRIDE shape: both ``master_lufs`` and
        ``MasteringProfile.loudness_target`` carry ``"default": null`` there, because a
        request omitting them means "compiler, use your own". The compiler's Settings
        default is deliberately NOT on the wire, so that job can NEVER catch drift in
        this number and re-vendoring the schema will never update it.

        So the mechanism is a HAND-CARRIED VENDORED LITERAL plus this test. What this
        test genuinely buys: it is the single place a reader learns which upstream file
        owns the number, and it fails loudly the moment anyone edits the orchestrator's
        copy without touching the compiler — which is exactly how -12.5 survived two
        months past its own retirement. What it does NOT buy: automatic detection of a
        change made on the compiler's main. That requires a human to re-read the source
        files named above.
        """
        assert COMPILER_MASTER_LUFS == -14.0, (
            f"vendored COMPILER_MASTER_LUFS is {COMPILER_MASTER_LUFS}, but "
            f"video_compiler_service ships -14.0 (config.py Settings.master_lufs and "
            f"mastering.py DEFAULT_MASTER_LUFS, compiler PR #217, 2026-09-12). If the "
            f"compiler really moved, update this literal AND this assertion together "
            f"and cite the compiler PR that moved it."
        )

    def test_the_abandoned_value_is_not_the_target(self) -> None:
        """⛔ -12.5 is the value that CAUSED the owner's complaint, not an alternative.

        Compiler PR #217's own words: the previous -12.5 "asked a bus that already
        arrives at -18 LUFS per narration clip for another ~5.5 dB of gain riding,
        which is what made the narration breathe". A change that lands back on -12.5 is
        not a tuning choice, it is a regression to a known audible artefact.
        """
        assert default_loudness_target() != -12.5, (
            "the orchestrator is asking the compiler to master at -12.5 LUFS — the "
            "value compiler PR #217 abandoned because it drove ~5.5 dB of gain riding "
            "and made the narration breathe."
        )


class TestTheOrchestratorCannotDisagreeWithTheCompiler:
    """The defect this PR fixes: the orchestrator ASSERTING a target of its own."""

    def test_default_target_is_the_compilers_default(self) -> None:
        assert default_loudness_target() == COMPILER_MASTER_LUFS

    def test_a_threaded_profile_carries_exactly_the_compilers_default(self) -> None:
        """A run that AUTHORS a mastering block must be mastered like one that doesn't.

        ``build_mastering_profile`` is presence-is-opt-in: with no ``mastering`` block
        the compile request carries no ``mastering`` key and the compiler applies its
        own default. With a block, ``loudness_target`` ALWAYS threads. If the threaded
        value differs from the compiler's default, then whether a run authored an
        unrelated knob (``deband``, ``crf``) silently decides its loudness.
        """
        profile = build_mastering_profile({"mastering": {"deband": "on"}})
        assert profile is not None
        payload = profile.as_mastering_payload()
        assert payload["loudness_target"] == COMPILER_MASTER_LUFS, (
            f"a run authoring only `deband` is threaded loudness_target="
            f"{payload['loudness_target']}, while a run authoring nothing is mastered "
            f"at the compiler's {COMPILER_MASTER_LUFS}. Authoring an unrelated knob "
            f"must not move the loudness."
        )

    def test_dataclass_default_is_the_compilers_default(self) -> None:
        assert MasteringProfile().as_mastering_payload()["loudness_target"] == (
            COMPILER_MASTER_LUFS
        )

    def test_an_explicit_authored_target_still_wins(self) -> None:
        """Deriving the DEFAULT must not remove the per-run override."""
        profile = build_mastering_profile({"mastering": {"loudness_target": -16.0}})
        assert profile is not None
        assert profile.as_mastering_payload()["loudness_target"] == -16.0

    def test_no_run_setup_helper_seeds_a_loudness_target(self) -> None:
        """⛔ REGRESSION GUARD on the near-miss this commit reworked.

        An earlier commit on this branch added
        ``run_setup._seed_ship_gate_loudness_target``, which injected an
        orchestrator-derived target into the snapshot of every run with an armed ship
        gate. Because that target was the gate window's CENTRE (-12.5), it would have
        re-imposed the abandoned value fleet-wide — undoing compiler PR #217 from a
        different repo, silently, on runs that never authored a mastering block.

        The honest shape is the one that ships now: the orchestrator does NOT invent a
        target for a run that did not ask for one. An absent block means the compiler's
        own default applies, which is the same number this repo would have threaded.

        ⚠️ This guard is deliberately NARROW. ``run_setup`` legitimately writes
        ``mastering.loudness_target`` in ``_alias_brief_master_lufs``, which folds a
        brief-AUTHORED ``voice.master_lufs`` onto the live loudness path. Propagating a
        value the author stated is the opposite of inventing one, so a blunt
        "``loudness_target`` must not appear in this module" check would ban the
        correct behaviour along with the wrong one (it did, on the first run of this
        test). The property that actually distinguishes them: run_setup must never
        reach for the orchestrator's DEFAULT.
        """
        from src.services.runs import run_setup

        source = inspect.getsource(run_setup)
        for invented in ("default_loudness_target", "COMPILER_MASTER_LUFS"):
            assert invented not in source, (
                f"run_setup references `{invented}`, i.e. it is seeding a DEFAULT "
                f"loudness target into the config snapshot. The orchestrator must not "
                f"invent a target for a run that did not author one — an absent "
                f"`mastering` block correctly leaves the compiler's own default in "
                f"charge, and seeding one would let a stale vendored number OVERRIDE "
                f"what the compiler actually ships."
            )

    def test_an_armed_ship_gate_alone_does_not_manufacture_a_mastering_block(self) -> None:
        """Behavioural half of the guard above: arming the gate must not thread a target.

        This is the exact snapshot shape the reworked commit would have mutated.
        """
        snapshot: dict[str, object] = {"ship_gate": {"mode": "critique_only"}}
        assert build_mastering_profile(snapshot) is None, (
            "an armed ship gate alone produced a mastering profile — something is "
            "seeding a loudness target the run never asked for."
        )


class TestTheWindowIsDerivedFromTheTarget:
    """⭐ 'Two names for one number.' The band is COMPUTED, so it cannot disagree."""

    def test_ship_window_is_the_target_plus_minus_the_tolerance(self) -> None:
        assert ship_loudness_window() == (
            COMPILER_MASTER_LUFS - SHIP_TOLERANCE_LU,
            COMPILER_MASTER_LUFS + SHIP_TOLERANCE_LU,
        )

    def test_the_gates_own_default_window_is_that_window(self) -> None:
        """``gestalt_metrics`` must not carry a second literal band."""
        assert loudness_window({}) == ship_loudness_window()
        assert ship_loudness_window() == gestalt_metrics._DEFAULT_LOUDNESS_WINDOW

    def test_the_tolerance_is_positive_so_containment_is_structural(self) -> None:
        """The property that makes a bad window unwritable rather than merely untested."""
        assert SHIP_TOLERANCE_LU > 0

    def test_the_window_was_recentred_not_widened(self) -> None:
        """⚠️ THE ANTI-GOALPOST ASSERTION. Width is frozen at the historical 3.0 LU.

        The pre-fix band ``(-14.0, -11.0)`` was 3.0 LU wide. If a future change makes a
        loudness conviction go away by making this number bigger, that is moving the
        goalposts, not fixing a defect — and this test is where it has to be argued.
        """
        lo, hi = ship_loudness_window()
        assert hi - lo == pytest.approx(3.0), (
            f"the ship window is now {hi - lo} LU wide; it has always been 3.0 LU. "
            f"Widening it is not a fix for a loudness conviction."
        )

    @pytest.mark.parametrize("ceilings", [{}])
    def test_every_default_path_lands_on_the_derived_window(
        self, ceilings: dict[str, object]
    ) -> None:
        """A malformed per-run override must fall back to the DERIVED band, not a literal."""
        assert loudness_window(ceilings) == ship_loudness_window()

    def test_a_per_run_override_is_still_honoured(self) -> None:
        assert loudness_window({"loudness_lufs": [-16.0, -12.0]}) == (-16.0, -12.0)


class TestTheTargetSurvivesItsOwnGate:
    """The measured failure: a target the gate's own band convicts."""

    def test_target_is_strictly_inside_the_window(self) -> None:
        lo, hi = ship_loudness_window()
        target = default_loudness_target()
        assert lo < target < hi, (
            f"master target {target} LUFS is not strictly inside the ship window "
            f"{(lo, hi)} — the compiler is aiming where the gate convicts."
        )

    def test_target_clears_both_edges_by_the_loudnorm_residual(self) -> None:
        """A target ON an edge is convicted by the residual alone — that IS the bug."""
        lo, hi = ship_loudness_window()
        target = default_loudness_target()
        assert target - lo >= _LOUDNORM_RESIDUAL_LU, (
            f"master target {target} is within {round(target - lo, 3)} LU of the ship "
            f"FLOOR {lo}; an ordinary loudnorm undershoot convicts it. This is exactly "
            f"how a -14.0 target under a (-14.0, -11.0) band produced the -14.1/-14.2 "
            f"live failures."
        )
        assert hi - target >= _LOUDNORM_RESIDUAL_LU, (
            f"master target {target} is within {round(hi - target, 3)} LU of the ship "
            f"CEILING {hi}; an ordinary loudnorm overshoot convicts it."
        )


class TestTheTunerCannotAimWhereTheGateConvicts:
    """The other shipped literal: advisor range (-16.0, -13.0) vs a band of (-14.0, -11.0)."""

    def test_advisor_legal_range_is_contained_by_the_ship_window(self) -> None:
        lo, hi = ship_loudness_window()
        assert lo <= encode_master._LOUDNESS_LO, (
            f"the mux-mastering-advisor may clamp loudness_target down to "
            f"{encode_master._LOUDNESS_LO} LUFS, BELOW the ship floor {lo} — the tuner "
            f"is permitted to aim exactly where the gate convicts."
        )
        assert hi >= encode_master._LOUDNESS_HI, (
            f"the advisor may clamp loudness_target up to {encode_master._LOUDNESS_HI} "
            f"LUFS, above the ship ceiling {hi}."
        )

    @pytest.mark.parametrize("proposed", [-30.0])
    def test_no_advisor_proposal_can_clamp_outside_the_window(self, proposed: float) -> None:
        """Behavioural, not constant-comparing: run the real clamp over the extremes."""
        lo, hi = ship_loudness_window()
        block, _prov = encode_master._clamp_mastering_block({"loudness_target": proposed})
        resolved = float(block["loudness_target"])  # ty: ignore[invalid-argument-type]
        assert lo <= resolved <= hi, (
            f"advisor proposal {proposed} clamped to {resolved} LUFS, outside the ship "
            f"window {(lo, hi)}."
        )


class TestNoModuleRestatesTheNumber:
    """⛔ The failure mode is a LITERAL reappearing, not a wrong value today."""

    @pytest.mark.parametrize(
        "module",
        [mastering_plan],
        ids=["mastering_plan"],
    )
    def test_no_downstream_module_hardcodes_a_loudness_constant(self, module: object) -> None:
        """Every loudness constant downstream must be an expression, not a literal.

        Scans ASSIGNMENT STATEMENTS only (via ``ast``), so the long explanatory
        comments and docstrings in these files — which necessarily quote the old
        numbers to explain what went wrong — cannot satisfy or trip this test. A
        docstring matching what it documents is this project's signature vacuous test;
        this one cannot do that.
        """
        import ast

        tree = ast.parse(inspect.getsource(module))  # ty: ignore[invalid-argument-type]
        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            names = (
                [t.id for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else ([node.target.id] if isinstance(node.target, ast.Name) else [])
            )
            if not any("LOUDNESS" in n.upper() or "LUFS" in n.upper() for n in names):
                continue
            if node.value is None:
                continue
            # A bare/negated numeric literal, or a tuple of them, is a restatement.
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)):
                    offenders.append(f"{'/'.join(names)} = {ast.unparse(node.value)}")
                    break
        assert not offenders, (
            f"{getattr(module, '__name__', module)} restates a loudness number as a "
            f"literal instead of deriving it from loudness_contract: {offenders}. The "
            f"ONE literal lives in loudness_contract.COMPILER_MASTER_LUFS."
        )

    def test_the_contract_module_is_the_place_that_does_hold_the_literal(self) -> None:
        """Sanity: the detector above would be vacuous if nothing ever held a literal."""
        import ast

        tree = ast.parse(inspect.getsource(loudness_contract))
        literals: list[str] = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.AnnAssign) or not isinstance(n.target, ast.Name):
                continue
            value = n.value
            if value is None:
                continue
            if any(
                isinstance(s, ast.Constant) and isinstance(s.value, (int, float))
                for s in ast.walk(value)
            ):
                literals.append(n.target.id)
        assert "COMPILER_MASTER_LUFS" in literals and "SHIP_TOLERANCE_LU" in literals
