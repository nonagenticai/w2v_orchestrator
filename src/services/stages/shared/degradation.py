"""The explicit DEGRADATION channel — a machine-readable quality signal.

Before this module the orchestrator had exactly one way to say "this run
finished green but shipped something broken": append a prose string to the
shared ``notes`` list, which ``api/routes/runs/_shared.py`` turns into the
``degraded`` boolean. That single channel is why several real defects were
invisible in the run record:

* ``notes`` is PROSE. It is written for a human reading one run, it has no
  stable identity, and no query can ask "how many runs shipped without their
  music bed last week". Every attempt to make it answerable ended in grepping
  substrings of English sentences.
* Because a note is the ONLY way to raise the flag, any signal too noisy to
  put in front of a human had to be suppressed ENTIRELY — losing the machine
  fact along with the prose. ``mastering_gate`` narrowed itself to blank
  pictures for exactly that reason (207 of 214 masters carry
  ``gate_pass=false``), and ``duration_conformance`` refused to emit anything
  at all, in a docstring that said so out loud.

So degradation gets a channel of its own: a list of STRUCTURED entries, each a
stable :data:`code` plus a human-readable ``detail``. ``notes`` keeps its job
(prose for a human) and keeps raising ``degraded`` for backward compatibility;
this channel is the one a query, an alert or a test can bind to.

Deliberate properties
---------------------
* **Additive.** The field defaults to ``[]`` on
  :class:`~src.services.stages.base.StageOutputWithDegradations`, and every
  reader (API, SQL predicate) treats an ABSENT key exactly like an empty list —
  so every stage output persisted before this module parses and behaves
  unchanged.
* **Never fatal.** Recording a degradation does not raise, does not change a
  stage's ``status``, and does not fail a run. It flips ``degraded`` to true.
  Turning currently-passing runs red is a separate decision.
* **A ``TypedDict``, not ``dict[str, Any]``.** ``tests/test_any_ratchet.py`` is
  a one-way ratchet on the ``Any`` surface; a new untyped blob would have to be
  paid for by typing something else. ``duration_conformance.DurationConformance``
  is a ``TypedDict`` for the same reason, and pydantic validates it as a field.
* **``code`` is typed ``str``, not a ``Literal``.** The known set is
  :data:`DEGRADATION_CODES` and :func:`degradation` asserts membership at the
  WRITE site, where a typo is a bug. Validation on the READ side would instead
  reject a row written by a newer pod during a rolling deploy — parsing a
  persisted output must never fail on a code this build has not heard of.
"""

from __future__ import annotations

from typing import TypedDict

#: The compiler's post-mux mastering gate reported ``gate_pass=false``.
MASTERING_GATE_FAILED = "mastering_gate_failed"
#: Delivered master length is outside the ordered duration's tolerance band.
DURATION_OUT_OF_TOLERANCE = "duration_out_of_tolerance"
#: ``reconcile_timeline`` cut footage tails that ran past their narration so the film
#: fits the order (owner ruling); ``detail`` names the seconds cut per beat.
FOOTAGE_TRIMMED = "footage_trimmed"
#: Word-timed captions were requested but the build soft-failed — the video
#: ships with NO subtitles at all.
CAPTIONS_MISSING = "captions_missing"
#: A music or ambient bed was chosen and then could not be resolved — the video
#: ships silent where it should have had a bed.
SOUND_BED_MISSING = "sound_bed_missing"
#: A music or ambient bed WAS resolved and sent to the compiler, but the compiler
#: reported it did not land in the master (``music_bed.applied is False`` or
#: ``ambient_fetch_failed``, from its ``_flag_unapplied_music_bed``) — e.g. the bed
#: URL 403'd because its presign expired mid-compile. Distinct from
#: :data:`SOUND_BED_MISSING`, which is the orchestrator's OWN pre-submit drop; this
#: is the compiler's post-render verdict, and before it existed the run shipped
#: ``completed`` with no bed and no signal at all.
MUSIC_BED_NOT_APPLIED = "music_bed_not_applied"
#: A slide deck came back empty as a STUB (nothing was authored to render).
SLIDES_STUBBED = "slides_stubbed"
#: A narration clip was stored at an unknown/unproven loudness.
AUDIO_UNNORMALIZED = "audio_unnormalized"
#: No duration was ORDERED for this deliverable, so no conformance verdict
#: could be given. Distinct from being outside the band: there is no band. A
#: null verdict is invisible to any ``within_tolerance = false`` query, so the
#: absence of an order is reported here instead of silently.
DURATION_NOT_ORDERED = "duration_not_ordered"
#: The ordered duration was arithmetically unreachable given a hard structural
#: floor (e.g. the guaranteed per-scenario recording length), so the plan was
#: refitted to what is actually achievable.
ORDERED_DURATION_INFEASIBLE = "ordered_duration_infeasible"
#: The structural-relief gate REFUSED to bill a corrective re-render. The run therefore
#: ships out of band with no automatic second attempt, and the user is the remaining
#: lever.
#:
#: ⚠️ THIS CODE REPORTS A VERDICT, NOT A CONSTRAINT. It says the gate declined; it does
#: NOT assert that the order is genuinely unsatisfiable, that the refusal is right, or
#: that the arithmetic behind it is sound. As of 2026-09-16 the lane that owns
#: ``structural_relief_policy`` is actively trying to refute its own refusal (the gate
#: enumerates against a FOOTAGE-derived ``measured_seconds_per_beat`` while the delivered
#: film may track NARRATION, which would make some refusals false). Read every instance
#: of this code as "the gate said no", and attribute it to the gate. Whether the gate is
#: right is that lane's question, and is deliberately not answered here.
#:
#: WHY IT NEEDS ITS OWN IDENTITY. ``structural_relief_policy.decide_relief_shape``
#: produces this verdict at plan time, before anything paid happens, and words a full
#: sentence of arithmetic onto ``ReliefShape.stop_detail`` — e.g. "no legal beat count
#: lands inside the +/-10% band [39.60s, 48.40s] of the 44.00s order at 12.42s
#: measured/beat ... refusing to bill a render that lands outside the band". Until
#: 2026-09-16 that sentence reached NOTHING: ``git grep`` for ``STOP_UNDERFULL`` /
#: ``stop_detail`` matched only the two files that produce them. So a run could ship out
#: of band with no visible explanation at all, and the verdict — sound or not — could
#: neither be acted on nor audited. Surfacing it is what makes it testable in either
#: direction.
#:
#: DISTINCT from :data:`ORDERED_DURATION_INFEASIBLE`, which reports that THIS run's
#: reconciled beats project past the order. This one reports only that no corrective
#: re-render will be attempted: the two are independent facts and a run can carry either
#: without the other.
#:
#: ⛔ VISIBILITY ONLY — it must never become terminal. A run that terminates never
#: reaches ``mark_completed``, and the relief hook is completion-gated, so failing here
#: would silently disable relief spawning outright. It also adds no newly-``degraded``
#: run: ``_cant_converge_note`` already lands prose in ``notes`` on exactly this path and
#: ``runs/_shared.py`` derives ``degraded`` from notes, so this strictly upgrades an
#: already-degraded run from prose to a queryable code.
STRUCTURAL_RELIEF_REFUSED = "structural_relief_refused"
#: The speech rate used to budget narration was outside the sanity band (a broken
#: probe, a unit mix-up, a non-positive or NaN value), so a substitute rate was
#: used. EVERY duration number on this run's narration — the estimate, the word
#: budget, the trim — was computed at a rate nobody measured. No existing code
#: says that: the estimate did not miss a band (``duration_out_of_tolerance``) and
#: no structural floor blocked the order (``ordered_duration_infeasible``); the
#: MEASUREMENT ITSELF is unsound, which is a different fact and invisible unless
#: it has its own identity.
SPEECH_RATE_UNUSABLE = "speech_rate_unusable"
#: A narration line was delivered DRAGGED: fitting it to its allotment needed a
#: speed BELOW the natural floor, so the fit clamped at the floor and the clip
#: still ships short of the seconds it was given. This is the owner's "the
#: narration weirdly slows down" complaint, and until 2026-09-15 nothing
#: reported it in either direction that mattered:
#:
#: * ``_RATE_DEFECT_SPEED_THRESHOLD`` (1.3) only fires on the FAST side — a line
#:   RUSHED to fit. The SLOW side had no signal at all beyond a prose note.
#: * the prose note it did raise is invisible to any query, so "how many runs
#:   shipped dragged narration" was unanswerable, exactly like the defects in
#:   this module's docstring.
#:
#: It is a DEGRADATION, not an advisory, on the same test the picture-truncation
#: code passes: it reports a defect in the artifact the customer paid for (the
#: voice audibly drags AND the beat ships under its allotment), not a gap in our
#: instrumentation. ``detail`` carries the UNCOVERED SECONDS — the part of the
#: allotment neither speech nor the clamped stretch fills — because "sounds slow"
#: and "the film is short" are one defect measured two ways and a reader cannot
#: size it without the seconds.
NARRATION_RATE_FLOOR_CLAMPED = "narration_rate_floor_clamped"

#: The delivered master's PICTURE stops before its AUDIO does — narration (or
#: music) plays over a frozen/absent frame for the remainder. Measured from the
#: compiler's mastering report, which probes the two streams SEPARATELY
#: (``video_stream_seconds`` / ``audio_stream_seconds``); no single reduced
#: duration scalar can express it, because ``actual_duration_seconds`` is
#: ``min(video_stream, container)`` and therefore AGREES with a truncated
#: picture (``video_compiler_service/src/services/renderer.py::_resolve_probed_duration``).
#:
#: It is a DEGRADATION, not an advisory, and the tier is deliberate. Every other
#: member of this module's advisory tier reports a gap in OUR INSTRUMENTATION
#: (:data:`DURATION_UNMEASURED`: nobody probed the file) or a PRE-production
#: guess (:data:`NARRATION_ESTIMATE_OFF_TARGET`). This one reports a defect in
#: the artifact the customer paid for: the film stops early. There is no
#: threshold at which that is the right call, which is the same test
#: ``mastering_gate._structural_reasons`` applies to ``av_coverage`` and a blank
#: picture.
#:
#: ONE-SIDED by construction — see
#: ``duration_conformance.picture_truncation_degradations``. Picture LONGER than
#: audio is normal here (live showcase masters run 2.3-5.6s long, and a title
#: card or a held final frame is authored, not broken); only audio outrunning
#: picture is the defect.
DURATION_PICTURE_TRUNCATED = "duration_picture_truncated"
#: A narration line claims a feature the delivered redesign does not contain (a
#: contact form that is not on the page, security headers "now live" on a static
#: mockup). Raised by ``website_review_script``'s deterministic grounding check
#: after its one grounding re-ask still left such a line: the video would SAY the
#: redesign does something the picture does not show.
SCRIPT_CLAIMS_UNGROUNDED = "script_claims_ungrounded"
#: ``media_discovery`` could not search (a search error, a failed source, no query), so
#: the run proceeds with fewer or no discovered media than it asked for.
MEDIA_DISCOVERY_INCOMPLETE = "media_discovery_incomplete"
#: A discovered/pinned media item that IS a downloadable image or clip could not
#: be re-published to run storage (egress, size cap, SSRF refusal), so it is not
#: on screen. NOT raised for an item that is simply not media (a YouTube watch
#: page, an HTML document) — that is a classification, recorded on the item.
MEDIA_NOT_COMPOSITED = "media_not_composited"
#: A lesson's ``ai_image_gen`` failed outright, so every slide renders as a plain
#: title + bullets slide with no hero image. The lesson (and its course) still ship.
AI_IMAGES_UNAVAILABLE = "ai_images_unavailable"
#: The footage critic was ASKED for a per-shot verdict and did not deliver one —
#: the analyzer never returned a reading, the critic batches returned no verdict,
#: or the batch budget was too small to admit a call at all. The deliverable still
#: ships, but it ships UNREVIEWED: nothing scored its shots, so no shot could be
#: flagged and no re-render pass could fire.
#:
#: MEASURED over 14 days fleet-wide: 56 of 115 completed ``footage_critique`` rows
#: (48.7%) carried ``critique_unavailable=true``, 55 of those scoring ZERO shots,
#: on 11 of 13 active days. All 56 were invisible: the stage raised prose ``notes``
#: only, so the DAG executor stamped them :data:`UNCLASSIFIED_NOTE` with
#: ``error IS NULL`` and ``runs.failure_class IS NULL``. "No flagged shots" has
#: therefore not meant "good footage" about half the time, and no query could tell
#: the two apart — the exact "recorded but unqueryable" failure this module exists
#: to remove.
#:
#: A DEGRADATION, not an advisory, and the tier is NOT a change: this path already
#: appended a note, which already flipped ``degraded`` and already landed in the
#: degradation tier as ``unclassified_note``. What is new is only its IDENTITY.
#: It also passes this module's own tier test on its merits — the shot-by-shot
#: review is a production step the customer paid for, and when it does not run the
#: picture ships un-improved, which is a fact about the artifact and not a gap in
#: our instrumentation. ``detail`` carries the critic's own reason sentence, because
#: "the analyzer never answered" and "the batch budget was below the critic floor"
#: are different defects and a reader cannot act on the code alone.
FOOTAGE_CRITIQUE_UNAVAILABLE = "footage_critique_unavailable"

#: The footage critic DID review the shots, and the best artifact the loop could
#: produce still scored BELOW the bar (``footage_critique._THRESHOLD``). The loop is
#: capped at two passes on purpose, so it stops and ships that best artifact.
#:
#: MEASURED on prod run ``01a0ab23-6c47-76a2-9d53-f16220e4792d`` (showcase_v1,
#: generated mode, 30s order): score 2.0 against a 4.0 bar, and the stage raised
#: only a prose note — so "the critic reviewed this and it failed" reached no query,
#: no badge, and no alert.
#:
#: ``detail`` carries the score, the bar, the loop's stop reason AND the re-render
#: outcome, because "plateau" alone conflates two different defects: the optimizer
#: returns the SAME draft when a re-render FAILS, so a failed re-render (delta 0)
#: and a re-render that ran and did not help both stop as ``plateau``. The
#: discriminator also rides the record itself as ``critique.rerender_outcome``.
#:
#: A DEGRADATION, and not a new flip: the same path already appended a note (⇒
#: ``degraded``). What is new is the identity. ⛔ It must never become terminal —
#: a run that terminates never reaches ``mark_completed``, which silently disables
#: structural-relief spawning. ⛔ It must never buy another pass either: the cap is
#: the cost control.
FOOTAGE_BELOW_BAR = "footage_below_bar"

#: ``final_critique`` (the ship gate) got NO reading from the video analyzer — a
#: failed / 404 / lost job (live run 01a16423-349f: a rollout orphaned the job) —
#: and, by its deliberate fail-open, shipped the film with an
#: ``insufficient_evidence`` verdict. Until this code that was an ERROR log only and
#: the run was NOT degraded. ``detail`` carries the analyzer error. The footage-level
#: twin is :data:`FOOTAGE_CRITIQUE_UNAVAILABLE` (footage_critique already records it).
#: ⛔ Never terminal, and never a resubmit here: video_analyzer re-claims orphaned
#: jobs itself, and an orchestrator-side retry would double-bill the analysis.
REVIEW_SKIPPED = "review_skipped"

#: An opt-in intro/outro title card (``shared/title_cards.build_title_card_shots``)
#: failed to render and was dropped — the video still compiles, just without that
#: card. Previously this landed ONLY as a prose ``notes`` sentence, which the DAG
#: executor stamps :data:`UNCLASSIFIED_NOTE` — indistinguishable from every other
#: kind of soft-failed stage note, so a query for "how many runs shipped without
#: their branded title card" could not be answered. ``detail`` carries which of the
#: two failure shapes it was — a poll TIMEOUT (``asyncio.TimeoutError`` /
#: ``httpx.ReadTimeout`` from ``wait_for_content_recording``) vs a genuine RENDER
#: error (a 4xx/5xx from ``video_production_service``) — because those point at
#: different owners (a slow/wedged recorder vs a bad template/params) and a reader
#: cannot act on the code alone. A DEGRADATION, not an advisory: a card the
#: customer configured is simply missing from the delivered file, which is a fact
#: about the artifact, not about our instrumentation.
TITLE_CARD_DROPPED = "title_card_dropped"
#: A promo's DEFAULT branded end card — the on-screen call-to-action — was planned
#: out because the narration left less room under the order's +10% ceiling than a
#: readable card needs (``promo/fill.py``). ``detail`` carries the numbers and the
#: order that would keep it. A DEGRADATION, not an advisory: the delivered film
#: ends without its CTA on screen, which is a conversion defect in the artifact.
END_CARD_DROPPED = "end_card_dropped"

#: A recorded scenario FAILED in the frontend test runner and its clip was still the
#: footage for a narration line. A failure on a Playwright wait (``Timed out 5000ms
#: waiting for expect(...)`` / ``Timeout 5000ms exceeded``) leaves that wait on the end
#: of the clip as a dead, motionless tail, and every beat is PINNED to its clip's
#: ``seconds`` — so the tail inflated the film.
#:
#: MEASURED on prod run ``01a0b007-9c2c-716a-883f-beefd14f4fd5`` (showcase, 30s order):
#: passing clips 9.56-9.88s, failing clips 14.56-15.28s; clip-1 was 9.64s when it
#: passed the run before and 14.56s failing (+4.92s = the 5000ms assert timeout), its
#: motion ~0 after 5.8s, and the film landed +24.3% on its order.
#:
#: ``demo_video_record`` now TRIMS such a clip to ``seconds - timeout`` (the start of
#: the failing wait — every action before it is kept) and keeps the file length on
#: ``DemoVideoClip.recorded_seconds``. ``detail`` says which it was: trimmed, or kept
#: whole because the tail could not be proven dead (no wait-timeout in the error, a
#: recorded action inside the would-be tail, or a timeout longer than the clip).
#: A DEGRADATION: the scenario did not do what the narration describes.
FAILED_SCENARIO_FOOTAGE = "failed_scenario_footage"

#: A GENERATED narration line still carried a URL or raw path (``contains_url``) when
#: it reached ``demo_video_narrate``, and the final speakable guard rewrote it
#: (``speakable_text``) before TTS. ``reconcile_timeline`` makes a generated spine
#: speakable, but SKIPS on a run with no ``direction.timing`` block; the rewrite then
#: happens here, at the last boundary.
#:
#: What is true, and all ``detail`` claims: the voice spoke different words than the
#: line's rate/word budget was computed on (that budget, where one exists, was
#: computed on pre-rewrite text). It does NOT claim the budget was *measured on the URL*: the speech-rate
#: sample never strips URLs, and text a later writer (script_doctor / claim_finish /
#: retention_plan) produced was never sampled at all. MEASURED on prod run
#: ``01a0b007``: script_doctor's clip-5 ``…application at "https://demo.playwright.dev/todomvc".``
#: A DEGRADATION: the delivered line is not the planned line.
URL_REWRITTEN_BEFORE_TTS = "url_rewritten_before_tts"

#: The ``growth_bundle`` companion produced no usable channel copy, after its
#: retries, on a run whose video had ALREADY been delivered. The run keeps the
#: video and reports the missing outbound copy here instead of failing outright
#: (showcase spike 01a0fd4d-9948 went ``failed`` with a finished 2560x1440 mp4).
#: A DEGRADATION: part of the deliverable is missing. A standalone
#: ``growth_bundle_v1`` run, where the copy IS the deliverable, still fails.
GROWTH_BUNDLE_EMPTY = "growth_bundle_empty"

#: A still shot's Ken Burns recording soft-failed and the shot ships as a FROZEN
#: still (``shared/ken_burns.apply_ken_burns``): a fetch error, an expired
#: presign, a render error, a poll that outlived the pass deadline. ``detail`` is
#: the shot's own prose note, exception class included. The queue-capacity case
#: has its own code, :data:`KEN_BURNS_STILL_QUEUE_SATURATED`. A DEGRADATION, and
#: not a new flip: the same path already appended a note (=> ``degraded``).
KEN_BURNS_STILL_SOFT_FAILED = "ken_burns_still_soft_failed"
#: The same frozen still, but lost to CAPACITY: video_production failed the
#: recording ``QUEUE_SATURATED`` (no per-pod recording slot within its timeout)
#: and the orchestrator's bounded re-submissions — or the pass budget — ran out.
#: Distinct because the remedy is different: nothing is wrong with the shot, the
#: recorder fleet was full (prod run 01a10fcf lost its fourth still this way).
KEN_BURNS_STILL_QUEUE_SATURATED = "ken_burns_still_queue_saturated"
#: The Ken Burns pass RECORDED nothing because less than its minimum budget was
#: left of the compile's pre-submit deadline (title cards ran long): every
#: uncached still ships frozen. One entry for the pass, not one per shot.
KEN_BURNS_SKIPPED_NO_BUDGET = "ken_burns_skipped_no_budget"
#: EVERY still of the film stayed a still — no motion was applied anywhere and
#: the delivered video is a slideshow of frozen frames. Raised once for the pass,
#: beside the per-shot codes, because "all of them" is the outage signature
#: (video_production down) and a query should not have to count shots to see it.
KEN_BURNS_ALL_STILLS_FROZEN = "ken_burns_all_stills_frozen"
#: A still the recorder failed could not be animated by the orchestrator's LOCAL
#: fallback either (``shared/ken_burns_local.rescue_frozen_stills``), so that
#: shot IS a frozen still. The success twin is the advisory
#: :data:`KEN_BURNS_LOCAL_FALLBACK`.
KEN_BURNS_LOCAL_FALLBACK_FAILED = "ken_burns_local_fallback_failed"
#: A cover-fit still could not be cropped to the frame's aspect before it was
#: animated, so it keeps its own aspect and the recorder letterboxes it inside a
#: blurred copy of itself (the hard vertical seam of run 01a10fcf).
STILL_COVER_CROP_FAILED = "still_cover_crop_failed"
#: The run authored a ``ken_burns.preset`` this build does not know (a typo, or a
#: preset from a newer build), so every still took the default ``zoom_in`` move
#: instead of the motion that was asked for. ``detail`` names the authored value.
#: A DEGRADATION: the film moves, but not the way its author ordered.
KEN_BURNS_PRESET_UNKNOWN = "ken_burns_preset_unknown"
#: A per-shot sound-effect cue was DROPPED at compile time because its catalog id
#: could not be presigned (storage outage, unknown id, library not seeded): the
#: film ships without that accent / transition sound.
SFX_CUE_DROPPED = "sfx_cue_dropped"
#: One or more shots had NO measurable duration, so ``assemble_shots``
#: substituted a fabricated hold to keep the compile legal — those shot lengths
#: are invented, not measured.
SHOT_DURATION_FABRICATED = "shot_duration_fabricated"
#: One or more shots measured a flash-cut length under the minimum and were
#: raised to the floor.
SHOT_DURATION_FLOORED = "shot_duration_floored"
#: The round-robin visual pool had fewer assets than beats needing one, so
#: visuals are REUSED across beats instead of unique.
VISUAL_ASSETS_REUSED = "visual_assets_reused"
#: One or more beats were UNDERSPOKEN (too few authored words for their
#: allotment even at the natural-rate floor) and the picture was held on its
#: last frame to cover the residual.
UNDERSPOKEN_BEAT_HELD = "underspoken_beat_held"
#: The compiler's DECLARED duration and its ffprobe of the encoded file disagree
#: beyond tolerance — a truncated or short-rendered master
#: (``compile_stage._compile_duration_qa_notes``). Distinct from
#: :data:`DURATION_OUT_OF_TOLERANCE`, which grades the delivery against the
#: ORDER; this grades the file against its own timeline.
COMPILE_DURATION_MISMATCH = "compile_duration_mismatch"
#: The compiler reported no usable ``actual_duration_seconds`` (its probe of the
#: delivered mux FAILED), so the declared-vs-encoded check never ran. This is the
#: coded twin of a note that already degrades the stage — it fires on a failed
#: probe, ~0 healthy runs — and is NOT the advisory :data:`DURATION_UNMEASURED`,
#: which is the conformance verdict's "nobody could probe this" on blueprints
#: that structurally have no probe.
COMPILE_DURATION_UNVERIFIED = "compile_duration_unverified"

#: A stage raised the legacy prose ``notes`` channel and no structured entry.
#: Stamped CENTRALLY by the DAG executor (:func:`ensure_degradation_codes`) so a
#: ``degraded`` stage can never reach the run record with ``degradation_codes: []``
#: — the FE warnings panel renders codes + details and showed nothing at all. A
#: row carrying this code names a producer that still needs a real code.
UNCLASSIFIED_NOTE = "unclassified_note"

#: ADVISORY (see :data:`ADVISORY_CODES`) — the delivered length was never
#: PROBED, so the conformance verdict beside this entry was computed against the
#: DECLARED timeline (the compiler's nominal sum of shot durations) rather than
#: against the encoded file. That is not a pass and not a fail: it is an
#: UNVERIFIED verdict. The plan checking itself cannot detect a master whose
#: picture stopped early — the declared number agrees with the plan by
#: construction — so the fact that no measurement backed the verdict needs an
#: identity of its own, or "unverified" stays byte-identical to "verified on
#: target".
#:
#: It is deliberately NOT a degradation, and that tiering is the whole point:
#: "we did not verify this length" is a statement about OUR INSTRUMENTATION, not
#: a defect in the artifact. Nothing about an unprobed master says the master is
#: wrong. Worse, it fires on EVERY stage that cannot reach a probe — and two
#: whole blueprints structurally cannot: ``demo_video`` and ``showcase`` route
#: through ``demo_video/encode_master.py``, whose ``EncodeMasterOutput`` has no
#: ``actual_duration_seconds`` to pass. As a degradation this code would
#: therefore flip ``degraded=True`` on 100% of two content types, indefinitely,
#: including on perfectly conforming masters. That is exactly the
#: 96.7%-of-masters near-universal-flag failure the advisory tier exists to
#: prevent. Raised IN ADDITION to :data:`DURATION_OUT_OF_TOLERANCE` where both
#: apply — on the ``advisories`` field beside that degradation — because a run
#: can be both out of band and unverified and those are distinct facts; the
#: tolerance breach keeps flipping ``degraded``, this never does.
DURATION_UNMEASURED = "duration_unmeasured"

#: ADVISORY (see :data:`ADVISORY_CODES`) — the PRE-AUDIO narration estimate was
#: off the steering target for this order. It is deliberately NOT a degradation:
#: the estimate is ``words / wpm``, its own error is several percent, and the
#: only number that can grade a delivery is the DELIVERED one. A lesson run
#: measured 2026-09-11 came in at +0.22% of its order while its script stage had
#: already stamped ``duration_out_of_tolerance`` from the estimate — a conformant
#: film reported degraded by a guess made before any audio existed. The compile
#: stages already raise the real, artifact-measured verdict
#: (``duration_conformance.py``), so this code carries the steering fact without
#: a second, worse opinion on the same question.
NARRATION_ESTIMATE_OFF_TARGET = "narration_estimate_off_target"
#: ADVISORY — ``website_review_script``'s narration structure is lopsided across
#: its acts (one act carries most of the words, another is a token line). A fact
#: about how the script was shaped, not proof the delivered film is wrong.
SCRIPT_ACT_IMBALANCE = "script_act_imbalance"
#: ADVISORY — no slide/card source exists for this run, so the deck stage returned an
#: empty deck without calling slide_service. Raised by ``website_review_slides`` (the
#: shotlist authored no ``slide`` shots; prod 01a0a054) and by ``showcase_slides`` for a
#: GENERATED-mode showcase, which has no ``card`` source at all (prod 01a0ab23). As the
#: ``slides_stubbed`` degradation both marked every such run degraded. A real render
#: outage hard-fails instead; an AUTHORED showcase with no cards stays ``slides_stubbed``.
SLIDES_NOT_AUTHORED = "slides_not_authored"
#: ADVISORY — ``website_review_slide_plan`` could not use the model's reply (it
#: failed, was malformed, or named no known narration line) and authored the deck
#: from the DETERMINISTIC critique projection instead. A real deck still ships and
#: nothing is missing from the video, so this is not a degradation — but the
#: fallback must never be silent: returning an empty/derived result on a malformed
#: reply with no trace is exactly ``html_render_service`` #132.
SLIDE_PLAN_LLM_FALLBACK = "slide_plan_llm_fallback"
#: ADVISORY — a discovered/pinned media result is not a media FILE (a YouTube or
#: Vimeo watch page, a URL answering ``text/html``), so it was kept for reference
#: and never composited. A classification of the search results, not a defect in
#: the video; the real "could not re-publish a media file" is
#: :data:`MEDIA_NOT_COMPOSITED`.
MEDIA_NOT_COMPOSITABLE = "media_not_compositable"
#: ADVISORY — the delivered master landed INSIDE the conformance band, but the PLAN
#: it was produced from had already been stamped infeasible upstream
#: (``reconcile_timeline`` classified ``convergence_outcome="cant_converge"`` and/or
#: raised :data:`ORDERED_DURATION_INFEASIBLE`). The band verdict is therefore not a
#: clean pass and must not read as one.
#:
#: Confirmed on prod run ``01a0a3c6-c8e7-73db-b13e-fded1a4cbb4f``: the timeline was
#: rejected (``feasible: [False, False, False]``, ``vo_rate`` already floored to 0.8
#: on half the beats) and ``demo_video_encode_master`` then graded the resulting
#: 32.038s against a 30.0s order as ``within_tolerance: True`` at +6.79%, purely
#: because its band is +/-10% while the plan had been refused with NO band at all.
#: Fleet-wide over 14 days, ``duration_out_of_tolerance`` (60) +
#: ``ordered_duration_infeasible`` (15) are 68% of ALL degradations emitted, so a
#: verdict that whitewashes the second with the first is hiding the dominant defect.
#:
#: ADVISORY, not a degradation, on this module's own tier test — and here the test is
#: about DOUBLE COUNTING rather than about instrumentation. The infeasibility itself
#: is ALREADY a degradation (:data:`ORDERED_DURATION_INFEASIBLE`) raised by the stage
#: that actually decided it, so the run is already flagged; re-raising it at the
#: encode exit would report one fault twice, from a stage that only observed it. What
#: is missing is not another flag — it is that the conformance record and its log line
#: said "within +/-10%" with nothing beside them. This code is that "beside them".
#: ⛔ It deliberately does NOT change ``within_tolerance`` (see
#: ``duration_conformance.duration_conformance``): post-rollout sample size is one
#: run, so this makes the truth VISIBLE and changes nothing about what passes.
DURATION_PLAN_INFEASIBLE = "duration_plan_infeasible"
#: ADVISORY — the compiler published an entry on ``MasteringReport.advisories``: a
#: mastering check it MEASURED as breached and deliberately did NOT gate. The first is
#: ``banding_residual`` (``video_compiler_service`` ``mastering.banding_advisories``):
#: run 01a0ab23 delivered ``gate_pass: true`` and ``violations: []`` while all three
#: shots requested deband (banding 0.262 / 0.259 / 0.5935), so the breach was measured
#: and invisible. ``detail`` names the entry's own ``stage/metric/value/limit/tier``.
#:
#: ADVISORY, not a degradation, because the compiler itself put it OUTSIDE
#: ``violations``: every ``violations`` entry is a check that cleared ``gate_pass``,
#: and a ``tunable`` banding ceiling is the 96.7%-of-masters noise
#: ``mastering_gate_degradations`` already refuses to record. One generic code so a
#: new compiler advisory is surfaced without an orchestrator change.
MASTERING_ADVISORY = "mastering_advisory"
#: A text-bearing motion-graphics item (e.g. a kinetic-typography claim card)
#: whose text matches no script line, so it was not rendered. Never placed on a
#: line it does not belong to; an advisory because the film itself is intact.
MOTION_ITEM_UNBOUND = "motion_item_unbound"
#: A motion-graphics clip was rendered KEYED to a line but the line shipped a
#: different visual (footage / curated media outranked it). A paid clip that
#: never reaches the film is recorded, never silent (run 01a10fcf dropped two).
MOTION_CLIP_UNUSED = "motion_clip_unused"
#: A still's Ken Burns recording failed (or came back near-static) and the shot
#: was animated by the orchestrator's local deterministic pan/zoom instead of
#: shipping a frozen frame. The film has motion; the designed clip is missing.
KEN_BURNS_LOCAL_FALLBACK = "ken_burns_local_fallback"
#: Burned-in captions were sent WITHOUT per-word karaoke timing although the
#: service has karaoke on: the product trades the word highlight for one
#: continuous caption box (the compiler boxes each karaoke run separately).
CAPTION_KARAOKE_OFF = "caption_karaoke_off"
#: A Ken Burns clip came back with almost no visible motion (first vs last frame
#: too similar) and no local fallback could replace it — a near-frozen shot.
KEN_BURNS_NEAR_STATIC = "ken_burns_near_static"
#: ``site_capture`` could not record one capture target (the recorder answered
#: "failed", or the per-path task raised). Other paths may still have produced
#: footage; this clip is missing from the render.
SITE_CAPTURE_CLIP_FAILED = "site_capture_clip_failed"
#: ``site_capture``'s per-screen pass failed (the recorder reported the job
#: failed, a 4xx we caused, or the returned bundle was unreadable), so some or
#: all screens are missing and the render falls back to slides.
SITE_CAPTURE_SCREENS_FAILED = "site_capture_screens_failed"
#: ``site_capture`` delivered ZERO real footage (no clips / no screens): the
#: render falls back to AI images, text cards or slides.
SITE_CAPTURE_NO_FOOTAGE = "site_capture_no_footage"
#: ``site_capture``'s page scrape produced NO grounding text (SSRF guard refused
#: the target, the fetch failed, or a 200 carried nothing readable), so the
#: script is written without copy from the live site.
SITE_PAGE_TEXT_MISSING = "site_page_text_missing"
#: ``site_capture``'s page scrape read text under the grounding floor (usually a
#: JavaScript-rendered shell), so the script has too little site copy to ground on.
SITE_PAGE_TEXT_THIN = "site_page_text_thin"

#: Every code this build knows how to raise. Membership is enforced by
#: :func:`degradation` at the write site (see the module docstring for why it is
#: deliberately NOT enforced when reading a persisted output back).
PROMO_HTML_DESIGN_FAILED = "promo_html_design_failed"
SOURCE_GROUNDING_MISSING = "source_grounding_missing"
#: ``promo_review``'s reviewer returned no scored criteria on every draw, so the
#: video was NOT reviewed (prod run 01a17e6f reported that as a 0.0 success).
PROMO_REVIEW_EMPTY = "promo_review_empty"

DEGRADATION_CODES = frozenset(
    {
        PROMO_HTML_DESIGN_FAILED,
        SOURCE_GROUNDING_MISSING,
        PROMO_REVIEW_EMPTY,
        MASTERING_GATE_FAILED,
        DURATION_OUT_OF_TOLERANCE,
        FOOTAGE_TRIMMED,
        CAPTIONS_MISSING,
        SOUND_BED_MISSING,
        MUSIC_BED_NOT_APPLIED,
        SLIDES_STUBBED,
        AUDIO_UNNORMALIZED,
        DURATION_NOT_ORDERED,
        ORDERED_DURATION_INFEASIBLE,
        STRUCTURAL_RELIEF_REFUSED,
        SPEECH_RATE_UNUSABLE,
        NARRATION_RATE_FLOOR_CLAMPED,
        DURATION_PICTURE_TRUNCATED,
        SCRIPT_CLAIMS_UNGROUNDED,
        MEDIA_DISCOVERY_INCOMPLETE,
        MEDIA_NOT_COMPOSITED,
        AI_IMAGES_UNAVAILABLE,
        FOOTAGE_CRITIQUE_UNAVAILABLE,
        FOOTAGE_BELOW_BAR,
        REVIEW_SKIPPED,
        TITLE_CARD_DROPPED,
        END_CARD_DROPPED,
        FAILED_SCENARIO_FOOTAGE,
        URL_REWRITTEN_BEFORE_TTS,
        GROWTH_BUNDLE_EMPTY,
        KEN_BURNS_STILL_SOFT_FAILED,
        KEN_BURNS_STILL_QUEUE_SATURATED,
        KEN_BURNS_SKIPPED_NO_BUDGET,
        KEN_BURNS_ALL_STILLS_FROZEN,
        KEN_BURNS_LOCAL_FALLBACK_FAILED,
        STILL_COVER_CROP_FAILED,
        KEN_BURNS_PRESET_UNKNOWN,
        SFX_CUE_DROPPED,
        SHOT_DURATION_FABRICATED,
        SHOT_DURATION_FLOORED,
        VISUAL_ASSETS_REUSED,
        UNDERSPOKEN_BEAT_HELD,
        COMPILE_DURATION_MISMATCH,
        COMPILE_DURATION_UNVERIFIED,
        UNCLASSIFIED_NOTE,
        SITE_CAPTURE_CLIP_FAILED,
        SITE_CAPTURE_SCREENS_FAILED,
        SITE_CAPTURE_NO_FOOTAGE,
        SITE_PAGE_TEXT_MISSING,
        SITE_PAGE_TEXT_THIN,
    }
)

#: Codes that are REPORTED but never count as degradation. An advisory entry
#: rides the identical :class:`Degradation` shape on its own ``advisories``
#: field (``base.StageOutputWithDegradations``), which :func:`is_degraded` and
#: ``run_repository.degraded_stage_clause`` both ignore.
#:
#: Why a second tier rather than "just do not record it": the two failure modes
#: this module already hit are BOTH live here. Suppressing the signal entirely
#: loses the machine fact (``duration_conformance`` recorded nothing at all for
#: exactly that reason); recording it as a degradation lights the flag on
#: healthy runs (``mastering_gate`` fired on 96.7% of masters). A pre-audio
#: estimate missing a 4% steering target is a real, queryable fact about how the
#: script was aimed — and it is NOT evidence the delivered film is wrong, which
#: is the only thing ``degraded`` is allowed to mean. :data:`DURATION_UNMEASURED`
#: is here for the same reason from the other direction: it reports a gap in our
#: INSTRUMENTATION (nobody probed the file), fires on every stage that cannot
#: reach a probe — 100% of ``demo_video``/``showcase`` masters, structurally —
#: and says nothing whatever about the artifact.
#:
#: Disjoint from :data:`DEGRADATION_CODES` by construction: :func:`degradation`
#: refuses an advisory code and :func:`advisory` refuses a degradation code, so
#: a code cannot silently change tier by being added to the wrong set.
ADVISORY_CODES: frozenset[str] = frozenset(
    {
        NARRATION_ESTIMATE_OFF_TARGET,
        DURATION_UNMEASURED,
        SCRIPT_ACT_IMBALANCE,
        SLIDES_NOT_AUTHORED,
        SLIDE_PLAN_LLM_FALLBACK,
        MEDIA_NOT_COMPOSITABLE,
        DURATION_PLAN_INFEASIBLE,
        MASTERING_ADVISORY,
        MOTION_ITEM_UNBOUND,
        MOTION_CLIP_UNUSED,
        KEN_BURNS_LOCAL_FALLBACK,
        KEN_BURNS_NEAR_STATIC,
        CAPTION_KARAOKE_OFF,
    }
)

if ADVISORY_CODES & DEGRADATION_CODES:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "a code cannot be in both tiers: "
        f"{sorted(ADVISORY_CODES & DEGRADATION_CODES)} — pick one in "
        "src/services/stages/shared/degradation.py"
    )


class Degradation(TypedDict):
    """One machine-readable quality defect on an otherwise-``completed`` stage."""

    #: Stable identifier from :data:`DEGRADATION_CODES`. THE queryable half:
    #: it must not carry run-specific values (put those in ``detail``).
    code: str
    #: One human-readable sentence naming the actual measurement that failed.
    #: A generic "quality problem" is a worse diagnostic than no entry at all.
    detail: str


def degradation(code: str, detail: str) -> Degradation:
    """Build one degradation entry.

    Raises ``ValueError`` on an unknown code. That is deliberate and it is NOT a
    contradiction of "never fatal": this fires only on a code that is not in
    :data:`DEGRADATION_CODES`, i.e. a typo in orchestrator source, which is a
    programming error caught by any test that walks the recording path — never
    something a production input can trigger, because the code argument is
    always a module constant at every call site.
    """
    if code in ADVISORY_CODES:
        raise ValueError(
            f"unknown degradation code {code!r} — it is an ADVISORY code; build it "
            "with advisory() and put it on the `advisories` field, or this run is "
            "reported degraded by a signal that is not evidence of a bad delivery"
        )
    if code not in DEGRADATION_CODES:
        raise ValueError(
            f"unknown degradation code {code!r} — add it to DEGRADATION_CODES "
            "in src/services/stages/shared/degradation.py"
        )
    return Degradation(code=code, detail=detail)


def advisory(code: str, detail: str) -> Degradation:
    """Build one ADVISORY entry — the same shape, the NON-degrading tier.

    Mirrors :func:`degradation` exactly, including the write-site membership
    check and its rationale, against :data:`ADVISORY_CODES` instead. Passing a
    DEGRADATION code raises rather than quietly downgrading it: routing a real
    quality defect onto the advisory field would make it invisible to
    :func:`is_degraded` and to ``degraded_stage_clause``, which is the exact
    "recorded but unqueryable" failure this channel exists to remove — so the
    mistake is a loud one at the write site, where it is a programming error.
    """
    if code in DEGRADATION_CODES:
        raise ValueError(
            f"unknown advisory code {code!r} — it is a DEGRADATION code; build it "
            "with degradation() and put it on the `degradations` field, or a real "
            "defect ships on a field nothing reads"
        )
    if code not in ADVISORY_CODES:
        raise ValueError(
            f"unknown advisory code {code!r} — add it to ADVISORY_CODES "
            "in src/services/stages/shared/degradation.py"
        )
    return Degradation(code=code, detail=detail)


def is_degraded(output: object) -> bool:
    """True when a PERSISTED stage output mapping carries the degraded signal.

    The single Python definition of "degraded", shared by the stage-detail
    response and anything else reading a raw output blob. It is the OR of the
    two channels:

    * a non-empty ``degradations`` list — the explicit channel above;
    * a non-empty ``notes`` list — the legacy prose channel, kept so nothing
      that reads degraded TODAY silently becomes clean.

    ``advisories`` is deliberately NOT a third term. An advisory is reported so
    it can be queried and never so it can raise the flag; adding it here would
    make ``degraded`` mean "something was noticed" instead of "something shipped
    broken", which is the near-universal-flag failure this channel exists to
    avoid. That omission is load-bearing and is pinned by a test.

    Deliberately tolerant of shape: ``output`` may be ``None`` (no blob), may
    lack either key (every row written before this module), or may carry a JSON
    ``null``. All of those are "not degraded", never an exception — a run's
    honesty flag must not be able to 500 the endpoint that reports it.

    ``run_repository.degraded_stage_clause`` is the SQL twin of this function
    and the two must be changed together.
    """
    if not isinstance(output, dict):
        return False
    return bool(output.get("degradations")) or bool(output.get("notes"))


def ensure_degradation_codes(stage_name: str, output: dict[str, object]) -> bool:
    """Make ``degraded ⇒ at least one coded entry`` hold for a completed output.

    The invariant the run record needs and nothing enforced: :func:`is_degraded`
    reads ``notes`` too, so a stage that only wrote prose was reported
    ``degraded=true`` with ``degradation_codes: []`` — prod run 01a09f82's
    ``media_discovery`` did exactly that, and the FE warnings panel (which renders
    codes + details) showed an unexplained warning.

    When ``notes`` is non-empty and ``degradations`` is empty/absent, one
    :data:`UNCLASSIFIED_NOTE` entry is added whose detail carries the notes
    verbatim. It never changes ``degraded`` (notes already raised it) and never
    touches an output that has a real code. Mutates ``output`` in place and
    returns whether it did, so the caller can name the producer in a log.
    """
    notes = output.get("notes")
    if not isinstance(notes, list) or not notes:
        return False
    if output.get("degradations"):
        return False
    detail = f"{stage_name}: " + "; ".join(str(note) for note in notes)
    output["degradations"] = [degradation(UNCLASSIFIED_NOTE, detail)]
    return True


# ── reconcile_timeline ────────────────────────────────────────────────────────
# Kept as a self-contained block at the END of the registry (rather than spliced into
# DEGRADATION_CODES above) so a parallel append by another lane merges trivially.
# Each code is the coded twin of one degrading prose note ``reconcile_timeline`` writes;
# before these existed every one of those notes reached the run record only as the
# executor's generic ``unclassified_note`` ("degraded via prose notes only"). The
# INFEASIBLE note already had its code (:data:`ORDERED_DURATION_INFEASIBLE`).

#: The reconciled timeline is UNDERFULL: even holding every beat at its maximum the
#: content fills less than the ordered target (outside tolerance). A PLAN-time fact —
#: distinct from :data:`DURATION_OUT_OF_TOLERANCE`, which grades the delivered master.
TIMELINE_UNDERFULL = "timeline_underfull"
#: The reconciled total misses the ordered target outside the tolerance band, in
#: either direction (the solver's headline guarantee did not hold).
TIMELINE_OFF_TARGET = "timeline_off_target"
#: One or more beats are UNSAYABLE: even at the maximum delivery rate their words
#: overrun the beat's allotment, so the narration overruns the shot.
NARRATION_UNSAYABLE = "narration_unsayable"
#: One or more beats are UNDERSPOKEN at plan time: too few words for the allotted
#: seconds, so the planned ``vo_rate`` clamps at the natural floor. The plan-time
#: twin of the delivery-time :data:`NARRATION_RATE_FLOOR_CLAMPED` /
#: :data:`UNDERSPOKEN_BEAT_HELD`, raised by different stages on different evidence.
TIMELINE_UNDERSPOKEN = "timeline_underspoken"
#: A footage beat's narration (word-count ESTIMATE) overruns its recorded clip by more
#: than the freeze-pad ceiling, so the pad is capped and the rate compresses the line.
FREEZE_PAD_CEILING_EXCEEDED = "freeze_pad_ceiling_exceeded"
#: An AUTHORED ``direction.timing.wpm.natural`` governed the word budget while the
#: run's voice was measured at a materially different rate.
AUTHORED_SPEECH_RATE_OVERRIDE = "authored_speech_rate_override"
#: The relief loop classified the timeline ``cant_converge``: it is over target and
#: every over-budget beat is claim-bearing / locked / footage-pinned.
TIMELINE_CANT_CONVERGE = "timeline_cant_converge"
#: A WM/LLM timing-relief pass (the relief loop or its structural pre-pass) could not
#: run; the deterministic allotments shipped un-relieved.
TIMELINE_RELIEF_UNAVAILABLE = "timeline_relief_unavailable"

_RECONCILE_TIMELINE_CODES: frozenset[str] = frozenset(
    {
        TIMELINE_UNDERFULL,
        TIMELINE_OFF_TARGET,
        NARRATION_UNSAYABLE,
        TIMELINE_UNDERSPOKEN,
        FREEZE_PAD_CEILING_EXCEEDED,
        AUTHORED_SPEECH_RATE_OVERRIDE,
        TIMELINE_CANT_CONVERGE,
        TIMELINE_RELIEF_UNAVAILABLE,
    }
)
DEGRADATION_CODES = DEGRADATION_CODES | _RECONCILE_TIMELINE_CODES
if ADVISORY_CODES & DEGRADATION_CODES:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "a code cannot be in both tiers: "
        f"{sorted(ADVISORY_CODES & DEGRADATION_CODES)} — pick one in "
        "src/services/stages/shared/degradation.py"
    )


# ── audit scorecards (seo_audit / cro_audit / marketing_scorecard) ───────────
# Self-contained block at the END of the registry, same convention as the
# reconcile_timeline block above, so a parallel append merges trivially.

#: The run asked for AI analysis (``brief.include_llm``) and analysis_service
#: reported it ``partial`` or ``unavailable`` — or reported no ``ai`` status at all
#: (an older pod), which is never read as complete. The deterministic scorecard
#: still shipped; the AI content did not. Design 2026-10-17 §3.3: never a silent green.
AUDIT_AI_ANALYSIS_UNAVAILABLE = "audit_ai_analysis_unavailable"

_AUDIT_SCORECARD_CODES: frozenset[str] = frozenset({AUDIT_AI_ANALYSIS_UNAVAILABLE})
DEGRADATION_CODES = DEGRADATION_CODES | _AUDIT_SCORECARD_CODES
if ADVISORY_CODES & DEGRADATION_CODES:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "a code cannot be in both tiers: "
        f"{sorted(ADVISORY_CODES & DEGRADATION_CODES)} — pick one in "
        "src/services/stages/shared/degradation.py"
    )
