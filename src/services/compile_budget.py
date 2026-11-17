"""Single source of truth for the video-compile wall-clock budget.

Anchored on ``video_compiler_service``'s OWN backstop
(``Settings.reclaim_max_lifetime_seconds``, default 7200s — "wall-clock ceiling
on a job's total RUNNING life across reclaims"), not a number picked on this
side. Two consumers read the values below:

* ``services/stages/primitives/compile_stage.py`` — the orchestrator's own
  give-up deadline for a compile job must never abandon a job the compiler
  still considers healthy. Production incident 2026-09-17 (Studio re-stitch
  run ``01a0aeae-…``, compile job 1602): the orchestrator's 1320s wait gave up
  on a compile the compiler was still happily heartbeating (its own subprocess
  watchdog is 1800s, plus reaper redispatch on top of that), triggering a full
  ``phase.start name=title_cards`` stage resume — re-rendering title cards
  (new MinIO objects, re-billed renders) — while the original compile was
  still healthily running.
* ``config.py`` (``StorageSettings.presign_expiry_seconds``) — the presigned
  MinIO URLs handed to the compiler as compile inputs must outlive the wait
  above. Same incident: a compile that legitimately ran past the old 1h TTL
  hit ``shot_asset_unfetchable`` (403 fetching a ``ken_burns`` input) exactly
  one hour after submit, and shipped a master missing that shot's narration.

Deliberately a LEAF module with no other project imports: ``config.py`` is
loaded before almost everything else in this service, and importing this
module from there must never risk an import cycle. Do not add imports here.
"""

from __future__ import annotations

# video_compiler_service Settings.reclaim_max_lifetime_seconds default — see
# video_compiler_service/src/config.py. The compiler force-fails (not
# reclaims) a job past this wall clock, so waiting at least this long before
# giving up can never be "premature"; it is the compiler's OWN worst case for
# an eventually-terminal job, not ours to second-guess or shorten.
COMPILER_RECLAIM_MAX_LIFETIME_SECONDS = 7200.0

# Reaper detection/redispatch latency the compiler can add on top of the above
# before a healthy-but-orphaned job resumes progressing: job_lease_seconds
# (120s default) + reaper_interval_seconds (30s default) + slack for the
# submit/poll HTTP round trip and clock skew between the two services.
_REAPER_DETECTION_MARGIN_SECONDS = 300.0

# The orchestrator's own give-up deadline for ONE compile()/_await_job wait.
# MUST stay >= COMPILER_RECLAIM_MAX_LIFETIME_SECONDS — see compile_stage.py's
# _COMPILE_TIMEOUT_SECONDS, which is bound directly to this constant.
COMPILE_WAIT_CEILING_SECONDS = (
    COMPILER_RECLAIM_MAX_LIFETIME_SECONDS + _REAPER_DETECTION_MARGIN_SECONDS
)  # 7500.0

# Presigned compile-input URLs must outlive the wait above: a compile that
# legitimately runs the full COMPILE_WAIT_CEILING_SECONDS must still be able
# to fetch its inputs in its final seconds. Margin covers the gap between
# "shots presigned" and "submit accepted" plus clock skew. MinIO/SigV4's own
# presign ceiling is 7 days (604800s — miniopy_async.api raises ValueError
# past it), comfortably above this.
PRESIGN_TTL_SECONDS = int(COMPILE_WAIT_CEILING_SECONDS + 300.0)  # 7800
