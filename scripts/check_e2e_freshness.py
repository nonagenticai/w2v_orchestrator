#!/usr/bin/env python3
"""Make the E2E evidence gap VISIBLE, because it cannot be made to close itself.

WHY THIS EXISTS — a measured gap, not a hypothetical.

Go-live for w2v is claimed as "ability to create any project type product fully, every
scenario". The evidence for that claim comes from ``e2e-post-promote.yaml``. Two
independent facts, both measured 2026-09-01 over the 28 scheduled runs from 2026-08-05
to 2026-09-01, make that evidence weaker than it reads:

1. **The cron is never dropped, but it is now hours LATE.** Every one of the 28 days has
   exactly one ``schedule`` run — GitHub missed nothing. What changed is delivery time:
   ``0 3 * * *`` used to land at 03:31-03:46Z, and since 2026-08-27 lands at 08:09-14:45Z
   (+5h to +11h45m). This is GitHub's documented "scheduled workflows may be delayed
   during periods of high load" behaviour and nothing in this repo can make it fire on
   time. It matters anyway, because ``e2e-post-promote.yaml`` sizes its own load against
   a premise that the delay destroys: "the 03:00 cron window is quiet, and a 4.6h matrix
   started overnight finishes by morning contending with almost nothing." A matrix that
   starts at 08:09 instead contends with ~36 concurrent sessions for a shared
   minRunners=2/maxRunners=8 org-wide ARC pool. On 2026-09-01 two shards of the
   scheduled run sat ``queued`` for 74+ minutes.

2. **The nightly has never been the go-live proof, and cannot become one.** ``matrix.json``
   carries 41 rows (verified 2026-10-06); 6 are ``nightly: true`` and 3 are
   ``rotation: false`` paid proofs reachable only by label. ``select_e2e_matrix.py``
   narrows the ``schedule`` event to those 6 plus a 3-scenario day-of-year rotation — 9
   shards. The full matrix (the 38 unattended rows) only ever runs from an explicit
   ``workflow_dispatch`` with an empty ``scenario_label``. So the "every scenario" claim
   has NO scheduled evidence source
   at all: it decays silently from the moment someone stops dispatching by hand.

WHY THIS ALERTS INSTEAD OF DISPATCHING

Making the run FIRE on time is not available, and every automated substitute is worse
than the gap it closes:

- An external dispatcher (in-cluster CronJob, following ``coredns-sync``) can only send
  ``workflow_dispatch``. ``select_e2e_matrix.py`` narrows on ``event_name == "schedule"``
  and a dispatch is never ``schedule``, so a cron-driven dispatch with no label runs the
  FULL-matrix fan-out — exactly the load that saturated pgbouncer and got this cron
  disabled on 2026-07-16. There is no input that reproduces the curated subset.
- A catch-up guard that dispatches on a detected miss destroys the evidence it protects:
  ``concurrency`` allows only ONE pending run per group, so a dispatch issued while the
  late scheduled run is still pending CANCELS it (observed 2026-08-28T10:33:17Z).
  It would also be firing on "late", not "missed" — nothing is being missed.

So this script adds no load to the pool and dispatches nothing. It reads the run history
and turns a silent decay into a red run. Judgement lives here, not in inline jq in the
workflow, so it is unit-tested and its failure arms are watched RED — same reason
``select_e2e_matrix.py`` exists.

This check is itself a scheduled workflow and is therefore subject to the same delivery
delay. A late alert is still an alert; a missing one is not.

WHY IT GOES DORMANT INSTEAD OF RED WHEN THE PRODUCER IS OFF (added 2026-09-06)

Everything above assumes ``e2e-post-promote.yaml`` is still producing evidence. On
2026-09-04, PR #954 removed its ``schedule:`` on an owner directive — the nightly was
paying LLM + render spend to re-measure a pipeline that has never completed (6
full-matrix attempts, 0 successes). From that moment this alarm was red BY
CONSTRUCTION: two of its three arms (no ``schedule`` run in history; no successful
full-matrix run) can no longer be satisfied by anything, because the thing they measure
is deliberately switched off. It duly failed all 11 of its own scheduled runs and never
once went green. #954 dealt with that by removing THIS workflow's cron too, and left a
comment asking whoever re-arms to remember to re-arm both, in the right order.

A comment is not a mechanism. So the precondition is now MEASURED, not remembered:
``evidence_source_armed()`` reads ``e2e-post-promote.yaml`` and asks whether it still
carries an active cron. If it does not, the correct verdict is not "the evidence is
stale" — it is "there is no evidence to be stale, and that is intentional" — so the run
reports one ``notice`` and exits 0.

This is NOT ``continue-on-error`` and NOT a deleted assertion. Every arm stays fully
armed and every failure message is unchanged; a run against an ARMED producer still
fails exactly as before. What changed is only that the alarm now knows whether it
applies. It also re-arms ITSELF: restoring the producer's cron re-arms this check on the
next run, with nothing here to edit and no ordering for a human to get right.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# A `schedule` run older than this means GitHub genuinely DROPPED a window, rather than
# merely delivering it late. Sized above 24h + the worst observed lag (11h45m) so normal
# lateness never fires it: only a real miss does.
DEFAULT_MAX_SCHEDULE_AGE_HOURS = 40.0

# The cron's nominal delivery hour (UTC), from `on.schedule` in e2e-post-promote.yaml.
CRON_HOUR_UTC = 3

# Lag above which the "03:00 window is quiet" load premise no longer holds and the
# matrix is competing with the working day for the shared 8-runner pool.
DEFAULT_MAX_LAG_HOURS = 4.0

# A run with at least this many jobs is a FULL-matrix run (38 scenario shards as of 2026-10-06 +
# gate + set-matrix). The curated nightly emits 9 shards + 2, so the two are far apart and no
# threshold in between is ambiguous. Counting jobs is used rather than reading the
# dispatch inputs because the REST run object does not carry `inputs`.
DEFAULT_FULL_MATRIX_MIN_JOBS = 30

# How stale the full-matrix "every scenario" proof may get before the go-live claim is unevidenced.
DEFAULT_MAX_FULL_MATRIX_AGE_DAYS = 8.0

# The workflow whose runs ARE the evidence. Its arming state is this alarm's
# precondition, so it is read from the file rather than duplicated into a constant here:
# a second copy of "is the nightly on?" is a second thing that can go stale.
DEFAULT_EVIDENCE_SOURCE = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "e2e-post-promote.yaml"
)


class MalformedRunError(ValueError):
    """A run record could not be interpreted. Never silently skipped: a run history that
    does not parse must not be reported as a healthy history."""


@dataclass(frozen=True)
class Finding:
    """One judgement. ``failed`` drives the exit code; warnings annotate but pass."""

    level: str  # "error" | "warning" | "notice"
    message: str

    @property
    def failed(self) -> bool:
        return self.level == "error"


@dataclass(frozen=True)
class Run:
    run_id: int
    event: str
    created_at: datetime
    status: str
    conclusion: str
    job_count: int | None


def _parse_timestamp(raw: str) -> datetime:
    # GitHub emits Zulu time; fromisoformat handles the offset form, not the bare "Z"
    # suffix on every Python this repo may run under.
    text = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MalformedRunError(f"run timestamp {raw!r} is not ISO-8601") from exc
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_runs(raw_runs: Sequence[dict[str, Any]]) -> list[Run]:
    """Normalise ``gh run list --json`` records, rejecting anything unreadable."""
    runs: list[Run] = []
    for raw in raw_runs:
        created = raw.get("createdAt") or raw.get("created_at")
        if not isinstance(created, str):
            raise MalformedRunError(f"run record has no usable createdAt: {raw!r}")
        job_count = raw.get("jobCount", raw.get("job_count"))
        if job_count is not None and not isinstance(job_count, int):
            raise MalformedRunError(f"jobCount must be an int or null, got {job_count!r}")
        runs.append(
            Run(
                run_id=int(raw.get("databaseId", raw.get("id", 0))),
                event=str(raw.get("event", "")),
                created_at=_parse_timestamp(created),
                status=str(raw.get("status", "")),
                conclusion=str(raw.get("conclusion") or ""),
                job_count=job_count,
            )
        )
    return runs


# An active `cron:` key. Matched anywhere on a non-comment line so that both the block
# form (`- cron: "0 3 * * *"`) and a flow-style `on: {schedule: [{cron: "..."}]}` count.
_ACTIVE_CRON = re.compile(r"(?:^|[\s{,\[])cron\s*:")


def evidence_source_armed(workflow: Path) -> bool:
    """Does ``e2e-post-promote.yaml`` still carry an ACTIVE ``schedule:`` trigger?

    A deliberately dumb comment-stripping line scan, for three reasons.

    1. It must run under the runner's bare ``python3``. The workflow invokes this script
       directly with no venv and no installed dependencies, so PyYAML is not available.
    2. PyYAML would be the wrong tool anyway. YAML 1.1 parses the key ``on:`` as the
       BOOLEAN ``True``, which is a well-worn way to write a workflow parser that reads
       every trigger as absent.
    3. A *commented-out* cron must read as ABSENT — and that is exactly how #954
       disabled this one, leaving behind a literal
       ``#   Re-arm by restoring:  schedule:\\n    - cron: "0 3 * * *"`` line that a
       naive substring search would happily count as armed. Dropping comment lines
       first gives the right answer for free.

    The scan errs toward ARMED: a false positive leaves the alarm live and loud, which
    is recoverable by reading the annotation, while a false negative would mute it
    silently and permanently. That is the direction a guard should fail in.
    """
    for line in workflow.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        if _ACTIVE_CRON.search(line):
            return True
    return False


def check_evidence_source_armed(workflow: Path) -> Finding | None:
    """Decide whether this alarm applies at all.

    ``None`` means the producer is armed and every judgement below should run. A
    ``Finding`` means the whole alarm is answered by this one fact and the other arms
    would only report the intended state as a defect.
    """
    if not workflow.is_file():
        return Finding(
            "error",
            f"cannot read the evidence source at {workflow} to decide whether this alarm "
            "applies. A renamed, moved or deleted e2e-post-promote.yaml must fail LOUD: "
            "assuming 'it must be switched off then' would silently mute this check "
            "forever, which is the exact failure mode it was written to prevent.",
        )
    if evidence_source_armed(workflow):
        return None
    return Finding(
        "notice",
        f"DORMANT — {workflow.name} carries no active `schedule:` trigger (removed "
        "2026-09-04 by PR #954, owner directive), so the nightly E2E evidence this alarm "
        "polices is deliberately NOT being produced. Staleness is the intended state "
        "here, not a defect; reporting it as one would be an alarm on a switched-off "
        "capability, which manufactures noise rather than confidence. Every arm of this "
        "check remains armed and unchanged, and it RE-ARMS ITSELF on the next run once "
        "that cron is restored — nothing in this repo needs editing to bring it back.",
    )


def scheduled_lag_hours(run: Run) -> float:
    """Hours between the cron's nominal window and when GitHub actually created the run.

    The nominal window is ``CRON_HOUR_UTC`` on the run's own date, except for a run
    created BEFORE that hour, which belongs to the previous day's window.
    """
    nominal = run.created_at.replace(hour=CRON_HOUR_UTC, minute=0, second=0, microsecond=0)
    if run.created_at < nominal:
        nominal -= timedelta(days=1)
    return (run.created_at - nominal).total_seconds() / 3600.0


def check_schedule_delivered(
    runs: Sequence[Run], now: datetime, max_age_hours: float = DEFAULT_MAX_SCHEDULE_AGE_HOURS
) -> Finding:
    """Did GitHub deliver the cron at all? Late is survivable; dropped is the real miss."""
    scheduled = [r for r in runs if r.event == "schedule"]
    if not scheduled:
        return Finding(
            "error",
            "no `schedule` run of e2e-post-promote.yaml appears in the queried history at "
            "all — the nightly cron is not being delivered.",
        )
    newest = max(scheduled, key=lambda r: r.created_at)
    age = (now - newest.created_at).total_seconds() / 3600.0
    if age > max_age_hours:
        return Finding(
            "error",
            f"the newest `schedule` run (#{newest.run_id}) is {age:.1f}h old, over the "
            f"{max_age_hours:.0f}h ceiling — GitHub DROPPED a nightly window rather than "
            "merely delaying it. The nightly E2E evidence has a hole.",
        )
    return Finding("notice", f"nightly cron delivered {age:.1f}h ago (run #{newest.run_id}).")


def check_schedule_lag(
    runs: Sequence[Run],
    sample: int = 7,
    max_lag_hours: float = DEFAULT_MAX_LAG_HOURS,
) -> Finding:
    """Is the matrix still landing in the quiet window its load budget assumes?"""
    scheduled = sorted(
        (r for r in runs if r.event == "schedule"), key=lambda r: r.created_at, reverse=True
    )[:sample]
    if not scheduled:
        return Finding("notice", "no scheduled runs to measure delivery lag from.")
    lags = [scheduled_lag_hours(r) for r in scheduled]
    median = statistics.median(lags)
    if median > max_lag_hours:
        return Finding(
            "warning",
            f"median cron delivery lag over the last {len(lags)} scheduled runs is "
            f"{median:.1f}h (worst {max(lags):.1f}h) against a {CRON_HOUR_UTC:02d}:00 UTC "
            f"cron. e2e-post-promote.yaml sizes max-parallel against a QUIET 03:00 window; "
            "at this lag the matrix runs during the working day and competes with the "
            "shared org-wide ARC pool. Expect queued shards, not a product regression.",
        )
    return Finding("notice", f"median cron delivery lag is {median:.1f}h — within budget.")


def check_full_matrix_evidence(
    runs: Sequence[Run],
    now: datetime,
    min_jobs: int = DEFAULT_FULL_MATRIX_MIN_JOBS,
    max_age_days: float = DEFAULT_MAX_FULL_MATRIX_AGE_DAYS,
) -> Finding:
    """Is the "every scenario" go-live claim still backed by a run that proved it?

    No trigger produces this automatically — the `schedule` event is narrowed to a 9-shard
    subset — so without this check the claim decays with no signal whatsoever.
    """
    full = [
        r
        for r in runs
        if r.conclusion == "success" and r.job_count is not None and r.job_count >= min_jobs
    ]
    if not full:
        return Finding(
            "error",
            "no SUCCESSFUL full-matrix (>= "
            f"{min_jobs} jobs) e2e run in the queried history. The go-live claim 'every "
            "scenario' has no evidence: the nightly cron only ever runs the curated 9. "
            "Dispatch e2e-post-promote.yaml with an EMPTY scenario_label.",
        )
    newest = max(full, key=lambda r: r.created_at)
    age_days = (now - newest.created_at).total_seconds() / 86400.0
    if age_days > max_age_days:
        return Finding(
            "error",
            f"the newest successful full-matrix run (#{newest.run_id}, {newest.job_count} "
            f"jobs) is {age_days:.1f} days old, over the {max_age_days:.0f}-day ceiling. "
            "The full-matrix go-live evidence is stale — the nightly cron does NOT "
            "refresh it (it runs only the curated subset). Dispatch with an EMPTY scenario_label.",
        )
    return Finding(
        "notice",
        f"full-matrix evidence is {age_days:.1f} days old (run #{newest.run_id}, "
        f"{newest.job_count} jobs).",
    )


def evaluate(
    raw_runs: Sequence[dict[str, Any]],
    now: datetime,
    *,
    max_schedule_age_hours: float = DEFAULT_MAX_SCHEDULE_AGE_HOURS,
    max_lag_hours: float = DEFAULT_MAX_LAG_HOURS,
    full_matrix_min_jobs: int = DEFAULT_FULL_MATRIX_MIN_JOBS,
    max_full_matrix_age_days: float = DEFAULT_MAX_FULL_MATRIX_AGE_DAYS,
    evidence_source: Path = DEFAULT_EVIDENCE_SOURCE,
) -> list[Finding]:
    """All three judgements. Every arm runs — one failure must not hide another.

    Unless the producer is disarmed, in which case the arming fact is the ONLY honest
    judgement available and is returned alone. See ``check_evidence_source_armed``.
    """
    gate = check_evidence_source_armed(evidence_source)
    if gate is not None:
        return [gate]
    runs = parse_runs(raw_runs)
    return [
        check_schedule_delivered(runs, now, max_schedule_age_hours),
        check_schedule_lag(runs, max_lag_hours=max_lag_hours),
        check_full_matrix_evidence(
            runs, now, min_jobs=full_matrix_min_jobs, max_age_days=max_full_matrix_age_days
        ),
    ]


_ANNOTATION = {"error": "::error::", "warning": "::warning::", "notice": "::notice::"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check E2E evidence freshness.")
    parser.add_argument(
        "--runs",
        default="-",
        help="Path to the JSON run array from `gh run list --json`, or - for stdin.",
    )
    parser.add_argument(
        "--max-schedule-age-hours", type=float, default=DEFAULT_MAX_SCHEDULE_AGE_HOURS
    )
    parser.add_argument("--max-lag-hours", type=float, default=DEFAULT_MAX_LAG_HOURS)
    parser.add_argument("--full-matrix-min-jobs", type=int, default=DEFAULT_FULL_MATRIX_MIN_JOBS)
    parser.add_argument(
        "--max-full-matrix-age-days", type=float, default=DEFAULT_MAX_FULL_MATRIX_AGE_DAYS
    )
    parser.add_argument(
        "--evidence-source",
        type=Path,
        default=DEFAULT_EVIDENCE_SOURCE,
        help=(
            "Path to the workflow that PRODUCES the evidence. When it carries no active "
            "cron this check reports DORMANT and passes, because there is nothing to be "
            "stale. Overridable for tests only."
        ),
    )
    args = parser.parse_args(argv)

    text = sys.stdin.read() if args.runs == "-" else Path(args.runs).read_text(encoding="utf-8")
    raw_runs = json.loads(text)
    if not isinstance(raw_runs, list):
        print("::error::run history must be a JSON array", file=sys.stderr)
        return 2

    findings = evaluate(
        raw_runs,
        datetime.now(UTC),
        max_schedule_age_hours=args.max_schedule_age_hours,
        max_lag_hours=args.max_lag_hours,
        full_matrix_min_jobs=args.full_matrix_min_jobs,
        max_full_matrix_age_days=args.max_full_matrix_age_days,
        evidence_source=args.evidence_source,
    )
    for finding in findings:
        print(f"{_ANNOTATION[finding.level]}{finding.message}")
    return 1 if any(f.failed for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
