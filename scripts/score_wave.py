#!/usr/bin/env python3
"""Score a w2v conformance wave from SNAPSHOT FILES, not the live database.

Why snapshots: production `runs`/`projects`/`stage_outputs` rows are hard-deleted with
no archive (lifetime runs 1706 inserted / 4974 deleted). Wave W14 was erased ~2h after
it finished, before anyone read it back. A scorer that queries the DB is therefore a
scorer that can silently return "no data" for work that really happened. This one reads
what the snapshotter captured while the wave was live.

Two rules this tool enforces, because getting either wrong has produced a published
wrong verdict in this workstream before:

  1. SPLIT BY LINEAGE ROLE. A structural-relief PARENT is out of band BY CONSTRUCTION --
     being out of band is the precondition for spawning a child. Averaging parents with
     children measures a fiction. `orig` = no parent and no child; `parent` = some other
     run names it; `child` = it names a parent.
  2. ONLY `delivered_basis == "playable"` ROWS ENTER A MEDIAN. A row without it is the
     plan grading itself, and is reported separately rather than silently included.

A zero-row result is reported as AMBIGUOUS, never as "no data".
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any, cast

TOLERANCE_PCT = 10.0  # DURATION_TOLERANCE_FRACTION = 0.10, duration_conformance.py:199


def _load_json_list(path: pathlib.Path) -> list[dict[str, Any]]:
    """Parse a JSON array of row objects, narrowing ``json.loads``'s ``Any`` return.

    The snapshot files are always ``json_agg(row_to_json(...))`` output from
    postgres -- a JSON array of row objects, or ``[]``. This is the ONE place the
    raw ``Any`` from ``json.loads`` is allowed to leak out; every caller downstream
    gets a real ``list[dict[str, Any]]`` instead of re-deriving the same union at
    every call site (which is what previously made `ty` unable to narrow `rows`
    down from a giant str|dict|list|None union at every field access).
    """
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        return []
    return cast("list[dict[str, Any]]", data)


def load_latest_runs(snapdir: pathlib.Path) -> list[dict[str, Any]]:
    files = sorted(snapdir.glob("runs-*.json"))
    if not files:
        sys.exit(f"AMBIGUOUS: no runs-*.json under {snapdir} -- was the snapshotter running?")
    return _load_json_list(files[-1])


def find_conformance(
    stage_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the duration_conformance payload and the stage that carried it.

    Searched generically rather than by a hardcoded stage name: the compile stage is
    named per blueprint (`demo_video_compile`, `website_review_compile`, ...), and
    guessing it wrong yields an empty result that reads exactly like a clean miss.
    """
    for row in stage_rows:
        out = row.get("output")
        if isinstance(out, dict) and isinstance(out.get("duration_conformance"), dict):
            return out["duration_conformance"], row.get("stage_name")
    return None, None


def find_relief_decisions(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in stage_rows if r.get("stage_name") == "structural_relief_decision"]


# Fields of a structural-relief decision are NOT at the top level of `output`; the
# producer writes them under output["structural_relief_decision"]. Reading the top
# level returned None for every field and the tool printed `outcome=None` rather than
# erroring -- a silent under-report. Verified 2026-09-16 against /tmp/w15-snap-BrrD12:
# `output` carries exactly one key, `structural_relief_decision`.
_DECISION_MARKERS = ("outcome", "refusal", "schema_version", "inputs")


def decision_payload(stage_row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Return (payload, where). `where` is "nested" | "top-level" | "" (unreadable).

    Falls back to a top-level payload so an older/flatter snapshot still scores, but
    anything it cannot recognise is returned as None so the caller can SHOUT about it.
    """
    out = stage_row.get("output")
    if not isinstance(out, dict):
        return None, ""
    nested = out.get("structural_relief_decision")
    if isinstance(nested, dict):
        return nested, "nested"
    if any(k in out for k in _DECISION_MARKERS):
        return out, "top-level"
    return None, ""


def decision_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull the reported fields, tolerating schema v1 AND v2.

    v1 carries `inputs.spawned_target_beats`; v2 drops it and the beat count must come
    from `inputs.shape.target_beats`. `inputs.shape.status_quo_seconds` is v2-ONLY and
    is therefore optional -- requiring it is what made every v1 row read as empty.
    `measured_seconds_per_beat` lives on `inputs.shape`, never on `inputs` itself.
    """
    inp = payload.get("inputs") or {}
    shape = inp.get("shape") or {}
    beats = inp.get("spawned_target_beats")
    if beats is None:
        beats = shape.get("target_beats")
    per_beat = shape.get("measured_seconds_per_beat")
    pred = None
    if per_beat is not None and beats is not None:
        try:
            pred = round(float(per_beat) * int(beats), 2)
        except TypeError, ValueError:
            pred = None
    return {
        "schema_version": payload.get("schema_version"),
        "outcome": payload.get("outcome"),
        "refusal": payload.get("refusal"),
        "verdict": shape.get("verdict"),
        "target_beats": beats,
        "per_beat": per_beat,
        "predicted": pred,
        "status_quo": shape.get("status_quo_seconds"),  # v2 only -- optional
        "child": payload.get("spawned_child_run_id"),
        "spawned_pass": payload.get("spawned_pass"),
    }


@dataclass
class ScoredRun:
    """One `runs` row plus whatever we could resolve from its stage snapshot.

    A plain dict here is what made every downstream field access resolve to the
    union of ALL the dict's value types (str | dict | list | None | ...) under
    `ty` -- dataclass fields keep each field's own type instead.
    """

    id: str
    blueprint: str | None
    status: str | None
    role: str
    srp: Any
    ordered: Any
    conf: dict[str, Any] | None
    stage: str | None
    decisions: list[dict[str, Any]]
    captured: bool


def build_rows(runs: list[dict[str, Any]], stagedir: pathlib.Path) -> list[ScoredRun]:
    parents = {r["parent_run_id"] for r in runs if r.get("parent_run_id")}
    rows: list[ScoredRun] = []
    for r in runs:
        rid = cast(str, r["id"])
        role = "child" if r.get("parent_run_id") else ("parent" if rid in parents else "orig")
        sf = stagedir / f"{rid}.json"
        stage_rows = _load_json_list(sf) if sf.exists() else []
        conf, stage = find_conformance(stage_rows)
        decisions = find_relief_decisions(stage_rows)
        rows.append(
            ScoredRun(
                id=rid,
                blueprint=r.get("blueprint_id"),
                status=r.get("status"),
                role=role,
                srp=r.get("structural_relief_pass"),
                ordered=r.get("ordered_seconds"),
                conf=conf,
                stage=stage,
                decisions=decisions,
                captured=sf.exists(),
            )
        )
    return rows


def print_run_table(rows: list[ScoredRun]) -> None:
    print(f"=== {len(rows)} runs ===")
    for x in sorted(rows, key=lambda x: (x.blueprint or "", x.role)):
        c = x.conf or {}
        dev = c.get("deviation_pct")
        basis = c.get("delivered_basis")
        flag = "" if x.captured else "  [NO STAGE SNAPSHOT]"
        print(
            f"{x.blueprint:<20} {x.id[:13]} {x.role:<6} {x.status:<10} "
            f"ord={x.ordered!s:>5} deliv={c.get('delivered_seconds')!s:>8} "
            f"dev={dev!s:>8} basis={basis!s:<10} srp={x.srp!s}{flag}"
        )


def print_role_summary(rows: list[ScoredRun]) -> None:
    print("\n=== scored by role (playable basis only) ===")
    for role in ("orig", "parent", "child"):
        sel = [
            x
            for x in rows
            if x.role == role
            and (x.conf or {}).get("delivered_basis") == "playable"
            and (x.conf or {}).get("deviation_pct") is not None
        ]
        if not sel:
            print(f"{role:<7} n=0  (AMBIGUOUS -- no playable rows, not proof of absence)")
            continue
        devs = [float((x.conf or {})["deviation_pct"]) for x in sel]
        inband = sum(1 for d in devs if abs(d) <= TOLERANCE_PCT)
        note = "  <- OUT OF BAND BY CONSTRUCTION, not a defect" if role == "parent" else ""
        print(
            f"{role:<7} n={len(devs):<3} median={statistics.median(devs):+7.2f}%  "
            f"min={min(devs):+7.2f}%  max={max(devs):+7.2f}%  in±10%={inband}/{len(devs)}{note}"
        )


def print_basis_notes(rows: list[ScoredRun]) -> None:
    excluded = [x for x in rows if (x.conf or {}).get("delivered_basis") not in (None, "playable")]
    missing = [x for x in rows if x.conf is None]
    if excluded:
        print(f"\nnon-playable basis (reported, NOT in any median): {len(excluded)}")
    if missing:
        print(f"no duration_conformance at all (never reached compile): {len(missing)}")


def print_relief_decisions(rows: list[ScoredRun]) -> int:
    """Print the persisted relief-decision rows. Returns the process exit code."""
    print("\n=== relief decisions (persisted by #1025) ===")
    n_rows = n_read = 0
    unreadable: list[tuple[str, list[str]]] = []
    top_level = 0
    for x in rows:
        for d in x.decisions:
            n_rows += 1
            payload, where = decision_payload(d)
            if payload is None:
                out = d.get("output")
                keys = sorted(out.keys()) if isinstance(out, dict) else [f"<{type(out).__name__}>"]
                unreadable.append((x.id, keys))
                print(f"{x.id[:13]} !! UNREADABLE DECISION PAYLOAD -- output keys={keys}")
                continue
            n_read += 1
            if where == "top-level":
                top_level += 1
            f = decision_fields(payload)
            sq = "" if f["status_quo"] is None else f" status_quo={f['status_quo']}"
            print(
                f"{x.id[:13]} v{f['schema_version']} outcome={f['outcome']} "
                f"refusal={f['refusal']} verdict={f['verdict']} "
                f"target_beats={f['target_beats']} per_beat={f['per_beat']} "
                f"predicted={f['predicted']}{sq} "
                f"child={(f['child'] or '-')[:13]} pass={f['spawned_pass']}"
            )

    if n_rows == 0:
        print("none captured yet (AMBIGUOUS -- the wave may not have reached relief)")
    else:
        print(f"\ndecision rows: {n_rows}, readable: {n_read}, unreadable: {len(unreadable)}")
        if top_level:
            print(f"  note: {top_level} row(s) used the TOP-LEVEL fallback shape.")
    if not unreadable:
        return 0

    sys.stdout.flush()  # keep the banner below the rows it is about
    # LOUD by design. The original defect was that a misnested payload scored as
    # None and looked like a clean 'no relief happened'. It must never be quiet.
    print(
        f"\n!!!!!! {len(unreadable)} of {n_rows} DECISION ROWS COULD NOT BE READ !!!!!!\n"
        "The expected payload output['structural_relief_decision'] was absent and no\n"
        "top-level decision shape was recognised. DO NOT read this section as\n"
        "'no relief decisions' -- the scorer is blind to these rows:",
        file=sys.stderr,
    )
    for rid, keys in unreadable:
        print(f"  {rid} output keys={keys}", file=sys.stderr)
    return 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapdir", type=pathlib.Path)
    ap.add_argument("--blueprint", help="filter, e.g. demo_video_v1")
    args = ap.parse_args()

    runs = load_latest_runs(args.snapdir)
    if args.blueprint:
        runs = [r for r in runs if r.get("blueprint_id") == args.blueprint]
    if not runs:
        sys.exit("AMBIGUOUS: zero runs after filtering -- not the same thing as 'no data'.")

    rows = build_rows(runs, args.snapdir / "stages")

    print_run_table(rows)
    print_role_summary(rows)
    print_basis_notes(rows)
    exit_code = print_relief_decisions(rows)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
