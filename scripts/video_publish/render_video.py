#!/usr/bin/env python3
"""Reusable RENDER harness + `w2v-video create` authoring front door.

TWO commands:

- ``render`` (legacy, the default when no subcommand is given): POST a full hand-
  authored brief. Runs INSIDE a w2v-orchestrator pod (it has httpx, network to
  localhost:6300, and the perpetual license token in $DOWNSTREAM_LICENSE_TOKEN).

      kubectl exec -i -n w2v <orch-pod> -c w2v-orchestrator -- python - < render_video.py --brief - < brief.json

- ``create`` (A0 authoring front door, great-video PR 2.2): POST a ~15-line
  ``VideoIntent`` to /api/projects/from-intent; the orchestrator computes the plan.
  Runs anywhere with $ORCH_URL + $DOWNSTREAM_LICENSE_TOKEN reachable (through the
  ingress — no kubectl exec needed):

      ORCH_URL=https://w2v.test DOWNSTREAM_LICENSE_TOKEN=... \
        python render_video.py create intent.yaml

Contract (snake_case, proven by scripts/e2e_validate.py + happy_path.json):
  POST /api/projects                      full brief          (render)
  POST /api/projects/from-intent          {intent: {...}}     (create)
  POST /api/deliverables/{id}/runs        autopilot           (render)
  GET  /api/runs/{id}/status              poll until terminal
  GET  /api/runs/{id}                     pull terminal artifact_url

Prints a final JSON line: {"run_id","status","artifact_url","project_id"}.
Exit 0 only on a terminal 'completed' with a non-empty artifact_url (or, with
``--no-wait``, once the run is started).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

BASE = os.environ.get("ORCH_URL", "http://localhost:6300")
TOKEN = os.environ.get("DOWNSTREAM_LICENSE_TOKEN", "")
HEADERS = {"X-Tenant-Id": f"Bearer {TOKEN}"}
TERMINAL_OK = {"completed"}
TERMINAL_BAD = {"failed", "cancelled"}
# terminal video stage outputs hold the publishable URL, highest priority first
TERMINAL_STAGES = ("demo_video_compile", "lesson_compile", "course_stitch")
VIDEO_KEYS = ("artifact_url", "course_artifact_url", "video_url")


def _artifact_url(detail: dict[str, Any]) -> str:
    by_stage = {s["stage_name"]: s for s in detail.get("stage_outputs", [])}
    for stage in TERMINAL_STAGES:
        out = (by_stage.get(stage) or {}).get("output") or {}
        for k in VIDEO_KEYS:
            v = out.get(k)
            if isinstance(v, str) and v and v != "None":
                return v
    return ""


def _read_source(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text()


def _poll_run(c: httpx.Client, run_id: str, project_id: str, *, poll: int, deadline_s: int) -> int:
    """Poll a started run to a terminal state; emit the final JSON line + exit code."""
    print(f"run={run_id} started; polling...", file=sys.stderr)
    deadline = time.time() + deadline_s
    last = ""
    while time.time() < deadline:
        s = c.get(f"/api/runs/{run_id}/status")
        if s.status_code != 200:
            print(f"poll {s.status_code}: {s.text[:200]}", file=sys.stderr)
            time.sleep(poll)
            continue
        st = s.json()
        status = st["status"]
        cur = st.get("current_stage")
        stages = ",".join(f"{x['stage_name']}:{x['status']}" for x in st.get("stage_outputs", []))
        line = f"{status} @{cur} [{stages}]"
        if line != last:
            print(f"  {int(time.time() - (deadline - deadline_s))}s {line}", file=sys.stderr)
            last = line
        if status in TERMINAL_OK or status in TERMINAL_BAD:
            detail = c.get(f"/api/runs/{run_id}").json()
            art = _artifact_url(detail)
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "project_id": project_id,
                        "status": status,
                        "artifact_url": art,
                        "error": detail.get("error"),
                    }
                )
            )
            return 0 if (status in TERMINAL_OK and art) else 1
        time.sleep(poll)
    print(
        json.dumps(
            {"run_id": run_id, "project_id": project_id, "status": "timeout", "artifact_url": ""}
        )
    )
    return 1


def cmd_render(c: httpx.Client, args: argparse.Namespace) -> int:
    """Legacy: POST a full hand-authored brief, start an autopilot run, poll."""
    brief = json.loads(_read_source(args.brief))
    r = c.post("/api/projects", json=brief)
    if r.status_code not in (200, 201):
        print(f"FATAL: create project {r.status_code}: {r.text[:500]}", file=sys.stderr)
        return 1
    proj = r.json()
    project_id = proj["id"]
    deliverable_id = proj["deliverables"][0]["id"]
    print(f"project={project_id} deliverable={deliverable_id}", file=sys.stderr)
    r = c.post(
        f"/api/deliverables/{deliverable_id}/runs",
        json={"pause_after_stages": [], "config_overrides": {}},
    )
    if r.status_code not in (200, 201):
        print(f"FATAL: start run {r.status_code}: {r.text[:500]}", file=sys.stderr)
        return 1
    return _poll_run(c, r.json()["id"], project_id, poll=args.poll, deadline_s=args.deadline)


def cmd_create(c: httpx.Client, args: argparse.Namespace) -> int:
    """A0 authoring front door: POST a VideoIntent to /api/projects/from-intent."""
    # yaml.safe_load parses JSON too, so an intent.json or intent.yaml both work.
    intent = yaml.safe_load(_read_source(args.intent))
    r = c.post("/api/projects/from-intent", json={"intent": intent})
    if r.status_code not in (200, 201):
        print(f"FATAL: from-intent {r.status_code}: {r.text[:500]}", file=sys.stderr)
        return 1
    body = r.json()
    print(json.dumps(body))  # {project_id, deliverable_id, run_id, plan_preview_url, events_url}
    if args.no_wait:
        return 0
    return _poll_run(
        c, body["run_id"], body["project_id"], poll=args.poll, deadline_s=args.deadline
    )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="w2v-video")
    sub = ap.add_subparsers(dest="cmd")

    p_render = sub.add_parser("render", help="render a full hand-authored brief (legacy)")
    p_render.add_argument("--brief", default="-", help="path to brief JSON, or - for stdin")
    p_render.add_argument("--deadline", type=int, default=2400, help="poll deadline seconds")
    p_render.add_argument("--poll", type=int, default=10)

    p_create = sub.add_parser("create", help="create a video from a VideoIntent (A0)")
    p_create.add_argument("intent", nargs="?", default="-", help="intent YAML/JSON, or - for stdin")
    p_create.add_argument("--deadline", type=int, default=2400, help="poll deadline seconds")
    p_create.add_argument("--poll", type=int, default=10)
    p_create.add_argument("--no-wait", action="store_true", help="start the run and exit, no poll")
    return ap


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv
    # Backward-compat: no subcommand (or a leading flag like `--brief`) ⇒ legacy render.
    if not raw or (raw[0] not in {"render", "create"} and raw[0].startswith("-")):
        raw = ["render", *raw]
    args = _build_parser().parse_args(raw)

    if not TOKEN:
        print("FATAL: DOWNSTREAM_LICENSE_TOKEN not in env", file=sys.stderr)
        return 2
    with httpx.Client(base_url=BASE, headers=HEADERS, timeout=30) as c:
        return cmd_create(c, args) if args.cmd == "create" else cmd_render(c, args)


if __name__ == "__main__":
    raise SystemExit(main())
