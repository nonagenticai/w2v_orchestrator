"""Unit test for the `w2v-video create` CLI (A0 authoring front door).

Loads scripts/video_publish/render_video.py by path (that dir is not a package)
and drives ``cmd_create`` through an httpx MockTransport — no network, no token,
so it runs hermetically in CI. Asserts the CLI posts the intent to the right
route with the right body, and that the legacy `render` default still routes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "video_publish" / "render_video.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_video_cli", _MOD_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cmd_create_posts_intent_body(tmp_path: Path) -> None:
    mod = _load()
    intent_file = tmp_path / "intent.yaml"
    intent_file.write_text(
        "type: showcase\ntitle: Demo\nurl: https://x.com\npackage: minio://w2v/p.zip\n"
    )
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "project_id": "p",
                "deliverable_id": "d",
                "run_id": "r",
                "plan_preview_url": "/api/runs/r",
                "events_url": "/api/runs/r/events",
                "pause_after_stages": [],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    args = SimpleNamespace(intent=str(intent_file), no_wait=True, poll=1, deadline=10)
    rc = mod.cmd_create(client, args)

    assert rc == 0
    assert str(captured["url"]).endswith("/api/projects/from-intent")
    assert captured["body"] == {
        "intent": {
            "type": "showcase",
            "title": "Demo",
            "url": "https://x.com",
            "package": "minio://w2v/p.zip",
        }
    }


def test_create_subcommand_parses() -> None:
    mod = _load()
    args = mod._build_parser().parse_args(["create", "intent.yaml", "--no-wait"])
    assert args.cmd == "create"
    assert args.intent == "intent.yaml"
    assert args.no_wait is True


def test_bare_flags_default_to_legacy_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """A leading flag (no subcommand) must still route to the legacy render path —
    exercised through the REAL main(), not a copy of its routing condition, so a
    regression in main() actually fails this (run_in_pod.py's legacy invocation)."""
    mod = _load()
    fired: dict[str, str] = {}
    monkeypatch.setattr(mod, "TOKEN", "tok")
    monkeypatch.setattr(mod, "cmd_render", lambda c, a: fired.update(cmd="render") or 0)
    monkeypatch.setattr(mod, "cmd_create", lambda c, a: fired.update(cmd="create") or 0)
    assert mod.main(["--brief", "-"]) == 0
    assert fired["cmd"] == "render"
    fired.clear()
    assert mod.main([]) == 0  # no args at all → still render
    assert fired["cmd"] == "render"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
