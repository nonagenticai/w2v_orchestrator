"""Pre-baked demo-video payloads for the AA trilogy recordings (E1-E3).

Six beats x one primary BDD job each. Split-screen beats (B2 / B4 / B6) carry
a `secondary_payload` that records the partner surface concurrently; the two
recordings are joined at compile time via `FFmpegOps.hstack()` in
`video_production_service.compose`.

Submission flow: `BDDPlaywrightClient.start_demo_video(payload)` for each
payload; poll to completion; collect the resulting MP4s. The B.4 blueprint
binding will lift this list and dispatch one BDD job per primary (+ secondary)
payload, then route artefacts through the lesson compile stage.

Auth: every payload pointed at `fr.nonagentic.ai` carries `auth_config` —
the bdd_playwright_service auto-generates `auth.setup.ts` and a Playwright
`storageState` from `AUTH_LOGIN_URL` / `AUTH_USERNAME` / `AUTH_PASSWORD`
env vars on the execution pod (Step 0 spike, plan §"Step 0").

The trilogy tenant is `aa110000-0000-7aa1-8aa1-000000000001` (seeded Wave 1).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class AuthConfig(TypedDict):
    login_url: str
    type: Literal["form"]


class NarrationOverride(TypedDict, total=False):
    enabled: bool
    tone: Literal["tutorial", "demo", "marketing", "technical"]
    llm_polish: bool


class DemoVideoPayload(TypedDict, total=False):
    url: str
    target_pages: list[str]
    focus_areas: list[str]
    max_scenarios_per_feature: int
    auth_config: AuthConfig
    narration: NarrationOverride
    feed_to_video: bool


class TrilogyBeat(TypedDict):
    id: str
    episode: int
    beat: str
    title: str
    compose: Literal["single", "hstack"]
    primary_payload: DemoVideoPayload
    secondary_payload: DemoVideoPayload | None


_FR_BASE = "https://fr.nonagentic.ai"
_N8N_BASE = "https://n8n.test"
_PLANKA_BASE = "https://kanban.nonagentic.ai"

_FR_AUTH: AuthConfig = {"login_url": f"{_FR_BASE}/", "type": "form"}
_PLANKA_AUTH: AuthConfig = {"login_url": f"{_PLANKA_BASE}/login", "type": "form"}

_DEMO_NARRATION: NarrationOverride = {
    "enabled": True,
    "tone": "demo",
    "llm_polish": True,
}


B1_LIFECYCLE_CLASS: TrilogyBeat = {
    "id": "B1",
    "episode": 1,
    "beat": "B",
    "title": "Create a project, pick a Lifecycle Class",
    "compose": "single",
    "primary_payload": {
        "url": f"{_FR_BASE}/projects/new",
        "target_pages": ["/projects/new"],
        "focus_areas": [
            "fill the project name field",
            "open the Lifecycle Class dropdown",
            "select 'execution', then re-open and select 'workflow-improvement'",
            "submit the form and land on the project detail page",
        ],
        "max_scenarios_per_feature": 4,
        "auth_config": _FR_AUTH,
        "narration": _DEMO_NARRATION,
        "feed_to_video": True,
    },
    "secondary_payload": None,
}


B2_READY_TO_DOING: TrilogyBeat = {
    "id": "B2",
    "episode": 1,
    "beat": "C",
    "title": "Drag READY → DOING; Doing-Orchestrate executes",
    "compose": "hstack",
    "primary_payload": {
        "url": f"{_FR_BASE}/board",
        "target_pages": ["/board"],
        "focus_areas": [
            "locate a card in the READY column on the trilogy board",
            "drag the card into the DOING column and drop it",
            "wait for the card's status pill to update to DOING",
        ],
        "max_scenarios_per_feature": 3,
        "auth_config": _FR_AUTH,
        "narration": _DEMO_NARRATION,
        "feed_to_video": True,
    },
    "secondary_payload": {
        "url": f"{_N8N_BASE}/workflow/AA-Pattern-Doing-Orchestrate/executions",
        "target_pages": ["/executions"],
        "focus_areas": [
            "open the most recent execution of AA Pattern - Doing Orchestrate",
            "observe the cascading green checkmarks: webhook → build context → Seeker fan-out",
            "stop recording before the Expansion node finishes (partial-capture window ~8s)",
        ],
        "max_scenarios_per_feature": 2,
        "narration": _DEMO_NARRATION,
        "feed_to_video": True,
    },
}


B3_GOVERNANCE_WALK: TrilogyBeat = {
    "id": "B3",
    "episode": 2,
    "beat": "B",
    "title": "Walk the 5D governance pages; hover backref counts",
    "compose": "single",
    "primary_payload": {
        "url": f"{_FR_BASE}/principles",
        "target_pages": [
            "/principles",
            "/requirements",
            "/decisions",
            "/procedures",
        ],
        "focus_areas": [
            "land on /principles, hover the first principle card and reveal its backref_card_count",
            "click through to /requirements, repeat the hover-backref interaction",
            "do the same at /decisions and /procedures",
            "scroll one row on each page to show more than the first item",
        ],
        "max_scenarios_per_feature": 6,
        "auth_config": _FR_AUTH,
        "narration": _DEMO_NARRATION,
        "feed_to_video": True,
    },
    "secondary_payload": None,
}


B4_APPROVE_DONE_FINALIZE: TrilogyBeat = {
    "id": "B4",
    "episode": 2,
    "beat": "C",
    "title": "Approve a pending trilogy approval; Done-Finalize fires",
    "compose": "hstack",
    "primary_payload": {
        "url": f"{_FR_BASE}/approvals",
        "target_pages": ["/approvals"],
        "focus_areas": [
            "land on /approvals and pick the first pending trilogy approval",
            "open its detail drawer and read the proposed change",
            "click Approve and confirm the modal",
            "wait for the 'lesson written' toast to appear",
        ],
        "max_scenarios_per_feature": 3,
        "auth_config": _FR_AUTH,
        "narration": _DEMO_NARRATION,
        "feed_to_video": True,
    },
    "secondary_payload": {
        "url": f"{_N8N_BASE}/workflow/AA-Pattern-Done-Finalize/executions",
        "target_pages": ["/executions"],
        "focus_areas": [
            "open the most recent execution of AA Pattern - Done Finalize",
            "observe Action notify firing fast (cascading green)",
            "stop before Compression completes (partial-capture window ~6s)",
        ],
        "max_scenarios_per_feature": 2,
        "narration": _DEMO_NARRATION,
        "feed_to_video": True,
    },
}


B5_SEEKER_READBACK: TrilogyBeat = {
    "id": "B5",
    "episode": 3,
    "beat": "B",
    "title": "Seeker reads back a pre-seeded KAG lesson",
    "compose": "single",
    "primary_payload": {
        "url": f"{_N8N_BASE}/workflow/AA-Activity-Seeker/executions",
        "target_pages": ["/executions"],
        "focus_areas": [
            "open the most recent Seeker execution for the trilogy tenant",
            "expand the Seeker output node to show the retrieved lesson row",
            "highlight the KAG hit count and the cited lesson title",
        ],
        "max_scenarios_per_feature": 3,
        "narration": _DEMO_NARRATION,
        "feed_to_video": True,
    },
    "secondary_payload": None,
}


B6_KAIZEN_TO_PLANKA: TrilogyBeat = {
    "id": "B6",
    "episode": 3,
    "beat": "D",
    "title": "Kaizen detector → improvement card in Planka",
    "compose": "hstack",
    "primary_payload": {
        "url": f"{_N8N_BASE}/workflow/AA-Kaizen-Detector/executions",
        "target_pages": ["/executions"],
        "focus_areas": [
            "open the most recent Kaizen Detector execution",
            "observe detector → router → Planka card-creation nodes lighting green",
            "let the run complete (this beat is fast enough for full capture)",
        ],
        "max_scenarios_per_feature": 3,
        "narration": _DEMO_NARRATION,
        "feed_to_video": True,
    },
    "secondary_payload": {
        "url": f"{_PLANKA_BASE}/projects/aa-trilogy/board",
        "target_pages": ["/board"],
        "focus_areas": [
            "land on the trilogy Planka board",
            "wait for a new improvement card to appear in the BACKLOG column",
            "click the new card to expand its description (cite the kaizen pattern)",
        ],
        "max_scenarios_per_feature": 3,
        "auth_config": _PLANKA_AUTH,
        "narration": _DEMO_NARRATION,
        "feed_to_video": True,
    },
}


TRILOGY_BEATS: tuple[TrilogyBeat, ...] = (
    B1_LIFECYCLE_CLASS,
    B2_READY_TO_DOING,
    B3_GOVERNANCE_WALK,
    B4_APPROVE_DONE_FINALIZE,
    B5_SEEKER_READBACK,
    B6_KAIZEN_TO_PLANKA,
)


TRILOGY_BEATS_BY_ID: dict[str, TrilogyBeat] = {beat["id"]: beat for beat in TRILOGY_BEATS}


def all_payloads() -> list[DemoVideoPayload]:
    """Flatten primary + secondary payloads into a single submission list."""
    out: list[DemoVideoPayload] = []
    for beat in TRILOGY_BEATS:
        out.append(beat["primary_payload"])
        secondary = beat["secondary_payload"]
        if secondary is not None:
            out.append(secondary)
    return out


def beat_payloads(beat_id: str) -> list[DemoVideoPayload]:
    """Return the BDD payload(s) for a single beat — 1 for single, 2 for hstack."""
    beat = TRILOGY_BEATS_BY_ID[beat_id]
    payloads: list[DemoVideoPayload] = [beat["primary_payload"]]
    secondary = beat["secondary_payload"]
    if secondary is not None:
        payloads.append(secondary)
    return payloads


def as_request_dicts(payloads: list[DemoVideoPayload]) -> list[dict[str, Any]]:
    """Coerce TypedDict payloads to plain dicts for HTTP submission."""
    return [{k: v for k, v in p.items()} for p in payloads]
