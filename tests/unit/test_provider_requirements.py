"""P8 requirements manifest — built from the slot registry + stage declarations.

Owner 2026-10-09: the platform scrape/search path is NOT a slot. The manifest lists
only the existing configurable slots (llm, tts, image, video_gen, device_farm) plus
the grant slot publish.youtube, and webscraper + media_discovery (which replaced the
retired KAG gateway) under ``dependencies`` as fixed platform dependencies.
"""

from __future__ import annotations

import src.services.blueprints  # noqa: F401  registers every blueprint
from src.services.blueprints.registry import get_registry
from src.services.providers import tester, tts_inprocess
from src.services.providers.manifest import build_manifest
from src.services.providers.registry import (
    ANTHROPIC_DEFAULT_BASE_URL,
    ELEVENLABS_DEFAULT_BASE_URL,
    OPENAI_DEFAULT_BASE_URL,
    SLOTS,
    VEO_DEFAULT_BASE_URL,
)
from src.services.providers.settings import ProviderPolicySettings


def _manifest(policy: ProviderPolicySettings | None = None) -> dict:
    return build_manifest(policy or ProviderPolicySettings(), get_registry()).model_dump()


def test_lists_only_existing_configurable_slots_and_kag_as_a_dependency() -> None:
    m = _manifest()
    assert (m["app"], m["manifest_version"]) == ("w2v", 1)
    assert [s["slot"] for s in m["slots"]] == [
        "llm",
        "vlm",
        "tts",
        "asr",
        "image",
        "video_gen",
        "ocr",
        "publish.youtube",
        "device_farm",
    ]
    for gone in ("scrape", "media_search", "doc_parse", "kag", "webscraper"):
        assert gone not in {s["slot"] for s in m["slots"]}
    deps = [(d["name"], d["type"], d["configurable"]) for d in m["dependencies"]]
    assert deps == [
        ("webscraper", "platform", False),
        ("media_discovery", "platform", False),
    ]


def test_grant_slot_has_no_kinds() -> None:
    yt = next(s for s in _manifest()["slots"] if s["slot"] == "publish.youtube")
    assert (yt["type"], yt["allowed_kinds"], yt["kinds"]) == ("grant", [], {})
    assert yt["used_by"] == []  # publishing happens after the run, not in a stage


def test_kind_fields_carry_the_server_rules() -> None:
    slots = {s["slot"]: s for s in _manifest()["slots"]}
    tts = slots["tts"]["kinds"]
    compat = {f["key"]: f["required"] for f in tts["openai-compatible"]["fields"]}
    # Exactly what a PUT refuses without: base_url, model, options.voice.
    assert compat == {"base_url": True, "model": True, "options.voice": True}
    assert tts["elevenlabs"]["fields"][0] == {
        "key": "options.voice",
        "label": "Voice ID",
        "required": True,
        "help": "the ElevenLabs voice_id",
        "format": "text",
    }
    bs = slots["device_farm"]["kinds"]["browserstack"]
    assert (bs["base_url"], bs["secret_label"]) == ("forbidden", "Access key")
    assert [f["key"] for f in bs["fields"]] == ["options.username"]
    assert slots["llm"]["default_kind_label"] == "Platform (Nonagentic)"
    # Every field's `required` agrees with the KindSpec the PUT validator enforces.
    for slot in slots.values():
        for kind, req in slot["kinds"].items():
            ks = SLOTS[slot["slot"]].tenant_kind_specs[kind]
            assert req["required_options"] == list(ks.required_options)
            for f in req["fields"]:
                assert f["required"] == ks.field_required(f["key"]), (slot["slot"], kind, f)


def test_policy_allow_list_filters_kinds() -> None:
    m = _manifest(ProviderPolicySettings(allowed_kinds=frozenset({"anthropic"})))
    llm = next(s for s in m["slots"] if s["slot"] == "llm")
    assert (llm["allowed_kinds"], list(llm["kinds"])) == (["anthropic"], ["anthropic"])


def test_default_base_urls_match_the_adapters() -> None:
    assert OPENAI_DEFAULT_BASE_URL == tester.OPENAI_DEFAULT_BASE_URL
    assert OPENAI_DEFAULT_BASE_URL == tts_inprocess.OPENAI_DEFAULT_BASE_URL
    assert ANTHROPIC_DEFAULT_BASE_URL == tester.ANTHROPIC_DEFAULT_BASE_URL
    assert ELEVENLABS_DEFAULT_BASE_URL == tester.ELEVENLABS_DEFAULT_BASE_URL
    assert ELEVENLABS_DEFAULT_BASE_URL == tts_inprocess.ELEVENLABS_DEFAULT_BASE_URL
    assert VEO_DEFAULT_BASE_URL == tester.VEO_DEFAULT_BASE_URL


def test_used_by_comes_from_stage_declarations() -> None:
    slots = {s["slot"]: s for s in _manifest()["slots"]}
    image = {(u["blueprint"], u["stage"], u["when"]) for u in slots["image"]["used_by"]}
    assert ("promo_v1", "ai_image_gen", "brief.ai_image_gen") in image
    assert any(u["stage"] == "course_thumbnail" for u in slots["image"]["used_by"])
    promo_llm = {u["stage"] for u in slots["llm"]["used_by"] if u["project_type"] == "promo"}
    assert "promo_script" in promo_llm
    assert {u["stage"] for u in slots["device_farm"]["used_by"]} == {"app_inspect"}
    # source_ingest uses webscraper (a dependency) — it appears under NO slot except
    # ``ocr`` (scanned-PDF pages, optional), since OCR became a provider slot.
    assert all(
        u["stage"] != "source_ingest" for n, s in slots.items() if n != "ocr" for u in s["used_by"]
    )
    assert {(u["stage"], u["when"]) for u in slots["ocr"]["used_by"]} == {
        ("source_ingest", "optional")
    }
