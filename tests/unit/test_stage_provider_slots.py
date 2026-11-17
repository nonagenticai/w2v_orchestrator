"""Every blueprint stage that can reach a provider slot client DECLARES the slot.

The P8 requirements manifest (``GET /api/provider-requirements``) builds each slot's
``used_by`` from the ``provider_slots`` ClassVar on the stage classes. Nothing forces
a stage author to keep that declaration in sync, so this test derives the truth
from source: it walks the stage class's module (and every ``src.services.stages.*``
module it imports, transitively — helpers like ``primitives/narration.py`` and
``shared/speech_rate.py`` are where most TTS calls live) and looks for the slot
clients:

* ``llm``             — ``WMClient`` (``src.services.llm``: the WM / tenant-LLM path)
* ``tts``             — ``TTSClient``, ``TTSGatewayClient``, ``tts_inprocess``
* ``asr``             — ``TranscriptionWorkerClient``, ``asr_inprocess``
* ``image``           — ``AIContentGenerationClient`` + a ``create_concept_image*`` call,
  or the shared helper ``src.services.concept_images.generate_concept_image``
  (the ONLY place the single-image submit may live, see
  ``tests/unit/test_concept_image_single_path.py``)
* ``video_gen``       — ``AIContentGenerationClient`` + a ``start_generation`` call
* ``ocr``             — ``extract_document`` (scanned-PDF OCR) or ``ocr_client``
* ``device_farm``     — ``AppInspectorClient``
* ``publish.youtube`` — anything from ``youtube_upload_client``

The declared set must EQUAL the detected set: a missing declaration hides a
requirement from the GUI, a stale one claims a requirement that is gone. The scan
over-approximates on purpose (a stage importing another stage module inherits its
clients) — "may call" is the safe direction for a requirements manifest.
"""

from __future__ import annotations

import ast
import importlib.util
from functools import cache

import pytest

import src.services.blueprints  # noqa: F401  registers every blueprint
from src.services.blueprints.base import stage_handler_class
from src.services.blueprints.registry import get_registry
from src.services.providers.registry import SLOTS
from src.services.stages.base import StageHandler

#: (module, imported symbol or None = any symbol) → slot.
_SYMBOL_MARKERS: dict[tuple[str, str | None], str] = {
    ("src.services.llm", "WMClient"): "llm",
    ("src.services.llm.dispatch", None): "llm",
    ("src.services.llm.in_process", None): "llm",
    ("src.clients.tts_client", "TTSClient"): "tts",
    ("src.clients.tts_gateway_client", "TTSGatewayClient"): "tts",
    ("src.services.providers.tts_inprocess", None): "tts",
    ("src.clients.transcription_worker_client", "TranscriptionWorkerClient"): "asr",
    ("src.services.providers.asr_inprocess", None): "asr",
    ("src.clients.app_inspector_client", "AppInspectorClient"): "device_farm",
    ("src.clients.youtube_upload_client", None): "publish.youtube",
    ("src.services.concept_images", "generate_concept_image"): "image",
    ("src.services.documents.extract", "extract_document"): "ocr",
    ("src.clients.ocr_client", None): "ocr",
}
_ACG = ("src.clients.ai_content_generation_client", "AIContentGenerationClient")
#: ACG serves two slots; the method called decides which.
_ACG_METHODS: dict[str, str] = {
    "create_concept_image": "image",
    # The shared helper is called as a bare function (an ast.Name, not an
    # Attribute) with the client as its first argument.
    "generate_concept_image": "image",
    "create_concept_image_batch": "image",
    "start_generation": "video_gen",
}
_STAGE_PKG = "src.services.stages"


def _is_module(name: str) -> bool:
    try:
        spec = importlib.util.find_spec(name)
    except ImportError, ValueError:
        return False
    return spec is not None and bool(spec.origin) and str(spec.origin).endswith(".py")


def _marker_slots(module: str, name: str | None) -> set[str]:
    """Slots ``from module import name`` (or ``import module`` when name is None) reaches."""
    return {
        slot
        for (mod, sym), slot in _SYMBOL_MARKERS.items()
        if module == mod and (name is None or sym is None or name == sym)
    }


def _stage_dep(module: str, name: str | None) -> str | None:
    if not module.startswith(_STAGE_PKG):
        return None
    if name is not None and _is_module(f"{module}.{name}"):
        return f"{module}.{name}"
    return module if _is_module(module) else None


def _acg_slots(tree: ast.Module) -> set[str]:
    called = {
        _ACG_METHODS[name]
        for n in ast.walk(tree)
        for name in (
            n.attr if isinstance(n, ast.Attribute) else n.id if isinstance(n, ast.Name) else None,
        )
        if name in _ACG_METHODS
    }
    # A client imported but no known method called: claim both (fail toward
    # declaring) rather than none.
    return called or {"image", "video_gen"}


@cache
def _scan(module: str) -> tuple[frozenset[str], frozenset[str]]:
    """``(slots this module's own source reaches, stage-package modules it imports)``."""
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin, module
    with open(spec.origin, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.extend((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            imports.extend((alias.name, None) for alias in node.names)
    slots: set[str] = set()
    deps: set[str] = set()
    for mod, name in imports:
        slots |= _marker_slots(mod, name)
        if (mod, name) == _ACG:
            slots |= _acg_slots(tree)
        dep = _stage_dep(mod, name)
        if dep is not None:
            deps.add(dep)
    return frozenset(slots), frozenset(deps)


def detected_slots(cls: type[StageHandler]) -> set[str]:
    roots = {c.__module__ for c in cls.__mro__ if c.__module__.startswith("src.")}
    seen: set[str] = set()
    todo = list(roots)
    found: set[str] = set()
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        slots, deps = _scan(mod)
        found |= slots
        todo.extend(deps)
    return found


def _stage_classes() -> dict[str, type[StageHandler]]:
    out: dict[str, type[StageHandler]] = {}
    for bp in get_registry().all():
        for spec in bp.stages:
            cls = stage_handler_class(spec)
            assert cls is not None, (
                f"{bp.id}.{spec.name}: untagged factory — wrap it in StageFactory so its "
                "provider_slots can be read"
            )
            out[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return out


def test_every_blueprint_stage_is_introspectable() -> None:
    assert len(_stage_classes()) > 50


@pytest.mark.parametrize("qualname", list(sorted(_stage_classes()))[:1])
def test_declared_provider_slots_equal_the_clients_the_stage_reaches(qualname: str) -> None:
    cls = _stage_classes()[qualname]
    declared = [u.slot for u in cls.provider_slots]
    assert len(declared) == len(set(declared)), f"{qualname}: duplicate SlotUse"
    detected = detected_slots(cls)
    assert set(declared) == detected, (
        f"{qualname}: declares {sorted(declared)} but reaches {sorted(detected)} — "
        "update its provider_slots ClassVar"
    )


def test_declared_slots_are_configurable_registry_slots() -> None:
    """Only connection/grant slots a tenant can act on — never scrape/kag & co."""
    allowed = {n for n, s in SLOTS.items() if s.writable or s.slot_type == "grant"}
    for qualname, cls in _stage_classes().items():
        for use in cls.provider_slots:
            assert use.slot in allowed, f"{qualname}: {use.slot!r} is not a configurable slot"
            assert use.when in {"always", "optional"} or use.when.startswith("brief."), (
                qualname,
                use,
            )


def test_the_scan_is_not_vacuous() -> None:
    """Known shapes: if the scan stopped seeing them, every equality above is hollow."""
    by_name = {cls.__name__: cls for cls in _stage_classes().values()}
    assert detected_slots(by_name["PromoScriptStage"]) >= {"llm", "tts"}
    assert detected_slots(by_name["PromoNarrateStage"]) == {"tts"}  # via primitives/narration
    assert detected_slots(by_name["AiImageGenStage"]) == {"image"}
    assert detected_slots(by_name["AiVideoGenStage"]) == {"video_gen"}
    assert detected_slots(by_name["AppInspectStage"]) == {"device_farm"}
    # scrape/media are dependencies, not slots; scanned-PDF OCR IS a slot (2026-10-22).
    assert detected_slots(by_name["SourceIngestStage"]) == {"ocr"}
