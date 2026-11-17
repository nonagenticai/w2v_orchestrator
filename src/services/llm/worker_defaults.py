"""Per-worker defaults (seed system prompt + sampling) for in-process LLM calls.

worker_management applies a worker's SEEDED system prompt and temperature to every
``/workers/{name}/message`` call; callers may omit both. When a run's ``llm`` slot is
a tenant connection the orchestrator calls the provider itself, so it needs the same
defaults. They are vendored from WM's ``seed/w2v_workers.yaml`` into
``worker_defaults.yaml`` next to this module (see that file's header for provenance
and the unseeded names).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml

log = logging.getLogger(__name__)

_DEFAULTS_PATH: Final = Path(__file__).with_name("worker_defaults.yaml")

#: WM's auto-create default (``message_endpoints.py``: temperature=0.1, no prompt).
AUTO_CREATE_TEMPERATURE: Final = 0.1


@dataclass(frozen=True, slots=True)
class WorkerDefaults:
    worker_name: str
    temperature: float = AUTO_CREATE_TEMPERATURE
    system_prompt: str | None = None
    #: False for names WM never seeded (it auto-created them with no prompt).
    seeded: bool = False
    #: OpenAI ``frequency_penalty`` for the PLATFORM backends (vLLM / inference-router,
    #: both forward it verbatim). ``None`` ⇒ the key is omitted from the body. Not a
    #: WM seed field: set only where a live replay measured a repetition loop.
    frequency_penalty: float | None = None


def _parse(raw: object) -> dict[str, WorkerDefaults]:
    if not isinstance(raw, dict) or not isinstance(raw.get("workers"), list):
        raise ValueError(f"{_DEFAULTS_PATH.name}: expected a top-level 'workers' list")
    out: dict[str, WorkerDefaults] = {}
    for entry in raw["workers"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("worker_name"), str):
            raise ValueError(f"{_DEFAULTS_PATH.name}: every worker needs a worker_name")
        name = entry["worker_name"]
        if name in out:
            raise ValueError(f"{_DEFAULTS_PATH.name}: duplicate worker {name!r}")
        temperature = entry.get("temperature", AUTO_CREATE_TEMPERATURE)
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            raise ValueError(f"{_DEFAULTS_PATH.name}: {name}: temperature must be a number")
        prompt = entry.get("system_prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"{_DEFAULTS_PATH.name}: {name}: system_prompt must be a string")
        penalty = entry.get("frequency_penalty")
        if penalty is not None and (
            isinstance(penalty, bool) or not isinstance(penalty, int | float) or abs(penalty) > 2
        ):
            raise ValueError(
                f"{_DEFAULTS_PATH.name}: {name}: frequency_penalty must be a number in [-2, 2]"
            )
        out[name] = WorkerDefaults(
            worker_name=name,
            temperature=float(temperature),
            system_prompt=prompt or None,
            seeded=bool(entry.get("seeded", False)),
            frequency_penalty=None if penalty is None else float(penalty),
        )
    return out


@lru_cache(maxsize=1)
def _load() -> dict[str, WorkerDefaults]:
    return _parse(yaml.safe_load(_DEFAULTS_PATH.read_text(encoding="utf-8")))


def known_workers() -> frozenset[str]:
    return frozenset(_load())


def get_worker_defaults(worker_name: str) -> WorkerDefaults:
    """Defaults for ``worker_name``; WM's auto-create defaults for an unknown name."""
    found = _load().get(worker_name)
    if found is not None:
        return found
    log.info(
        "in-process LLM: worker %r has no vendored defaults; using WM auto-create "
        "defaults (temperature=%s, no seed prompt)",
        worker_name,
        AUTO_CREATE_TEMPERATURE,
    )
    return WorkerDefaults(worker_name=worker_name)
