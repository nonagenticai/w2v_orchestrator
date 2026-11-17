"""KAG is retired (2026-10): no orchestrator code may address it again.

Comments may still NAME it historically; what must not come back is a call path —
a ``"kag"`` discovery/lane name, a ``kag.test`` host, or an import of the
deleted ``kag_client`` / ``schemas.kag`` modules / w2v_common ``KagScrapeClient``.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
_FORBIDDEN = re.compile(
    r"""["']kag["']|kag\.test|kag\.kag\.svc|kag_client|schemas\.kag|KagScrapeClient|KAGClient"""
)


def test_no_code_path_to_kag() -> None:
    hits = [
        f"{path.relative_to(_SRC.parent)}:{n}: {line.strip()}"
        for path in sorted(_SRC.rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if _FORBIDDEN.search(line)
    ]
    assert hits == [], "KAG is retired; remove these call paths:\n" + "\n".join(hits)


def test_guard_is_not_vacuous() -> None:
    assert _FORBIDDEN.search('service_url("kag")')
    assert _FORBIDDEN.search("from src.clients.kag_client import KAGClient")
    assert not _FORBIDDEN.search("# the retired KAG gateway used to proxy this")
