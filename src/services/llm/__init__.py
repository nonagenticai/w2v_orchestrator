"""LLM access for the orchestrator: ``WMClient`` with per-run ``llm`` slot dispatch (P3).

Import ``WMClient`` from HERE, never from ``w2v_common.clients`` (ruff TID251).
"""

from src.services.llm.dispatch import WMClient, WMResponse

__all__ = ["WMClient", "WMResponse"]
