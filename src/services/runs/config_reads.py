"""Record which run-snapshot keys a stage actually reads while it runs.

A Restart after the user changed a project/run input (voice, verbatim items, the
deliverable, …) must re-run exactly the stages that READ the changed input and
their dependents — and reuse the rest (``restart_plan``). A static map of "which
stage reads which key" cannot be kept honest: 100+ stage modules reach the
snapshot through shared helpers (``quality_tier(ctx.config_snapshot)``,
``film_frame_dims(...)``), so a hand-written table would silently go stale and a
stale table reuses a stage that should have re-run.

So the executor hands every stage its own :class:`RecordingSnapshot` — a ``dict``
subclass with the snapshot's content — and persists the keys the stage touched on
its ``stage_outputs`` row (``config_reads``, migration 0102).

FAIL-SAFE BY CONSTRUCTION. Over-recording only re-runs more; under-recording reuses
a stale stage, so every access path that could reveal content without naming a key
records :data:`WHOLE_SNAPSHOT` instead:

* key reads — ``[]``, ``get``, ``in``, ``setdefault``, ``pop`` — record the key;
* bulk reads — iteration, ``keys``/``items``/``values``, ``copy``, ``==``, ``|``,
  ``repr`` — record ``"*"``. ``dict(snap)`` / ``{**snap}`` / ``json.dumps`` /
  ``copy.deepcopy`` / pickling all go through ``keys()``/``items()`` for a dict
  subclass whose ``__iter__`` is overridden, so they land on ``"*"`` too
  (pinned by ``tests/services/runs/test_config_reads.py``).

Granularity is the TOP-LEVEL key: a stage that reads ``brief.source_url`` records
``brief``, so any brief change re-runs it. Coarser than ideal, never wrong.
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView

#: Recorded when a stage read the snapshot wholesale: any input change dirties it.
WHOLE_SNAPSHOT = "*"

_MISSING = object()


class RecordingSnapshot(dict[str, object]):
    """A copy of the run snapshot that remembers which top-level keys were read."""

    __slots__ = ("_reads",)

    def __init__(self, data: Mapping[str, object] | None = None) -> None:
        super().__init__(data or {})
        self._reads: set[str] = set()

    # --- key reads ------------------------------------------------------------
    def __getitem__(self, key: str) -> object:
        self._reads.add(key)
        return super().__getitem__(key)

    def get(self, key: str, default: object = None) -> object:  # ty: ignore[invalid-method-override]  # why: dict.get's overloads narrow the default; runtime-identical
        self._reads.add(key)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            self._reads.add(key)
        return super().__contains__(key)

    def setdefault(self, key: str, default: object = None) -> object:
        self._reads.add(key)
        return super().setdefault(key, default)

    def pop(self, key: str, default: object = _MISSING) -> object:  # ty: ignore[invalid-method-override]  # why: dict.pop's overloads narrow the default; runtime-identical
        self._reads.add(key)
        if default is _MISSING:
            return super().pop(key)
        return super().pop(key, default)

    # --- bulk reads: content seen without a key name --------------------------
    def _whole(self) -> None:
        self._reads.add(WHOLE_SNAPSHOT)

    def __iter__(self) -> Iterator[str]:
        self._whole()
        return super().__iter__()

    def keys(self) -> KeysView[str]:  # ty: ignore[invalid-method-override]  # why: returns the real dict_keys view; the stub names the private class
        self._whole()
        return super().keys()

    def items(self) -> ItemsView[str, object]:  # ty: ignore[invalid-method-override]  # why: same as keys()
        self._whole()
        return super().items()

    def values(self) -> ValuesView[object]:  # ty: ignore[invalid-method-override]  # why: same as keys()
        self._whole()
        return super().values()

    def copy(self) -> dict[str, object]:
        self._whole()
        return dict(super().items())

    def __eq__(self, other: object) -> bool:
        self._whole()
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        self._whole()
        return super().__ne__(other)

    def __repr__(self) -> str:
        self._whole()
        return super().__repr__()

    def __reduce_ex__(
        self, protocol: object
    ) -> tuple[type[dict[str, object]], tuple[dict[str, object]]]:
        # copy / deepcopy / pickle: hand back a PLAIN dict so a copy never keeps
        # recording into (or aliasing) this stage's read set.
        self._whole()
        return (dict, (dict(super().items()),))

    # --- what the executor persists --------------------------------------------
    def reads(self) -> list[str]:
        """The keys read so far, sorted (``["*"]`` alone when read wholesale)."""
        if WHOLE_SNAPSHOT in self._reads:
            return [WHOLE_SNAPSHOT]
        return sorted(self._reads)
