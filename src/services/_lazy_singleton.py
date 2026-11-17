"""Process-wide async lazy singleton.

StageHandlers and routes receive a ``RunContext`` / ``Request``, not
``app.state``, so a shared resource (the MinIO-backed ``StorageService``)
can't be threaded through the registry without a wider refactor. This holds
one lazily-built instance behind a double-checked ``asyncio.Lock`` and lets
tests inject a fake. ``tts_storage`` and ``manuscript_storage`` both use it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


class AsyncLazySingleton[T]:
    """Lazily build a single ``T`` on first ``get()``, then reuse it."""

    def __init__(self, factory: Callable[[], Awaitable[T]]) -> None:
        self._factory = factory
        self._value: T | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> T:
        if self._value is not None:
            return self._value
        async with self._lock:
            if self._value is None:
                self._value = await self._factory()
        # Lock-protected assignment guarantees _value is set, but the type
        # checker can't carry that across the async-with boundary — narrow.
        value = self._value
        if value is None:
            raise RuntimeError("async lazy singleton factory returned None")
        return value

    def set_for_tests(self, value: T | None) -> None:
        """Override (or reset, with ``None``) the held instance for tests."""
        self._value = value
