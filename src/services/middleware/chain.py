from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from src.services.middleware.base import Middleware

T = TypeVar("T")


def build_chain(
    middlewares: Sequence[Middleware],
) -> Callable[[Callable[[], Awaitable[T]]], Awaitable[T]]:
    """Compose middlewares so the first in the list is outermost."""

    async def terminal(call: Callable[[], Awaitable[T]]) -> T:
        return await call()

    chain: Callable[[Callable[[], Awaitable[T]]], Awaitable[T]] = terminal
    for mw in reversed(middlewares):
        nxt = chain

        async def wrapped(
            call: Callable[[], Awaitable[T]],
            mw: Middleware = mw,
            nxt: Callable[[Callable[[], Awaitable[T]]], Awaitable[T]] = nxt,
        ) -> T:
            # why: T is bound through the closure default args above, so the
            # Middleware Protocol's Awaitable[T] result is the declared return type.
            return await mw(call, nxt)

        chain = wrapped
    return chain
