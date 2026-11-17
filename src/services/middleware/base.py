from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
NextFn = Callable[[Callable[[], Awaitable[T]]], Awaitable[T]]


class Middleware(ABC):
    @abstractmethod
    async def __call__(
        self,
        call: Callable[[], Awaitable[T]],
        next_: NextFn[T],
    ) -> T: ...
