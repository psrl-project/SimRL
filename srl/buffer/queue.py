"""Buffer and queue data structures between pipeline stages."""

from __future__ import annotations

from collections import deque

from ..component.rollout.engine import Request


class CompletionBuffer:
    """
    Holds completed rollout requests waiting for consumption by the trainer.
    """

    def __init__(self, max_size: int | None = None, drop_policy: str = "drop_oldest"):
        self.max_size = max_size
        self.drop_policy = drop_policy
        self._buffer: deque[Request] = deque()

    def push(self, req: Request) -> Request | None:
        if self.max_size is not None and len(self._buffer) >= self.max_size:
            if self.drop_policy == "drop_oldest":
                dropped = self._buffer.popleft()
                self._buffer.append(req)
                return dropped
            elif self.drop_policy == "drop_newest":
                return req
            else:
                return None
        self._buffer.append(req)
        return None

    def pop_batch(self, n: int) -> list[Request] | None:
        if len(self._buffer) < n:
            return None
        return [self._buffer.popleft() for _ in range(n)]

    def peek(self) -> list[Request]:
        return list(self._buffer)

    @property
    def size(self) -> int:
        return len(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)


class RecomputeQueue:
    """Queue of batches waiting for log-prob recomputation."""

    def __init__(self) -> None:
        self._queue: deque[list[Request]] = deque()

    def push_batch(self, batch: list[Request]) -> None:
        self._queue.append(batch)

    def pop_batch(self) -> list[Request] | None:
        return self._queue.popleft() if self._queue else None

    @property
    def size(self) -> int:
        return len(self._queue)
