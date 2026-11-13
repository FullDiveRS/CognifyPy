"""Event bus: the mind's observable inner life."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from ..util import now


@dataclass(frozen=True)
class MindEvent:
    type: str
    cycle: int
    ts: float
    payload: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[cycle {self.cycle}] {self.type}: {self.payload}"


Callback = Callable[[MindEvent], None]


class EventBus:
    """Tiny synchronous pub/sub with a bounded replay history."""

    def __init__(self, history_size: int = 500) -> None:
        self._subs: Dict[str, List[Callback]] = {}
        self._wildcards: List[Callback] = []
        self.history: Deque[MindEvent] = deque(maxlen=history_size)

    def subscribe(self, event_type: str, callback: Callback) -> Callable[[], None]:
        """Subscribe; use ``"*"`` for all events. Returns an unsubscribe fn."""
        if event_type == "*":
            self._wildcards.append(callback)
            def unwild() -> None:
                if callback in self._wildcards:
                    self._wildcards.remove(callback)
            return unwild
        self._subs.setdefault(event_type, []).append(callback)
        def unsub() -> None:
            lst = self._subs.get(event_type, [])
            if callback in lst:
                lst.remove(callback)
        return unsub

    def emit(self, event_type: str, cycle: int = 0, **payload: object) -> MindEvent:
        event = MindEvent(type=event_type, cycle=cycle, ts=now(), payload=payload)
        self.history.append(event)
        for cb in self._subs.get(event_type, []):
            cb(event)
        for cb in self._wildcards:
            cb(event)
        return event

    def clear(self) -> None:
        self._subs.clear()
        self._wildcards.clear()
        self.history.clear()
