"""Drives, goals, and the goal manager."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..util import clamp, tokenize

_ids = itertools.count(1)


@dataclass
class Drive:
    """An internal need whose deficit grows over time and motivates goals."""

    name: str
    level: float = 0.3              # current deficit 0..1 (higher = more urgent)
    growth: float = 0.02            # deficit added per cycle

    def tick(self) -> None:
        self.level = clamp(self.level + self.growth)

    def satisfy(self, amount: float) -> None:
        self.level = clamp(self.level - amount)


@dataclass
class Goal:
    description: str
    priority: float = 0.5
    progress: float = 0.0
    done: bool = False
    source: str = "seed"
    id: int = field(default_factory=lambda: next(_ids))
    drive: str = "curiosity"

    def advance(self, amount: float) -> None:
        self.progress = clamp(self.progress + amount)
        if self.progress >= 0.999:
            self.done = True


class GoalManager:
    """Selects and tracks goals, driven by internal drive deficits."""

    def __init__(self, drives: Optional[Dict[str, Drive]] = None) -> None:
        self.drives = drives or {
            "curiosity": Drive("curiosity", 0.35, 0.025),
            "competence": Drive("competence", 0.25, 0.015),
            "connection": Drive("connection", 0.2, 0.01),
        }
        self.goals: List[Goal] = []
        self.active: Optional[Goal] = None

    def tick(self) -> None:
        for d in self.drives.values():
            d.tick()

    def consider(self, text: str) -> Optional[Goal]:
        """Seed goals from explicit intents in the stimulus."""
        t = (text or "").lower()
        for marker in ("i want to learn", "teach me", "help me learn", "learn to"):
            if marker in t:
                idx = t.index(marker)
                tail = t[idx + len(marker):].strip(" .!,?")
                topic = tail.split()[0] if tail.split() else tail.strip()
                if topic:
                    return self.add_goal(f"learn about {topic}", priority=0.85,
                                         drive="curiosity", source="user")
        if "?" in t:
            return self.add_goal("answer the user's question", priority=0.6,
                                 drive="competence", source="question", transient=True)
        return None

    def add_goal(self, description: str, priority: float = 0.5,
                 drive: str = "curiosity", source: str = "seed",
                 transient: bool = False) -> Goal:
        for g in self.goals:
            if g.description == description and not g.done:
                g.priority = clamp(max(g.priority, priority))
                return g
        goal = Goal(description=description, priority=clamp(priority), drive=drive,
                    source=source)
        if transient:
            goal.advance(0.0)
        self.goals.append(goal)
        return goal

    def select_active(self) -> Optional[Goal]:
        open_goals = [g for g in self.goals if not g.done]
        if not open_goals:
            # synthesize a curiosity goal when the drive deficit is high
            if self.drives["curiosity"].level > 0.75:
                g = Goal(description="explore something new", priority=0.5,
                         drive="curiosity", source="drive")
                self.goals.append(g)
                return g
            self.active = None
            return None
        drive_level = self.drives.get(open_goals[0].drive, Drive("x")).level
        best = max(open_goals,
                   key=lambda g: g.priority * (0.6 + 0.4 * self.drives.get(
                       g.drive, Drive(g.drive)).level) + 0.05 * drive_level)
        self.active = best
        return best

    def report_learning(self, new_facts: int) -> None:
        """Called when the mind learns something: satisfies curiosity."""
        if new_facts > 0:
            self.drives["curiosity"].satisfy(0.08 * new_facts)
            self.drives["competence"].satisfy(0.03 * new_facts)
        for g in self.goals:
            if not g.done and g.drive == "curiosity":
                g.advance(0.15 * new_facts)

    def report_tool_success(self, ok: bool) -> None:
        self.drives["competence"].satisfy(0.15 if ok else -0.05)
        for g in self.goals:
            if not g.done and g.drive == "competence":
                g.advance(0.2 if ok else 0.0)

    def progress_text(self) -> str:
        g = self.active
        if g is None:
            return ""
        pct = int(g.progress * 100)
        return f"goal '{g.description}' ({pct}%)"

    def to_list(self) -> List[dict]:
        return [{"description": g.description, "priority": g.priority,
                 "progress": g.progress, "done": g.done, "drive": g.drive,
                 "source": g.source} for g in self.goals]

    def load(self, items: List[dict]) -> None:
        self.goals = [Goal(description=i["description"], priority=i.get("priority", .5),
                           progress=i.get("progress", 0.0), done=i.get("done", False),
                           drive=i.get("drive", "curiosity"), source=i.get("source", "seed"))
                      for i in items]
        self.drives_state = None
