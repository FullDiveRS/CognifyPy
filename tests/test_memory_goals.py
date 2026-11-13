"""Tests for associative memory and the goal system."""
import pytest

from cognify.goals.goals import Drive, Goal, GoalManager
from cognify.memory.memory import AssociativeMemory, MemoryRecord


class TestMemory:
    def test_add_and_recall_exact(self):
        mem = AssociativeMemory()
        mem.add("the cat sat on the mat")
        hits = mem.recall("cat sat mat")
        assert hits and hits[0][1] > 0.3

    def test_recall_miss(self):
        mem = AssociativeMemory()
        mem.add("the cat sat on the mat")
        assert mem.recall("quantum thermodynamics") == []

    def test_importance_ranks_higher(self):
        mem = AssociativeMemory()
        mem.add("python is a language", importance=0.2)
        mem.add("python powers cognify minds", importance=0.9)
        hits = mem.recall("python")
        assert hits[0][0].content == "python powers cognify minds"

    def test_associative_hop(self):
        mem = AssociativeMemory()
        mem.add("birds fly in the morning sky")
        mem.add("the morning sky is bright")     # linked via 3 shared tokens
        hits = mem.recall("birds fly", k=2)
        contents = {r.content for r, _ in hits}
        assert "the morning sky is bright" in contents

    def test_rehearsal_boosts_strength(self):
        mem = AssociativeMemory()
        rec = mem.add("a fact to remember")
        before = rec.strength
        mem.touch(rec)
        assert rec.strength > before

    def test_tick_decays(self):
        mem = AssociativeMemory()
        rec = mem.add("fading memory", importance=0.1)
        strong = rec.strength
        for _ in range(20):
            mem.tick(0.05)
        assert rec.strength < strong
        assert rec.strength >= 0.1               # floor

    def test_capacity_eviction(self):
        mem = AssociativeMemory(capacity=12)
        keep = mem.add("keep me, I am very very important", importance=1.0)
        for i in range(30):
            mem.add(f"filler memory number {i}", importance=0.1)
        assert mem.size() <= 13
        assert mem.find("keep me, I am very very important") is not None

    def test_similarity_fn_used(self):
        calls = []

        def fake_sim(a, b):
            calls.append(1)
            return 0.9

        mem = AssociativeMemory(similarity=fake_sim)
        mem.add("alpha")
        mem.add("beta")
        assert calls                                # custom similarity consulted

    def test_empty_content_rejected(self):
        with pytest.raises(ValueError):
            AssociativeMemory().add("   ")

    def test_kind_filter(self):
        mem = AssociativeMemory()
        mem.add("an episode", kind="episode")
        mem.add("a fact", kind="fact")
        assert mem.size() == 2
        assert len(mem.by_kind("fact")) == 1

    def test_record_kind_validated(self):
        rec = MemoryRecord("x", kind="bogus")
        assert rec.kind == "episode"


class TestGoals:
    def test_drives_grow(self):
        mgr = GoalManager()
        before = mgr.drives["curiosity"].level
        mgr.tick()
        assert mgr.drives["curiosity"].level > before

    def test_goal_from_intent(self):
        mgr = GoalManager()
        goal = mgr.consider("I want to learn photography")
        assert goal is not None
        assert "photography" in goal.description

    def test_question_goal(self):
        mgr = GoalManager()
        goal = mgr.consider("what is a bird?")
        assert goal is not None

    def test_no_goal_from_statement(self):
        mgr = GoalManager()
        assert mgr.consider("birds are animals") is None

    def test_select_active_priority(self):
        mgr = GoalManager()
        mgr.add_goal("low prio", priority=0.2)
        high = mgr.add_goal("high prio", priority=0.95)
        assert mgr.select_active() is high

    def test_progress_and_completion(self):
        g = Goal("learn stuff", priority=0.9)
        g.advance(0.6)
        assert not g.done
        g.advance(0.6)
        assert g.done

    def test_duplicate_goal_merged(self):
        mgr = GoalManager()
        a = mgr.add_goal("learn chess", priority=0.5)
        b = mgr.add_goal("learn chess", priority=0.8)
        assert a is b and a.priority == 0.8

    def test_report_learning_satisfies_curiosity(self):
        mgr = GoalManager()
        mgr.tick()
        before = mgr.drives["curiosity"].level
        mgr.report_learning(3)
        assert mgr.drives["curiosity"].level < before

    def test_tool_success_satisfies_competence(self):
        mgr = GoalManager()
        mgr.tick()
        before = mgr.drives["competence"].level
        mgr.report_tool_success(True)
        assert mgr.drives["competence"].level < before

    def test_drive_satisfy_clamped(self):
        d = Drive("curiosity", level=0.1)
        d.satisfy(0.5)
        assert d.level == 0.0

    def test_list_roundtrip(self):
        mgr = GoalManager()
        mgr.add_goal("learn chess", priority=0.7)
        mgr.add_goal("write poem", priority=0.4).advance(0.5)
        clone = GoalManager()
        clone.load(mgr.to_list())
        assert {g.description for g in clone.goals} == \
               {g.description for g in mgr.goals}
