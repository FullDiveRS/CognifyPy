"""Tests for the self-improvement module."""
import pytest

from cognify.core.builder import MindBuilder
from cognify.learning.self_improvement import SelfImprovement, classify_feedback


def test_classify_feedback_detects_approval_and_disapproval():
    assert classify_feedback("good job, that helped a lot") == 1
    assert classify_feedback("thank you!") == 1
    assert classify_feedback("no, that is wrong") == -1
    assert classify_feedback("stop saying that") == -1
    # Questions are not grades, even when they contain cue words.
    assert classify_feedback("are you sure that is right?") == 0
    assert classify_feedback("tell me about birds") == 0


def test_negative_feedback_credits_previous_action():
    mind = (MindBuilder("Learner")
            .with_preset("curious_child")
            .with_memory_capacity(20)
            .build())
    improvement = mind.improvement

    mind.ask("what is a bird?")
    replies_before = mind.learner.agent.stats()["updates"] \
        if "updates" in mind.learner.agent.stats() else None

    mind.ask("no, that is wrong, stop saying that")

    assert improvement.negatives == 1
    assert improvement._prev is not None


def test_positive_feedback_detected():
    mind = (MindBuilder("Learner")
            .with_preset("curious_child")
            .with_memory_capacity(20)
            .build())
    improvement = mind.improvement

    mind.ask("what is a bird?")
    mind.ask("good job, exactly right, thank you")

    assert improvement.positives == 1


def test_feedback_only_grades_the_immediately_previous_reply():
    mind = (MindBuilder("Learner")
            .with_preset("curious_child")
            .with_memory_capacity(20)
            .build())
    improvement = mind.improvement

    mind.ask("what is a bird?")
    first_template = improvement._prev["template"]
    mind.ask("tell me about cats")      # a newer reply now exists
    mind.ask("no, that is wrong")       # grades the CATS reply, not birds

    assert improvement.negatives == 1
    assert improvement.template_penalty(first_template) == 0.0


def test_self_improve_retires_disliked_sayings():
    mind = (MindBuilder("Learner")
            .with_preset("curious_child")
            .with_memory_capacity(20)
            .build())
    improvement = mind.improvement

    mind.phrases.add("I am a boring broken record line")
    assert "I am a boring broken record line" in mind.phrases.all()

    # The host grades this specific wording directly (deterministic); three
    # clear rejections push its reputation past the retirement threshold.
    for _ in range(3):
        improvement.record_feedback("I am a boring broken record line", -1.0)

    report = mind.self_improve()
    assert "I am a boring broken record line" not in mind.phrases.all()
    assert report.sayings_pruned == 1


def test_self_improve_keeps_approved_sayings():
    mind = (MindBuilder("Learner")
            .with_preset("curious_child")
            .with_memory_capacity(20)
            .build())
    improvement = mind.improvement
    mind.phrases.add("Users love this charming line")

    improvement.record_feedback("Users love this charming line", 1.0)
    mind.self_improve()

    assert "Users love this charming line" in mind.phrases.all()


def test_improvement_ledger_persists_through_snapshot():
    from cognify.persistence.store import SqliteMindStore
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "snap.db")
    store = SqliteMindStore(path)
    mind = (MindBuilder("Sleepy")
            .with_preset("curious_child")
            .with_memory_capacity(20)
            .build())
    mind.ask("what is a bird?")
    mind.ask("no, that is wrong")
    store.save(mind, "sleepy")
    clone = store.load("sleepy")
    assert clone.improvement.negatives == 1
    store.close()


def test_template_scores_move_towards_rewards():
    improvement = SelfImprovement()
    improvement.record_feedback("some reply wording", 1.0)
    improvement.record_feedback("some reply wording", 1.0)
    assert improvement.template_boost(
        improvement._template_scores and "some reply wording") > 0
