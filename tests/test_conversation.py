"""Tests for the conversational dial (rate) and the ConversationTracker —
the mind's running understanding of the chat (topics, open questions,
familiarity). Uses the committed fluent API: ``rate_amount`` /
``understand_shifts`` on ``ConversationalOptions``.
"""
from cognify.core.builder import MindBuilder
from cognify.core.conversation import ConversationTracker
from cognify.core.options import ConversationalOptions


def make_mind():
    return (MindBuilder("Chatty")
            .with_reasoning()
            .with_memory()
            .with_learning()
            .with_emotions()
            .build())


# ------------------------------------------------------------------ dial

def test_rate_amount_clamps_to_valid_range():
    c = ConversationalOptions()
    c.rate_amount(99)
    assert c.rate == 3.0
    c.rate_amount(-5)
    assert c.rate == 0.5
    c.rate_amount(1.7)
    assert c.rate == 1.7


def test_understand_shifts_fluent_switch():
    c = ConversationalOptions()
    c.understand_shifts(False)
    assert c.understand is False
    c.understand_shifts(True)
    assert c.understand is True


def test_conversational_options_persist_round_trip():
    mind = MindBuilder("Persist").with_conversational(lambda c: c.rate_amount(2.5)).build()
    data = mind.to_dict()
    restored = ConversationalOptions.from_dict(data["conversational"])
    assert restored.rate == 2.5


# -------------------------------------------------------------- tracker

def test_tracker_collects_topics_turns_and_familiarity():
    mind = make_mind()
    tracker = ConversationTracker()
    tracker.observe(mind, "I love gardening very much.", "that sounds nice!", 1)
    tracker.observe(mind, "Do tulips need much sun?", "Tulips are flowers that like light.", 2)
    tracker.observe(mind, "Gardening makes me happy.", "it is a calm hobby", 3)

    assert tracker.turns == 3
    assert any("garden" in t for t in tracker.topics)
    assert 0 < tracker.familiarity < 1
    summary = " ".join(tracker.summary())
    assert "garden" in summary.lower()


def test_tracker_keeps_unanswered_questions_as_open_threads():
    mind = make_mind()
    tracker = ConversationTracker()

    tracker.observe(mind, "Do tulips need much sun?", "nice weather today", 1)
    assert any("tulips" in q for q in tracker.pending_questions)

    # A reply that shares content words with the question closes the thread.
    tracker.observe(mind, "Do roses need water?", "Roses do need water, yes.", 2)
    assert not any("roses" in q for q in tracker.pending_questions)


def test_tracker_open_threads_fade_after_eight_cycles():
    mind = make_mind()
    tracker = ConversationTracker()
    tracker.observe(mind, "Who wrote that book about mountains?", "I am not sure!", 1)
    assert tracker.pending_questions
    tracker.observe(mind, "just chatting a bit here", "cool", 10)
    assert not tracker.pending_questions      # cycle 10 - cycle 1 > 8 -> faded


def test_tracker_writes_topics_into_long_term_memory():
    mind = make_mind()
    tracker = ConversationTracker()
    for i in range(1, 8):
        tracker.observe(mind, "More about astronomy please.", "space is fascinating", i)
    # Every 6th turn the top topics land in long-term memory, tagged as
    # conversation understanding (mirrors the C# source="conversation").
    assert any("talking about" in m.content and "conversation" in m.tags
               for m in mind.memory.all())


def test_tracker_ignores_empty_input():
    mind = make_mind()
    tracker = ConversationTracker()
    tracker.observe(mind, "   ", "hi", 1)
    assert tracker.turns == 0
