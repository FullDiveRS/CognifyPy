"""Tests for the train option: facts become knowledge, counting sequences
become the number line, and plain conversational lines become sayings — the
mind's spoken vocabulary. A bigger phrase pool directly widens reply variety,
which is the structural fix for "she keeps repeating the same sentence".

Mirrors the C# TrainingTests.
"""
import pytest

from cognify import Mind, MindBuilder, TrainingReport
from cognify.core.phrases import PhrasePool
from cognify.reasoning.facts import FactParser


def make_mind(**kw):
    return (MindBuilder("Yume")
            .with_seed(7)
            .with_reasoning()
            .with_tools()
            .build())


def teach(mind, sentence):
    for fact in FactParser().parse(sentence):
        mind.kb.add(fact)


class TestTrainSplit:
    def test_train_splits_facts_sayings_and_number_lines(self):
        mind = make_mind()
        report = mind.train_all([
            "Birds are animals.",
            "1 + 1 = 2.",
            "The number system is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.",
            "I find the stars endlessly fascinating.",
        ])
        assert report.lines == 4
        assert report.facts_learned == 3  # birds + 1+1=2 + number-line sentence
        assert report.sayings_added == 1  # only the conversational line
        assert report.sequences_taught == 1
        assert len(mind.phrases) == 1
        assert mind.kb.size() == 3

    def test_train_skips_questions_and_conditionals_as_sayings(self):
        mind = make_mind()
        report = mind.train_all([
            "What is a bird?",
            "if i have no apples and i buy an apple, how many do i have?",
            "Tell me about the weather.",
            "I enjoy rainy days.",
        ])
        assert report.lines == 4
        assert report.facts_learned == 0
        assert report.sayings_added == 1
        assert mind.phrases.all()[0] == "I enjoy rainy days"

    def test_train_never_duplicates_a_saying(self):
        mind = make_mind()
        mind.train("I find the stars endlessly fascinating.")
        second = mind.train("i find the stars endlessly fascinating!")
        assert second.sayings_added == 0
        assert len(mind.phrases) == 1


class TestTrainFile:
    def test_train_file_reads_corpus_and_ignores_comments_and_blanks(self, tmp_path):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text(
            "# a comment line\n"
            "\n"
            "Cats are animals.\n"
            "What is love?\n"
            "I enjoy quiet mornings.\n",
            encoding="utf-8")
        mind = make_mind()
        report = mind.train_file(str(corpus))
        assert report.lines == 3
        assert report.facts_learned == 1
        assert report.sayings_added == 1

    def test_train_file_missing_file_raises(self):
        mind = make_mind()
        with pytest.raises(FileNotFoundError):
            mind.train_file("no/such/corpus.txt")


class TestSayingPersistence:
    def test_sayings_survive_save_and_restore(self, tmp_path):
        import json
        mind = make_mind()
        mind.train_all(["I enjoy quiet mornings.",
                        "Rainy afternoons put me in a quiet mood."])
        snapshot = mind.to_dict()
        assert snapshot["sayings"] == ["I enjoy quiet mornings",
                                       "Rainy afternoons put me in a quiet mood"]

        restored = Mind.from_dict(snapshot)
        assert len(restored.phrases) == 2
        assert any(s.startswith("I enjoy quiet") for s in restored.phrases.all())


class TestVocabularyInAction:
    # Cross-implementation sayings: none of these parse as facts in either
    # the C# or the Python fact parser (no copulas / fact verbs).
    SAYINGS = [
        "I find the stars endlessly fascinating.",
        "Old stories open doors into other lives.",
        "I enjoy quiet mornings.",
        "Rainy afternoons put me in a quiet mood.",
        "The ocean sounds endless and calm to me.",
        "Bread baking reminds me of a happy home.",
    ]

    def test_trained_sayings_are_spoken_in_the_minds_own_voice(self):
        mind = make_mind()
        report = mind.train_all(self.SAYINGS)
        assert report.sayings_added == len(self.SAYINGS)

        # Unanswerable small talk — nothing in the knowledge base, no math.
        # The Q-agent explores, so within a handful of turns a trained saying
        # must surface verbatim.
        prompts = [
            "tell me something you think about",
            "what do you wonder about",
            "share a thought with me",
            "i am curious what you like",
            "say whatever is on your mind",
            "what feels nice to think about",
            "go on, surprise me",
            "is there something you find lovely",
            "what occupies your dreams",
            "talk to me about anything",
        ]
        expected = {s.rstrip(".!").lower() for s in self.SAYINGS}
        spoken = [mind.process(p).text.rstrip(".!").lower() for p in prompts]
        assert any(text in expected for text in spoken)


class TestWordProblemReasoning:
    def test_word_problem_starting_from_nothing_is_reasoned_out(self):
        mind = make_mind()
        teach(mind, "1 + 1 = 2")  # taught, but NO "0 + 1" anywhere

        resp = mind.process("if i have no apples and i buy an apple, "
                            "how many apples do i have?")

        # "no apples" = 0 is stated by the problem itself: 0 + 1 is an
        # identity the mind can reason out — no calculator, no refusal.
        assert "1 apple" in resp.text
        assert "used the calculator" not in resp.text

    def test_quoted_word_problem_is_answered(self):
        mind = make_mind()
        teach(mind, "1 + 1 = 2")

        resp = mind.process('what is the answer to "if i have no apples and '
                            'i buy an apple. how many do i currently have?"')
        assert "1" in resp.text
        assert "apple" in resp.text

    def test_unanswerable_word_problem_falls_back_to_calculator_transparently(self):
        mind = make_mind()  # nothing taught at all
        resp = mind.process("if I have 4 apples and buy 3 more, "
                            "how many apples do I have?")
        # Taught reasoning could not work this out, so the calculator fills
        # in — but it must SAY so openly instead of passing mechanical math
        # off as knowledge the mind reasoned out itself.
        assert "not been taught" in resp.text
        assert "7" in resp.text


class TestWordMeanings:
    # "X means Y." lines are dictionary-style training data: they must land
    # as real (word, means, meaning) knowledge — recallable via "what does
    # X mean?" / "what is the meaning of X?" / "define X" — never as mere
    # sayings. Mirrors the C# TrainingTests word-meaning section.

    def test_means_lines_become_facts_not_sayings(self):
        mind = make_mind()
        report = mind.train_all([
            "Joy means a bright feeling of gladness deep inside.",
            "Happiness means feeling glad about being alive.",
        ])
        assert report.lines == 2
        assert report.facts_learned == 2
        assert report.sayings_added == 0
        assert mind.kb.get("joy", "means",
                           "bright feeling of gladness deep inside") is not None
        assert mind.kb.get("happiness", "means",
                           "feeling glad about being alive") is not None

    def test_what_does_x_mean_answers_with_the_definition(self):
        mind = make_mind()
        mind.train_all(["Joy means a bright feeling of gladness deep inside."])
        ans = mind.engine.answer(mind.kb, "what does joy mean?")
        assert ans is not None and "gladness" in ans.text

    def test_meaning_of_and_define_routes(self):
        mind = make_mind()
        mind.train_all(["Patience means the calm strength of being able to wait."])
        for question in ("what is the meaning of patience?", "define patience"):
            ans = mind.engine.answer(mind.kb, question)
            assert ans is not None and "wait" in ans.text

    def test_describe_leads_with_the_meaning(self):
        mind = make_mind()
        mind.train_all(["A fox is an animal.",
                        "Fox means a clever red rustler of the night."])
        ans = mind.engine.answer(mind.kb, "what is a fox?")
        assert ans is not None
        assert ans.text.startswith("Fox means")
        assert "animal" in ans.text

    def test_ness_words_keep_their_tail(self):
        from cognify.util import singularize
        assert singularize("kindness") == "kindness"
        assert singularize("happiness") == "happiness"
        mind = make_mind()
        mind.train_all(["Kindness means the gentle art of caring for people."])
        assert mind.kb.get("kindness", "means",
                           "gentle art of caring for people") is not None
        ans = mind.engine.answer(mind.kb, "what does kindness mean?")
        assert ans is not None and "caring" in ans.text

    def test_unknown_word_gets_a_graceful_teach_me(self):
        mind = make_mind()
        ans = mind.engine.answer(mind.kb, "what does unicorn mean?")
        assert ans is not None
        assert "unicorn" in ans.text and "teach" in ans.text.lower()

    def test_conversational_mean_questions_are_not_dictionary_lookups(self):
        mind = make_mind()
        # "what do you mean?" is conversation, not a lookup for the word "you"
        assert mind.engine.answer(mind.kb, "what do you mean?") is None

    def test_causes_question_still_works_next_to_mean_questions(self):
        mind = make_mind()
        mind.teach("rain causes floods")
        ans = mind.engine.answer(mind.kb, "what causes flood?")
        assert ans is not None and "rain" in ans.text

    def test_shipped_corpus_trains_end_to_end(self):
        import pathlib
        corpus = (pathlib.Path(__file__).resolve().parents[1]
                  / "corpus" / "words-and-meanings.txt")
        if not corpus.exists():
            pytest.skip("corpus file not present")
        mind = make_mind()
        report = mind.train_file(str(corpus))
        # Every shipped entry is knowledge: nothing leaks into sayings, and
        # both parsers (this one and the C# mirror) must agree line for line.
        assert report.lines == report.facts_learned
        assert report.sayings_added == 0
        assert mind.kb.size() == report.facts_learned
        ans = mind.engine.answer(mind.kb, "what does patience mean?")
        assert ans is not None and "wait" in ans.text
        ans = mind.engine.answer(mind.kb, "what does kindness mean?")
        assert ans is not None and "caring" in ans.text


class TestPhrasePool:
    def test_add_normalizes_and_rejects_junk(self):
        pool = PhrasePool()
        assert pool.add("  Hello   there!  ")
        assert not pool.add("hello there.")       # duplicate (case/punct)
        assert not pool.add("ok")                  # too short
        assert not pool.add("")                    # empty
        assert pool.all() == ["Hello there"]

    def test_pick_relevant_prefers_overlapping_sayings(self):
        import random
        pool = PhrasePool()
        pool.restore(["The ocean sounds like breathing to me.",
                      "Bread baking smells like a happy home.",
                      "I find the stars endlessly fascinating."])
        picks = pool.pick_relevant("tell me about the ocean and the stars",
                                   random.Random(1), 2)
        assert picks
        joined = " ".join(picks).lower()
        assert ("ocean" in joined) or ("stars" in joined)

    def test_report_str_is_human_readable(self):
        report = TrainingReport()
        report.lines, report.facts_learned = 10, 4
        assert "10 line(s)" in str(report)
        assert "4 fact(s)" in str(report)
