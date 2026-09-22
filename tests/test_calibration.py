# tests/test_calibration.py
"""
Real pytest unit tests for the calibration hard negative logic added to
engine.py: extract_content_words(), generate_hard_negatives(), and the
parts of calibrate_threshold() that depend on them.

No real model, no real file scoring. sentence_transformers is stubbed
before import, the same technique test_engine.py and test_classifier.py
use, and the classifier is a hand written fake whose score_batch returns
a fixed list, so thresholds are exactly predictable. Runs in well under
a second.

The emphasis is on the degenerate and failure cases, because the whole
safety argument for this change is that generation is additive: when it
produces nothing, for any reason, calibration must fall back to the
previous generic bank behaviour, and calibrate_threshold() must never
raise during session startup.
"""
import json
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

_fake_sentence_transformers = types.ModuleType("sentence_transformers")
_fake_sentence_transformers.CrossEncoder = MagicMock(name="CrossEncoder_class")
sys.modules.setdefault("sentence_transformers", _fake_sentence_transformers)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import engine
from engine import (
    CALIBRATION_CEILING,
    CALIBRATION_MARGIN,
    DEFAULT_THRESHOLD,
    MAX_CONTENT_WORDS,
    calibrate_threshold,
    extract_content_words,
    generate_hard_negatives,
)


class FakeClassifier:
    """Returns a caller supplied score for every text, and records the
    sample list it was handed so tests can assert on what calibration
    actually scored."""

    def __init__(self, score=0.02):
        self.score = score
        self.last_samples = None

    def score_batch(self, goal, texts):
        self.last_samples = list(texts)
        return [self.score] * len(texts)


@pytest.fixture
def samples_file(tmp_path):
    path = tmp_path / "workspace_samples.json"
    path.write_text(json.dumps({"focused": [], "distraction": ["Netflix - Browse", "Instagram - Reels"]}))
    return str(path)


@pytest.fixture
def templates_file(tmp_path):
    path = tmp_path / "hard_negative_templates.json"
    path.write_text(json.dumps({"templates": ["Safari - {word} - Google Search", "Instagram - {word} memes"]}))
    return str(path)


# --- extract_content_words -------------------------------------------------

def test_extracts_content_words_from_a_spoken_goal():
    assert extract_content_words("I have to write a report.") == ["write", "report"]


def test_strips_punctuation_and_lowercases():
    assert extract_content_words("Writing my DISSERTATION, literature review!") == [
        "writing", "dissertation", "literature", "review"
    ]


def test_deduplicates_while_preserving_spoken_order():
    assert extract_content_words("report the report about the report") == ["report"]


def test_drops_tokens_shorter_than_three_characters():
    # "b2" is two characters and carries no topical weight on its own.
    assert "b2" not in extract_content_words("Practising Spanish for my B2 exam")
    assert extract_content_words("Practising Spanish for my B2 exam") == ["practising", "spanish", "exam"]


def test_caps_the_number_of_content_words():
    goal = "alpha bravo charlie delta echo foxtrot golf hotel"
    assert len(extract_content_words(goal)) == MAX_CONTENT_WORDS


def test_respects_an_explicit_max_words_argument():
    assert extract_content_words("alpha bravo charlie", max_words=2) == ["alpha", "bravo"]


def test_handles_an_apostrophe_inside_a_word():
    assert extract_content_words("Organising my sister's wedding") == ["organising", "sister", "wedding"]


@pytest.mark.parametrize("goal", ["", "   ", "\n\t "])
def test_empty_goal_yields_no_content_words(goal):
    assert extract_content_words(goal) == []


def test_goal_made_entirely_of_stopwords_yields_no_content_words():
    assert extract_content_words("I need to do this now") == []


def test_one_word_goal_yields_that_word():
    assert extract_content_words("dissertation") == ["dissertation"]


@pytest.mark.parametrize("goal", [None, 42, ["a", "list"], {"a": "dict"}])
def test_non_string_goal_yields_no_content_words_instead_of_raising(goal):
    assert extract_content_words(goal) == []


# --- generate_hard_negatives ----------------------------------------------

def test_generates_one_sample_per_template_per_content_word(templates_file):
    generated = generate_hard_negatives("I have to write a report.", templates_path=templates_file)
    assert len(generated) == 2 * 2  # 2 templates x 2 content words
    assert "Safari - write - Google Search" in generated
    assert "Safari - report - Google Search" in generated
    assert "Instagram - report memes" in generated


def test_substitutes_every_placeholder_leaving_none_behind(templates_file):
    for sample in generate_hard_negatives("Editing the video", templates_path=templates_file):
        assert "{word}" not in sample


def test_generates_from_the_real_shipped_template_file():
    # Guards the actual data file: if an edit breaks its structure or drops
    # the {word} placeholder, this fails rather than silently generating
    # nothing in production.
    generated = generate_hard_negatives("I have to write a report.")
    assert generated
    assert all("{word}" not in s for s in generated)
    assert any("report" in s for s in generated)


def test_the_shipped_templates_include_a_video_tutorial_pattern():
    # Specifically justified by the evaluation: for the video editing goal a
    # "how I do this" tutorial outscored the real project file, and that
    # pattern generalises past video editing.
    with open(engine.HARD_NEGATIVE_TEMPLATES_PATH) as f:
        templates = json.load(f)["templates"]
    assert any("YouTube" in t for t in templates)


@pytest.mark.parametrize("goal", ["", "I need to do this now", None])
def test_no_content_words_generates_nothing(goal, templates_file):
    assert generate_hard_negatives(goal, templates_path=templates_file) == []


def test_missing_template_file_generates_nothing_instead_of_raising(tmp_path):
    missing = str(tmp_path / "does_not_exist.json")
    assert generate_hard_negatives("write a report", templates_path=missing) == []


def test_malformed_template_file_generates_nothing_instead_of_raising(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{ this is not valid json")
    assert generate_hard_negatives("write a report", templates_path=str(path)) == []


def test_template_file_with_no_templates_key_generates_nothing(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"_about": "no templates here"}))
    assert generate_hard_negatives("write a report", templates_path=str(path)) == []


def test_templates_without_a_placeholder_are_skipped(tmp_path):
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps({"templates": ["Safari - {word} - Google Search", "Netflix - Browse TV Shows", 42]}))
    generated = generate_hard_negatives("report", templates_path=str(path))
    assert generated == ["Safari - report - Google Search"]


# --- the live path has generation switched off -----------------------------

def test_generation_is_switched_off_by_default_for_the_live_app():
    # The rejected calibration experiment must not run in the live app. This
    # is the switch, see GENERATED_HARD_NEGATIVES_DEFAULT in engine.py.
    assert engine.GENERATED_HARD_NEGATIVES_DEFAULT is None


def test_a_falsy_templates_path_generates_nothing_without_touching_the_disk():
    # Checked explicitly rather than by letting open(None) raise into the
    # catch-all, so narrowing that except clause later cannot silently turn
    # the experiment back on or collapse calibration to the fixed default.
    for path in (None, "", False):
        assert generate_hard_negatives("I have to write a report.", templates_path=path) == []


def test_live_calibration_scores_the_generic_bank_alone(samples_file):
    classifier = FakeClassifier()
    calibrate_threshold(classifier, "I have to write a report.", samples_path=samples_file)
    assert classifier.last_samples == ["Netflix - Browse", "Instagram - Reels"]


def test_calibration_still_includes_traps_when_a_path_is_passed_explicitly(samples_file):
    classifier = FakeClassifier()
    calibrate_threshold(classifier, "I have to write a report.",
                        samples_path=samples_file,
                        templates_path=engine.HARD_NEGATIVE_TEMPLATES_PATH)
    assert len(classifier.last_samples) > 2
    assert any("report" in s for s in classifier.last_samples[2:])


# --- calibrate_threshold ---------------------------------------------------

def test_calibration_scores_generated_hard_negatives_alongside_the_bank(samples_file, templates_file):
    classifier = FakeClassifier()
    calibrate_threshold(classifier, "I have to write a report.",
                        samples_path=samples_file, templates_path=templates_file)

    # 2 bank distractions + (2 templates x 2 content words)
    assert len(classifier.last_samples) == 6
    assert "Netflix - Browse" in classifier.last_samples
    assert "Safari - report - Google Search" in classifier.last_samples


def test_bank_samples_are_still_scored_and_come_first(samples_file, templates_file):
    classifier = FakeClassifier()
    calibrate_threshold(classifier, "write a report",
                        samples_path=samples_file, templates_path=templates_file)
    assert classifier.last_samples[:2] == ["Netflix - Browse", "Instagram - Reels"]


def test_threshold_is_the_percentile_plus_the_margin(samples_file, templates_file):
    classifier = FakeClassifier(score=0.03)
    threshold = calibrate_threshold(classifier, "write a report",
                                    samples_path=samples_file, templates_path=templates_file)
    assert threshold == pytest.approx(0.03 + CALIBRATION_MARGIN)


def test_a_missing_template_file_falls_back_to_bank_only_calibration(samples_file, tmp_path):
    # The pre fix behaviour. Generation contributing nothing must not stop
    # calibration, it must just calibrate on the generic bank as before.
    classifier = FakeClassifier(score=0.03)
    threshold = calibrate_threshold(classifier, "write a report",
                                    samples_path=samples_file,
                                    templates_path=str(tmp_path / "missing.json"))
    assert classifier.last_samples == ["Netflix - Browse", "Instagram - Reels"]
    assert threshold == pytest.approx(0.03 + CALIBRATION_MARGIN)


def test_a_goal_with_no_content_words_falls_back_to_bank_only_calibration(samples_file, templates_file):
    classifier = FakeClassifier(score=0.03)
    threshold = calibrate_threshold(classifier, "I need to do this now",
                                    samples_path=samples_file, templates_path=templates_file)
    assert classifier.last_samples == ["Netflix - Browse", "Instagram - Reels"]
    assert threshold == pytest.approx(0.03 + CALIBRATION_MARGIN)


def test_threshold_is_capped_by_the_sanity_ceiling(samples_file, templates_file):
    # A pathological goal whose words score high everywhere must not push the
    # line so high that genuine work reads as distraction.
    classifier = FakeClassifier(score=0.9)
    threshold = calibrate_threshold(classifier, "write a report",
                                    samples_path=samples_file, templates_path=templates_file)
    assert threshold == CALIBRATION_CEILING


def test_missing_samples_file_returns_the_default_threshold(tmp_path, templates_file):
    threshold = calibrate_threshold(FakeClassifier(), "write a report",
                                    samples_path=str(tmp_path / "missing.json"),
                                    templates_path=templates_file)
    assert threshold == DEFAULT_THRESHOLD


def test_empty_distraction_bank_returns_the_default_threshold(tmp_path, templates_file):
    path = tmp_path / "empty_bank.json"
    path.write_text(json.dumps({"distraction": []}))
    threshold = calibrate_threshold(FakeClassifier(), "write a report",
                                    samples_path=str(path), templates_path=templates_file)
    assert threshold == DEFAULT_THRESHOLD


def test_a_classifier_that_raises_returns_the_default_threshold(samples_file, templates_file):
    class ExplodingClassifier:
        def score_batch(self, goal, texts):
            raise RuntimeError("model died")

    threshold = calibrate_threshold(ExplodingClassifier(), "write a report",
                                    samples_path=samples_file, templates_path=templates_file)
    assert threshold == DEFAULT_THRESHOLD


@pytest.mark.parametrize("goal", ["", "   ", None, 42, "I need to do this now", "dissertation"])
def test_calibration_never_raises_on_a_degenerate_goal(goal, samples_file, templates_file):
    threshold = calibrate_threshold(FakeClassifier(), goal,
                                    samples_path=samples_file, templates_path=templates_file)
    assert isinstance(threshold, float)
    assert 0.0 < threshold <= CALIBRATION_CEILING
