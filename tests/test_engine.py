# tests/test_engine.py
"""
Real pytest unit tests for CogAlignEngine's own new logic from Phase 3: the
scoring_text composition in run_tick() that combines the existing app and
title text with the new accessibility based visible text read, alongside
the existing goal string, not replacing either. Not the real classifier
model, real Ollama, or real AppleScript. sentence_transformers is stubbed
before import, the same technique test_classifier.py uses, so no real
model ever loads, and tracker/classifier/nudge_engine are mocked directly
on the constructed instance. Runs in well under a second.
"""
import os
import sys
import time
import types
from unittest.mock import MagicMock

import pytest

# Stub sentence_transformers in sys.modules BEFORE classifier.py (imported
# transitively by engine.py) is ever imported, so its heavy import chain
# (torch, transformers, ...) never runs. See test_classifier.py for the
# full rationale. setdefault so this is a no-op if the real package
# somehow already got imported first in this test session.
_fake_sentence_transformers = types.ModuleType("sentence_transformers")
_fake_sentence_transformers.CrossEncoder = MagicMock(name="CrossEncoder_class")
sys.modules.setdefault("sentence_transformers", _fake_sentence_transformers)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from engine import CogAlignEngine


@pytest.fixture
def engine():
    """
    A real CogAlignEngine, with the pretrained model stubbed out (see
    above) so construction is instant, and classifier.is_related plus
    nudge_engine.generate_nudge replaced with controllable fakes, so
    run_tick()'s own composition logic is exercised directly, in
    isolation from real scoring and real Ollama. Each test still sets its
    own tracker.get_active_window / get_cached_visible_text.
    """
    eng = CogAlignEngine(goal="Writing my dissertation literature review")
    eng.classifier.is_related = MagicMock(return_value=(False, 0.0))
    eng.nudge_engine.generate_nudge = MagicMock(return_value="a nudge")
    return eng


def test_scoring_text_combines_window_text_and_visible_text(engine):
    engine.tracker.get_active_window = MagicMock(return_value=("Safari", "Cognitive load - Wikipedia"))
    engine.tracker.get_cached_visible_text = MagicMock(
        return_value="Cognitive load refers to the used amount of working memory resources."
    )

    engine.run_tick()

    scoring_text = engine.classifier.is_related.call_args.args[0]
    assert scoring_text == (
        "Safari - Cognitive load - Wikipedia "
        "Cognitive load refers to the used amount of working memory resources."
    )


def test_scoring_text_falls_back_to_window_text_alone_when_visible_text_empty(engine):
    # Chrome almost always, or a timed out / capped read on any app.
    engine.tracker.get_active_window = MagicMock(return_value=("Google Chrome", "New Tab"))
    engine.tracker.get_cached_visible_text = MagicMock(return_value="")

    engine.run_tick()

    scoring_text = engine.classifier.is_related.call_args.args[0]
    assert scoring_text == "Google Chrome - New Tab"


def test_get_visible_text_called_with_same_app_name_as_the_title_read(engine):
    # Confirms the cached visible text lookup targets the exact app_name
    # the title read already resolved, not a fresh independent lookup that
    # could race against a focus change between the two calls.
    engine.tracker.get_active_window = MagicMock(return_value=("Safari", "Some Title"))
    engine.tracker.get_cached_visible_text = MagicMock(return_value="")

    engine.run_tick()

    engine.tracker.get_cached_visible_text.assert_called_once_with("Safari")


def test_unreliable_read_skips_visible_text_entirely():
    # The existing hold state contract: an unreliable title read must
    # never reach get_cached_visible_text() or is_related() at all.
    eng = CogAlignEngine(goal="test goal")
    eng.classifier.is_related = MagicMock(return_value=(False, 0.0))
    eng.tracker.get_active_window = MagicMock(return_value=("Unknown", "No Active Window"))
    eng.tracker.get_cached_visible_text = MagicMock(return_value="")

    eng.run_tick()

    eng.tracker.get_cached_visible_text.assert_not_called()
    eng.classifier.is_related.assert_not_called()


def test_nudge_distraction_context_stays_app_and_title_only(engine):
    # window_text (app + title alone) must still drive the nudge's
    # "distracted by" context, the enrichment is for scoring only.
    engine.tracker.get_active_window = MagicMock(return_value=("Safari", "Reddit"))
    engine.tracker.get_cached_visible_text = MagicMock(
        return_value="lots of extra page text that must not leak into the nudge"
    )
    engine.classifier.is_related = MagicMock(return_value=(False, 0.5))  # off task

    # Force a Level 1 nudge on this very tick instead of waiting real time.
    engine.state_machine.state = "DISTRACTED_BUFFER"
    engine.state_machine.distraction_start_time = time.time() - 10  # past buffer_threshold=5

    engine.run_tick()

    engine.nudge_engine.generate_nudge.assert_called_once()
    distraction_arg = engine.nudge_engine.generate_nudge.call_args.kwargs["distraction"]
    assert distraction_arg == "Safari - Reddit"
    assert "extra page text" not in distraction_arg
