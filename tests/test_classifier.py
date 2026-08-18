# tests/test_classifier.py
"""
Real pytest unit tests for WorkspaceClassifier's own logic: normalization,
thresholding, batching, and the empty-input edge case. Not the pretrained
model's correctness, that's covered separately by
results/classifier_evaluation.json and tests/compare_crossencoder_models.py.

sentence_transformers.CrossEncoder is mocked entirely via the
mock_classifier fixture below, so no real model ever loads. Runs in well
under a second, not the 280+ seconds the full untargeted pytest run takes
elsewhere in this project because of the other test_*.py files' side-effect
model loads.
"""
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub sentence_transformers in sys.modules BEFORE classifier.py is ever
# imported, so its `from sentence_transformers import CrossEncoder` binds to
# this fake module instead of triggering the real package's import chain
# (which pulls in torch, transformers, ...). Patching CrossEncoder after a
# real import would still pay that cost on every run; this is what actually
# gets these tests down to well under a second. setdefault so this is a
# no-op if the real package somehow already got imported first.
_fake_sentence_transformers = types.ModuleType("sentence_transformers")
_fake_sentence_transformers.CrossEncoder = MagicMock(name="CrossEncoder_class")
sys.modules.setdefault("sentence_transformers", _fake_sentence_transformers)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import classifier as classifier_module
from classifier import WorkspaceClassifier


@pytest.fixture
def mock_classifier():
    """
    A real WorkspaceClassifier instance with sentence_transformers.CrossEncoder
    patched out at the point classifier.py imports it, so __init__ never loads
    a real model. Returns (classifier_instance, mock_model) so each test can
    set mock_model.predict.return_value directly.
    """
    with patch.object(classifier_module, "CrossEncoder") as mock_cross_encoder_cls:
        mock_model = MagicMock()
        mock_cross_encoder_cls.return_value = mock_model
        clf = WorkspaceClassifier(model_name="fake-model-never-loaded")
        yield clf, mock_model


# --- _normalize() ---

def test_normalize_score_inside_range_returned_unchanged():
    assert WorkspaceClassifier._normalize(0.5) == 0.5


def test_normalize_large_positive_logit_squashed_toward_one():
    result = WorkspaceClassifier._normalize(10.0)
    assert 0.99 < result < 1.0


def test_normalize_large_negative_logit_squashed_toward_zero():
    result = WorkspaceClassifier._normalize(-10.0)
    assert 0.0 < result < 0.01


def test_normalize_boundary_zero_returned_unchanged_not_sigmoided():
    # score == 0 fails both `score < 0` and `score > 1`, so it must be
    # returned as-is. Sigmoid(0) would be 0.5, which would prove the
    # boundary was wrongly included in the squashed branch.
    result = WorkspaceClassifier._normalize(0)
    assert result == 0
    assert result != 0.5


def test_normalize_boundary_one_returned_unchanged_not_sigmoided():
    # Same reasoning at the top edge. Sigmoid(1) would be roughly 0.731.
    result = WorkspaceClassifier._normalize(1)
    assert result == 1
    assert abs(result - 0.7310585786300049) > 1e-9


# --- is_related() ---

def test_is_related_score_above_threshold_returns_true(mock_classifier):
    clf, mock_model = mock_classifier
    mock_model.predict.return_value = [0.5]
    predicted, score = clf.is_related("window text", "core goal", threshold=0.3)
    assert predicted is True
    assert score == 0.5


def test_is_related_score_at_threshold_returns_false(mock_classifier):
    # Comparison is strictly `>`, so a score equal to the threshold must
    # be False, not True.
    clf, mock_model = mock_classifier
    mock_model.predict.return_value = [0.3]
    predicted, score = clf.is_related("window text", "core goal", threshold=0.3)
    assert predicted is False
    assert score == 0.3


def test_is_related_score_below_threshold_returns_false(mock_classifier):
    clf, mock_model = mock_classifier
    mock_model.predict.return_value = [0.1]
    predicted, score = clf.is_related("window text", "core goal", threshold=0.3)
    assert predicted is False
    assert score == 0.1


def test_is_related_calls_predict_with_correctly_shaped_pairs(mock_classifier):
    clf, mock_model = mock_classifier
    mock_model.predict.return_value = [0.5]
    clf.is_related("window text", "core goal", threshold=0.01)
    mock_model.predict.assert_called_once_with([["core goal", "window text"]])


def test_is_related_normalizes_a_raw_logit_before_thresholding(mock_classifier):
    clf, mock_model = mock_classifier
    mock_model.predict.return_value = [10.0]  # raw logit, well outside [0, 1]
    predicted, score = clf.is_related("window text", "core goal", threshold=0.5)
    assert predicted is True
    assert 0.99 < score < 1.0


# --- the empty-string bug ---
# is_related("", ...) currently does `return False`, a bare bool instead of
# the (bool, float) tuple every other path returns. Every real caller does
# `predicted, score = classifier.is_related(...)`, confirmed by grepping the
# whole repo, nothing depends on the old bare-False shape, so this is a pure
# bug fix, not a behavior callers rely on.

def test_empty_string_returns_a_tuple_not_bare_false(mock_classifier):
    clf, mock_model = mock_classifier
    result = clf.is_related("", "core goal")
    assert isinstance(result, tuple), f"expected a (bool, float) tuple, got {result!r}"
    assert result == (False, 0.0)


def test_whitespace_only_string_returns_a_tuple_not_bare_false(mock_classifier):
    clf, mock_model = mock_classifier
    result = clf.is_related("   \n\t  ", "core goal")
    assert isinstance(result, tuple), f"expected a (bool, float) tuple, got {result!r}"
    assert result == (False, 0.0)


def test_empty_string_does_not_call_predict(mock_classifier):
    clf, mock_model = mock_classifier
    clf.is_related("", "core goal")
    mock_model.predict.assert_not_called()


# --- score_batch() ---

def test_score_batch_empty_list_returns_empty_list_without_calling_predict(mock_classifier):
    clf, mock_model = mock_classifier
    result = clf.score_batch("core goal", [])
    assert result == []
    mock_model.predict.assert_not_called()


def test_score_batch_calls_predict_once_with_correctly_shaped_pairs(mock_classifier):
    clf, mock_model = mock_classifier
    mock_model.predict.return_value = [0.1, 0.5, 0.9]
    texts = ["text one", "text two", "text three"]
    clf.score_batch("core goal", texts)
    mock_model.predict.assert_called_once_with([["core goal", t] for t in texts])


def test_score_batch_returns_normalized_scores_same_order_same_length(mock_classifier):
    clf, mock_model = mock_classifier
    mock_model.predict.return_value = [10.0, -10.0, 0.5]  # mixed raw logits and in-range
    result = clf.score_batch("core goal", ["a", "b", "c"])
    assert len(result) == 3
    assert 0.99 < result[0] < 1.0   # squashed toward 1
    assert 0.0 < result[1] < 0.01   # squashed toward 0
    assert result[2] == 0.5         # already in range, unchanged


# --- consistency between is_related() and score_batch() ---

def test_is_related_and_score_batch_normalize_identically(mock_classifier):
    clf, mock_model = mock_classifier
    raw_score = 3.7  # a raw logit outside [0, 1], must go through sigmoid

    mock_model.predict.return_value = [raw_score]
    _, single_score = clf.is_related("window text", "core goal", threshold=0.01)

    mock_model.predict.return_value = [raw_score]
    batch_scores = clf.score_batch("core goal", ["window text"])

    assert single_score == batch_scores[0]
