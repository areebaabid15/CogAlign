# tests/test_window_tracker.py
"""
Real pytest unit tests for WindowTracker's own logic: output parsing, the
empty-title retry added alongside the Word bug fix, and the rolling
history deque. Not macOS or AppleScript itself, subprocess.run is mocked
entirely for _query_window_once()'s own tests, and _query_window_once is
mocked directly for get_active_window()'s retry-logic tests so retry
behavior is tested in isolation from AppleScript output parsing.
time.sleep is also mocked so retry-delay tests don't actually wait in real
time. Runs in well under a second.
"""
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from window_tracker import WindowTracker


def make_proc(stdout=""):
    proc = MagicMock()
    proc.stdout = stdout
    return proc


def make_tracker_with_sequence(sequence):
    """A WindowTracker whose _query_window_once() yields each item in
    sequence in turn, one per call, so retry behavior can be driven with a
    controlled script instead of depending on real AppleScript output."""
    tracker = WindowTracker()
    tracker._query_window_once = MagicMock(side_effect=sequence)
    return tracker


# --- _query_window_once(): subprocess.run mocked ---

def test_query_window_once_splits_and_strips_on_triple_pipe():
    tracker = WindowTracker()
    stdout = " Google Chrome ||| Reddit - The heart of the internet \n"
    with patch("window_tracker.subprocess.run", return_value=make_proc(stdout)):
        app, title = tracker._query_window_once()
    assert app == "Google Chrome"
    assert title == "Reddit - The heart of the internet"


def test_query_window_once_no_triple_pipe_returns_unknown():
    tracker = WindowTracker()
    with patch("window_tracker.subprocess.run", return_value=make_proc("garbage, no delimiter")):
        app, title = tracker._query_window_once()
    assert (app, title) == ("Unknown", "No Active Window")


def test_query_window_once_timeout_returns_timeout_label():
    tracker = WindowTracker()
    timeout_err = subprocess.TimeoutExpired(cmd="osascript", timeout=3)
    with patch("window_tracker.subprocess.run", side_effect=timeout_err):
        app, title = tracker._query_window_once()
    assert (app, title) == ("Timeout", "AppleScript timed out")


def test_query_window_once_generic_exception_returns_error_label_with_message():
    tracker = WindowTracker()
    with patch("window_tracker.subprocess.run", side_effect=RuntimeError("boom")):
        app, title = tracker._query_window_once()
    assert app == "Error"
    assert title == "boom"


# --- get_active_window(): retry logic, _query_window_once mocked directly ---

def test_first_call_non_empty_title_returns_immediately_no_retry():
    tracker = make_tracker_with_sequence([("Google Chrome", "Reddit")])
    with patch("window_tracker.time.sleep") as mock_sleep:
        app, title = tracker.get_active_window()
    assert (app, title) == ("Google Chrome", "Reddit")
    assert tracker._query_window_once.call_count == 1
    mock_sleep.assert_not_called()


def test_empty_title_for_real_app_retries_and_recovers():
    tracker = make_tracker_with_sequence([
        ("Microsoft Word", ""),
        ("Microsoft Word", "literature_review.docx"),
    ])
    with patch("window_tracker.time.sleep") as mock_sleep:
        app, title = tracker.get_active_window()
    assert (app, title) == ("Microsoft Word", "literature_review.docx")
    assert tracker._query_window_once.call_count == 2
    mock_sleep.assert_called_once_with(0.15)


def test_both_retries_empty_gives_up_after_max_retries_plus_one_calls():
    tracker = make_tracker_with_sequence([
        ("Microsoft Word", ""),
        ("Microsoft Word", ""),
        ("Microsoft Word", ""),
    ])
    with patch("window_tracker.time.sleep") as mock_sleep:
        app, title = tracker.get_active_window()  # default max_retries=2
    assert (app, title) == ("Microsoft Word", "")
    assert tracker._query_window_once.call_count == 3  # 1 initial + 2 retries
    assert mock_sleep.call_count == 2


@pytest.mark.parametrize("label", WindowTracker.FALLBACK_APP_LABELS)
def test_fallback_labels_with_empty_title_do_not_retry(label):
    tracker = make_tracker_with_sequence([(label, "")])
    with patch("window_tracker.time.sleep") as mock_sleep:
        app, title = tracker.get_active_window()
    assert (app, title) == (label, "")
    assert tracker._query_window_once.call_count == 1
    mock_sleep.assert_not_called()


def test_max_retries_zero_disables_retry_even_on_empty_real_app_title():
    tracker = make_tracker_with_sequence([("Microsoft Word", "")])
    with patch("window_tracker.time.sleep") as mock_sleep:
        app, title = tracker.get_active_window(max_retries=0)
    assert (app, title) == ("Microsoft Word", "")
    assert tracker._query_window_once.call_count == 1
    mock_sleep.assert_not_called()


# --- update_history() ---

def test_update_history_appends_dict_with_expected_keys():
    tracker = WindowTracker(window_size=5)
    tracker.get_active_window = MagicMock(return_value=("Google Chrome", "Reddit"))
    state = tracker.update_history()
    assert set(state.keys()) == {"timestamp", "app", "title"}
    assert state["app"] == "Google Chrome"
    assert state["title"] == "Reddit"
    assert isinstance(state["timestamp"], float)
    assert len(tracker.history) == 1
    assert tracker.history[-1] == state


def test_update_history_respects_window_size_ejects_oldest():
    tracker = WindowTracker(window_size=2)
    tracker.get_active_window = MagicMock(side_effect=[
        ("App1", "Title1"),
        ("App2", "Title2"),
        ("App3", "Title3"),
    ])
    tracker.update_history()
    tracker.update_history()
    tracker.update_history()
    assert len(tracker.history) == 2
    remaining_apps = [entry["app"] for entry in tracker.history]
    assert remaining_apps == ["App2", "App3"]  # App1 ejected, oldest out first
