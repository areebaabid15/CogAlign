# tests/test_nudge_engine.py
"""
Real pytest unit tests for QwenNudgeEngine's own logic: persona selection,
prompt assembly, and response handling. Not Ollama itself, requests.post is
mocked entirely, no real network call ever happens. Runs in well under a
second.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nudge_engine import QwenNudgeEngine, _is_real_distraction, PLACEHOLDER_DISTRACTIONS


def make_response(status_code=200, content="A nudge."):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"message": {"content": content}}
    return resp


@pytest.fixture
def engine():
    return QwenNudgeEngine()


def user_message_of(mock_post):
    return mock_post.call_args.kwargs["json"]["messages"][1]["content"]


def system_message_of(mock_post):
    return mock_post.call_args.kwargs["json"]["messages"][0]["content"]


# --- _is_real_distraction() ---

def test_is_real_distraction_none_is_false():
    assert _is_real_distraction(None) is False


def test_is_real_distraction_empty_string_is_false():
    assert _is_real_distraction("") is False


@pytest.mark.parametrize("prefix", PLACEHOLDER_DISTRACTIONS)
def test_is_real_distraction_placeholder_prefixes_are_false(prefix):
    assert _is_real_distraction(prefix + "something") is False


def test_is_real_distraction_real_distraction_is_true():
    assert _is_real_distraction("Google Chrome - Reddit") is True


# --- generate_nudge(): invalid level falls back to level 1 ---

@pytest.mark.parametrize("bad_level", [0, 5, -1])
def test_invalid_level_falls_back_to_level_1_persona(engine, bad_level):
    with patch("nudge_engine.requests.post", return_value=make_response()) as mock_post:
        engine.generate_nudge(bad_level, goal="test goal")
    assert system_message_of(mock_post) == engine.personas[1]["system"]


# --- generate_nudge(): goal prefixing ---

def test_goal_prefixes_user_message(engine):
    with patch("nudge_engine.requests.post", return_value=make_response()) as mock_post:
        engine.generate_nudge(1, goal="writing my dissertation")
    expected = f"My goal is: 'writing my dissertation'. {engine.personas[1]['user']}"
    assert user_message_of(mock_post) == expected


def test_no_goal_leaves_user_message_unprefixed(engine):
    with patch("nudge_engine.requests.post", return_value=make_response()) as mock_post:
        engine.generate_nudge(1, goal=None)
    assert user_message_of(mock_post) == engine.personas[1]["user"]


# --- generate_nudge(): level 2 distraction clause ---

def test_level2_real_distraction_appends_clause(engine):
    with patch("nudge_engine.requests.post", return_value=make_response()) as mock_post:
        engine.generate_nudge(2, distraction="Google Chrome - Reddit")
    msg = user_message_of(mock_post)
    assert "I am currently distracted by: 'Google Chrome - Reddit'." in msg


def test_level2_placeholder_distraction_does_not_append_clause(engine):
    with patch("nudge_engine.requests.post", return_value=make_response()) as mock_post:
        engine.generate_nudge(2, distraction="Unknown - No Active Window")
    msg = user_message_of(mock_post)
    assert "I am currently distracted by" not in msg
    assert msg == engine.personas[2]["user"]


@pytest.mark.parametrize("level", [1, 3])
def test_level1_and_level3_never_append_distraction_clause(engine, level):
    with patch("nudge_engine.requests.post", return_value=make_response()) as mock_post:
        engine.generate_nudge(level, distraction="Google Chrome - Reddit")
    msg = user_message_of(mock_post)
    assert "I am currently distracted by" not in msg
    assert msg == engine.personas[level]["user"]


# --- response handling ---

def test_200_response_returns_stripped_quote_stripped_content(engine):
    with patch("nudge_engine.requests.post", return_value=make_response(200, '  "A nudge sentence."  ')):
        result = engine.generate_nudge(1)
    assert result == "A nudge sentence."


def test_non_200_response_returns_formatted_error(engine):
    with patch("nudge_engine.requests.post", return_value=make_response(500)):
        result = engine.generate_nudge(1)
    assert result == "[Nudge unavailable: Ollama error 500]"


def test_timeout_exception_returns_formatted_message(engine):
    with patch("nudge_engine.requests.post", side_effect=requests.exceptions.Timeout()):
        result = engine.generate_nudge(1)
    assert result == "[Nudge unavailable: Ollama timed out]"


def test_generic_exception_returns_formatted_message_with_real_text(engine):
    with patch("nudge_engine.requests.post", side_effect=RuntimeError("connection refused")):
        result = engine.generate_nudge(1)
    assert result == "[Nudge unavailable: connection refused]"


# --- payload shape, consistent across all levels ---

@pytest.mark.parametrize("level", [1, 2, 3])
def test_payload_shape_consistent_across_levels(engine, level):
    with patch("nudge_engine.requests.post", return_value=make_response()) as mock_post:
        engine.generate_nudge(level)
    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "qwen2.5:1.5b"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.85
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
