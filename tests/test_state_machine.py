# tests/test_state_machine.py
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from state_machine import CogAlignStateMachine


def make_sm(buffer=5, l2=10, l3=15, l3_interval=10):
    sm = CogAlignStateMachine(
        buffer_threshold=buffer, level_2_threshold=l2, level_3_threshold=l3
    )
    sm.level_3_interval = l3_interval
    return sm


def enter_escalating(sm, buffer=5):
    """Drive the machine from FOCUS through the buffer into ESCALATING (Level 1 fired)."""
    sm.evaluate_state(False)  # FOCUS -> DISTRACTED_BUFFER
    sm.distraction_start_time = time.time() - (buffer + 1)
    action, level = sm.evaluate_state(False)  # DISTRACTED_BUFFER -> ESCALATING, fires L1
    assert action == "NUDGE" and level == 1
    return sm


def test_goal_oriented_stays_silent_and_focused():
    sm = make_sm()
    action, level = sm.evaluate_state(True)
    assert (action, level) == ("SILENT", 0)
    assert sm.state == "FOCUS"


def test_first_distraction_enters_buffer_silently():
    sm = make_sm()
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("SILENT", 0)
    assert sm.state == "DISTRACTED_BUFFER"


def test_buffer_stays_silent_before_threshold():
    sm = make_sm(buffer=5)
    sm.evaluate_state(False)
    sm.distraction_start_time = time.time() - 2  # only 2s elapsed, buffer is 5s
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("SILENT", 0)
    assert sm.state == "DISTRACTED_BUFFER"


def test_level1_fires_after_buffer_threshold():
    sm = make_sm(buffer=5)
    sm.evaluate_state(False)
    sm.distraction_start_time = time.time() - 6
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("NUDGE", 1)
    assert sm.state == "ESCALATING"
    assert sm.last_nudge_level_fired == 1


def test_level2_fires_when_elapsed_enters_window():
    sm = enter_escalating(make_sm(buffer=5, l2=10, l3=15))
    sm.distraction_start_time = time.time() - 11  # elapsed=11, inside [10, 15)
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("NUDGE", 2)
    assert sm.last_nudge_level_fired == 2


def test_level3_fires_after_level2():
    sm = enter_escalating(make_sm(buffer=5, l2=10, l3=15))
    sm.distraction_start_time = time.time() - 11
    sm.evaluate_state(False)  # fires L2
    sm.distraction_start_time = time.time() - 16  # elapsed=16, past level_3_threshold
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("NUDGE", 3)
    assert sm.last_nudge_level_fired == 3
    assert sm.level_3_nudge_count == 1


def test_level3_repeats_then_enters_passive_log():
    sm = enter_escalating(make_sm(buffer=5, l2=10, l3=15, l3_interval=10))
    sm.distraction_start_time = time.time() - 11
    sm.evaluate_state(False)  # L2
    sm.distraction_start_time = time.time() - 16
    sm.evaluate_state(False)  # L3 #1, count=1

    sm.last_l3_time = time.time() - 11  # interval (10s) has passed
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("NUDGE", 3)
    assert sm.level_3_nudge_count == 2

    sm.last_l3_time = time.time() - 11
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("NUDGE", 3)
    assert sm.level_3_nudge_count == 3

    # A 4th L3 interval elapsing should trip PASSIVE_LOG instead of nudging again.
    sm.last_l3_time = time.time() - 11
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("SILENT", 0)
    assert sm.state == "PASSIVE_LOG"


def test_level3_does_not_repeat_before_interval_elapses():
    sm = enter_escalating(make_sm(buffer=5, l2=10, l3=15, l3_interval=10))
    sm.distraction_start_time = time.time() - 16
    sm.evaluate_state(False)  # L3 #1, count=1

    sm.last_l3_time = time.time() - 3  # only 3s since last L3, interval is 10s
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("SILENT", 0)
    assert sm.level_3_nudge_count == 1


def test_passive_log_stays_silent():
    sm = make_sm()
    sm.state = "PASSIVE_LOG"
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("SILENT", 0)
    assert sm.state == "PASSIVE_LOG"


def test_refocus_resets_from_escalating():
    sm = enter_escalating(make_sm())
    action, level = sm.evaluate_state(True)
    assert (action, level) == ("SILENT", 0)
    assert sm.state == "FOCUS"
    assert sm.last_nudge_level_fired == 0
    assert sm.distraction_start_time is None
    assert sm.level_3_nudge_count == 0


def test_refocus_resets_from_passive_log():
    sm = make_sm()
    sm.state = "PASSIVE_LOG"
    sm.last_nudge_level_fired = 3
    sm.level_3_nudge_count = 3
    action, level = sm.evaluate_state(True)
    assert (action, level) == ("SILENT", 0)
    assert sm.state == "FOCUS"
    assert sm.last_nudge_level_fired == 0
    assert sm.level_3_nudge_count == 0


def test_skipped_level2_window_still_reaches_level3():
    """
    Regression test for the stuck-at-Level-1 bug: the old implementation required
    a tick to land exactly inside [level_2_threshold, level_3_threshold) while
    last_nudge_level_fired == 1. If no tick landed in that window, Level 2 never
    fired, and Level 3 never fired either since it required last_nudge_level_fired
    == 2 first, so the machine got stuck at Level 1 permanently. A live run
    reproduced this by sitting stuck at Level 1 for 50+ ticks.

    The fix must catch up directly to Level 3 on the next tick, regardless of how
    much elapsed time was skipped over.
    """
    sm = enter_escalating(make_sm(buffer=5, l2=10, l3=15))
    assert sm.last_nudge_level_fired == 1

    # Simulate a large gap that jumps straight past the Level 2 window.
    sm.distraction_start_time = time.time() - 50
    action, level = sm.evaluate_state(False)

    assert (action, level) == ("NUDGE", 3)
    assert sm.last_nudge_level_fired == 3
    assert sm.state == "ESCALATING"


def test_stuck_at_level1_does_not_persist_across_repeated_ticks():
    """Same regression, but hammer it with repeated ticks at the skipped-over
    elapsed time to confirm it doesn't take more than one tick to recover, and
    that it doesn't then get double-fired on every subsequent tick either."""
    sm = enter_escalating(make_sm(buffer=5, l2=10, l3=15))

    sm.distraction_start_time = time.time() - 50
    first_action, first_level = sm.evaluate_state(False)
    assert (first_action, first_level) == ("NUDGE", 3)

    # Immediately following ticks at the same elapsed time should stay silent
    # (no repeat until level_3_interval has passed), not re-fire Level 3 every tick.
    for _ in range(5):
        sm.distraction_start_time = time.time() - 50
        action, level = sm.evaluate_state(False)
        assert action == "SILENT"
        assert sm.state == "ESCALATING"
