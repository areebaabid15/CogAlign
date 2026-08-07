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
    sm.evaluate_state(False)  # climbs to L2 (one level per qualifying tick)
    sm.evaluate_state(False)  # climbs to L3, count=1 (elapsed still ~16)

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


def test_skipped_level2_window_climbs_to_level2_then_level3():
    """
    Regression test for the stuck-at-Level-1 bug, updated for the one-level-per-tick
    design. The original implementation required a tick to land exactly inside
    [level_2_threshold, level_3_threshold) while last_nudge_level_fired == 1, so a
    skipped window meant Level 2 (and therefore Level 3) never fired, and the machine
    got stuck at Level 1 permanently. A live run reproduced this by sitting stuck at
    Level 1 for 50+ ticks.

    The first fix jumped straight to the computed target level, which never got stuck
    but let Level 2 be skipped entirely. This was then observed live twice in a row.
    The current design instead climbs exactly one level per qualifying tick, so even
    when a tick skips straight past the Level 2 window in elapsed time, Level 2 still
    fires (a bit late) before Level 3 does, on the next qualifying tick.
    """
    sm = enter_escalating(make_sm(buffer=5, l2=10, l3=15))
    assert sm.last_nudge_level_fired == 1

    # Simulate a large gap that jumps straight past the Level 2 window in elapsed time.
    sm.distraction_start_time = time.time() - 50
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("NUDGE", 2)
    assert sm.last_nudge_level_fired == 2
    assert sm.state == "ESCALATING"

    # Next qualifying tick (elapsed still far past level_3_threshold) climbs to
    # Level 3, not stuck at Level 2.
    sm.distraction_start_time = time.time() - 50
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("NUDGE", 3)
    assert sm.last_nudge_level_fired == 3
    assert sm.level_3_nudge_count == 1


def test_never_gets_stuck_regardless_of_skipped_thresholds():
    """
    As long as target_level > last_nudge_level_fired, forward progress must happen
    on every call to evaluate_state, no matter how large the elapsed time skip is.
    From Level 1, it must take exactly two qualifying ticks (Level 2, then Level 3)
    to reach Level 3 even under an absurdly large skip, never more, and it must
    never silently stay stuck while behind target_level.
    """
    sm = enter_escalating(make_sm(buffer=5, l2=10, l3=15))

    sm.distraction_start_time = time.time() - 500  # absurdly large skip
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("NUDGE", 2)

    sm.distraction_start_time = time.time() - 500
    action, level = sm.evaluate_state(False)
    assert (action, level) == ("NUDGE", 3)
    assert sm.last_nudge_level_fired == 3

    # Now at the max level; further ticks at the same elapsed time must not
    # re-fire immediately, only once level_3_interval has passed.
    sm.distraction_start_time = time.time() - 500
    action, level = sm.evaluate_state(False)
    assert action == "SILENT"
    assert sm.state == "ESCALATING"
