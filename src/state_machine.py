# src/state_machine.py
import time


class CogAlignStateMachine:
    def __init__(
        self, buffer_threshold=30, level_2_threshold=90, level_3_threshold=150
    ):
        """
        Manages focus and distraction states with an escalation ladder.
        """
        self.state = "FOCUS"
        self.distraction_start_time = None
        self.last_nudge_level_fired = 0
        self.level_3_nudge_count = 0  # Track repeated level 3 nudges
        self.last_l3_time = 0.0  # Explicitly initialized to clear IDE warnings

        # Timing thresholds (in seconds)
        self.buffer_threshold = buffer_threshold  # Grace period
        self.level_2_threshold = level_2_threshold  # Switch to Level 2
        self.level_3_threshold = level_3_threshold  # Switch to Level 3
        self.level_3_interval = 60  # 1 minute gap between consecutive L3s

    def reset_focus(self):
        self.state = "FOCUS"
        self.distraction_start_time = None
        self.last_nudge_level_fired = 0
        self.level_3_nudge_count = 0
        self.last_l3_time = 0.0

    def evaluate_state(self, is_goal_oriented):
        now = time.time()

        # Step A: Absolute Override (User is working)
        if is_goal_oriented:
            self.reset_focus()
            return "SILENT", 0

        # Step B: User just slid out of focus
        if self.state == "FOCUS":
            self.state = "DISTRACTED_BUFFER"
            self.distraction_start_time = now
            return "SILENT", 0

        # Calculate elapsed distraction duration safely
        elapsed = now - (
            self.distraction_start_time if self.distraction_start_time else now
        )

        # Step C: Handling the Grace Buffer
        if self.state == "DISTRACTED_BUFFER":
            if elapsed < self.buffer_threshold:
                return "SILENT", 0
            else:
                self.state = "ESCALATING"
                self.last_nudge_level_fired = 1
                return "NUDGE", 1

        # Step D: Climbing the Escalation Ladder & Waving the White Flag
        elif self.state == "ESCALATING":
            # Work out which level we should be at given elapsed time alone,
            # not whether a tick happened to land inside a specific window.
            # This guarantees we catch up to the correct level on the very next
            # tick even if earlier ticks were skipped or delayed past a threshold.
            if elapsed >= self.level_3_threshold:
                target_level = 3
            elif elapsed >= self.level_2_threshold:
                target_level = 2
            else:
                target_level = 1

            # Climb straight to the target level if we're behind it.
            if target_level > self.last_nudge_level_fired:
                self.last_nudge_level_fired = target_level
                if target_level == 3:
                    self.level_3_nudge_count = 1
                    self.last_l3_time = now
                return "NUDGE", target_level

            # Check for Repeated Level 3 Nudges (Every level_3_interval seconds)
            if self.last_nudge_level_fired == 3:
                # Have we hit the maximum limit of 3 nudges?
                if self.level_3_nudge_count >= 3:
                    self.state = "PASSIVE_LOG"
                    return "SILENT", 0

                # Check if the interval has passed since the last Level 3 notification
                if now - self.last_l3_time >= self.level_3_interval:
                    self.level_3_nudge_count += 1
                    self.last_l3_time = now
                    return "NUDGE", 3

        # Step E: Waved White Flag - Silent log state
        elif self.state == "PASSIVE_LOG":
            return "SILENT", 0

        return "SILENT", 0
