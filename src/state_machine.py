# src/state_machine.py
import time


class CogAlignStateMachine:
    def __init__(
        self, buffer_threshold=30, level_2_threshold=90, level_3_threshold=150
    ):
        """
        Manages the distraction state transitions and tracking timers.
        All thresholds are calculated in seconds from the initial distraction timestamp.
        """
        self.state = "FOCUS"
        self.distraction_start_time = None
        self.last_nudge_level_fired = 0

        # Timing configurations (Seconds relative to distraction start)
        self.buffer_threshold = buffer_threshold  # 30s grace period
        self.level_2_threshold = level_2_threshold  # 1.5 mins total distraction
        self.level_3_threshold = level_3_threshold  # 2.5 mins total distraction

    def reset_focus(self):
        """Resets all metrics back to a clean focus profile when the user returns to work."""
        self.state = "FOCUS"
        self.distraction_start_time = None
        self.last_nudge_level_fired = 0

    def evaluate_state(self, is_goal_oriented):
        """
        Executes a single tick of the state machine logic based on
        whether the current window layout matches workspace goals.

        Returns: Tuple (action_to_take, nudge_level)
        """
        now = time.time()

        # --- FOCUS STATE RESET ---
        # The "User Returns to" line on our diagram
        if is_goal_oriented:
            self.reset_focus()
            return "SILENT", 0

        # --- ENTERING THE BUFFER ---
        # Initial slip from Focus into the Yellow Buffer zone
        if self.state == "FOCUS":
            self.state = "DISTRACTED_BUFFER"
            self.distraction_start_time = now
            return "SILENT", 0

        # Calculate exactly how many seconds we've been off-track
        elapsed = now - self.distraction_start_time

        # --- BUFFER STATE LOGIC ---
        if self.state == "DISTRACTED_BUFFER":
            # If within 30s grace period, stay silent ("User Refocuses" path remains open)
            if elapsed < self.buffer_threshold:
                return "SILENT", 0
            else:
                # Buffer breached! Drop into the Escalating container at Level 1
                self.state = "ESCALATING"
                self.last_nudge_level_fired = 1
                return "NUDGE", 1

        # --- ESCALATING CONTAINER LOGIC ---
        elif self.state == "ESCALATING":
            # Level 2 Threshold Check (90 seconds total distraction)
            if elapsed >= self.level_2_threshold and self.last_nudge_level_fired == 1:
                self.last_nudge_level_fired = 2
                return "NUDGE", 2

            # Level 3 Threshold Check (150 seconds total distraction)
            elif elapsed >= self.level_3_threshold and self.last_nudge_level_fired == 2:
                self.last_nudge_level_fired = 3
                return "NUDGE", 3

        # Default fallback: Keep waiting without double-firing alerts
        return "SILENT", 0


if __name__ == "__main__":
    # Quick simulation testing block with accelerated step timers
    machine = CogAlignStateMachine(
        buffer_threshold=4, level_2_threshold=8, level_3_threshold=12
    )
    print("=== Simulating State Machine (Fast-Tracked Timers) ===")

    # 1. Simulate 5 continuous ticks of being distracted (every 2 seconds)
    for tick in range(1, 6):
        action, level = machine.evaluate_state(is_goal_oriented=False)
        print(
            f"Tick {tick} (Distracted): State='{machine.state}' -> Action={action}, Level={level}"
        )
        time.sleep(2)

    # 2. Simulate returning to work
    action, level = machine.evaluate_state(is_goal_oriented=True)
    print(
        f"\nTick 6 (Returned to Work): State='{machine.state}' -> Action={action}, Level={level}"
    )
