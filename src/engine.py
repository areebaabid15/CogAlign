# src/engine.py
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from classifier import WorkspaceClassifier
from state_machine import CogAlignStateMachine
from window_tracker import WindowTracker
from nudge_engine import QwenNudgeEngine
from notifier import send_notification

PERSONA_NAMES = ['', 'Empathetic Coach', 'Sarcastic Friend', 'Deadline Supervisor']


class CogAlignEngine:

    def __init__(self, goal: str):
        print("Loading CogAlign Engine...")
        self.core_goal = goal
        self.classifier = WorkspaceClassifier()
        self.tracker = WindowTracker()
        self.nudge_engine = QwenNudgeEngine()
        self.state_machine = CogAlignStateMachine(
            buffer_threshold=5,
            level_2_threshold=10,
            level_3_threshold=15
        )
        self.state_machine.level_3_interval = 10
        self.max_wasted_allowed = 120

        print(f"\n✅ CogAlign Engine Ready")
        print(f"🎯 Goal: '{self.core_goal}'\n")

    def run_tick(self):
        window_info = self.tracker.get_active_window()
        if not window_info:
            return

        app_name, window_title = window_info[0], window_info[1]
        window_text = f"{app_name} - {window_title}"

        is_goal_oriented, score = self.classifier.is_related(
            window_text, self.core_goal, threshold=0.01
        )

        action, level = self.state_machine.evaluate_state(is_goal_oriented)

        print(f"\n[App]: {app_name}")
        print(f"[Window]: {window_title}")
        print(f"[Score]: {score:.2f} | [State]: {self.state_machine.state}")

        if action == "NUDGE":
            print(f"\n🚨 Distraction detected — generating Level {level} nudge...")
            nudge = self.nudge_engine.generate_nudge(level=level, goal=self.core_goal,distraction=window_text)
            persona_name = PERSONA_NAMES[level]
            print(f"\n💬 [{persona_name}]: {nudge}\n")
            send_notification(persona_name, nudge)

        elif self.state_machine.state == "PASSIVE_LOG":
            wasted = time.time() - self.state_machine.distraction_start_time
            mins, secs = int(wasted // 60), int(wasted % 60)
            print(f"🏳️ Passive Log | Wasted: {mins}m {secs}s")

            if wasted >= self.max_wasted_allowed:
                print("\n🛑 CogAlign shutting down — prolonged distraction detected.")
                sys.exit(0)
        else:
            if self.state_machine.state == "FOCUS":
                print("✅ On task — monitoring silently")
            elif self.state_machine.state == "DISTRACTED_BUFFER":
                print("⏳ Distraction buffer — watching...")
            elif self.state_machine.state == "ESCALATING":
                print("⚠️ Escalating — nudge incoming soon...")

    def start(self, interval=3):
        print(f"⏱️ Checking every {interval}s. Press Ctrl+C to stop.\n")
        try:
            while True:
                self.run_tick()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


if __name__ == "__main__":
    engine = CogAlignEngine(goal="writing my dissertation literature review")
    engine.start(interval=3)