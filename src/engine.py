# src/engine.py
import json
import os
import sys
import time

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from classifier import WorkspaceClassifier
from state_machine import CogAlignStateMachine
from window_tracker import WindowTracker
from nudge_engine import QwenNudgeEngine
from notifier import send_notification

PERSONA_NAMES = ['', 'Empathetic Coach', 'Sarcastic Friend', 'Deadline Supervisor']

DEFAULT_THRESHOLD = 0.01
CALIBRATION_PERCENTILE = 90
CALIBRATION_MARGIN = 0.005
DISTRACTION_SAMPLES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "workspace_dumps", "workspace_samples.json"
)


def calibrate_threshold(
    classifier,
    goal,
    samples_path=DISTRACTION_SAMPLES_PATH,
    percentile=CALIBRATION_PERCENTILE,
    margin=CALIBRATION_MARGIN,
    default=DEFAULT_THRESHOLD,
):
    """
    Calibrates a per-goal threshold instead of using the fixed 0.01 default, which
    was only ever validated against one goal (see results/classifier_evaluation.json)
    and does not generalize, raw score scale varies wildly by goal (see
    tests/diagnose_goal_sensitivity.py).

    Reuses the existing goal-agnostic bank of generic distraction examples in
    workspace_samples.json, scores the session's goal against every one of them,
    and takes a high percentile (90th by default, not the max, so one outlier
    sample doesn't distort it) as this goal's "distraction ceiling". The live
    threshold is set just above that ceiling.

    Runs once per session at startup, not per tick. Falls back to the fixed
    default threshold if calibration fails for any reason (file missing, empty
    scores, etc.) so the system stays functional either way.
    """
    try:
        with open(samples_path, "r") as f:
            data = json.load(f)

        distractions = data.get("distraction", [])
        if not distractions:
            return default

        scores = classifier.score_batch(goal, distractions)
        if not scores:
            return default

        ceiling = float(np.percentile(scores, percentile))
        return ceiling + margin
    except Exception:
        return default


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

        self.classifier_threshold = calibrate_threshold(self.classifier, self.core_goal)

        print(f"\n✅ CogAlign Engine Ready")
        print(f"🎯 Goal: '{self.core_goal}'")
        print(f"📏 Calibrated distraction threshold: {self.classifier_threshold:.4f}\n")

    def run_tick(self):
        window_info = self.tracker.get_active_window()
        if not window_info:
            return

        app_name, window_title = window_info[0], window_info[1]
        window_text = f"{app_name} - {window_title}"

        is_goal_oriented, score = self.classifier.is_related(
            window_text, self.core_goal, threshold=self.classifier_threshold
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

    def start(self, interval=3, stop_event=None):
        """
        stop_event is optional (defaults to None) so existing direct callers,
        like pipeline.py, are unaffected: when it's None the loop only ever
        exits via KeyboardInterrupt, exactly as before. Pass a threading.Event
        to allow a caller running this in a background thread to request a
        clean exit, since KeyboardInterrupt is only ever delivered to the
        main thread and does not work there.
        """
        print(f"⏱️ Checking every {interval}s. Press Ctrl+C to stop.\n")
        try:
            while stop_event is None or not stop_event.is_set():
                self.run_tick()
                time.sleep(interval)
            print("\nMonitoring stopped (stop requested).")
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


if __name__ == "__main__":
    engine = CogAlignEngine(goal="writing my dissertation literature review")
    engine.start(interval=3)