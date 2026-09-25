# src/engine.py
import json
import os
import re
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
HARD_NEGATIVE_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "workspace_dumps", "hard_negative_templates.json"
)
WORK_LIKE_DISTRACTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "workspace_dumps", "work_like_distractions.json"
)

# Generated hard negatives: OFF on the live path (set this to
# HARD_NEGATIVE_TEMPLATES_PATH to turn them on). Tested and rejected: they
# caught more distractions but wrongly flagged more genuine work, the wrong
# trade for a focus tool. Kept so the experiment can be rerun. See
# tests/evaluate_calibration_fix.py, results/classifier_calibration_fix.json
# and results/final_eval/e2_calibration.json.
GENERATED_HARD_NEGATIVES_DEFAULT = None

# Work-like distraction bank: also OFF on the live path (set this to
# WORK_LIKE_DISTRACTIONS_PATH to turn it on). Tested and rejected: it changed
# nothing on held-out goals, and in sample it wrongly flagged more genuine
# work. Kept so the experiment can be rerun. See tests/evaluate_worklike_fix.py
# and results/final_eval/e2_calibration.json.
WORK_LIKE_DEFAULT = None

# Upper bound on the calibrated threshold, so a goal whose words appear in
# almost any title cannot push the line above the user's own work.
CALIBRATION_CEILING = 0.15

# Cap on content words taken from the goal. Each word multiplies the number of
# generated titles, and calibration runs while the user waits at startup.
MAX_CONTENT_WORDS = 5

# Words with no topical content, including the framing people speak into the
# microphone ("I have to...", "I need to finish..."). A fixed set rather than a
# model, so extraction stays deterministic and unit testable.
GOAL_STOPWORDS = frozenset({
    "a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "been", "before", "being", "but", "by",
    "can", "could",
    "did", "do", "does", "doing", "done", "down", "due",
    "each", "every",
    "few", "finally", "finish", "finished", "finishing", "for", "from",
    "get", "getting", "go", "going", "gonna", "got",
    "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his",
    "how", "however",
    "i", "if", "in", "into", "is", "it", "its",
    "just",
    "keep", "kind",
    "later", "least", "less", "like", "little", "lot",
    "make", "makes", "making", "many", "may", "me", "might", "mine", "more",
    "most", "much", "must", "my", "myself",
    "need", "needs", "needed", "never", "new", "next", "no", "nor", "not", "now",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "out",
    "over", "own",
    "per", "put",
    "really", "right",
    "same", "she", "should", "so", "some", "soon", "start", "started",
    "starting", "still", "such", "sure",
    "take", "than", "that", "the", "their", "theirs", "them", "then", "there",
    "these", "they", "thing", "things", "this", "those", "through", "time",
    "to", "today", "tomorrow", "tonight", "too", "try", "trying",
    "under", "until", "up", "us", "use", "using",
    "very",
    "want", "wanted", "wanna", "was", "way", "we", "week", "well", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "would",
    "yet", "you", "your", "yours",
})

# Below this length a token is almost never a usable topic word, and short
# fragments make noisy traps. Applied after stopword removal.
MIN_CONTENT_WORD_LENGTH = 3


def extract_content_words(goal, max_words=MAX_CONTENT_WORDS):
    """
    Returns the topical words of a spoken goal, in spoken order, for building
    trap window titles. Lowercases, splits on anything but letters, digits and
    apostrophes, drops stopwords and short tokens, removes duplicates and
    keeps at most max_words.

    "I have to write a report." -> ["write", "report"]

    Returns [] for an empty goal or one made only of stopwords. Callers treat
    that as "no traps for this goal", not as an error.
    """
    if not isinstance(goal, str) or not goal.strip():
        return []

    tokens = re.findall(r"[a-z0-9']+", goal.lower())

    content_words = []
    for token in tokens:
        word = token.strip("'")
        # Drop a possessive ("sister's" becomes "sister") so the templates
        # read as natural window titles.
        if word.endswith("'s"):
            word = word[:-2]
        if len(word) < MIN_CONTENT_WORD_LENGTH:
            continue
        if word in GOAL_STOPWORDS:
            continue
        if word in content_words:
            continue
        content_words.append(word)
        if len(content_words) >= max_words:
            break

    return content_words


def generate_hard_negatives(goal, templates_path=HARD_NEGATIVE_TEMPLATES_PATH, max_words=MAX_CONTENT_WORDS):
    """
    Builds goal-specific trap titles by putting each content word of the goal
    into every off-task template in hard_negative_templates.json, giving
    windows that share the goal's vocabulary but are off task.

    A falsy templates_path means generation is switched off, the live default
    (see GENERATED_HARD_NEGATIVES_DEFAULT), and returns [] without reading any
    file. Also returns [] on any failure, so calibration falls back to the
    generic bank instead of breaking.
    """
    if not templates_path:
        return []

    try:
        content_words = extract_content_words(goal, max_words=max_words)
        if not content_words:
            return []

        with open(templates_path, "r") as f:
            data = json.load(f)

        templates = data.get("templates", [])
        if not templates:
            return []

        generated = []
        for template in templates:
            if not isinstance(template, str) or "{word}" not in template:
                continue
            for word in content_words:
                generated.append(template.replace("{word}", word))

        return generated
    except Exception:
        return []


def load_work_like_distractions(path=WORK_LIKE_DISTRACTIONS_PATH):
    """
    Loads the work-like distraction bank: windows that look like work but are
    not the stated goal, such as email, other documents and admin screens. The
    generic bank holds only entertainment, social media and shopping, so
    windows like these could clear the calibrated threshold.

    Kept in its own file so that workspace_samples.json, which
    results/classifier_evaluation.json was computed from, stays unchanged.
    Off on the live path, see WORK_LIKE_DEFAULT. Returns [] on any failure.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return list(data.get("distraction", []))
    except Exception:
        return []


def calibrate_threshold(
    classifier,
    goal,
    samples_path=DISTRACTION_SAMPLES_PATH,
    percentile=CALIBRATION_PERCENTILE,
    margin=CALIBRATION_MARGIN,
    default=DEFAULT_THRESHOLD,
    templates_path=GENERATED_HARD_NEGATIVES_DEFAULT,
    max_threshold=CALIBRATION_CEILING,
    work_like_path=WORK_LIKE_DEFAULT,
):
    """
    Sets a per-goal threshold once per session, because one fixed threshold
    does not transfer between goals (see tests/diagnose_goal_sensitivity.py).

    Scores the goal against a bank of off-task windows, takes the 90th
    percentile (not the maximum, so one outlier cannot distort it), adds a
    small margin and caps the result at max_threshold. On the live path the
    bank is the generic distraction set in workspace_samples.json alone.

    templates_path adds generated hard negatives and work_like_path adds the
    work-like bank. Both are off on the live path, see the constants above for
    why, and tests/test_calibration.py switches them on to test them.

    Falls back to the fixed default threshold if anything fails, so the system
    always starts.
    """
    try:
        with open(samples_path, "r") as f:
            data = json.load(f)

        distractions = data.get("distraction", [])
        if not distractions:
            return default

        # The generic bank always, plus the two optional populations, which
        # return [] when switched off or on any failure.
        calibration_samples = (
            list(distractions)
            + load_work_like_distractions(path=work_like_path)
            + generate_hard_negatives(goal, templates_path=templates_path)
        )

        scores = classifier.score_batch(goal, calibration_samples)
        if not scores:
            return default

        ceiling = float(np.percentile(scores, percentile))
        threshold = ceiling + margin

        # Cap the threshold: set too high, it would flag the user's own work as
        # distraction and nudge them continuously, worse than a missed one.
        return min(threshold, max_threshold)
    except Exception:
        return default


class CogAlignEngine:

    def __init__(self, goal: str):
        print("Loading CogAlign Engine...")
        self.core_goal = goal
        self.classifier = WorkspaceClassifier()
        self.tracker = WindowTracker()
        self.nudge_engine = QwenNudgeEngine()
        # Demonstration profile: nudges after 5, 10 and 15 seconds of drift,
        # with Level 3 repeating every 10 seconds up to three times, and the
        # session closes after 2 minutes of continued drift. This compresses a
        # session so it can be shown and tested in minutes. The defaults in
        # state_machine.py (30, 90 and 150 seconds, 60 second repeat) are the
        # slower production profile.
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

        # No reliable signal this tick: an empty title even after
        # window_tracker's retries, or one of its failure labels. Skip scoring
        # and hold state, so a bad read can neither fire a nudge nor reset
        # genuine progress.
        if not window_title.strip() or app_name in WindowTracker.FALLBACK_APP_LABELS:
            print(f"\n[App]: {app_name}")
            print("⚠️ No reliable window signal, holding state")
            return

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
            nudge = self.nudge_engine.generate_nudge(level=level, goal=self.core_goal, distraction=window_text)
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
        Runs run_tick() every interval seconds until Ctrl+C or until stop_event
        is set. Pass a threading.Event when running in a background thread, as
        app.py does, since Ctrl+C only reaches the main thread.
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