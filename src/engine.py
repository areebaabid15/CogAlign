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

# Generated hard negatives are OFF in the live system. Setting this to
# HARD_NEGATIVE_TEMPLATES_PATH turns them back on.
#
# They were built, measured and deliberately rejected, they are not unfinished
# work. Adding them to calibration does what it was designed to do: on a
# held-out set of three goals it never saw, the vocabulary-sharing leak halved,
# 41.7% down to 20.8%. But raising the line above those traps necessarily
# raises it above genuine work using the same words, and the false alarm rate
# nearly tripled with it, 12.5% up to 33.3%, taking held-out F1 down from 76.4%
# to 71.1%. For a focus tool that trade is backwards: being interrupted while
# working destroys trust in the system, while missing one distraction costs
# almost nothing, so the cheaper error is the one to keep making. The damage
# also falls unevenly, goals whose vocabulary is common off task ("wedding",
# "report", "video") are hurt worst, and a goal already leaking nothing is
# charged the cost for a problem it does not have.
#
# The code, templates, tests and evaluation runner all stay so the experiment
# stays reproducible. See notes/CogAlign_Model_Testing_Log.md section 8 and
# results/classifier_calibration_fix.json for the full per goal numbers.
GENERATED_HARD_NEGATIVES_DEFAULT = None

# The work-like bank is OFF on the live path too. Setting this to
# WORK_LIKE_DISTRACTIONS_PATH turns it back on.
#
# Measured on both labelled sets, see results/final_eval/e2_calibration.json.
# On the three held-out goals it changes nothing at all: thresholds move by
# 0.0003 to 0.002 and not one sample changes class. On the six in-sample goals
# it cuts the hard negative leak from 50.0% to 33.3%, but raises false alarms
# on genuine work from 20.8% to 33.3% and takes mean per goal F1 from 0.69
# down to 0.65. That is the same backwards trade as the generated hard
# negatives above: it buys silence on distractions by interrupting real work.
#
# The loader, data/workspace_dumps/work_like_distractions.json and
# tests/evaluate_worklike_fix.py all stay so the experiment stays reproducible.
WORK_LIKE_DEFAULT = None

# Upper bound on the calibrated threshold, see calibrate_threshold() for the
# empirical justification. A pathological goal must not be able to push the
# line so high that genuine work reads as distraction and the user is nudged
# continuously, which is worse than missing a distraction.
CALIBRATION_CEILING = 0.15

# Cap on how many content words are pulled from the goal. Every extra word
# multiplies by the template count, and calibration is startup latency the
# user waits through. Five covers every goal tested without the scoring call
# growing enough to be noticeable.
MAX_CONTENT_WORDS = 5

# Words carrying no topical content, so substituting them into a trap
# template would produce a string that tells us nothing about this goal.
# Covers ordinary English function words plus the intention framing people
# actually speak into the microphone ("I have to...", "I need to finish...").
# Deliberately a plain hardcoded set, not a model or an external corpus, so
# extraction stays deterministic, dependency free and unit testable.
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
    Pulls the topical words out of a spoken goal, so trap window titles can be
    built from them. Deterministic string handling only, no model and no new
    dependency: lowercase, split on anything that is not a letter, digit or
    internal apostrophe, drop stopwords and very short tokens, de-duplicate
    while preserving spoken order, and keep at most max_words.

    "I have to write a report." -> ["write", "report"]

    Returns [] for the degenerate cases (empty goal, non-string goal, a goal
    made entirely of stopwords such as "I need to do this now"). Callers must
    treat an empty list as "no traps can be generated for this goal" and carry
    on, never as an error.
    """
    if not isinstance(goal, str) or not goal.strip():
        return []

    tokens = re.findall(r"[a-z0-9']+", goal.lower())

    content_words = []
    for token in tokens:
        word = token.strip("'")
        # Drop the possessive suffix so "sister's" becomes "sister". The
        # trap templates read as natural window titles with the bare noun
        # and as nonsense with the possessive still attached.
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
    Builds goal-specific trap window titles by substituting each content word
    from the goal into every generic off-task template in
    hard_negative_templates.json.

    These are the samples the generic distraction bank has never contained:
    windows that share the goal's vocabulary but are genuinely off task, which
    is exactly the population the calibrated threshold was failing to sit above
    (see notes/CogAlign_Model_Testing_Log.md section 7).

    A falsy templates_path means "generation is switched off", and returns []
    without touching the filesystem. That is how the live app runs, see
    GENERATED_HARD_NEGATIVES_DEFAULT above. It is checked explicitly rather
    than left to open(None) raising into the catch-all below, so that the
    disabled path stays distinguishable from a genuine failure and does not
    depend on the breadth of an except clause.

    Returns [] on any failure or degenerate goal, never raises. Calibration
    then simply proceeds on the generic bank alone, which is the pre-existing
    behaviour, so this can only ever add signal, never remove a working path.
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
    Loads the goal-agnostic "looks like work but is not the stated goal" bank:
    email, other documents, admin, system chrome. workspace_samples.json's
    original 50 distraction samples are entertainment, social media and
    shopping only, so the calibrated threshold sat just above that level and
    a CV, a Terminal window, or an inbox, on task or not, cleared it easily.
    Confirmed live: with the goal "Writing my project proposal", an unrelated
    CV in Microsoft Word scored 0.03 against a calibrated threshold of 0.0232
    and read as on task.

    Kept in its own file rather than merged into workspace_samples.json,
    since results/classifier_evaluation.json (report Table 6) was computed
    against that file's original 50 samples and must stay reproducible from
    it unchanged.

    Off on the live path, like the hard negative templates, see
    WORK_LIKE_DEFAULT for the measured reason. The assumption behind it was
    that goal-agnostic samples would raise the threshold evenly rather than
    punishing genuine work the way the hard negative experiment did. Measured
    against the labelled sets that did not hold: in sample the false alarm rate
    rose from 20.8% to 33.3% while F1 fell, and on the held-out goals it made
    no difference at all.

    Returns [] on any failure, same contract as generate_hard_negatives, so a
    missing or broken file degrades to the previous generic-bank-only
    calibration rather than failing startup.
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
    Calibrates a per-goal threshold instead of using the fixed 0.01 default, which
    was only ever validated against one goal (see results/classifier_evaluation.json)
    and does not generalize, raw score scale varies wildly by goal (see
    tests/diagnose_goal_sensitivity.py).

    Scores the session's goal against a population of off-task windows and takes
    a high percentile (90th by default, not the max, so one outlier sample
    doesn't distort it) as this goal's "distraction ceiling". The live threshold
    is set just above that ceiling.

    As shipped, that population is the goal-agnostic bank of generic distraction
    examples in workspace_samples.json alone: Netflix, Instagram, food delivery,
    none of which share vocabulary with any goal.

    Two further populations exist and both are off on the live path. Each is
    switched on explicitly by tests/test_calibration.py and by its own
    evaluation runner, so both rejected experiments stay reproducible.

    Passing templates_path adds goal-specific hard negatives, synthesised by
    substituting the goal's own content words into the off-task templates in
    hard_negative_templates.json. See GENERATED_HARD_NEGATIVES_DEFAULT above
    for why it is off, and tests/evaluate_calibration_fix.py. It halved the
    vocabulary-sharing leak and nearly tripled the false alarm rate on genuine
    work, which is the wrong trade for a focus tool.

    Passing work_like_path adds the goal-agnostic work-like bank: email, other
    documents, admin, system chrome. See WORK_LIKE_DEFAULT above for why it is
    off, and tests/evaluate_worklike_fix.py. It changed nothing on the held-out
    goals, and in sample it bought a lower hard negative leak with a higher
    false alarm rate and lower F1.

    See notes/CogAlign_Model_Testing_Log.md section 8 and
    results/final_eval/e2_calibration.json.

    Runs once per session at startup, not per tick. Falls back to the fixed
    default threshold if calibration fails for any reason (file missing, empty
    scores, etc.) so the system stays functional either way. Both optional
    populations are additive and independently guarded: if either yields
    nothing, for any reason including a goal with no extractable content words
    or a missing file, calibration proceeds on whatever remains, and with both
    off that is the generic bank alone.
    """
    try:
        with open(samples_path, "r") as f:
            data = json.load(f)

        distractions = data.get("distraction", [])
        if not distractions:
            return default

        # Three populations, all additive. The generic bank (always), the
        # work-like distractions and the generated hard negatives (both off on
        # the live path, work_like_path and templates_path both default to
        # None). Each loader swallows its own failures and returns [], so a
        # missing or broken file degrades calibration one population at a time
        # rather than failing startup.
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

        # Sanity ceiling. A pathological goal, one whose content words appear
        # in almost any window title, could otherwise push the threshold above
        # the score its own genuine work earns, so every window reads as
        # distraction and the user is nudged continuously while working. That
        # failure is worse than missing a distraction, because it destroys
        # trust in the system immediately. See the log for how 0.15 was chosen.
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

        # window_tracker.py already retries an empty title a couple of times
        # internally before returning. If it's still empty here, or the app
        # name is one of its own failure labels, there's no reliable signal
        # for this tick, confirmed live against Microsoft Word returning an
        # empty title mid-session while genuinely still frontmost. Don't
        # score it and don't touch the state machine at all this tick, a
        # real distraction will still be caught on the next good read, and a
        # false empty read must not fire a nudge or reset genuine progress.
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