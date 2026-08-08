# tests/diagnose_goal_sensitivity.py
"""
Diagnostic tool, not a pytest test: does WorkspaceClassifier's fixed threshold
(0.01, picked in results/classifier_evaluation.json) generalize to goals it was
never calibrated against, or does it only work reliably for the one goal
("Writing my dissertation literature review") that sweep actually used? And does
the per-goal calibrate_threshold() in engine.py actually fix that?

Tests three goals spanning different domains against clear, substantive window
text on both sides (on-task examples tailored per goal, the same off-task
examples reused across all three so any variation is attributable to the goal
text, not the window text). Deliberately avoids empty/near-empty window titles,
that is the separate, already-documented window_tracker.py issue, not this one.

For each goal, runs the same 6 on-task/off-task combinations twice: once against
the fixed 0.01 threshold, once against calibrate_threshold()'s per-goal result,
and prints a before/after accuracy table.

Usage: python3 tests/diagnose_goal_sensitivity.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier import WorkspaceClassifier
from engine import calibrate_threshold

FIXED_THRESHOLD = 0.01

# Held constant across all three goals so a change in whether these get
# correctly rejected is attributable to the goal text, not the window text.
OFF_TASK = [
    "YouTube - funny cat videos",
    "Instagram - Explore Page",
    "Reddit - r/all",
]

GOALS = {
    # Control: the one goal results/classifier_evaluation.json was actually
    # calibrated against. Expected to behave correctly under both thresholds.
    "Writing my dissertation literature review": [
        "Microsoft Word - Literature Review Chapter 2.docx",
        "Google Scholar - search results for NLP papers",
        "Zotero - My Library - dissertation references",
    ],
    "Writing my project proposal": [
        "Microsoft Word - Project Proposal Draft.docx",
        "Google Docs - Project Proposal Outline",
        "Notes - project proposal budget and timeline",
    ],
    # A genuinely different domain, not just another academic-writing task.
    "Preparing for a job interview": [
        "Notes - interview questions and answers",
        "LinkedIn - Company Research - Acme Corp",
        "Google Docs - STAR method answers draft",
    ],
}


def evaluate(classifier, goal, on_task_examples, threshold):
    """Runs the same 6 on-task/off-task combinations for one goal against one
    threshold. Returns a list of (window, label, score, predicted, correct)."""
    rows = []

    for window in on_task_examples:
        predicted, score = classifier.is_related(window, goal, threshold=threshold)
        rows.append((window, "on-task", score, predicted, predicted is True))

    for window in OFF_TASK:
        predicted, score = classifier.is_related(window, goal, threshold=threshold)
        rows.append((window, "off-task", score, predicted, predicted is False))

    return rows


def print_rows(rows):
    for window, label, score, predicted, correct in rows:
        status = "correct  " if correct else "WRONG    "
        print(f"  [{status}] {label:<8} score={score:.4f} predicted={str(predicted):5} -> '{window}'")


def accuracy(rows):
    correct = sum(1 for r in rows if r[-1])
    return correct, len(rows)


def run():
    print("=" * 78)
    print("  COGALIGN — GOAL SENSITIVITY DIAGNOSTIC")
    print("  Fixed threshold=0.01 vs calibrate_threshold()'s per-goal result")
    print("=" * 78)

    classifier = WorkspaceClassifier()

    fixed_results = {}      # goal -> rows
    calibrated_results = {}  # goal -> rows
    calibrated_thresholds = {}  # goal -> threshold value

    for goal, on_task_examples in GOALS.items():
        calibrated = calibrate_threshold(classifier, goal)
        calibrated_thresholds[goal] = calibrated

        print(f"\n--- Goal: '{goal}' ---")
        print(f"Calibrated threshold: {calibrated:.4f}  (fixed threshold: {FIXED_THRESHOLD})")

        print("\n  Fixed 0.01:")
        fixed_rows = evaluate(classifier, goal, on_task_examples, FIXED_THRESHOLD)
        print_rows(fixed_rows)
        fixed_results[goal] = fixed_rows

        print("\n  Calibrated:")
        calibrated_rows = evaluate(classifier, goal, on_task_examples, calibrated)
        print_rows(calibrated_rows)
        calibrated_results[goal] = calibrated_rows

    print("\n" + "=" * 78)
    print("  BEFORE / AFTER — accuracy per goal, fixed 0.01 vs per-goal calibration")
    print("=" * 78)
    print(f"{'Goal':<45} {'Fixed 0.01':>12} {'Calibrated':>12}")
    print("-" * 78)

    total_fixed_correct = total_fixed_total = 0
    total_cal_correct = total_cal_total = 0

    for goal in GOALS:
        fc, ft = accuracy(fixed_results[goal])
        cc, ct = accuracy(calibrated_results[goal])
        total_fixed_correct += fc
        total_fixed_total += ft
        total_cal_correct += cc
        total_cal_total += ct
        print(f"{goal:<45} {fc}/{ft:<9} {fc/ft:>6.1%}  {cc}/{ct:<9} {cc/ct:>6.1%}")

    print("-" * 78)
    print(
        f"{'OVERALL':<45} "
        f"{total_fixed_correct}/{total_fixed_total:<9} {total_fixed_correct/total_fixed_total:>6.1%}  "
        f"{total_cal_correct}/{total_cal_total:<9} {total_cal_correct/total_cal_total:>6.1%}"
    )

    print("\n" + "=" * 78)
    print("  CALIBRATED THRESHOLDS USED")
    print("=" * 78)
    for goal, t in calibrated_thresholds.items():
        print(f"  {goal:<45} -> {t:.4f}")

    print("\n" + "=" * 78)
    print("  CONCLUSION")
    print("=" * 78)
    control_goal = "Writing my dissertation literature review"
    control_fixed_acc = accuracy(fixed_results[control_goal])
    control_cal_acc = accuracy(calibrated_results[control_goal])
    control_fixed_pct = control_fixed_acc[0] / control_fixed_acc[1]
    control_cal_pct = control_cal_acc[0] / control_cal_acc[1]

    print(
        f"Control goal ('{control_goal}'): fixed 0.01 = {control_fixed_pct:.0%}, "
        f"calibrated = {control_cal_pct:.0%}."
    )
    print(
        f"Overall across all three goals: fixed 0.01 = {total_fixed_correct/total_fixed_total:.0%}, "
        f"calibrated = {total_cal_correct/total_cal_total:.0%}."
    )


if __name__ == "__main__":
    run()
