# tests/diagnose_goal_sensitivity.py
"""
Diagnostic tool, not a pytest test: does WorkspaceClassifier's threshold (0.01,
picked in results/classifier_evaluation.json) generalize to goals it was never
calibrated against, or does it only work reliably for the one goal
("Writing my dissertation literature review") that sweep actually used?

Tests three goals spanning different domains against clear, substantive window
text on both sides (on-task examples tailored per goal, the same off-task
examples reused across all three so any variation is attributable to the goal
text, not the window text). Deliberately avoids empty/near-empty window titles,
that is the separate, already-documented window_tracker.py issue, not this one.

Usage: python3 tests/diagnose_goal_sensitivity.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier import WorkspaceClassifier

THRESHOLD = 0.01

# Held constant across all three goals so a change in whether these get
# correctly rejected is attributable to the goal text, not the window text.
OFF_TASK = [
    "YouTube - funny cat videos",
    "Instagram - Explore Page",
    "Reddit - r/all",
]

GOALS = {
    # Control: the one goal results/classifier_evaluation.json was actually
    # calibrated against. Expected to behave correctly.
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


def run():
    print("=" * 72)
    print("  COGALIGN — GOAL SENSITIVITY DIAGNOSTIC")
    print("  Does threshold=0.01 generalize beyond the one goal it was")
    print("  calibrated against?")
    print("=" * 72)

    classifier = WorkspaceClassifier()
    results = []  # (goal, window, label, score, predicted, correct)

    for goal, on_task_examples in GOALS.items():
        print(f"\n--- Goal: '{goal}' ---")

        for window in on_task_examples:
            predicted, score = classifier.is_related(window, goal, threshold=THRESHOLD)
            correct = predicted is True
            results.append((goal, window, "on-task", score, predicted, correct))
            status = "correct  " if correct else "WRONG    "
            print(f"  [{status}] on-task  score={score:.4f} predicted={str(predicted):5} -> '{window}'")

        for window in OFF_TASK:
            predicted, score = classifier.is_related(window, goal, threshold=THRESHOLD)
            correct = predicted is False
            results.append((goal, window, "off-task", score, predicted, correct))
            status = "correct  " if correct else "WRONG    "
            print(f"  [{status}] off-task score={score:.4f} predicted={str(predicted):5} -> '{window}'")

    print("\n" + "=" * 72)
    print("  SUMMARY — accuracy per goal at threshold=0.01")
    print("=" * 72)
    print(f"{'Goal':<45} {'Correct':>9} {'Total':>7} {'Accuracy':>10}")
    print("-" * 72)

    per_goal_correct, per_goal_total = {}, {}
    for goal, _, _, _, _, correct in results:
        per_goal_total[goal] = per_goal_total.get(goal, 0) + 1
        per_goal_correct[goal] = per_goal_correct.get(goal, 0) + (1 if correct else 0)

    for goal in GOALS:
        correct, total = per_goal_correct[goal], per_goal_total[goal]
        print(f"{goal:<45} {correct:>9} {total:>7} {correct / total:>9.1%}")

    total_correct = sum(per_goal_correct.values())
    total_total = sum(per_goal_total.values())
    print("-" * 72)
    print(f"{'OVERALL':<45} {total_correct:>9} {total_total:>7} {total_correct / total_total:>9.1%}")

    print("\n" + "=" * 72)
    print("  CONCLUSION")
    print("=" * 72)
    control_goal = "Writing my dissertation literature review"
    control_acc = per_goal_correct[control_goal] / per_goal_total[control_goal]
    other_goals = [g for g in GOALS if g != control_goal]
    other_correct = sum(per_goal_correct[g] for g in other_goals)
    other_total = sum(per_goal_total[g] for g in other_goals)
    other_acc = other_correct / other_total

    if control_acc == 1.0 and other_acc < control_acc:
        print(
            f"Threshold=0.01 is fully accurate on the control goal ({control_acc:.0%}) but "
            f"drops to {other_acc:.0%} across the two goals it was never calibrated against. "
            f"This threshold does not generalize, it is overfit to the single goal used in "
            f"results/classifier_evaluation.json."
        )
    elif control_acc == other_acc == 1.0:
        print("Threshold=0.01 was fully accurate across all three goals in this sample.")
    else:
        print(
            f"Control goal accuracy: {control_acc:.0%}. Other two goals combined accuracy: "
            f"{other_acc:.0%}. See the per-example results above for exactly which examples "
            f"were misclassified."
        )


if __name__ == "__main__":
    run()
