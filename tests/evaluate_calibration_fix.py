# tests/evaluate_calibration_fix.py
"""
Measures the calibration fix: synthesising goal-specific hard negatives at
session start and letting the calibrated threshold be drawn above them,
instead of above the generic distraction bank alone.

Reports BEFORE and AFTER, per goal and pooled, on two datasets kept strictly
apart:

  IN SAMPLE  evaluation_set.json         6 goals. Its hard negatives were
                                         written by the same author as the
                                         trap templates, so improvement here
                                         is partly an artifact.
  HELD OUT   evaluation_set_holdout.json 3 goals in unseen domains, hard
                                         negatives written fresh. This is the
                                         honest number.

Two rates are reported side by side throughout. The leak rate is the benefit,
the false alarm rate is the cost, and the cost is the reason the fix might not
be worth shipping, so it is never reported on its own.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier import WorkspaceClassifier
from engine import calibrate_threshold, extract_content_words, generate_hard_negatives

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'workspace_dumps')
IN_SAMPLE_PATH = os.path.join(DATA_DIR, 'evaluation_set.json')
HELD_OUT_PATH = os.path.join(DATA_DIR, 'evaluation_set_holdout.json')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'classifier_calibration_fix.json')

# To measure the BEFORE threshold, calibrate_threshold() is called with a
# templates path that does not exist. generate_hard_negatives() swallows the
# failure and returns [], so the function falls through to scoring the generic
# bank alone, which is exactly the pre-fix behaviour. This exercises the real
# shipped function on its real fallback path rather than reimplementing the
# old logic, so the two arms cannot drift apart.
NO_TEMPLATES_PATH = os.path.join(DATA_DIR, '__no_such_templates_file__.json')

EXPECTED_ON_TASK = {"focused": True, "distraction": False, "hard_negative": False}
CATEGORIES = ["focused", "distraction", "hard_negative"]


def score_goal(classifier, goal_block):
    """Scores every sample for one goal once. Thresholds are applied to the
    cached scores afterwards, so the cross-encoder runs once per sample, not
    once per sample per threshold arm."""
    scored = []
    for category in CATEGORIES:
        samples = goal_block.get(category, [])
        if not samples:
            continue
        scores = classifier.score_batch(goal_block["goal"], samples)
        for sample, score in zip(samples, scores):
            scored.append({
                "sample": sample,
                "label": category,
                "expected": EXPECTED_ON_TASK[category],
                "score": float(score),
            })
    return scored


def metrics_from(results):
    tp = sum(1 for r in results if r["expected"] == True  and r["predicted"] == True)
    tn = sum(1 for r in results if r["expected"] == False and r["predicted"] == False)
    fp = sum(1 for r in results if r["expected"] == False and r["predicted"] == True)
    fn = sum(1 for r in results if r["expected"] == True  and r["predicted"] == False)

    total = len(results)
    accuracy  = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def rates_from(results):
    """
    The two headline rates.

    leak_rate:        hard negatives wrongly called ON TASK. The benefit. The
                      system stays silent while the user is off task.
    false_alarm_rate: focused samples wrongly called OFF TASK. The cost. The
                      system nudges while the user is genuinely working.
    """
    hard = [r for r in results if r["label"] == "hard_negative"]
    focused = [r for r in results if r["label"] == "focused"]
    distraction = [r for r in results if r["label"] == "distraction"]

    leaked = sum(1 for r in hard if r["predicted"] == True)
    false_alarms = sum(1 for r in focused if r["predicted"] == False)
    distraction_leaked = sum(1 for r in distraction if r["predicted"] == True)

    return {
        "hard_negative_n": len(hard),
        "hard_negative_leaked": leaked,
        "leak_rate": round(leaked / len(hard), 4) if hard else 0.0,
        "focused_n": len(focused),
        "focused_false_alarms": false_alarms,
        "false_alarm_rate": round(false_alarms / len(focused), 4) if focused else 0.0,
        "distraction_n": len(distraction),
        "distraction_leaked": distraction_leaked,
        "distraction_leak_rate": round(distraction_leaked / len(distraction), 4) if distraction else 0.0,
    }


def apply_threshold(scored, threshold):
    return [
        {**r, "predicted": bool(r["score"] > threshold), "correct": bool(r["score"] > threshold) == r["expected"]}
        for r in scored
    ]


def evaluate_dataset(classifier, path, dataset_label):
    with open(path, 'r') as f:
        data = json.load(f)

    per_goal = []
    pooled_before, pooled_after = [], []

    for goal_block in data["goals"]:
        goal = goal_block["goal"]
        print(f"  scoring '{goal}' ...")

        before_threshold = calibrate_threshold(classifier, goal, templates_path=NO_TEMPLATES_PATH)
        after_threshold = calibrate_threshold(classifier, goal)

        scored = score_goal(classifier, goal_block)
        before = apply_threshold(scored, before_threshold)
        after = apply_threshold(scored, after_threshold)
        pooled_before.extend(before)
        pooled_after.extend(after)

        per_goal.append({
            "goal": goal,
            "domain": goal_block.get("domain"),
            "content_words": extract_content_words(goal),
            "n_generated_hard_negatives": len(generate_hard_negatives(goal)),
            "threshold_before": round(float(before_threshold), 4),
            "threshold_after": round(float(after_threshold), 4),
            "before": {**metrics_from(before), **rates_from(before), "per_sample": before},
            "after": {**metrics_from(after), **rates_from(after), "per_sample": after},
        })

    return {
        "dataset": dataset_label,
        "path": os.path.relpath(path, os.path.join(os.path.dirname(__file__), '..')),
        "n_goals": len(data["goals"]),
        "n_samples": len(pooled_before),
        "per_goal": per_goal,
        "pooled": {
            "before": {**metrics_from(pooled_before), **rates_from(pooled_before)},
            "after": {**metrics_from(pooled_after), **rates_from(pooled_after)},
        },
    }


def print_dataset_report(block, caveat):
    print()
    print("=" * 104)
    print(f"  {block['dataset']}  ({block['n_goals']} goals, {block['n_samples']} samples)")
    print(f"  {caveat}")
    print("=" * 104)

    print(f"\n{'Goal':<40} {'Threshold':>17}   {'LEAK RATE (benefit)':>21}   {'FALSE ALARM (cost)':>21}")
    print(f"{'':<40} {'before':>8} {'after':>8}   {'before':>10} {'after':>10}   {'before':>10} {'after':>10}")
    print("-" * 104)
    for g in block["per_goal"]:
        b, a = g["before"], g["after"]
        print(f"{g['goal'][:39]:<40} {g['threshold_before']:>8.4f} {g['threshold_after']:>8.4f}   "
              f"{b['leak_rate']:>9.1%} {a['leak_rate']:>10.1%}   "
              f"{b['false_alarm_rate']:>9.1%} {a['false_alarm_rate']:>10.1%}")
    pb, pa = block["pooled"]["before"], block["pooled"]["after"]
    print("-" * 104)
    print(f"{'POOLED':<40} {'':>8} {'':>8}   "
          f"{pb['leak_rate']:>9.1%} {pa['leak_rate']:>10.1%}   "
          f"{pb['false_alarm_rate']:>9.1%} {pa['false_alarm_rate']:>10.1%}")
    print(f"{'':<40} {'':>8} {'':>8}   "
          f"{pb['hard_negative_leaked']:>5}/{pb['hard_negative_n']:<4} {pa['hard_negative_leaked']:>5}/{pa['hard_negative_n']:<4}   "
          f"{pb['focused_false_alarms']:>5}/{pb['focused_n']:<4} {pa['focused_false_alarms']:>5}/{pa['focused_n']:<4}")

    print(f"\n{'Goal':<40} {'Accuracy':>15} {'Precision':>15} {'Recall':>15} {'F1':>15}")
    print(f"{'':<40} {'before':>7} {'after':>7} {'before':>7} {'after':>7} {'before':>7} {'after':>7} {'before':>7} {'after':>7}")
    print("-" * 104)
    for g in block["per_goal"]:
        b, a = g["before"], g["after"]
        print(f"{g['goal'][:39]:<40} {b['accuracy']:>7.1%} {a['accuracy']:>7.1%} {b['precision']:>7.1%} {a['precision']:>7.1%} "
              f"{b['recall']:>7.1%} {a['recall']:>7.1%} {b['f1']:>7.1%} {a['f1']:>7.1%}")
    print("-" * 104)
    print(f"{'POOLED':<40} {pb['accuracy']:>7.1%} {pa['accuracy']:>7.1%} {pb['precision']:>7.1%} {pa['precision']:>7.1%} "
          f"{pb['recall']:>7.1%} {pa['recall']:>7.1%} {pb['f1']:>7.1%} {pa['f1']:>7.1%}")

    print(f"\n  Plain distraction leak, pooled: before {pb['distraction_leak_rate']:.1%}, after {pa['distraction_leak_rate']:.1%}")


def run_evaluation():
    print("=" * 104)
    print("   COGALIGN — CALIBRATION FIX: GOAL SPECIFIC HARD NEGATIVES")
    print("=" * 104)

    classifier = WorkspaceClassifier()

    print("\nIn sample set:")
    in_sample = evaluate_dataset(classifier, IN_SAMPLE_PATH, "IN SAMPLE — evaluation_set.json")
    print("\nHeld out set:")
    held_out = evaluate_dataset(classifier, HELD_OUT_PATH, "HELD OUT — evaluation_set_holdout.json")

    print_dataset_report(
        in_sample,
        "Hard negatives written by the same author as the trap templates. Improvement here is partly an artifact.",
    )
    print_dataset_report(
        held_out,
        "Unseen domains, hard negatives written fresh, no template pattern reproduced. THIS IS THE HONEST NUMBER.",
    )

    output = {
        "fix": "calibrate_threshold() now scores goal-specific synthesised hard negatives alongside the generic distraction bank",
        "templates": "data/workspace_dumps/hard_negative_templates.json",
        "before_arm": "calibrate_threshold() with a non existent templates path, falling through to generic bank only",
        "in_sample": in_sample,
        "held_out": held_out,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Full results saved to: results/classifier_calibration_fix.json")
    print("   (classifier_evaluation.json and classifier_evaluation_multigoal.json untouched)")


if __name__ == "__main__":
    run_evaluation()
