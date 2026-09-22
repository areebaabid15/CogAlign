# tests/evaluate_classifier_multigoal.py
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier import WorkspaceClassifier
from engine import calibrate_threshold, DEFAULT_THRESHOLD

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'workspace_dumps', 'evaluation_set.json')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'classifier_evaluation_multigoal.json')

# "focused" is the only category the system should call ON TASK. Both
# negative categories are kept separate rather than merged, because the
# whole point of this set is to report hard_negative on its own.
EXPECTED_ON_TASK = {"focused": True, "distraction": False, "hard_negative": False}
CATEGORIES = ["focused", "distraction", "hard_negative"]


def score_goal(classifier, goal_block):
    """
    Scores every sample for one goal exactly once. Scores don't depend on the
    threshold, only the predictions do, so scoring happens here and every
    threshold is then applied to the cached numbers. Avoids running the
    cross-encoder twice over the same 24 samples.
    """
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
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def category_breakdown(results):
    """
    Per-category leak rate. For the two negative categories this is the
    fraction wrongly called ON TASK, which for hard_negative is the exact
    number this evaluation exists to produce. For "focused" the same field
    is the fraction correctly called on task, i.e. recall.
    """
    breakdown = {}
    for category in CATEGORIES:
        rows = [r for r in results if r["label"] == category]
        if not rows:
            continue
        on_task = sum(1 for r in rows if r["predicted"] == True)
        breakdown[category] = {
            "n": len(rows),
            "predicted_on_task": on_task,
            "rate_on_task": round(on_task / len(rows), 4),
            "mean_score": round(sum(r["score"] for r in rows) / len(rows), 4),
        }
    return breakdown


def evaluate_at_threshold(scored, threshold):
    results = [
        {**r, "predicted": bool(r["score"] > threshold), "correct": bool(r["score"] > threshold) == r["expected"]}
        for r in scored
    ]
    return {
        "threshold": round(float(threshold), 4),
        **metrics_from(results),
        "by_category": category_breakdown(results),
        "per_sample": results,
    }


def print_metric_table(title, rows):
    print(f"\n{title}")
    print(f"{'Goal':<45} {'Thresh':>7} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("-" * 87)
    for label, r in rows:
        print(f"{label:<45} {r['threshold']:>7.4f} {r['accuracy']:>8.1%} {r['precision']:>8.1%} {r['recall']:>8.1%} {r['f1']:>8.1%}")


def print_hard_negative_table(rows):
    print(f"\nHARD NEGATIVE LEAK RATE (off task samples wrongly called ON TASK)")
    print(f"{'Goal':<45} {'Fixed 0.01':>18} {'Calibrated':>18}")
    print("-" * 83)
    for label, fixed, calib in rows:
        f = fixed["by_category"].get("hard_negative")
        c = calib["by_category"].get("hard_negative")
        if not f or not c:
            continue
        print(f"{label:<45} {f['predicted_on_task']:>6}/{f['n']:<3} {f['rate_on_task']:>7.1%} "
              f"{c['predicted_on_task']:>6}/{c['n']:<3} {c['rate_on_task']:>7.1%}")


def run_evaluation():
    print("=" * 87)
    print("   COGALIGN — MULTI GOAL CLASSIFIER EVALUATION (held out set)")
    print("=" * 87)

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    goals = data["goals"]
    classifier = WorkspaceClassifier()

    per_goal = []
    pooled_scored = []

    for goal_block in goals:
        goal = goal_block["goal"]
        print(f"\nScoring '{goal}' ...")

        # Per-goal threshold from the live calibration path, imported from
        # engine.py rather than reimplemented, so this measures what actually
        # ships. It calibrates against workspace_samples.json, a different
        # file from the one being evaluated here, which is the whole point.
        calibrated = calibrate_threshold(classifier, goal)

        scored = score_goal(classifier, goal_block)
        pooled_scored.extend(scored)

        per_goal.append({
            "goal": goal,
            "domain": goal_block.get("domain"),
            "n_samples": len(scored),
            "calibrated_threshold": round(float(calibrated), 4),
            "fixed": evaluate_at_threshold(scored, DEFAULT_THRESHOLD),
            "calibrated": evaluate_at_threshold(scored, calibrated),
        })

    # Pooled figures use the fixed threshold for the fixed row. There is no
    # single calibrated threshold across goals by definition, so the pooled
    # calibrated row re-uses each sample's own goal threshold instead.
    pooled_fixed = evaluate_at_threshold(pooled_scored, DEFAULT_THRESHOLD)

    pooled_calibrated_rows = []
    for g, block in zip(per_goal, goals):
        pooled_calibrated_rows.extend(g["calibrated"]["per_sample"])
    pooled_calibrated = {
        "threshold": float("nan"),
        **metrics_from(pooled_calibrated_rows),
        "by_category": category_breakdown(pooled_calibrated_rows),
    }

    print_metric_table(
        f"FIXED THRESHOLD {DEFAULT_THRESHOLD}",
        [(g["goal"][:44], g["fixed"]) for g in per_goal],
    )
    print("-" * 87)
    print(f"{'POOLED (all goals)':<45} {pooled_fixed['threshold']:>7.4f} {pooled_fixed['accuracy']:>8.1%} "
          f"{pooled_fixed['precision']:>8.1%} {pooled_fixed['recall']:>8.1%} {pooled_fixed['f1']:>8.1%}")

    print_metric_table(
        "PER GOAL CALIBRATED THRESHOLD (engine.calibrate_threshold)",
        [(g["goal"][:44], g["calibrated"]) for g in per_goal],
    )
    print("-" * 87)
    print(f"{'POOLED (each goal at its own threshold)':<45} {'  n/a':>7} {pooled_calibrated['accuracy']:>8.1%} "
          f"{pooled_calibrated['precision']:>8.1%} {pooled_calibrated['recall']:>8.1%} {pooled_calibrated['f1']:>8.1%}")

    print_hard_negative_table([(g["goal"][:44], g["fixed"], g["calibrated"]) for g in per_goal])
    print("-" * 83)
    pf = pooled_fixed["by_category"]["hard_negative"]
    pc = pooled_calibrated["by_category"]["hard_negative"]
    print(f"{'POOLED':<45} {pf['predicted_on_task']:>6}/{pf['n']:<3} {pf['rate_on_task']:>7.1%} "
          f"{pc['predicted_on_task']:>6}/{pc['n']:<3} {pc['rate_on_task']:>7.1%}")

    print(f"\nFor comparison, plain distraction leak rate, pooled: "
          f"fixed {pooled_fixed['by_category']['distraction']['rate_on_task']:.1%}, "
          f"calibrated {pooled_calibrated['by_category']['distraction']['rate_on_task']:.1%}")

    output = {
        "dataset": "data/workspace_dumps/evaluation_set.json",
        "calibration_source": "data/workspace_dumps/workspace_samples.json (via engine.calibrate_threshold)",
        "fixed_threshold": DEFAULT_THRESHOLD,
        "n_goals": len(goals),
        "n_samples": len(pooled_scored),
        "per_goal": per_goal,
        "pooled": {
            "fixed": {k: v for k, v in pooled_fixed.items() if k != "per_sample"},
            "calibrated_per_goal": {k: v for k, v in pooled_calibrated.items() if k != "threshold"},
        },
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Full results saved to: results/classifier_evaluation_multigoal.json")
    print("   (results/classifier_evaluation.json is untouched)")


if __name__ == "__main__":
    run_evaluation()
