# tests/evaluate_worklike_fix.py
"""
Measures the work-like distractions fix: adding a goal-agnostic bank of
"looks like work but is not the stated goal" windows (email, other documents,
admin, system chrome) to the calibration population, on top of the generic
entertainment/social/shopping bank in workspace_samples.json.

The motivation was live: with the goal "Writing my project proposal", an
unrelated CV in Microsoft Word and a Terminal window both scored above a
threshold calibrated against entertainment alone, and read as on task.

The risk is the one that sank the goal-specific hard negative experiment
(see tests/evaluate_calibration_fix.py and notes section 8): adding samples
raises the threshold, and a higher threshold catches more real distractions
but also flags more genuine work as distraction. That experiment halved the
leak and nearly tripled the false alarm rate, which is the wrong trade for a
focus tool. This script exists to check whether this fix does the same thing,
before it is trusted.

Reports BEFORE and AFTER, per goal and pooled, on two datasets kept apart:

  IN SAMPLE  evaluation_set.json         6 goals.
  HELD OUT   evaluation_set_holdout.json 3 goals in unseen domains.

Unlike the hard negative experiment, the work-like bank is goal-agnostic: it
was not written from any of these goals' vocabulary. So neither dataset is
"in sample" with respect to the bank itself, and the in sample / held out
split is carried over here only for comparability with the prior report.

Two rates are reported side by side throughout. The leak rate is the benefit,
the false alarm rate is the cost, and the cost is the reason the fix might not
be worth shipping, so it is never reported on its own.

BEFORE arm: calibrate_threshold(..., work_like_path=None), the generic bank
alone, which is exactly the behaviour shipped before this change.
AFTER arm:  calibrate_threshold(...) at its current default, generic bank plus
the work-like bank. Both arms leave templates_path at its shipped default of
None, so the rejected hard negative experiment is off in both and this
measures the work-like bank's effect in isolation.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier import WorkspaceClassifier
from engine import calibrate_threshold, load_work_like_distractions, WORK_LIKE_DISTRACTIONS_PATH

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'workspace_dumps')
IN_SAMPLE_PATH = os.path.join(DATA_DIR, 'evaluation_set.json')
HELD_OUT_PATH = os.path.join(DATA_DIR, 'evaluation_set_holdout.json')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'worklike_fix_evaluation.json')

EXPECTED_ON_TASK = {"focused": True, "distraction": False, "hard_negative": False}
CATEGORIES = ["focused", "distraction", "hard_negative"]

# The two windows observed live tonight that were wrongly read as on task
# against this goal, before the fix.
LIVE_GOAL = "Writing my project proposal"
LIVE_OBSERVED = [
    "Microsoft Word - Areeba_Abid_AIAutomation_CV",
    "Terminal - CogAlign_copy osascript python3 src/app.py 80x24",
]


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

        before_threshold = calibrate_threshold(classifier, goal, work_like_path=None)
        after_threshold = calibrate_threshold(classifier, goal)

        scored = score_goal(classifier, goal_block)
        before = apply_threshold(scored, before_threshold)
        after = apply_threshold(scored, after_threshold)
        pooled_before.extend(before)
        pooled_after.extend(after)

        per_goal.append({
            "goal": goal,
            "domain": goal_block.get("domain"),
            "threshold_before": round(float(before_threshold), 4),
            "threshold_after": round(float(after_threshold), 4),
            "threshold_delta": round(float(after_threshold) - float(before_threshold), 4),
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


def evaluate_live_observations(classifier):
    """
    The two windows observed live tonight, scored against the live goal under
    both arms. Kept deliberately outside the formal datasets: n is 2, they were
    hand picked because they failed, and they are the examples that motivated
    the fix, so they cannot also be evidence that it works. Reported for
    traceability, not as a result.
    """
    before_threshold = float(calibrate_threshold(classifier, LIVE_GOAL, work_like_path=None))
    after_threshold = float(calibrate_threshold(classifier, LIVE_GOAL))
    scores = classifier.score_batch(LIVE_GOAL, LIVE_OBSERVED)

    rows = []
    for sample, score in zip(LIVE_OBSERVED, scores):
        score = float(score)
        rows.append({
            "sample": sample,
            "score": round(score, 4),
            "expected": "off task",
            "before": {
                "threshold": round(before_threshold, 4),
                "verdict": "on task" if score > before_threshold else "off task",
                "correct": bool(score <= before_threshold),
                "margin": round(score - before_threshold, 4),
            },
            "after": {
                "threshold": round(after_threshold, 4),
                "verdict": "on task" if score > after_threshold else "off task",
                "correct": bool(score <= after_threshold),
                "margin": round(score - after_threshold, 4),
            },
        })
    return {
        "goal": LIVE_GOAL,
        "threshold_before": round(before_threshold, 4),
        "threshold_after": round(after_threshold, 4),
        "note": "Motivating examples, not evidence. n=2, hand picked because they failed.",
        "observations": rows,
    }


def print_live_report(live):
    print()
    print("=" * 104)
    print("  SEPARATE SECTION, NOT PART OF THE FORMAL DATASET")
    print("  Two windows observed live tonight. These motivated the fix, so they cannot also justify it.")
    print("=" * 104)
    print(f"\n  Goal: '{live['goal']}'")
    print(f"  Calibrated threshold: before {live['threshold_before']:.4f}, after {live['threshold_after']:.4f}")
    print(f"\n{'Window':<58} {'Score':>8} {'BEFORE':>16} {'AFTER':>16}")
    print("-" * 104)
    for r in live["observations"]:
        b, a = r["before"], r["after"]
        bmark = "OK" if b["correct"] else "WRONG"
        amark = "OK" if a["correct"] else "WRONG"
        print(f"{r['sample'][:57]:<58} {r['score']:>8.4f} {b['verdict']:>9} {bmark:>6} {a['verdict']:>9} {amark:>6}")
    print("-" * 104)
    print("  Both are expected to read OFF TASK. 'margin' in the JSON is score minus threshold,")
    print("  so a negative margin after the fix means it clears the line structurally, not marginally.")
    for r in live["observations"]:
        print(f"    {r['sample'][:50]:<52} margin before {r['before']['margin']:+.4f}, after {r['after']['margin']:+.4f}")


def run_evaluation():
    print("=" * 104)
    print("   COGALIGN: WORK LIKE DISTRACTIONS FIX")
    print("=" * 104)

    bank = load_work_like_distractions()
    print(f"\nWork-like bank: {len(bank)} samples from {os.path.relpath(WORK_LIKE_DISTRACTIONS_PATH, os.path.join(os.path.dirname(__file__), '..'))}")
    if not bank:
        print("!! Empty bank. The AFTER arm is identical to BEFORE, results are meaningless.")

    classifier = WorkspaceClassifier()

    print("\nIn sample set:")
    in_sample = evaluate_dataset(classifier, IN_SAMPLE_PATH, "IN SAMPLE, evaluation_set.json")
    print("\nHeld out set:")
    held_out = evaluate_dataset(classifier, HELD_OUT_PATH, "HELD OUT, evaluation_set_holdout.json")

    print_dataset_report(
        in_sample,
        "Six goals. The work-like bank is goal-agnostic, so it was not tuned on these either.",
    )
    print_dataset_report(
        held_out,
        "Three unseen domains, hard negatives written fresh. Carried over for comparability with the prior report.",
    )

    live = evaluate_live_observations(classifier)
    print_live_report(live)

    output = {
        "fix": "calibrate_threshold() adds a goal-agnostic work-like distraction bank as a third calibration population",
        "work_like_bank": "data/workspace_dumps/work_like_distractions.json",
        "work_like_bank_n": len(bank),
        "before_arm": "calibrate_threshold(..., work_like_path=None), generic bank alone, the previously shipped behaviour",
        "after_arm": "calibrate_threshold(...) at current default, generic bank plus work-like bank",
        "both_arms": "templates_path left at its shipped default of None, so generated hard negatives are off in both arms",
        "in_sample": in_sample,
        "held_out": held_out,
        "live_observations": live,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to: results/worklike_fix_evaluation.json")
    print("   (classifier_evaluation.json, classifier_evaluation_multigoal.json and")
    print("    classifier_calibration_fix.json all untouched)")


if __name__ == "__main__":
    run_evaluation()
