# tests/evaluate_classifier.py
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier import WorkspaceClassifier

GOAL = "Writing my dissertation literature review"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'workspace_dumps', 'workspace_samples.json')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'classifier_evaluation.json')

def evaluate_at_threshold(classifier, data, threshold):
    results = []

    for sample in data["focused"]:
        predicted, score = classifier.is_related(sample, GOAL, threshold=threshold)
        results.append({
            "sample": sample,
            "label": "focused",
            "expected": True,
            "predicted": predicted,
            "score": score,
            "correct": predicted == True
        })

    for sample in data["distraction"]:
        predicted, score = classifier.is_related(sample, GOAL, threshold=threshold)
        results.append({
            "sample": sample,
            "label": "distraction",
            "expected": False,
            "predicted": predicted,
            "score": score,
            "correct": predicted == False
        })

    tp = sum(1 for r in results if r["expected"] == True  and r["predicted"] == True)
    tn = sum(1 for r in results if r["expected"] == False and r["predicted"] == False)
    fp = sum(1 for r in results if r["expected"] == False and r["predicted"] == True)
    fn = sum(1 for r in results if r["expected"] == True  and r["predicted"] == False)

    total = len(results)
    accuracy  = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "threshold": threshold,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "per_sample": results
    }


def run_evaluation():
    print("=" * 55)
    print("   COGALIGN — CLASSIFIER THRESHOLD SWEEP")
    print("=" * 55)
    print(f"Goal: '{GOAL}'\n")

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    classifier = WorkspaceClassifier()

    thresholds = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]
    all_results = []

    print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}")
    print("-" * 75)

    for t in thresholds:
        r = evaluate_at_threshold(classifier, data, t)
        all_results.append(r)
        print(f"{r['threshold']:>10.2f} {r['accuracy']:>10.1%} {r['precision']:>10.1%} {r['recall']:>10.1%} {r['f1']:>10.1%} {r['tp']:>5} {r['tn']:>5} {r['fp']:>5} {r['fn']:>5}")

    # Find best F1
    best = max(all_results, key=lambda x: x["f1"])
    print(f"\n✅ Best threshold by F1: {best['threshold']} — F1: {best['f1']:.1%}, Accuracy: {best['accuracy']:.1%}")

    # Save full results
    output = {
        "goal": GOAL,
        "sweep": all_results
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Full results saved to: results/classifier_evaluation.json")


if __name__ == "__main__":
    run_evaluation()