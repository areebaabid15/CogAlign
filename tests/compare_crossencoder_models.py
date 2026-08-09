# tests/compare_crossencoder_models.py
"""
Real comparative evidence between the two cross-encoder models associated with
this project: cross-encoder/ms-marco-MiniLM-L-6-v2 (the model named in the
report) and mixedbread-ai/mxbai-rerank-xsmall-v1 (the model actually running
in production, src/classifier.py's default). Diagnostic only, does not modify
src/classifier.py or any other production code, only loads WorkspaceClassifier
twice with different model_name values.

Measures, for each model:
1. Accuracy/F1 on the full 50 focused + 50 distraction set in
   data/workspace_dumps/workspace_samples.json against the control goal, same
   methodology as evaluate_classifier.py (threshold sweep, report the best F1),
   except the sweep range is widened and log-spaced after a real exploratory
   check showed the two models sit on very different raw score scales
   (ms-marco-MiniLM-L-6-v2's distraction scores cluster around 1e-5, three
   orders of magnitude below mxbai-rerank-xsmall-v1's). Reusing the original
   evaluate_classifier.py sweep (0.01-0.15) would only be fair to the
   production model.
2. Cross-goal generalization: reuses the three-goal setup from
   diagnose_goal_sensitivity.py, held at each model's own best-on-control
   threshold from step 1, exactly mirroring how that script demonstrated the
   production model's fixed threshold does not generalize.
3. Average wall-clock time per single-prediction is_related() call, since this
   project is CPU-only and runs three models concurrently.
4. Real parameter count read directly from each loaded model's underlying
   transformers model, not estimated from memory or a model card.

Usage: python3 tests/compare_crossencoder_models.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from classifier import WorkspaceClassifier

CONTROL_GOAL = "Writing my dissertation literature review"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'workspace_dumps', 'workspace_samples.json')

MODELS = [
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "MiniLM (named in report)"),
    ("mixedbread-ai/mxbai-rerank-xsmall-v1", "mxbai-xsmall (in production)"),
]

# Log-spaced and wide enough to cover both models' real observed score ranges
# (checked directly before writing this: ms-marco's distraction scores sit
# around 1e-5, mxbai's on-task scores go up to ~0.2), not the narrow
# evaluate_classifier.py sweep, which was only ever tuned for mxbai.
THRESHOLD_SWEEP = [
    0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
    0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
]

# Held constant across goals, same list diagnose_goal_sensitivity.py uses.
OFF_TASK = [
    "YouTube - funny cat videos",
    "Instagram - Explore Page",
    "Reddit - r/all",
]

GOALS = {
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
    "Preparing for a job interview": [
        "Notes - interview questions and answers",
        "LinkedIn - Company Research - Acme Corp",
        "Google Docs - STAR method answers draft",
    ],
}


def evaluate_at_threshold(classifier, data, goal, threshold):
    """Same tp/tn/fp/fn/accuracy/precision/recall/f1 methodology as evaluate_classifier.py."""
    results = []
    for sample in data["focused"]:
        predicted, score = classifier.is_related(sample, goal, threshold=threshold)
        results.append((True, predicted))
    for sample in data["distraction"]:
        predicted, score = classifier.is_related(sample, goal, threshold=threshold)
        results.append((False, predicted))

    tp = sum(1 for exp, pred in results if exp and pred)
    tn = sum(1 for exp, pred in results if not exp and not pred)
    fp = sum(1 for exp, pred in results if not exp and pred)
    fn = sum(1 for exp, pred in results if exp and not pred)

    total = len(results)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "threshold": threshold, "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def sweep_best(classifier, data, goal, thresholds):
    sweep = [evaluate_at_threshold(classifier, data, goal, t) for t in thresholds]
    best = max(sweep, key=lambda r: r["f1"])
    return best, sweep


def goal_accuracy(classifier, goal, on_task_examples, threshold):
    correct = 0
    total = 0
    for window in on_task_examples:
        predicted, _ = classifier.is_related(window, goal, threshold=threshold)
        correct += 1 if predicted is True else 0
        total += 1
    for window in OFF_TASK:
        predicted, _ = classifier.is_related(window, goal, threshold=threshold)
        correct += 1 if predicted is False else 0
        total += 1
    return correct, total


def measure_inference_time(classifier, goal, text, n=30):
    classifier.is_related(text, goal)  # warm-up call, excluded from timing
    start = time.perf_counter()
    for _ in range(n):
        classifier.is_related(text, goal)
    elapsed = time.perf_counter() - start
    return elapsed / n


def count_parameters(classifier):
    try:
        return sum(p.numel() for p in classifier.model.model.parameters())
    except Exception:
        return None


def run():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    model_results = {}

    for model_name, label in MODELS:
        print("=" * 78)
        print(f"  {label}: {model_name}")
        print("=" * 78)
        classifier = WorkspaceClassifier(model_name=model_name)

        print(f"\nThreshold sweep against control goal ({CONTROL_GOAL!r}):")
        best, sweep = sweep_best(classifier, data, CONTROL_GOAL, THRESHOLD_SWEEP)
        print(f"{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        for r in sweep:
            print(f"{r['threshold']:>10.5f} {r['accuracy']:>10.1%} {r['precision']:>10.1%} {r['recall']:>10.1%} {r['f1']:>10.1%}")
        print(f"\nBest for this model: threshold={best['threshold']}, F1={best['f1']:.1%}, accuracy={best['accuracy']:.1%}")

        print(f"\nCross-goal generalization, held at threshold={best['threshold']}:")
        cross_goal = {}
        for goal, examples in GOALS.items():
            correct, total = goal_accuracy(classifier, goal, examples, best["threshold"])
            cross_goal[goal] = (correct, total)
            print(f"  {goal:<45} {correct}/{total} = {correct/total:.1%}")

        avg_time = measure_inference_time(
            classifier, CONTROL_GOAL, "Microsoft Word - Literature Review Chapter 2.docx", n=30
        )
        print(f"\nAverage inference time per is_related() call (n=30, warm): {avg_time * 1000:.2f}ms")

        param_count = count_parameters(classifier)
        print(f"Parameter count (measured from the loaded model): "
              f"{param_count:,}" if param_count else "Parameter count: could not measure")

        model_results[model_name] = {
            "label": label, "best": best, "cross_goal": cross_goal,
            "avg_time": avg_time, "param_count": param_count,
        }

        del classifier

    print("\n" + "=" * 78)
    print("  SIDE BY SIDE COMPARISON")
    print("=" * 78)

    names = [m for m, _ in MODELS]
    col_w = 34
    print(f"{'':<32}{names[0]:<{col_w}}{names[1]:<{col_w}}")
    r0, r1 = model_results[names[0]], model_results[names[1]]

    def row(label, get):
        print(f"{label:<32}{str(get(r0)):<{col_w}}{str(get(r1)):<{col_w}}")

    row("Best threshold (control goal)", lambda r: r["best"]["threshold"])
    row("Best F1 (control goal)", lambda r: f"{r['best']['f1']:.1%}")
    row("Best accuracy (control goal)", lambda r: f"{r['best']['accuracy']:.1%}")
    row("Best precision (control goal)", lambda r: f"{r['best']['precision']:.1%}")
    row("Best recall (control goal)", lambda r: f"{r['best']['recall']:.1%}")
    for goal in GOALS:
        c0, t0 = r0["cross_goal"][goal]
        c1, t1 = r1["cross_goal"][goal]
        label = f"  Cross-goal: {goal}"
        print(f"{label:<32}{f'{c0}/{t0} = {c0/t0:.1%}':<{col_w}}{f'{c1}/{t1} = {c1/t1:.1%}':<{col_w}}")
    row("Avg inference time / call", lambda r: f"{r['avg_time'] * 1000:.2f}ms")
    row("Parameter count", lambda r: f"{r['param_count']:,}" if r['param_count'] else "unknown")

    print("\n" + "=" * 78)
    print("  CONCLUSION")
    print("=" * 78)

    ms, mx = model_results[names[0]], model_results[names[1]]
    ms_overall_correct = sum(c for c, t in ms["cross_goal"].values()) + 0
    ms_overall_total = sum(t for c, t in ms["cross_goal"].values())
    mx_overall_correct = sum(c for c, t in mx["cross_goal"].values())
    mx_overall_total = sum(t for c, t in mx["cross_goal"].values())

    print(f"ms-marco-MiniLM-L-6-v2: control F1={ms['best']['f1']:.1%}, "
          f"cross-goal overall={ms_overall_correct}/{ms_overall_total} "
          f"({ms_overall_correct/ms_overall_total:.1%}), "
          f"{ms['avg_time']*1000:.2f}ms/call, {ms['param_count']:,} parameters.")
    print(f"mxbai-rerank-xsmall-v1: control F1={mx['best']['f1']:.1%}, "
          f"cross-goal overall={mx_overall_correct}/{mx_overall_total} "
          f"({mx_overall_correct/mx_overall_total:.1%}), "
          f"{mx['avg_time']*1000:.2f}ms/call, {mx['param_count']:,} parameters.")


if __name__ == "__main__":
    run()
