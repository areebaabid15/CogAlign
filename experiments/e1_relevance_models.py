"""E1. Relevance-model comparison on the labelled control set.

Compares the three candidate relevance scorers on the same 50 focused + 50
distraction window texts against the control goal:

  mxbai      cross-encoder  mixedbread-ai/mxbai-rerank-xsmall-v1   (deployed)
  msmarco    cross-encoder  cross-encoder/ms-marco-MiniLM-L-6-v2
  minilm_bi  bi-encoder     sentence-transformers/all-MiniLM-L6-v2  (cosine similarity)

For each model it records, per sample: the model's default output, its raw
(identity-activation) output for cross-encoders, and the score production's
WorkspaceClassifier._normalize() would produce. Threshold-free metrics (ROC-AUC,
Mann-Whitney U) are computed from these later, so the comparison does not
depend on the models' very different score scales.

It also records cold load time, warm single-call latency at several input
lengths (production now scores window title plus on-screen text, so input
length matters), parameter count, and peak memory. Each model runs in its own
subprocess so the memory figure is not contaminated by the others.

Output: results/final_eval/e1_<model>.json
Usage:  python experiments/e1_relevance_models.py            (all three, about 5 min)
        python experiments/e1_relevance_models.py --model mxbai
"""
import argparse
import json
import os
import subprocess
import sys
import time

from _common import CONTROL_GOAL, banner, load_samples, out_path, peak_rss_mb, run_meta, save_json, summarize

MODELS = {
    "mxbai": ("cross", "mixedbread-ai/mxbai-rerank-xsmall-v1"),
    "msmarco": ("cross", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "minilm_bi": ("bi", "sentence-transformers/all-MiniLM-L6-v2"),
}

TITLE = "Safari - Digital distraction and self-regulation - Google Scholar"
FILLER = (
    "Search results list recent studies on attention, task switching and interruption "
    "in knowledge work, including field studies of office workers, diary studies of "
    "task resumption, and experiments on the timing of notifications. Each result shows "
    "the title, authors, venue, year, a short abstract snippet and the number of citations. "
)
LENGTH_VARIANTS_WORDS = [0, 60, 120, 240, 480]  # words of on-screen text appended to the title
N_WARMUP, N_TIMED = 3, 30


def text_of_length(n_words):
    if n_words == 0:
        return TITLE
    words = []
    filler_words = FILLER.split()
    while len(words) < n_words:
        words.extend(filler_words)
    return TITLE + " " + " ".join(words[:n_words])


def run_one(key):
    kind, name = MODELS[key]
    samples = load_samples()
    texts = samples["focused"] + samples["distraction"]
    labels = ["focused"] * len(samples["focused"]) + ["distraction"] * len(samples["distraction"])
    rss_before = peak_rss_mb()

    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    result = {"model_key": key, "model_name": name, "kind": kind, "goal": CONTROL_GOAL, "meta": run_meta()}

    t0 = time.perf_counter()
    if kind == "cross":
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(name)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(name)
    result["load_seconds"] = time.perf_counter() - t0

    hf_model = model.model if kind == "cross" else model
    try:
        result["parameters"] = int(sum(p.numel() for p in hf_model.parameters()))
    except Exception as e:
        result["parameters"] = f"unavailable ({e})"
    result["max_length"] = getattr(model, "max_length", None) or getattr(model, "max_seq_length", None)

    rows = []
    if kind == "cross":
        act = getattr(model, "activation_fn", None) or getattr(model, "default_activation_function", None)
        result["default_activation"] = type(act).__name__ if act is not None else None
        pairs = [[CONTROL_GOAL, t] for t in texts]
        default = model.predict(pairs, batch_size=32, show_progress_bar=False)
        try:
            raw = model.predict(pairs, batch_size=32, show_progress_bar=False, activation_fn=torch.nn.Identity())
        except TypeError:
            raw = model.predict(pairs, batch_size=32, show_progress_bar=False, activation_fct=torch.nn.Identity())
        from classifier import WorkspaceClassifier
        production = [float(WorkspaceClassifier._normalize(float(s))) for s in default]
        inside = [0.0 <= float(s) <= 1.0 for s in default]
        result["default_outputs_inside_unit_interval"] = int(sum(inside))
        result["normalize_mixes_identity_and_sigmoid"] = bool(0 < sum(inside) < len(inside))
        for t, lab, d, r, p in zip(texts, labels, default, raw, production):
            rows.append({"text": t, "label": lab, "default": float(d), "raw": float(r), "production": p})
    else:
        from sentence_transformers import util
        g = model.encode(CONTROL_GOAL, convert_to_tensor=True, normalize_embeddings=True)
        e = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        cos = util.cos_sim(g, e)[0].tolist()
        for t, lab, c in zip(texts, labels, cos):
            rows.append({"text": t, "label": lab, "default": float(c), "raw": float(c), "production": None})
    result["samples"] = rows

    # Warm single-call latency, the way production calls it: one pair per tick.
    latency = {}
    for n_words in LENGTH_VARIANTS_WORDS:
        text = text_of_length(n_words)
        if kind == "cross":
            call = lambda: model.predict([[CONTROL_GOAL, text]], show_progress_bar=False)
            try:
                n_tokens = len(model.tokenizer(CONTROL_GOAL, text, truncation=True,
                                               max_length=result["max_length"] or 512)["input_ids"])
            except Exception:
                n_tokens = None
        else:
            call = lambda: model.encode(text, convert_to_tensor=True, normalize_embeddings=True)  # goal embedding is cached once per session
            try:
                n_tokens = len(model.tokenizer(text, truncation=True, max_length=result["max_length"] or 256)["input_ids"])
            except Exception:
                n_tokens = None
        for _ in range(N_WARMUP):
            call()
        times = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            call()
            times.append(time.perf_counter() - t0)
        latency[str(n_words)] = {"appended_words": n_words, "input_tokens": n_tokens, **summarize(times)}
        print(f"  {key:9s} +{n_words:3d} words ({n_tokens} tokens): median {latency[str(n_words)]['median']*1000:7.1f} ms")
    result["latency_single_call"] = latency
    result["peak_rss_mb"] = peak_rss_mb()
    result["peak_rss_mb_before_import"] = rss_before
    save_json(f"e1_{key}.json", result)


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(pos, neg, alternative="greater")
    return u / (len(pos) * len(neg)), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    args = ap.parse_args()
    if args.model:
        run_one(args.model)
        return

    banner("E1  Relevance-model comparison (each model in its own process)")
    for key in MODELS:
        print(f"\n--- {key}: {MODELS[key][1]}")
        rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--model", key]).returncode
        if rc != 0:
            print(f"  !! {key} failed with exit code {rc}, continuing with the others")

    print("\nSummary (threshold-free, higher AUC = better separation):")
    for key in MODELS:
        path = out_path(f"e1_{key}.json")
        if not os.path.exists(path):
            print(f"  {key:9s} no result")
            continue
        r = json.load(open(path))
        pos = [s["default"] for s in r["samples"] if s["label"] == "focused"]
        neg = [s["default"] for s in r["samples"] if s["label"] == "distraction"]
        a, p = auc(pos, neg)
        med = r["latency_single_call"]["0"]["median"] * 1000
        print(f"  {key:9s} AUC {a:.3f} (Mann-Whitney p {p:.1e})  load {r['load_seconds']:.1f}s  "
              f"call {med:.1f} ms  params {r['parameters']:,}  peak RSS {r['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
