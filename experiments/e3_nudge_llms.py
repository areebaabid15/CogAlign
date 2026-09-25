"""E3. Nudge-generation model comparison and nudge quality.

Generates nudges with the exact production prompts (captured from
nudge_engine.QwenNudgeEngine.generate_nudge, so nothing is re-typed) for
several goals, all three personas, and several local models served by Ollama:

  qwen2.5:1.5b   deployed model
  llama3.2:1b    benchmarked early in the project (tests/test_ollama_nudges.py)
  qwen2.5:0.5b   smaller candidate named in the report as a latency fix

Goals deliberately include non-dissertation goals, because the Level 1 and 3
prompts mention a dissertation regardless of the user's goal.

Per generation it stores the text (with production's own post-processing),
wall-clock latency and Ollama's timing counters (load, prompt and generation
durations, token counts). Quality metrics are computed afterwards from the
stored text. Resumable: re-running skips generations already on disk.

Output: results/final_eval/e3_nudges.jsonl, e3_models.json
Usage:  python experiments/e3_nudge_llms.py                 (about 60 to 90 min)
        python experiments/e3_nudge_llms.py --reps 4        (quicker)
        python experiments/e3_nudge_llms.py --models qwen2.5:1.5b
"""
import argparse
import copy
import random
import time

import requests

from _common import OLLAMA, append_jsonl, banner, ollama_has, ollama_models, ollama_post, read_jsonl, run_meta, save_json

DEFAULT_MODELS = ["qwen2.5:1.5b", "llama3.2:1b", "qwen2.5:0.5b"]

# goal -> the distraction window text passed as context (production passes
# "App - Title" of the current window; only Level 2 uses it).
GOALS = {
    "Writing my dissertation literature review": "Google Chrome - YouTube - Top 10 Funniest Cat Videos 2024",
    "Debugging my Python pipeline code": "Safari - Reddit - r/AskReddit Hot",
    "Revising for my statistics exam": "Google Chrome - Instagram - Reels",
    "Preparing for a job interview": "Safari - Amazon - Today's Lightning Deals",
    "Planning my small business budget": "Google Chrome - Netflix - Continue Watching",
}
LEVELS = [1, 2, 3]


def capture_production_payload(level, goal, distraction):
    """Run the real generate_nudge() with requests.post intercepted, and return
    exactly the payload production would send to Ollama."""
    import nudge_engine
    box = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"message": {"content": "captured"}}

    def fake_post(url, json=None, timeout=None, **kw):
        box.update({"url": url, "payload": copy.deepcopy(json), "timeout": timeout})
        return _Resp()

    original = nudge_engine.requests.post
    nudge_engine.requests.post = fake_post
    try:
        nudge_engine.QwenNudgeEngine().generate_nudge(level=level, goal=goal, distraction=distraction)
    finally:
        nudge_engine.requests.post = original
    return box


def generate(model, payload, timeout=300):
    p = copy.deepcopy(payload)
    p["model"] = model
    t0 = time.perf_counter()
    r = requests.post(OLLAMA + "/api/chat", json=p, timeout=timeout)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    j = r.json()
    raw = j.get("message", {}).get("content", "")
    return {
        "raw_text": raw,
        "text": raw.strip().strip("\"'"),  # identical to production post-processing
        "wall_s": wall,
        "total_s": (j.get("total_duration") or 0) / 1e9,
        "load_s": (j.get("load_duration") or 0) / 1e9,
        "prompt_tokens": j.get("prompt_eval_count"),
        "prompt_eval_s": (j.get("prompt_eval_duration") or 0) / 1e9,
        "output_tokens": j.get("eval_count"),
        "eval_s": (j.get("eval_duration") or 0) / 1e9,
        "done_reason": j.get("done_reason"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    banner("E3  Nudge generation: model comparison and quality")
    available = ollama_models()
    if available is None:
        print("  !! Ollama is not reachable on localhost:11434. Start it (ollama serve) and re-run.")
        return
    models = [m for m in args.models if ollama_has(available, m)]
    for m in args.models:
        if m not in models:
            print(f"  -- {m} is not pulled, skipping (to include it: ollama pull {m})")
    if not models:
        return

    info = {"meta": run_meta(), "models": {}, "goals": GOALS, "reps": args.reps}
    for m in models:
        try:
            d = ollama_post("/api/show", {"model": m}).get("details", {})
        except Exception as e:
            d = {"error": str(e)}
        info["models"][m] = d
    payloads = {}
    for goal, distraction in GOALS.items():
        for level in LEVELS:
            box = capture_production_payload(level, goal, distraction)
            payloads[(goal, level)] = box["payload"]
    info["production_payload_examples"] = {
        f"L{lvl}": payloads[(next(iter(GOALS)), lvl)] for lvl in LEVELS
    }
    info["production_request_timeout_s"] = box.get("timeout")
    save_json("e3_models.json", info)

    done = {(r["model"], r["goal"], r["level"], r["rep"]) for r in read_jsonl("e3_nudges.jsonl")}
    rng = random.Random(args.seed)
    for m in models:
        conditions = [(g, lvl, rep) for g in GOALS for lvl in LEVELS for rep in range(args.reps)]
        rng.shuffle(conditions)
        todo = [c for c in conditions if (m, c[0], c[1], c[2]) not in done]
        print(f"\n--- {m}: {len(todo)} generations to run ({len(conditions) - len(todo)} already on disk)")
        if not todo:
            continue
        # Cold start: first request loads the model. Recorded separately, not in the sample.
        try:
            cold = generate(m, payloads[(next(iter(GOALS)), 1)])
            append_jsonl("e3_coldstart.jsonl", {"model": m, **cold, "timestamp": time.time()})
            print(f"  cold start: {cold['wall_s']:.1f}s (load {cold['load_s']:.1f}s)")
        except Exception as e:
            print(f"  !! cold start failed for {m}: {e}")
            continue
        t_start = time.time()
        for i, (goal, level, rep) in enumerate(todo, 1):
            try:
                res = generate(m, payloads[(goal, level)])
            except Exception as e:
                res = {"error": str(e)}
            append_jsonl("e3_nudges.jsonl", {
                "model": m, "goal": goal, "level": level, "rep": rep,
                "distraction": GOALS[goal], **res, "timestamp": time.time(),
            })
            if i % 5 == 0 or i == len(todo):
                rate = (time.time() - t_start) / i
                print(f"  {i}/{len(todo)}  last {res.get('wall_s', float('nan')):.1f}s  "
                      f"eta {rate * (len(todo) - i) / 60:.1f} min  | L{level}: {res.get('text', res.get('error'))[:90]}")
    print("\nDone. Raw nudges are in results/final_eval/e3_nudges.jsonl")


if __name__ == "__main__":
    main()
