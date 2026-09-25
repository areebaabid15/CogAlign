"""Shared helpers for the final-report experiments.

Every script in experiments/ only collects data on the target machine and
writes it to results/final_eval/. Nothing here modifies src/. Analysis,
statistics and figures are produced separately from these raw outputs.
"""
import datetime
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

OUT_DIR = os.path.join(ROOT, "results", "final_eval")
os.makedirs(OUT_DIR, exist_ok=True)

CONTROL_GOAL = "Writing my dissertation literature review"
SAMPLES_PATH = os.path.join(ROOT, "data", "workspace_dumps", "workspace_samples.json")
GOAL_STRINGS_PATH = os.path.join(ROOT, "data", "task_profiles", "goal_strings.json")
OLLAMA = "http://localhost:11434"

# Demo profile is what engine.py runs today. Production profile is the
# CogAlignStateMachine class defaults. Both are exercised by E4.
PROFILES = {
    "demo": {"buffer": 5, "l2": 10, "l3": 15, "l3_interval": 10},
    "production": {"buffer": 30, "l2": 90, "l3": 150, "l3_interval": 60},
}


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def out_path(name):
    return os.path.join(OUT_DIR, name)


def save_json(name, obj):
    path = out_path(name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  saved {os.path.relpath(path, ROOT)}")
    return path


def append_jsonl(name, obj):
    with open(out_path(name), "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def read_jsonl(name):
    path = out_path(name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_samples():
    with open(SAMPLES_PATH) as f:
        return json.load(f)


def load_goal_strings():
    with open(GOAL_STRINGS_PATH) as f:
        return json.load(f)


def peak_rss_mb():
    """Peak resident set size of this process. ru_maxrss is bytes on macOS
    and kilobytes on Linux."""
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def summarize(values):
    """Descriptive summary of a list of timings (seconds)."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return None


def run_meta():
    return {
        "timestamp": now_iso(),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def ollama_get(path, timeout=10):
    import requests
    r = requests.get(OLLAMA + path, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ollama_post(path, payload, timeout=10):
    import requests
    r = requests.post(OLLAMA + path, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def ollama_models():
    """Names of locally available Ollama models, or None if Ollama is not reachable."""
    try:
        return [m["name"] for m in ollama_get("/api/tags").get("models", [])]
    except Exception:
        return None


def ollama_has(models, name):
    if models is None:
        return False
    return name in models or f"{name}:latest" in models


def banner(text):
    print("\n" + "=" * 72)
    print("  " + text)
    print("=" * 72)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
