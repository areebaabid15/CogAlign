"""E4. State-machine conformance: property-based testing, mutation analysis,
and measured escalation timing on the live system.

Part A (--part spec, no models needed, about 1 min)
  Drives src/state_machine.py with thousands of randomised tick traces on a
  simulated clock, for both timing profiles (demo 5/10/15/10 s as deployed in
  engine.py, production 30/90/150/60 s as the class defaults), and checks every
  tick against properties taken from the specification:
    P1 reset       any on-task tick returns SILENT and leaves the machine in FOCUS
    P2 grace       no nudge before the buffer threshold has elapsed in an episode
    P3 order       nudge levels within an episode follow 1, 2, 3, 3, 3 exactly
    P4 lower bound Level 2 never before the L2 threshold, first Level 3 never before L3
    P5 spacing     repeated Level 3 nudges are at least the repeat interval apart
    P6 cap         at most three Level 3 nudges per episode, then PASSIVE_LOG and silence
    P7 progress    whenever a nudge is due under the specification, the tick delivers it
                   (this is the property the Level 1 deadlock fixed in commit 5044ff3 broke)
  Then applies the same property suite, and the existing example-based pytest
  suite, to deliberately faulty variants of the state machine (mutants),
  including a reconstruction of the historical deadlock, and reports which
  suite detects each fault.

Part B (--part live, needs the classifier and Ollama, about 8 min)
  Runs the real engine loop (real classifier, real Qwen via Ollama, demo
  profile) against a scripted window source, and timestamps every tick,
  state change and nudge. Sustained-distraction runs measure when each level
  actually fires versus the 5/10/15 s specification, since nudge generation
  blocks the loop. Refocus runs switch the window back on-task the moment a
  Level 1 nudge starts generating and measure how stale the delivered nudge is.
  Notifications are recorded, not displayed.

Output: results/final_eval/e4_spec.json, e4_live.json
Usage:  python experiments/e4_state_machine_conformance.py --part spec
        python experiments/e4_state_machine_conformance.py --part live
"""
import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from collections import Counter

from _common import CONTROL_GOAL, PROFILES, ROOT, SRC, banner, run_meta, save_json

SM_PATH = os.path.join(SRC, "state_machine.py")
TEST_PATH = os.path.join(ROOT, "tests", "test_state_machine.py")

# ----------------------------------------------------------------- Part A


class FakeTime:
    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now


def load_module(source, name):
    mod = types.ModuleType(name)
    mod.__file__ = name
    exec(compile(source, name, "exec"), mod.__dict__)
    return mod


# Pre-5044ff3 ESCALATING logic, reconstructed from that commit's diff. It only
# advanced to Level 2 if a tick happened to land inside [L2, L3) seconds, so a
# late tick (for example after slow nudge generation) left it stuck at Level 1.
HISTORICAL_BLOCK = '''            # Check for Level 2
            if (
                self.level_2_threshold <= elapsed < self.level_3_threshold
                and self.last_nudge_level_fired == 1
            ):
                self.last_nudge_level_fired = 2
                return "NUDGE", 2

            # Check for the Initial Level 3 Nudge
            if elapsed >= self.level_3_threshold and self.last_nudge_level_fired == 2:
                self.last_nudge_level_fired = 3
                self.level_3_nudge_count = 1
                self.last_l3_time = now
                return "NUDGE", 3

'''


def make_mutants(src):
    """Return {name: (description, mutated_source or None if not applicable)}."""
    def rep(old, new):
        return src.replace(old, new, 1) if old in src else None

    hist = None
    m = re.search(r"( +# Work out which level we should be at.*?)( +# Check for Repeated Level 3 Nudges)", src, re.S)
    if m:
        hist = src[:m.start(1)] + HISTORICAL_BLOCK + src[m.start(2):]
    return {
        "M1_grace_halved": ("buffer ends at half the threshold",
                            rep("if elapsed < self.buffer_threshold:", "if elapsed < self.buffer_threshold * 0.5:")),
        "M2_cap_off_by_one": ("allows a fourth Level 3 nudge",
                              rep("if self.level_3_nudge_count >= 3:", "if self.level_3_nudge_count >= 4:")),
        "M3_repeat_too_soon": ("Level 3 repeats at half the interval",
                               rep("if now - self.last_l3_time >= self.level_3_interval:",
                                   "if now - self.last_l3_time >= self.level_3_interval * 0.5:")),
        "M4_skips_levels": ("jumps straight to the target level",
                            rep("next_level = self.last_nudge_level_fired + 1", "next_level = target_level")),
        "M5_no_reset": ("an on-task tick no longer resets the episode",
                        rep('            self.reset_focus()\n            return "SILENT", 0',
                            '            return "SILENT", 0')),
        "M6_level2_early": ("Level 2 threshold replaced by the buffer threshold",
                            rep("elif elapsed >= self.level_2_threshold:", "elif elapsed >= self.buffer_threshold:")),
        "M7_historical_deadlock": ("pre-5044ff3 logic that could stay at Level 1 forever", hist),
    }


def random_trace(rng, profile):
    """A list of (dt_seconds, on_task) ticks."""
    n = rng.randint(30, 260)
    p_on = rng.choice([0.0, 0.0, 0.02, 0.05, 0.1, 0.3])  # some traces never refocus
    trace = []
    for _ in range(n):
        r = rng.random()
        if r < 0.78:
            dt = rng.uniform(2.5, 3.5)            # normal 3 s tick with jitter
        elif r < 0.95:
            dt = rng.uniform(5.0, 25.0)           # loop blocked by nudge generation
        else:
            dt = rng.uniform(0.01, 0.5)           # near-simultaneous ticks
        if profile == "production" and rng.random() < 0.1:
            dt *= rng.uniform(2, 6)               # reach the longer thresholds too
        trace.append((dt, rng.random() < p_on))
    return trace


def check_trace(mod, profile, trace, transitions=None):
    """Run one trace; return the name of the first violated property, or None."""
    cfg = PROFILES[profile]
    T1, T2, T3, R = cfg["buffer"], cfg["l2"], cfg["l3"], cfg["l3_interval"]
    clock = FakeTime()
    mod.time = clock
    sm = mod.CogAlignStateMachine(buffer_threshold=T1, level_2_threshold=T2, level_3_threshold=T3)
    sm.level_3_interval = R

    ep_start = None           # time of the first off-task tick of the current episode
    fired = []                # levels fired this episode
    l3_times = []
    eps = 1e-9
    for dt, on_task in trace:
        clock.now += dt
        t = clock.now
        before = sm.state
        # What the specification says is due on this tick (computed before the call).
        due = False
        if not on_task and ep_start is not None:
            e = t - ep_start
            if not fired:
                due = e >= T1
            elif fired[-1] == 1:
                due = e >= T2
            elif fired[-1] == 2:
                due = e >= T3
            elif len(l3_times) < 3:
                due = (t - l3_times[-1]) >= R
        action, level = sm.evaluate_state(on_task)
        after = sm.state
        if transitions is not None:
            transitions[(before, after)] += 1

        if on_task:
            if (action, level) != ("SILENT", 0) or after != "FOCUS":
                return "P1_reset"
            ep_start, fired, l3_times = None, [], []
            continue
        if ep_start is None:
            ep_start = t
            if action == "NUDGE":
                return "P2_grace"
            continue
        e = t - ep_start
        if action == "NUDGE":
            if e < T1 - eps:
                return "P2_grace"
            expected_next = [1, 2, 3, 3, 3][len(fired)] if len(fired) < 5 else None
            if level != expected_next:
                return "P3_order" if len(fired) < 5 else "P6_cap"
            if level == 2 and e < T2 - eps:
                return "P4_lower_bound"
            if level == 3 and not l3_times and e < T3 - eps:
                return "P4_lower_bound"
            if level == 3 and l3_times and (t - l3_times[-1]) < R - eps:
                return "P5_spacing"
            fired.append(level)
            if level == 3:
                l3_times.append(t)
            if len(l3_times) > 3:
                return "P6_cap"
        else:
            if due:
                return "P7_progress"
            if len(l3_times) >= 3 and after != "PASSIVE_LOG":
                return "P6_cap"
    return None


def run_properties(mod, n_traces, seed, stop_at_first=False, collect=False):
    out = {}
    for profile in PROFILES:
        rng = random.Random(f"{seed}-{profile}")
        transitions = Counter() if collect else None
        violations = Counter()
        ticks = 0
        for _ in range(n_traces):
            trace = random_trace(rng, profile)
            ticks += len(trace)
            v = check_trace(mod, profile, trace, transitions)
            if v:
                violations[v] += 1
                if stop_at_first:
                    break
        out[profile] = {"traces": n_traces, "ticks": ticks, "violations": dict(violations)}
        if collect:
            out[profile]["transitions"] = {f"{a}->{b}": c for (a, b), c in sorted(transitions.items())}
    return out


def run_example_suite_against(source):
    """Run the existing tests/test_state_machine.py against a given source. Returns True if all pass."""
    tmp = tempfile.mkdtemp(prefix="cogalign_mut_")
    try:
        os.makedirs(os.path.join(tmp, "src"))
        os.makedirs(os.path.join(tmp, "tests"))
        with open(os.path.join(tmp, "src", "state_machine.py"), "w") as f:
            f.write(source)
        shutil.copy(TEST_PATH, os.path.join(tmp, "tests", "test_state_machine.py"))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            os.path.join(tmp, "tests", "test_state_machine.py")],
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def part_spec(n_traces, seed):
    banner("E4 Part A  Property-based conformance and mutation analysis")
    src = open(SM_PATH).read()
    real = load_module(src, "state_machine_under_test")
    t0 = time.time()
    res = run_properties(real, n_traces, seed, collect=True)
    for p, r in res.items():
        print(f"  {p:10s} {r['traces']} traces, {r['ticks']} ticks, violations: {r['violations'] or 'none'}")
    print(f"  ({time.time() - t0:.1f}s)")

    ok, summary = run_example_suite_against(src)
    print(f"  example suite on the real code: {'PASS' if ok else 'FAIL'}  ({summary})")

    mutants = {}
    for name, (desc, msrc) in make_mutants(src).items():
        if msrc is None:
            mutants[name] = {"description": desc, "applicable": False}
            print(f"  {name:24s} not applicable (source pattern not found)")
            continue
        mod = load_module(msrc, name)
        pres = run_properties(mod, min(n_traces, 2000), seed, stop_at_first=True)
        killed_by_props = any(r["violations"] for r in pres.values())
        first = {p: list(r["violations"].keys()) for p, r in pres.items() if r["violations"]}
        ex_ok, ex_summary = run_example_suite_against(msrc)
        mutants[name] = {
            "description": desc, "applicable": True,
            "killed_by_property_suite": killed_by_props, "property_violations": first,
            "killed_by_example_suite": not ex_ok, "example_suite_summary": ex_summary,
        }
        print(f"  {name:24s} properties: {'KILLED' if killed_by_props else 'survived'} {first}   "
              f"examples: {'KILLED' if not ex_ok else 'survived'}")
    save_json("e4_spec.json", {"meta": run_meta(), "profiles": PROFILES, "seed": seed,
                               "real_implementation": res, "example_suite_real": {"pass": ok, "summary": summary},
                               "mutants": mutants})

# ----------------------------------------------------------------- Part B


class ScriptedTracker:
    OFF = ("Google Chrome", "YouTube - Top 10 Funniest Cat Videos 2024")
    ON = ("Microsoft Word", "Literature Review Chapter 2.docx")
    FALLBACK_APP_LABELS = ("Unknown", "Timeout", "Error")

    def __init__(self):
        self.mode = "off"

    def get_active_window(self, *a, **k):
        return self.OFF if self.mode == "off" else self.ON

    def get_cached_visible_text(self, *a, **k):
        return ""

    def start_background_visible_text_updates(self, *a, **k):
        pass

    def stop_background_visible_text_updates(self, *a, **k):
        pass


def part_live(runs, refocus_runs, max_seconds):
    banner("E4 Part B  Measured escalation timing on the live engine (demo profile)")
    import engine as eng_mod
    from _common import ollama_has, ollama_models
    if not ollama_has(ollama_models(), "qwen2.5:1.5b"):
        print("  !! Ollama with qwen2.5:1.5b is required for Part B. Skipping.")
        return

    notifications = []
    eng_mod.send_notification = lambda title, msg, *a, **k: notifications.append((time.time(), title, msg)) or True
    e = eng_mod.CogAlignEngine(goal=CONTROL_GOAL)
    tracker = ScriptedTracker()
    e.tracker = tracker
    sm = e.state_machine
    cfg = {"buffer": sm.buffer_threshold, "l2": sm.level_2_threshold, "l3": sm.level_3_threshold,
           "l3_interval": sm.level_3_interval, "max_wasted_allowed": e.max_wasted_allowed,
           "calibrated_threshold": e.classifier_threshold, "tick_interval": 3}
    ticks, gens = [], []
    ctx = {"refocus_on_l1": False, "refocus_time": None}

    orig_eval = sm.evaluate_state
    orig_score = e.classifier.is_related

    def eval_wrapped(is_goal):
        before = sm.state
        t = time.time()
        res = orig_eval(is_goal)
        ticks.append({"t": t, "before": before, "after": sm.state, "action": res[0], "level": res[1],
                      "on_task": is_goal, "score": ctx.get("last_score")})
        return res

    def score_wrapped(text, goal, threshold=0.01):
        out = orig_score(text, goal, threshold=threshold)
        ctx["last_score"] = out[1]
        return out

    orig_gen = e.nudge_engine.generate_nudge

    def gen_wrapped(level, goal=None, distraction=None):
        t0 = time.time()
        if ctx["refocus_on_l1"] and level == 1 and ctx["refocus_time"] is None:
            tracker.mode = "on"            # the user returns to work as generation starts
            ctx["refocus_time"] = t0
        text = orig_gen(level=level, goal=goal, distraction=distraction)
        gens.append({"level": level, "start": t0, "end": time.time(), "text": text})
        return text

    sm.evaluate_state = eval_wrapped
    e.classifier.is_related = score_wrapped
    e.nudge_engine.generate_nudge = gen_wrapped

    def one_run(kind, idx):
        ticks.clear(); gens.clear(); notifications.clear()
        sm.reset_focus()
        tracker.mode = "off"
        ctx.update({"refocus_on_l1": kind == "refocus", "refocus_time": None})
        t_start = time.time()
        stop_reason = "time_limit"
        after_refocus_ticks = 0
        while time.time() - t_start < max_seconds:
            try:
                e.run_tick()
            except SystemExit:
                stop_reason = "auto_shutdown"
                break
            if kind == "refocus" and ctx["refocus_time"] is not None and gens:
                after_refocus_ticks += 1
                if after_refocus_ticks > 4:
                    stop_reason = "refocus_observed"
                    break
            time.sleep(3)
        ep = next((tk["t"] for tk in ticks if tk["before"] == "FOCUS" and tk["after"] == "DISTRACTED_BUFFER"), t_start)
        rel = lambda x: None if x is None else round(x - ep, 2)
        run = {
            "kind": kind, "index": idx, "stop_reason": stop_reason,
            "ticks": [{**tk, "t": rel(tk["t"])} for tk in ticks],
            "generations": [{**g, "start": rel(g["start"]), "end": rel(g["end"]),
                             "duration": round(g["end"] - g["start"], 2)} for g in gens],
            "notifications": [{"t": rel(t), "title": ti, "message": m} for t, ti, m in notifications],
            "refocus_time": rel(ctx["refocus_time"]),
        }
        fires = [(tk["level"], tk["t"]) for tk in run["ticks"] if tk["action"] == "NUDGE"]
        run["nudge_fire_times"] = fires
        pl = next((tk["t"] for tk in run["ticks"] if tk["after"] == "PASSIVE_LOG"), None)
        run["passive_log_time"] = pl
        print(f"  {kind:9s} run {idx}: fires {fires}  PASSIVE_LOG at {pl}  stop: {stop_reason}")
        if kind == "refocus" and run["notifications"] and ctx["refocus_time"] is not None:
            run["stale_by_seconds"] = round(run["notifications"][0]["t"] - run["refocus_time"], 2)
            print(f"            nudge delivered {run['stale_by_seconds']}s after the user was already back on task")
        return run

    results = [one_run("sustained", i + 1) for i in range(runs)]
    results += [one_run("refocus", i + 1) for i in range(refocus_runs)]
    save_json("e4_live.json", {"meta": run_meta(), "config": cfg, "goal": CONTROL_GOAL,
                               "scripted_windows": {"off": ScriptedTracker.OFF, "on": ScriptedTracker.ON},
                               "runs": results})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["spec", "live", "all"], default="spec")
    ap.add_argument("--traces", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--refocus-runs", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=150)
    a = ap.parse_args()
    if a.part in ("spec", "all"):
        part_spec(a.traces, a.seed)
    if a.part in ("live", "all"):
        part_live(a.runs, a.refocus_runs, a.max_seconds)


if __name__ == "__main__":
    main()
