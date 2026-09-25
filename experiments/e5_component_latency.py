"""E5. Component latency and window-read reliability on the target Mac.

Updated after the on-screen text enrichment was removed from the pipeline on
23 September: window_tracker.py no longer exposes get_visible_text() or the
background cache, and engine.run_tick() scores the app name and window title
alone. This script measures only what the shipped pipeline actually does.

  1. Window read: WindowTracker.get_active_window(), 40 calls, including how
     often the read comes back unreliable (empty title or a fallback app label),
     which is the condition engine.run_tick() holds state on.
  2. Raw single query: _query_window_once(), 40 calls, so the cost of the retry
     policy (up to 2 retries, 0.15 s apart) can be separated from the base cost.
  3. Full tick without nudging: CogAlignEngine.run_tick() with the real tracker
     and real classifier, the state machine held silent, 30 ticks. This is the
     per-tick budget against the 3 second tick interval.
  4. Session startup: constructing CogAlignEngine, which loads the classifier and
     runs per-goal threshold calibration, measured separately.
  5. Notification: notifier.send_notification(), 3 calls (3 banners appear).

The terminal app running this needs macOS Accessibility permission, the same
permission CogAlign itself needs. Failures are recorded, not fatal.

Output: results/final_eval/e5_latency.json
Usage:  python experiments/e5_component_latency.py            (about 3 min)
"""
import time

from _common import CONTROL_GOAL, banner, run_meta, save_json, summarize


def main():
    banner("E5  Component latency on this machine")
    from window_tracker import WindowTracker
    res = {"meta": run_meta(), "goal": CONTROL_GOAL}
    wt = WindowTracker()

    # 1 and 2: window reads
    for label, fn, n in [("window_read_with_retries", wt.get_active_window, 40),
                         ("window_read_single_query", wt._query_window_once, 40)]:
        times, apps, unreliable, empty_titles = [], [], 0, 0
        for _ in range(n):
            t0 = time.perf_counter()
            app, title = fn()
            times.append(time.perf_counter() - t0)
            apps.append(app)
            if app in WindowTracker.FALLBACK_APP_LABELS:
                unreliable += 1
            elif not title.strip():
                empty_titles += 1
            time.sleep(0.15)
        res[label] = {**summarize(times), "unreliable_app_label": unreliable,
                      "empty_titles": empty_titles, "apps_seen": sorted(set(apps))}
        print(f"  {label:26s} median {res[label]['median']*1000:6.0f} ms  p95 {res[label]['p95']*1000:6.0f} ms  "
              f"unreliable {unreliable}/{n}  empty titles {empty_titles}/{n}")

    # 4 then 3: startup, then steady-state ticks
    try:
        import engine as eng_mod
        t0 = time.perf_counter()
        e = eng_mod.CogAlignEngine(goal=CONTROL_GOAL)
        res["session_startup_seconds"] = time.perf_counter() - t0
        res["calibrated_threshold"] = e.classifier_threshold
        print(f"  session startup (model load + calibration): {res['session_startup_seconds']:.1f}s, "
              f"threshold {e.classifier_threshold:.4f}")

        e.state_machine.evaluate_state = lambda is_goal: ("SILENT", 0)
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            e.run_tick()
            times.append(time.perf_counter() - t0)
            time.sleep(0.5)
        res["tick"] = summarize(times)
        print(f"  full tick (window read + classify): median {res['tick']['median']*1000:.0f} ms, "
              f"p95 {res['tick']['p95']*1000:.0f} ms, against a 3000 ms tick interval")

        # Calibration on its own, since it runs once per session and is the
        # only startup cost that scales with the size of the calibration banks.
        from engine import calibrate_threshold
        t0 = time.perf_counter()
        calibrate_threshold(e.classifier, CONTROL_GOAL)
        res["calibration_seconds"] = time.perf_counter() - t0
        print(f"  threshold calibration alone: {res['calibration_seconds']:.2f}s")
    except Exception as ex:
        res["engine_error"] = str(ex)
        print(f"  !! engine measurement failed: {ex}")

    # 5: notification
    try:
        from notifier import send_notification
        times, oks = [], []
        for i in range(3):
            t0 = time.perf_counter()
            oks.append(send_notification("CogAlign latency test", f"Test banner {i + 1} of 3"))
            times.append(time.perf_counter() - t0)
            time.sleep(1)
        res["notification"] = {**summarize(times), "succeeded": oks}
        print(f"  notification: median {res['notification']['median']*1000:.0f} ms, ok {oks}")
    except Exception as ex:
        res["notification_error"] = str(ex)

    save_json("e5_latency.json", res)


if __name__ == "__main__":
    main()
