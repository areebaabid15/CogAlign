# src/window_tracker.py
import subprocess
import threading
from collections import deque
import time


class WindowTracker:
    # Generous timeout for the background updater's own read, not the
    # tick-bound 1.5s default get_visible_text() otherwise uses. Originally
    # set to 5s based on live timings for Notes (about 4.4s) and a Netflix
    # page (2.3 to 3.9s), but a live smoke test against a real, dense
    # Wikipedia article found `entire contents of window 1` alone took
    # 7.02s just to count its 4,063 elements, well past 5s, before the
    # element cap even gets a chance to bound the slower per-element query
    # loop that follows. The full read succeeded in 12.15s once given more
    # room. Raised to 20s: the constraint that justified a tight 5s no
    # longer applies, this genuinely doesn't compete for tick budget
    # anymore, so there's no cost to affording real, content-heavy pages
    # the time they actually need. An unbounded read is still not safe
    # though, a genuinely pathological page hung for 25+ seconds in earlier
    # testing even with the element cap in place, so this stays a real,
    # finite backstop rather than being removed entirely.
    BACKGROUND_READ_TIMEOUT = 20.0

    # How long a cached visible text read stays usable before a tick treats
    # it as unavailable and falls back to app and title alone. Matches the
    # order of magnitude of the real production escalation timing
    # (buffer_threshold is 30s before a Level 1 nudge even becomes
    # possible), and the background updater gets several
    # real attempts inside any 30s window at its normal several second
    # cadence, so a fresh result is realistically available. Long enough
    # to ride out one slow or empty background cycle without needlessly
    # discarding a still relevant result, short enough that switching to a
    # genuinely different task and being scored against stale content from
    # the previous one for up to half a minute is not a meaningful risk,
    # real task switches this system cares about play out over minutes,
    # not seconds. Starting suggestion per its own design brief, not load
    # bearing elsewhere, safe to retune later.
    VISIBLE_TEXT_MAX_AGE = 30

    def __init__(self, window_size=5):
        """
        Initializes the tracker with a rolling window history.
        The deque automatically ejects older states as new ones arrive.
        """
        self.history = deque(maxlen=window_size)

        # Shared, thread-safe cache the background visible-text updater
        # writes to and the main tick reads from instantly, no waiting.
        # Guarded by a lock since two different threads touch it.
        self._visible_text_lock = threading.Lock()
        self._visible_text_cache = {"app_name": None, "text": "", "timestamp": 0.0}
        self._background_stop_event = threading.Event()
        self._background_thread = None
        # Gets the frontmost app name via System Events, then reads window 1's
        # name, also via System Events, for every app, Chrome and Safari
        # included. This used to route Chrome and Safari through their own
        # AppleScript dictionaries ("tell application Google Chrome/Safari to
        # ...") instead, since that was the only way confirmed to get the real
        # active tab title, System Events' generic window name read was
        # believed to come back empty for them specifically. Live retesting
        # found that belief no longer holds (it may have reflected an
        # unrelated Accessibility permission problem at the time, not a real
        # limitation of the method itself): "name of window 1" via System
        # Events returns the real tab title for both browsers, matching the
        # direct dictionary call exactly for Safari, and for Chrome once its
        # " - Google Chrome" window title suffix is stripped. Confirmed live
        # across 16 back to back trials switching real Chrome and Safari tabs
        # and windows, 0 failures, consistently under 0.4s, actually a little
        # faster than the two-dictionary approach it replaces. The payoff is
        # that Chrome and Safari no longer need the separate Automation
        # permission prompt at all, this file now only ever needs the
        # Accessibility permission get_visible_text() already requires.
        self.applescript_cmd = (
            'tell application "System Events" to tell (first application process whose frontmost is true) '
            "to set currentApp to name\n"
            'tell application "System Events" to tell process currentApp to set currentWindow to name of window 1\n'
            'if currentApp is "Google Chrome" and currentWindow ends with " - Google Chrome" then\n'
            '    set currentWindow to text 1 thru (-1 - (length of " - Google Chrome")) of currentWindow\n'
            "end if\n"
            'return currentApp & "|||" & currentWindow'
        )

    # App names that mean "the query itself didn't reach a real app", as
    # opposed to a real app whose title happened to come back empty. Retrying
    # after one of these wouldn't help, the call already failed or timed out.
    FALLBACK_APP_LABELS = ("Unknown", "Timeout", "Error")

    def _query_window_once(self):
        """
        A single AppleScript query attempt. Unchanged from the original
        get_active_window() logic, just factored out so get_active_window()
        can call it more than once.
        """
        try:
            proc = subprocess.run(
                ["osascript", "-e", self.applescript_cmd],
                capture_output=True,
                text=True,
                timeout=3,
            )
            output = proc.stdout.strip()

            if "|||" in output:
                app_name, window_title = output.split("|||", 1)
                return app_name.strip(), window_title.strip()
            return "Unknown", "No Active Window"

        except subprocess.TimeoutExpired:
            return "Timeout", "AppleScript timed out"
        except Exception as e:
            return "Error", str(e)

    def get_active_window(self, max_retries=2, retry_delay=0.15):
        """
        Fetches the active application and window title via AppleScript,
        entirely through System Events for every app, Chrome and Safari
        included (see the comment on applescript_cmd for why, and for the
        Chrome title suffix stripping).

        Confirmed live against Microsoft Word: a non-empty app name paired
        with an empty window title is very likely a timing race in when the
        accessibility tree is queried relative to window state, not a real
        "no window" case, and not specific to any one app. If that happens,
        retry the same call up to max_retries more times with a short delay
        before accepting the empty result.
        """
        app_name, window_title = self._query_window_once()

        attempts = 0
        while (
            window_title == ""
            and app_name not in self.FALLBACK_APP_LABELS
            and attempts < max_retries
        ):
            time.sleep(retry_delay)
            app_name, window_title = self._query_window_once()
            attempts += 1

        return app_name, window_title

    @staticmethod
    def _escape_for_applescript(text):
        """Escapes backslashes and double quotes so text is safe to embed
        inside an AppleScript double-quoted string literal."""
        return text.replace("\\", "\\\\").replace('"', '\\"')

    def get_visible_text(self, app_name, max_elements=75, timeout=1.5):
        """
        Reads a bounded snippet of visible on-screen text from app_name's
        frontmost window's accessibility tree, via System Events (not
        `tell application <app> to ...`), the same Accessibility permission
        the title read in get_active_window() also uses, so nothing in this
        file needs the separate Automation permission prompt anymore.

        app_name is the value get_active_window() already resolved for
        this tick, not looked up again independently here, so a focus
        change between the two separate osascript calls can't point this
        read at a different app than the title it's paired with.

        Confirmed live: Chrome (Chromium) does not expose real page text
        this way unless it detects an assistive technology client active,
        so it will realistically almost always return "" and the caller
        falls back to app name and title alone. Safari (WebKit) and native
        Mac apps do expose real content.

        Bounded on two axes, both required, not optional. `entire contents
        of window 1` is cheap on a small tree (under a second for 600+
        elements) but scales with tree size, confirmed live to take about
        7s just to enumerate a dense Wikipedia article's 4,063 elements.
        The other slow part is querying role and value on each element
        one at a time, confirmed live to hang for 25+ seconds on a real,
        complex page even with recursion depth capped at 6. So only the first max_elements of the tree are ever
        attribute queried, regardless of how large the real tree is, and
        the whole read additionally sits behind a hard subprocess timeout
        as a backstop. Any failure, timeout, or a genuinely empty result
        returns "" quietly. This never raises and never blocks a tick.

        Also filters out the literal string "missing value", AppleScript's
        own coercion artifact when `value of elem as string` is applied to
        an element whose value is genuinely absent, not real on-screen text,
        confirmed live to otherwise leak into the collected text as noise.
        """
        safe_app_name = self._escape_for_applescript(app_name)
        script = f'''
tell application "System Events"
    tell process "{safe_app_name}"
        set allElems to entire contents of window 1
    end tell
end tell
set elemCount to count of allElems
if elemCount > {max_elements} then set elemCount to {max_elements}
set collectedText to {{}}
repeat with i from 1 to elemCount
    set elem to item i of allElems
    tell application "System Events"
        try
            if role of elem is "AXStaticText" then
                set v to (value of elem as string)
                if v is not "" and v is not "missing value" then set end of collectedText to v
            end if
        end try
    end tell
end repeat
set AppleScript's text item delimiters to " "
return collectedText as string
'''
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                return ""
            return proc.stdout.strip()
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""

    def get_cached_visible_text(self, app_name, max_age=None):
        """
        Instantly returns whatever the background updater thread last found
        for app_name. No subprocess call on this path at all, so a tick
        never waits for this.

        Returns "" (meaning: unavailable, caller falls back to app name and
        title alone) if there is no cached result yet, the cached result
        belongs to a different app than app_name (the user switched apps
        faster than the background updater has refreshed since), or the
        cached result is older than max_age (VISIBLE_TEXT_MAX_AGE by
        default, see its docstring for why 30s).
        """
        if max_age is None:
            max_age = self.VISIBLE_TEXT_MAX_AGE

        with self._visible_text_lock:
            cache = dict(self._visible_text_cache)

        if cache["app_name"] != app_name:
            return ""
        if time.time() - cache["timestamp"] > max_age:
            return ""
        return cache["text"]

    def _run_one_background_update(self, poll_interval=3, backoff_interval=8):
        """
        Does exactly one background update cycle and returns the delay the
        caller should wait before the next one. Factored out from the
        actual while loop specifically so a test can call this directly and
        assert on the resulting cache state and chosen delay, without a
        real thread, a real sleep, or a timing race, the same way
        run_tick() is tested independently of start()'s own while loop.

        Finds the frontmost app via the existing cheap, single-shot title
        query (not the full retrying get_active_window(), a flaky read
        here just means trying again next cycle a few seconds later, no
        need to pay for retries), then reads its visible text with the
        generous background timeout, not the tick-bound one.

        Paces itself: a successful, non-empty read is worth checking again
        soon (poll_interval). An unreliable app read or an empty visible
        text result (Chrome, almost always, confirmed live) backs off
        further (backoff_interval) instead of hammering a window that
        isn't going to yield anything new.
        """
        app_name, window_title = self._query_window_once()

        if app_name in self.FALLBACK_APP_LABELS or not window_title.strip():
            return backoff_interval

        text = self.get_visible_text(app_name, timeout=self.BACKGROUND_READ_TIMEOUT)

        if text:
            with self._visible_text_lock:
                self._visible_text_cache = {
                    "app_name": app_name,
                    "text": text,
                    "timestamp": time.time(),
                }
            return poll_interval

        return backoff_interval

    def _background_loop(self, poll_interval, backoff_interval):
        # Deliberately no lock or preemption between this thread's own
        # osascript calls and the tick's, even though both go through
        # System Events, which serializes AppleEvent handling, so a long
        # running read here (up to BACKGROUND_READ_TIMEOUT) can occasionally
        # make the tick's own concurrent get_active_window() call stall and
        # time out. Confirmed live. Not fixed here on purpose: a timed out
        # tick already falls under the existing "unreliable read holds
        # state" contract in engine.py, the same outcome as any other flaky
        # AppleScript read, so this doesn't need special handling, and a
        # lock would just reintroduce the tick blocking on this read that
        # the whole background thread exists to avoid.
        while not self._background_stop_event.is_set():
            delay = self._run_one_background_update(poll_interval, backoff_interval)
            self._background_stop_event.wait(delay)

    def start_background_visible_text_updates(self, poll_interval=3, backoff_interval=8):
        """
        Starts the background updater thread, a no-op if one is already
        running. Daemon, so it never blocks process exit on its own, but
        engine.py still stops it explicitly on session end (see start()),
        so repeated Start and Stop cycles in the dashboard don't each leave
        one more thread running forever.

        Uses the exact same Accessibility read get_visible_text() already
        performs, System Events, not `tell application <app> to ...`, just
        rescheduled off the tick's own thread. Genuinely requires no new
        permission of any kind.
        """
        if self._background_thread is not None and self._background_thread.is_alive():
            return
        self._background_stop_event.clear()
        self._background_thread = threading.Thread(
            target=self._background_loop,
            args=(poll_interval, backoff_interval),
            daemon=True,
        )
        self._background_thread.start()

    def stop_background_visible_text_updates(self):
        """Signals the background thread to stop and waits briefly for it
        to actually exit. Safe to call even if it was never started."""
        self._background_stop_event.set()
        if self._background_thread is not None:
            self._background_thread.join(timeout=2)

    def update_history(self):
        """
        Gathers the current active window state and slides it into our rolling window.
        """
        app, title = self.get_active_window()
        timestamp = time.time()

        state = {"timestamp": timestamp, "app": app, "title": title}
        self.history.append(state)
        return state


if __name__ == "__main__":
    # Quick module self-test loop
    tracker = WindowTracker(window_size=3)
    print("Starting rolling tracker test loop. Switch windows to see updates...")

    for _ in range(3):
        current = tracker.update_history()
        print(f"\nCaptured State: App='{current['app']}' | Title='{current['title']}'")
        print(f"Current Rolling Memory Size: {len(tracker.history)}")
        time.sleep(2)
