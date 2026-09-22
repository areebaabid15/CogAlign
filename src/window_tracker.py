# src/window_tracker.py
import subprocess
from collections import deque
import time


class WindowTracker:
    def __init__(self, window_size=5):
        """
        Initializes the tracker with a rolling window history.
        The deque automatically ejects older states as new ones arrive.
        """
        self.history = deque(maxlen=window_size)
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
        # permission prompt at all, so the app asks for three permissions
        # rather than four, and this file only ever needs Accessibility.
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
