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
        # Reads the frontmost app and its window title through System Events
        # for every app, Chrome and Safari included, then strips Chrome's
        # " - Google Chrome" suffix. This replaced per-browser AppleScript
        # calls that needed a separate Automation permission, so the app now
        # asks for three permissions instead of four.
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
        """One AppleScript query, split out so get_active_window() can retry it."""
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
        Returns (app name, window title) for the frontmost window.

        An app name with an empty title is usually a timing race in the
        accessibility query rather than a real "no window" case (seen live
        with Microsoft Word), so the query is retried up to max_retries times
        before an empty title is accepted.
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
