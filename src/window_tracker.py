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
        # Gets the frontmost app name via System Events first, then branches.
        # Chrome and Safari's generic "name of window 1" reliably comes back
        # empty for the frontmost browser tab, so those two are routed
        # through their own AppleScript dictionaries to get the real active
        # tab title instead. Every other app keeps using the original
        # generic System Events method, unchanged.
        self.applescript_cmd = (
            'tell application "System Events" to tell (first application process whose frontmost is true) '
            "to set currentApp to name\n"
            'if currentApp is "Google Chrome" then\n'
            '    tell application "Google Chrome" to set currentWindow to title of active tab of front window\n'
            'else if currentApp is "Safari" then\n'
            '    tell application "Safari" to set currentWindow to name of front document\n'
            "else\n"
            '    tell application "System Events" to tell (first application process whose frontmost is true) '
            "to set currentWindow to name of window 1\n"
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
        Fetches the active application and window title via AppleScript.
        Chrome and Safari are routed through their own AppleScript
        dictionaries to get the real active tab title, since generic System
        Events UI scripting reliably returns an empty window name for them.

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
