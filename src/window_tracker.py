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
        # Flat single-line script verified to clear macOS permission checks
        self.applescript_cmd = (
            'tell application "System Events" to tell (first application process whose frontmost is true) '
            "to set currentWindow to name of window 1\n"
            'tell application "System Events" to tell (first application process whose frontmost is true) '
            "to set currentApp to name\n"
            'return currentApp & "|||" & currentWindow'
        )

    def get_active_window(self):
        """
        Fetches the active application and window title via AppleScript.
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
