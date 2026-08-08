# src/notifier.py
import subprocess

TERMINAL_NOTIFIER = "terminal-notifier"


def send_notification(title: str, message: str, timeout: int = 3) -> bool:
    """
    Triggers a macOS notification banner via the terminal-notifier CLI.

    osascript's `display notification` reaches usernoted (the request itself
    succeeds, no exception, no error) but on modern macOS it never registers an
    entry in Notification Center when called from a background thread rather
    than an interactive session, so nothing actually appears on screen.
    terminal-notifier registers as its own real app with Notification Center,
    which is what makes the banner actually appear.

    Uses subprocess.run with an argument list, not a shell string, so title
    and message reach the OS directly with no shell involved. That means no
    quote or backslash escaping is needed the way the old osascript string
    embedding required, each argument is passed through as-is.

    Returns True if terminal-notifier exited with status 0, False otherwise,
    including if the binary isn't installed or the call raises for any other
    reason. This only confirms the CLI call itself succeeded, not that the
    banner was actually shown.
    """
    try:
        result = subprocess.run(
            [TERMINAL_NOTIFIER, "-title", str(title), "-message", str(message)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    # Quick module self-test
    ok = send_notification("CogAlign", 'Test notification with a "quote" and a \\ backslash')
    print(f"send_notification returned: {ok}")
