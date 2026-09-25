# src/notifier.py
import subprocess

TERMINAL_NOTIFIER = "terminal-notifier"


def send_notification(title: str, message: str, timeout: int = 3) -> bool:
    """
    Shows a macOS notification banner through the terminal-notifier CLI.

    osascript's display notification does not appear when called from a
    background thread on current macOS, while terminal-notifier registers as
    its own app and does. Arguments go in as a list, not through a shell, so
    no quote escaping is needed.

    Returns True if the CLI call succeeded. That confirms the call ran, not
    that the banner was seen.
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
