# src/notifier.py
import subprocess


def _escape_for_applescript(text: str) -> str:
    """
    Escapes backslashes and double quotes so text is safe to embed inside an
    AppleScript double-quoted string literal.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


def send_notification(title: str, message: str, timeout: int = 3) -> bool:
    """
    Triggers a macOS notification banner via osascript's `display notification`,
    using the same subprocess.run pattern as window_tracker.py.

    Returns True if osascript ran without raising, False otherwise. This only
    confirms the call itself didn't error out, not that the banner was actually
    shown (e.g. Notification permissions may be denied for the calling process).
    """
    safe_title = _escape_for_applescript(title)
    safe_message = _escape_for_applescript(message)
    script = f'display notification "{safe_message}" with title "{safe_title}"'

    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Quick module self-test
    ok = send_notification("CogAlign", 'Test notification with a "quote" and a \\ backslash')
    print(f"send_notification returned: {ok}")
