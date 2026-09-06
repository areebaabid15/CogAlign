# CogAlign

A local, privacy-preserving AI pipeline that listens to a user's spoken goal,
monitors their active workspace content, detects semantic distraction in real time,
and delivers escalating persona-driven nudges to restore focus.

CM3020 Artificial Intelligence — Final Project
University of London
Student: Areeba Abid

**macOS only.** Window tracking uses native AppleScript, there is no Windows
or Linux equivalent in this codebase.

## Quickstart for reviewers

```bash
./setup.sh
source venv/bin/activate
python3 src/app.py
```

Then open http://127.0.0.1:5050 in your browser.

`setup.sh` installs everything it can automatically: Ollama plus the
qwen2.5:1.5b model it needs, terminal-notifier, and the Python environment
with the exact pinned dependency versions this project is tested against.
It's safe to re-run, already-installed pieces are skipped.

One thing it can't do for you: macOS will ask you to approve a few things
the first time you use them. These are one-time prompts, click Allow:

- Microphone access, for speaking your goal
- Accessibility access, for reading the active window's title
- Automation access, separately, the first time it reads a Chrome or Safari
  window specifically
- Notifications, so the nudges actually appear as banners

If a prompt is dismissed or denied by accident, it can be re-enabled under
System Settings > Privacy & Security.

## Manual setup

If you'd rather not run the script, or already have some of this installed,
here's what it does, step by step. The app is silently broken without these,
they're required, not optional.

1. Install [Ollama](https://ollama.com) if you don't already have it, then pull the model the nudge engine calls:

```bash
ollama pull qwen2.5:1.5b
```

Ollama needs to be running (`ollama serve`, or the Ollama.app menu bar service) whenever CogAlign is running.

2. Install terminal-notifier via Homebrew, used to show the escalating nudges as native macOS notification banners:

```bash
brew install terminal-notifier
```

3. Install the Python dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

- `src/` — pipeline code
- `data/` — audio recordings, workspace dumps, task profiles
- `results/` — model evaluation logs
- `tests/` — automated pytest suites and model comparison scripts
