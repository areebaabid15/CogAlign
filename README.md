# CogAlign

A local, privacy-preserving AI pipeline that listens to a user's spoken goal,
checks the title of the active window every three seconds, detects semantic
distraction in real time, and delivers escalating persona-driven nudges to
restore focus.

CM3020 Artificial Intelligence, Final Project
University of London
Student: Areeba Abid

**macOS only.** Window tracking uses native AppleScript, there is no Windows
or Linux equivalent in this codebase.

## How it works

Three pre-trained models run on the user's own machine, and nothing leaves it:

- **Whisper** (`tiny`) transcribes the spoken goal once, at session start.
- **mxbai-rerank-xsmall-v1**, a cross-encoder, scores the active window's app name and title against the goal.
- **Qwen2.5 1.5B** (`qwen2.5:1.5b`, served by Ollama) writes each nudge, in one of three personas.

A deterministic state machine (`src/state_machine.py`) decides when to nudge, and the relevance threshold is calibrated per goal at session start (`calibrate_threshold()` in `src/engine.py`). The app runs a demonstration timing profile, nudging after 5, 10 and 15 seconds of drift, so a session can be shown in minutes.

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

- `src/`: the application (`app.py` web interface, `pipeline.py` command line version, `engine.py` monitoring loop and threshold calibration)
- `data/`: labelled window title sets, the calibration banks, the 20 goal sentences used for the Whisper test (the recordings stay local), and anonymised user study responses
- `experiments/`: the final evaluation scripts, E0 to E5 and H1 (see `experiments/README.md`)
- `results/`: raw outputs of the evaluations, with the final runs in `results/final_eval/`
- `tests/`: pytest suites plus manual model comparison scripts (see below)

## Running the tests

```bash
source venv/bin/activate
python -m pytest tests/
```

This runs 111 tests across five suites: `test_calibration.py`, `test_classifier.py`, `test_nudge_engine.py`, `test_state_machine.py` and `test_window_tracker.py`. The other `test_*.py` files in `tests/` (the Whisper, Qwen, Ollama, bi-encoder and cross-encoder comparisons) are manual scripts run directly with Python. They need the real models and contain no pytest tests.

To rerun the automated evaluations, run `bash experiments/run_automated.sh`, which logs to `results/final_eval/run_log.txt`.
