#!/bin/bash
# setup.sh — one-command setup for CogAlign reviewers.
#
# Automates everything that CAN be automated: Homebrew presence check,
# Ollama install + model pull, terminal-notifier install, Python venv
# creation, and the pinned requirements.txt install.
#
# What this script CANNOT do, on purpose: click through macOS's permission
# dialogs. The first time you run the app, macOS will ask you to allow
# Terminal (or your IDE) to access the microphone, and separately to allow
# it to read other apps' windows via Accessibility, for window tracking,
# plus notifications. Those are one-time prompts you have to click Allow
# on yourself, no script can do that for you. This script tells you what
# to expect at the end.
#
# macOS only. CogAlign's window tracking uses native AppleScript, there is
# no Windows or Linux equivalent in this codebase.

set -e

echo "=== CogAlign setup ==="
echo

# --- 1. Platform check ---
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This script only works on macOS."
    echo "CogAlign's window tracking uses native AppleScript (osascript),"
    echo "which has no Windows or Linux equivalent in this codebase."
    exit 1
fi
echo "[1/7] macOS detected."

# --- 2. Python version check ---
# Prefer a versioned "python3.12" binary if one exists, since plain
# "python3" commonly resolves to whatever a system Python manager (conda,
# pyenv, etc.) put first on $PATH, which may not be 3.12 even if a correct
# 3.12 install is present elsewhere. Confirmed live: on the project's own
# dev machine, plain python3 resolves to conda's 3.13, not the 3.12 this
# project needs, even though 3.12 is also installed.
if command -v python3.12 &> /dev/null; then
    PYTHON_BIN="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    echo "No Python 3 found. Install Python 3.12 from https://python.org first."
    exit 1
fi
PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" != 3.12* ]]; then
    echo "Found Python $PYTHON_VERSION ($PYTHON_BIN), but this project is pinned against Python 3.12."
    echo "Other versions may not have matching prebuilt wheels for the pinned"
    echo "dependencies (numba, torch, numpy) and are not tested."
    echo "If you have multiple Python installs (e.g. conda alongside a plain"
    echo "install), make sure a 'python3.12' command is on your PATH, or run:"
    echo "  brew install python@3.12"
    echo "and re-run this script."
    exit 1
fi
echo "[2/7] Python $PYTHON_VERSION confirmed (using $PYTHON_BIN)."

# --- 3. Homebrew check ---
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Install it first from https://brew.sh, then re-run this script."
    exit 1
fi
echo "[3/7] Homebrew found."

# --- 4. Ollama: install, start, pull model ---
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama via Homebrew..."
    brew install ollama
fi
if ! ollama list &> /dev/null; then
    echo "Starting the Ollama service..."
    brew services start ollama
    sleep 3
fi
echo "Pulling qwen2.5:1.5b (safe to re-run, no-ops if already present)..."
ollama pull qwen2.5:1.5b
echo "[4/7] Ollama ready, qwen2.5:1.5b available."

# --- 5. terminal-notifier ---
if ! command -v terminal-notifier &> /dev/null; then
    echo "Installing terminal-notifier via Homebrew..."
    brew install terminal-notifier
fi
echo "[5/7] terminal-notifier ready."

# --- 6. Python virtual environment + pinned dependencies ---
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $PYTHON_BIN..."
    "$PYTHON_BIN" -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip --quiet
echo "Installing pinned Python dependencies (this can take a few minutes on first run)..."
pip install -r requirements.txt
deactivate
echo "[6/7] Python environment ready."

# --- 7. Data folder safety net ---
mkdir -p data/audio
echo "[7/7] Data folders confirmed."

echo
echo "=== Setup complete ==="
echo
echo "To run CogAlign:"
echo "  source venv/bin/activate"
echo "  python3 src/app.py"
echo "  then open http://127.0.0.1:5050 in your browser"
echo
echo "First-run heads-up, macOS will ask you to approve a few things the"
echo "first time you use them. These are one-time prompts, click Allow:"
echo "  - Microphone access, for speaking your goal"
echo "  - Accessibility access, for reading the active window's title"
echo "  - Notifications, so the nudges actually appear as banners"
echo "If a prompt is dismissed or denied by accident, it can be re-enabled"
echo "under System Settings > Privacy & Security."
