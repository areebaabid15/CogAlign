# CogAlign

A local, privacy-preserving AI pipeline that listens to a user's spoken goal,
monitors their active workspace content, detects semantic distraction in real time,
and delivers escalating persona-driven nudges to restore focus.

CM3020 Artificial Intelligence — Final Project
University of London
Student: Areeba Abid

## Setup

The app is silently broken without these. Both are required, not optional.

1. Install [Ollama](https://ollama.com) if you don't already have it, then pull the model the nudge engine calls:

\`\`\`bash
ollama pull qwen2.5:1.5b
\`\`\`

Ollama needs to be running (\`ollama serve\`, or the Ollama.app menu bar service) whenever CogAlign is running.

2. Install terminal-notifier via Homebrew, used to show the escalating nudges as native macOS notification banners:

\`\`\`bash
brew install terminal-notifier
\`\`\`

3. Install the Python dependencies:

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Project Structure

- \`src/\` — pipeline code
- \`data/\` — audio recordings, workspace dumps, task profiles
- \`results/\` — model evaluation logs
- \`notes/\` — working notes and Model Testing Log
