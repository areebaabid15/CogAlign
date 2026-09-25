# src/app.py
import threading

from flask import Flask, render_template, request, jsonify, redirect, url_for

from pipeline import record_goal_audio, transcribe_goal
from engine import CogAlignEngine

app = Flask(__name__)

# Single-user local app, so a module-level dict is enough to hold the current
# session's state.
state = {
    "tracking": False,
    "goal": None,
    "thread": None,
}

# Stops the engine loop, which runs in a background thread where Ctrl+C never
# arrives.
stop_event = threading.Event()


def _run_engine(goal):
    """
    Runs the monitoring loop for the session in a daemon thread, so it never
    blocks the web server. /stop ends it cleanly through stop_event.
    """
    engine = CogAlignEngine(goal=goal)
    engine.start(interval=3, stop_event=stop_event)


@app.route("/")
def index():
    return render_template("goal.html", tracking=state["tracking"], goal=state["goal"])


@app.route("/record", methods=["POST"])
def record():
    """Reuses pipeline.py's own record + transcribe flow, no duplicated logic."""
    try:
        record_goal_audio()
        goal = transcribe_goal()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not goal:
        return jsonify({"error": "Could not transcribe audio. Please try again."})

    return jsonify({"goal": goal})


@app.route("/start", methods=["POST"])
def start():
    goal = request.form.get("goal", "").strip()
    if not goal:
        return redirect(url_for("index"))

    stop_event.clear()
    state["tracking"] = True
    state["goal"] = goal

    thread = threading.Thread(target=_run_engine, args=(goal,), daemon=True)
    thread.start()
    state["thread"] = thread

    return redirect(url_for("index"))


@app.route("/stop", methods=["POST"])
def stop():
    stop_event.set()
    state["tracking"] = False
    state["goal"] = None
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=False, port=5050)
