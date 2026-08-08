# src/app.py
import threading

from flask import Flask, render_template, request, jsonify, redirect, url_for

from pipeline import record_goal_audio, transcribe_goal
from engine import CogAlignEngine

app = Flask(__name__)

# This is a single-user local app (one person, one machine), so a module-level
# dict is enough state to track the current session, no need for a real session
# store or database for this scope.
state = {
    "tracking": False,
    "goal": None,
}


def _run_engine(goal):
    """
    Runs CogAlignEngine's monitoring loop for the lifetime of the process.
    This is a daemon thread: it prints to the server console and fires
    notifications exactly as running engine.py directly would, but it is
    not woven into Flask's own request/response cycle at all, so it can't
    block the web server. Note that engine.start()'s KeyboardInterrupt
    handling only ever catches Ctrl+C on the main thread, so stopping the
    Flask process is currently the only way to stop a running session.
    """
    engine = CogAlignEngine(goal=goal)
    engine.start(interval=3)


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

    state["tracking"] = True
    state["goal"] = goal

    thread = threading.Thread(target=_run_engine, args=(goal,), daemon=True)
    thread.start()

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
