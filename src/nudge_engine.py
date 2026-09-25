# src/nudge_engine.py
import requests
import time

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"

# window_tracker.py falls back to these when it can't read the real window.
# Referencing them literally would make the model invent something instead
# of describing an actual distraction, so treat them as no distraction at all.
PLACEHOLDER_DISTRACTIONS = ("Unknown - ", "Timeout - ", "Error - ")


def _is_real_distraction(distraction: str) -> bool:
    """True for a real window, False for a window_tracker failure label."""
    return bool(distraction) and not distraction.startswith(PLACEHOLDER_DISTRACTIONS)

class QwenNudgeEngine:
    """
    Writes one sentence nudges with Qwen2.5 1.5B (qwen2.5:1.5b), served
    locally by Ollama. Each escalation level has its own persona: 1 the
    Empathetic Coach, 2 the Sarcastic Friend, 3 the Deadline Supervisor.
    """

    def __init__(self):
        self.personas = {
            1: {
                "system": (
                    "You are a warm, supportive mentor. "
                    "Output exactly one single sentence. Speak directly to the user. "
                    "Example: 'Every minute you spend on this right now is a step closer to finishing what you set out to do.'"
                ),
                "user": "I am supposed to be working on my goal but I got distracted. Give me a 1-sentence motivational nudge."
            },
            2: {
                "system": (
                    "You are a witty, sarcastic best friend. Your humor is sharp and direct. "
                    "Output exactly one single sentence. No introductory words or quotation marks. "
                    "The sentence must directly reference what the user is distracted by. "
                    "Example: 'I'm sure all that scrolling is exactly what you meant to be doing right now.'"
                ),
                "user": "I am supposed to be working on my goal but I got distracted by something unrelated. Roast me for getting distracted instead of working."
            },
            3: {
                "system": (
                    "You are a strict supervisor. Formal, urgent, direct. "
                    "Output exactly one single sentence giving a professional order to return to work. "
                    "Example: 'Cease all distractions immediately and return your full attention to the task you committed to.'"
                ),
                "user": "I am distracted instead of working on my goal. Give me a 1-sentence formal command."
            }
        }

    def generate_nudge(self, level: int, goal: str = None, distraction: str = None) -> str:
        """
        Returns one nudge sentence for the given level (1 to 3, anything else
        falls back to 1). The user's goal is added to the prompt, and Level 2
        also names the current distraction. If Ollama fails or times out, it
        returns a "[Nudge unavailable: ...]" message instead of raising.
        """
        if level not in self.personas:
            level = 1

        context = self.personas[level]
        user_msg = context["user"]
        if goal:
            user_msg = f"My goal is: '{goal}'. " + user_msg
        if level == 2 and _is_real_distraction(distraction):
            user_msg += f" I am currently distracted by: '{distraction}'."

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": context["system"]},
                {"role": "user", "content": user_msg}
            ],
            "stream": False,
            "options": {"temperature": 0.85}
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["message"]["content"].strip().strip("\"'")
            else:
                return f"[Nudge unavailable: Ollama error {response.status_code}]"
        except requests.exceptions.Timeout:
            return "[Nudge unavailable: Ollama timed out]"
        except Exception as e:
            return f"[Nudge unavailable: {str(e)}]"


if __name__ == "__main__":
    engine = QwenNudgeEngine()
    for level in [1, 2, 3]:
        print(f"\n--- Level {level} ---")
        start = time.time()
        nudge = engine.generate_nudge(level, goal="writing my dissertation literature review")
        print(f"Nudge: {nudge}")
        print(f"Time: {time.time() - start:.1f}s")