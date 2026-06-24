# src/nudge_engine.py
import requests
import time

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:1.5b"

class QwenNudgeEngine:
    def __init__(self):
        self.personas = {
            1: {
                "system": (
                    "You are a warm, supportive academic mentor. "
                    "Output exactly one single sentence. Speak directly to the user. "
                    "Example: 'Every paragraph you write right now is a step closer to completing your degree.'"
                ),
                "user": "I am supposed to be working on my dissertation but I got distracted. Give me a 1-sentence motivational nudge."
            },
            2: {
                "system": (
                    "You are a witty, sarcastic best friend. Your humor is sharp and direct. "
                    "Output exactly one single sentence. No introductory words or quotation marks. "
                    "Example: 'I'm sure the admissions board will find your scrolling habits incredibly relevant to your degree.'"
                ),
                "user": "I am supposed to be working on my dissertation but I got distracted. Give me a 1-sentence sarcastic roast."
            },
            3: {
                "system": (
                    "You are a strict university thesis director. Formal, urgent, direct. "
                    "Output exactly one single sentence giving a professional order to return to work. "
                    "Example: 'Cease all distractions immediately and return your full attention to your academic work.'"
                ),
                "user": "I am distracted instead of writing my dissertation. Give me a 1-sentence formal command."
            }
        }

    def generate_nudge(self, level: int, goal: str = None) -> str:
        if level not in self.personas:
            level = 1

        context = self.personas[level]
        user_msg = context["user"]
        if goal:
            user_msg = f"My goal is: '{goal}'. " + user_msg

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
                return f"[Nudge unavailable — Ollama error {response.status_code}]"
        except requests.exceptions.Timeout:
            return "[Nudge unavailable — Ollama timed out]"
        except Exception as e:
            return f"[Nudge unavailable — {str(e)}]"


if __name__ == "__main__":
    engine = QwenNudgeEngine()
    for level in [1, 2, 3]:
        print(f"\n--- Level {level} ---")
        start = time.time()
        nudge = engine.generate_nudge(level, goal="writing my dissertation literature review")
        print(f"Nudge: {nudge}")
        print(f"Time: {time.time() - start:.1f}s")