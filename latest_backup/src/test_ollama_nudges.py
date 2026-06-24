import requests
import json
import time

# Ollama runs a local server on port 11434 automatically
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:1b"

scenarios = {
    "Level 1: Empathetic Coach": {
        "system": (
            "You are a warm, supportive academic coach. Your job is to give a short, encouraging push. "
            "You must output exactly one single sentence. Speak directly to the user.\n"
            "Example: 'Remember why you started this journey; every sentence you write right now brings you one step closer to earning your degree.'"
        ),
        "user": "I am supposed to be working on my dissertation, but I am caught watching cat videos. Give me a 1-sentence motivational nudge.",
    },
    "Level 2: Sarcastic Friend": {
        "system": (
            "You are a witty, sarcastic best friend. Your humor is sharp and direct. "
            "You must output exactly one single sentence of dialogue. Do not include any introductory words or commentary.\n"
            "Example 1: 'I'm sure the admissions board will find your deep knowledge of funny cat reactions incredibly relevant to your degree.'\n"
            "Example 2: 'Wow, watching cats run into walls is definitely going to write that conclusion chapter for you.'"
        ),
        "user": "I am supposed to be working on my dissertation, but I am watching cat videos. Give me a 1-sentence sarcastic roast.",
    },
    "Level 3: Deadline Supervisor": {
        "system": (
            "You are a strict, serious university thesis director. Your tone is formal and direct. "
            "You must output exactly one single sentence giving a professional order to return to work.\n"
            "Example: 'Cease all digital distractions immediately and direct your full attention back to your pending academic writing.'"
        ),
        "user": "I am caught watching cat videos instead of writing my dissertation. Give me a 1-sentence formal command.",
    },
}

print("\n--- STARTING OLLAMA BENCHMARKS ---")

for name, context in scenarios.items():
    print(f"\nGenerating for {name}...")

    # Structure the payload exactly how Ollama's API expects it
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": context["system"]},
            {"role": "user", "content": context["user"]},
        ],
        "stream": False,  # We want the full sentence back all at once
        "options": {"temperature": 0.85},
    }

    start = time.time()
    response = requests.post(OLLAMA_URL, json=payload)
    elapsed = time.time() - start

    if response.status_code == 200:
        result = response.json()
        output_text = result["message"]["content"].strip()
        print(f'[{name} Output]:\n"{output_text}"')
    else:
        print(f"Error connecting to Ollama: {response.status_code}")

    print(f"Time taken: {elapsed:.2f}s")
    print("-" * 40)
