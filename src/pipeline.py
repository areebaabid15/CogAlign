# src/pipeline.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
from engine import CogAlignEngine

AUDIO_FILE = "data/audio/temp_goal.wav"
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
TICK_INTERVAL = 3  # seconds between workspace checks


def record_goal_audio():
    print("\n🎙️ [WHISPER] Speak your focus goal now!")
    print(f"   You have {RECORD_SECONDS} seconds...")
    print("   Example: 'I am writing my dissertation literature review'")
    print("-" * 50)

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    os.makedirs(os.path.dirname(AUDIO_FILE), exist_ok=True)
    wav.write(AUDIO_FILE, SAMPLE_RATE, audio)
    print("✅ Recording captured.\n")


def transcribe_goal():
    print("🧠 [WHISPER] Transcribing your goal...")
    model = whisper.load_model("tiny")
    result = model.transcribe(AUDIO_FILE)
    goal = result["text"].strip()

    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)

    return goal


def confirm_goal(goal):
    print(f"✨ [WHISPER] Goal transcribed: '{goal}'")
    correction = input(
        "Did I hear that right? Press Enter to confirm, or type the correct goal and press Enter: "
    ).strip()
    return correction if correction else goal


def run_pipeline():
    print("=" * 50)
    print("       COGALIGN — FOCUS MONITORING SYSTEM")
    print("=" * 50)

    # --- MODEL 1: Whisper ---
    record_goal_audio()
    goal = transcribe_goal()

    if not goal:
        print("❌ Could not transcribe audio. Please try again.")
        return

    goal = confirm_goal(goal)
    print("\n" + "=" * 50)

    # --- MODELS 2 + 3: Cross-encoder + Qwen via Engine ---
    engine = CogAlignEngine(goal=goal)
    engine.start(interval=TICK_INTERVAL)


if __name__ == "__main__":
    run_pipeline()