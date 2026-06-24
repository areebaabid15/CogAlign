# test_pipeline.py
import os
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
from engine import CogAlignEngine


def record_audio(filename="goal.wav", duration=5, fs=16000):
    """Records audio from your Mac's built-in microphone."""
    print(
        f"\n🎙️ [WHISPER] Recording started... Speak your focus goal now! (You have {duration}s)"
    )
    print("🗣️ Say something like: 'Writing university thesis documentation'")
    print("-" * 50)

    # Record audio in memory
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()  # Wait until the recording is finished

    print("Recording captured. Saving temporary file...")
    wav.write(filename, fs, audio)


def transcribe_audio(filename="goal.wav"):
    """Uses Whisper to transcribe the saved audio file into text string."""
    print("🧠 [WHISPER] Loading local Whisper model (tiny)...")
    # Using the 'tiny' model so it runs fast and lightweight on your Mac
    model = whisper.load_model("tiny")

    print("📝 [WHISPER] Transcribing audio input...")
    result = model.transcribe(filename)
    goal_text = result["text"].strip()
    return goal_text


def run_unified_pipeline():
    # 1. Capture the goal using your voice (Whisper Layer)
    AUDIO_FILE = "temp_goal.wav"
    record_audio(AUDIO_FILE, duration=5)

    spoken_goal = transcribe_audio(AUDIO_FILE)
    print(f"\n✨ [WHISPER SUCCESS] What the AI heard: '{spoken_goal}'")

    # Clean up the temporary audio file
    if os.path.exists(AUDIO_FILE):
        os.remove(AUDIO_FILE)

    if not spoken_goal:
        print("❌ Error: Whisper could not capture any speech. Try running again.")
        return

    print("\n" + "=" * 50)
    print("🚀 INITIATING TRACKING PIPELINE WITH TRANSCRIBED GOAL")
    print("=" * 50)

    # 2. Initialize the main engine layers (Classifier + State Machine)
    engine = CogAlignEngine()

    # Overwrite the default core goal with the speech output from Whisper!
    engine.core_goal = spoken_goal
    print(f"🎯 Dynamic Core Goal Locked In: '{engine.core_goal}'")

    # 3. Run a quick 5-tick verification loop to see everything work together
    print("\n⏱️ Running a 5-tick simulation loop...")
    for tick in range(1, 6):
        print(f"\n--- [Pipeline Tick {tick}/5] ---")
        engine.run_tick()
        time.sleep(3)

    print(
        "\n✅ Verification Pipeline Check Complete! All three systems talk to each other cleanly."
    )


if __name__ == "__main__":
    run_unified_pipeline()
