import whisper
import time

# Loading the 'small' model variant
model = whisper.load_model("small")

start = time.time()
result = model.transcribe("data/voicetest.m4a")
elapsed = time.time() - start

print("Transcript:", result["text"])
print(f"Time taken: {elapsed:.1f}s")
