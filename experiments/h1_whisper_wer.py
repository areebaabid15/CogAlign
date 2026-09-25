"""H1. Whisper word error rate on real spoken goals (needs you for the recording step).

Step 1, record (run this yourself in Terminal, it is interactive, about 10 min):
    python experiments/h1_whisper_wer.py record
  Shows each of the 20 goals in data/task_profiles/goal_strings.json, waits for
  Enter, then records exactly as CogAlign does: 5 seconds, 16 kHz, mono, int16.
  You can replay-check and redo any take. Recordings are saved to
  data/whisper_eval/ (keep them local, do not commit your voice to the repo).
  Speak naturally, at normal speed, in your usual environment.

Step 2, evaluate (no interaction needed, about 10 min):
    python experiments/h1_whisper_wer.py evaluate
  For whisper tiny, base and small: cold model-load time, per-utterance
  transcription time, and the transcript, using the same transcribe() call as
  pipeline.py (default options). Tiny is also run with language="en" to test
  whether fixing the language helps. Word error rate is computed afterwards.

Output: data/whisper_eval/manifest.json, results/final_eval/h1_whisper.json
"""
import json
import os
import sys
import time

from _common import ROOT, banner, load_goal_strings, run_meta, save_json

REC_DIR = os.path.join(ROOT, "data", "whisper_eval")
MANIFEST = os.path.join(REC_DIR, "manifest.json")
SECONDS, RATE = 5, 16000  # identical to src/pipeline.py


def record():
    import sounddevice as sd
    import scipy.io.wavfile as wav
    os.makedirs(REC_DIR, exist_ok=True)
    # Keep voice recordings out of git, since the repository will be public.
    with open(os.path.join(REC_DIR, ".gitignore"), "w") as f:
        f.write("*\n")
    goals = load_goal_strings()
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {"items": []}
    done = {it["index"] for it in manifest["items"]}
    banner("H1  Recording spoken goals (5 s each, same settings as CogAlign)")
    for i, goal in enumerate(goals):
        if i in done:
            continue
        while True:
            print(f"\n[{i + 1}/{len(goals)}]  Say:  \"{goal}\"")
            input("  Press Enter, then speak within 5 seconds...")
            audio = sd.rec(int(SECONDS * RATE), samplerate=RATE, channels=1, dtype="int16")
            sd.wait()
            ans = input("  Enter = keep,  p = play back,  r = redo: ").strip().lower()
            if ans == "p":
                sd.play(audio, RATE); sd.wait()
                ans = input("  Enter = keep,  r = redo: ").strip().lower()
            if ans != "r":
                break
        path = os.path.join(REC_DIR, f"goal_{i:02d}.wav")
        wav.write(path, RATE, audio)
        manifest["items"].append({"index": i, "file": os.path.basename(path), "reference": goal})
        json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"\nAll recordings saved in {os.path.relpath(REC_DIR, ROOT)}")


def evaluate():
    import whisper
    banner("H1  Whisper evaluation")
    if not os.path.exists(MANIFEST):
        print("  !! No recordings yet. Run:  python experiments/h1_whisper_wer.py record")
        return
    items = json.load(open(MANIFEST))["items"]
    out = {"meta": run_meta(), "n_utterances": len(items), "conditions": {}}
    conditions = [("tiny", None), ("tiny", "en"), ("base", None), ("small", None)]
    for size, lang in conditions:
        key = f"{size}" + (f"_lang_{lang}" if lang else "")
        t0 = time.perf_counter()
        model = whisper.load_model(size)
        load_s = time.perf_counter() - t0
        rows = []
        for it in items:
            path = os.path.join(REC_DIR, it["file"])
            t0 = time.perf_counter()
            kw = {"language": lang} if lang else {}
            r = model.transcribe(path, **kw)
            rows.append({"index": it["index"], "reference": it["reference"], "hypothesis": r["text"].strip(),
                         "detected_language": r.get("language"), "seconds": time.perf_counter() - t0})
        out["conditions"][key] = {"model": size, "language": lang, "load_seconds": load_s, "utterances": rows}
        print(f"  {key:12s} load {load_s:.1f}s, median transcribe "
              f"{sorted(x['seconds'] for x in rows)[len(rows)//2]:.2f}s")
        del model
    save_json("h1_whisper.json", out)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "record":
        record()
    elif mode == "evaluate":
        evaluate()
    else:
        print(__doc__)
