# Final-report experiments

Data-collection scripts for the final report's evaluation. They read `src/`
and `data/` but never modify them, and write raw results to
`results/final_eval/`. Statistics and figures are produced from those raw
files afterwards.

| Script | What it measures | Needs | Time |
|---|---|---|---|
| `e0_environment.py` | Hardware, OS, package versions, Ollama models and quantisation | Ollama running | seconds |
| E6 (pytest, inside `run_automated.sh`) | Unit test count and pass status on this Mac | nothing | seconds |
| `e4_state_machine_conformance.py --part spec` | 10,000 random traces per timing profile checked against 7 specification properties, plus mutation analysis | nothing | about 1 min |
| `e1_relevance_models.py` | mxbai vs ms-marco cross-encoders vs MiniLM bi-encoder on the 100 labelled samples; load time, per-call latency by input length, parameters, memory | HuggingFace models (downloads on first run) | about 5 min |
| `e5_component_latency.py` | Window read and its reliability, full tick, session startup, calibration, notification | Accessibility permission | about 3 min |
| `e4_state_machine_conformance.py --part live` | When each nudge level actually fires on the live engine, and how stale a nudge is if the user refocuses during generation | Ollama with qwen2.5:1.5b | about 8 min |
| `h1_whisper_wer.py record` | Your 20 spoken goals (interactive, run it yourself in Terminal) | microphone | about 10 min |
| `h1_whisper_wer.py evaluate` | Whisper tiny/base/small transcripts and timings | recordings | about 10 min |
| `e2_calibration_configs.py` | Six threshold-calibration configurations (including the shipped work-like bank and a pruned version of it) on every labelled evaluation set | the evaluation sets in `data/workspace_dumps/` | about 4 min |
| `e3_nudge_llms.py` | qwen2.5:1.5b vs llama3.2:1b vs qwen2.5:0.5b with production prompts, 5 goals x 3 personas x 8 reps | Ollama models | 60 to 90 min |

Run everything automated, in order, with one log:

```bash
bash experiments/run_automated.sh
```

It finds the project's venv automatically (or set `PY=/path/to/venv/bin/python`).
E3 is resumable; if it is interrupted, just run the command again.


## Note on the 23 September code changes

The on-screen text enrichment was removed from the pipeline, so E5 no longer measures
`get_visible_text()` or the background cache, and E4's live runs score app name and window
title alone, matching `engine.run_tick()`. E2 was re-pointed from a separate held-out goal
set to the project's own `evaluation_set.json` and `evaluation_set_holdout.json`, and now
compares calibration configurations rather than a single one.
