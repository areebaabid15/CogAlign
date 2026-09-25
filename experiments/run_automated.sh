#!/usr/bin/env bash
# Runs every automated experiment in a sensible order and logs everything to
# results/final_eval/run_log.txt. Safe to re-run: E3 resumes where it stopped.
#
# Usage (from the repo root):   bash experiments/run_automated.sh
# Optional: PY=/path/to/python bash experiments/run_automated.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p results/final_eval
LOG=results/final_eval/run_log.txt

if [ -z "${PY:-}" ]; then
  for c in venv/bin/python ../CogAlign/venv/bin/python; do
    if [ -x "$c" ] && "$c" -c "import sentence_transformers, whisper, requests, scipy" 2>/dev/null; then PY="$c"; break; fi
  done
fi
if [ -z "${PY:-}" ]; then echo "No working Python environment found. Set PY=/path/to/venv/bin/python"; exit 1; fi

run() { echo; echo ">>> $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

echo "==== CogAlign final-report experiments, $(date) ====" | tee -a "$LOG"
echo "Python: $PY" | tee -a "$LOG"

run "$PY" experiments/e0_environment.py

echo; echo ">>> E6 unit tests" | tee -a "$LOG"
"$PY" -m pytest tests/test_calibration.py tests/test_classifier.py tests/test_nudge_engine.py \
  tests/test_state_machine.py tests/test_window_tracker.py -q -p no:cacheprovider \
  --junitxml=results/final_eval/e6_pytest.xml 2>&1 | tee results/final_eval/e6_pytest.txt | tee -a "$LOG"

run "$PY" experiments/e4_state_machine_conformance.py --part spec
run "$PY" experiments/e1_relevance_models.py
run "$PY" experiments/e5_component_latency.py
run "$PY" experiments/e4_state_machine_conformance.py --part live

if [ -f data/whisper_eval/manifest.json ]; then
  run "$PY" experiments/h1_whisper_wer.py evaluate
else
  echo; echo ">>> H1 skipped: no recordings yet (run: $PY experiments/h1_whisper_wer.py record)" | tee -a "$LOG"
fi

run "$PY" experiments/e2_calibration_configs.py

run "$PY" experiments/e3_nudge_llms.py

echo; echo "==== finished $(date) ====" | tee -a "$LOG"
ls -la results/final_eval | tee -a "$LOG"
