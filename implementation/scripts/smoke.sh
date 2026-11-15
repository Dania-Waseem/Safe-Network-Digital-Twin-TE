#!/usr/bin/env bash
# Quick check (~1 min): unit tests + 2-case A-F smoke evaluation (writes to a temp dir).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests -q -p no:cacheprovider
out=$(mktemp -d)
python -m safety.improvements.run_eval --smoke --no-resume --output-dir "$out"
git -C .. checkout -- results/safety_gate/improvements/smoke_test_tmp 2>/dev/null || true
echo "smoke output: $out"
