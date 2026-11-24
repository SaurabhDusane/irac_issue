#!/bin/bash
# Full train_fit pipeline: rules -> clean-up -> validation -> test.
# Works against any OpenAI-compatible endpoint (see llm.py for the env vars).
# Every stage resumes from its cached output if interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
FIT=out/graph_gemini-001_0.9_fit          # built by: python build_graph.py gemini-001 0.9 30 1.0 --leiden --sub-res 2 --fit
V2=out/graph_gemini-001_0.9v2_fit

[ -f "$FIT/subcommunities.jsonl" ] || $PY build_graph.py gemini-001 0.9 30 1.0 --leiden --sub-res 2 --fit
echo "== 1/5 rules for subcommunities >= 3 $(date +%T)";  $PY generalize.py "$FIT" 3
echo "== 2/5 split mixed clusters + add pairs $(date +%T)"; $PY refine.py split "$FIT" "$V2"
echo "== 3/5 merge duplicate rules $(date +%T)";          $PY refine.py merge "$V2" 0.88 1000
echo "== 4/5 validation (train_val) $(date +%T)";         $PY validate.py "$V2" --tau 0.90 --k 3 --baseline
echo "== 5/5 test applicability $(date +%T)";             $PY evaluate.py "$V2" 3
echo "DONE $(date +%T)"
