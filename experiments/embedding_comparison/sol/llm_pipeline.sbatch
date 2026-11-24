#!/bin/bash
# Sol: serve gpt-oss-120b with vLLM on one GPU and run the train_fit pipeline against it.
# CHECK BEFORE SUBMITTING: partition/QOS/GPU names and the conda env name for your Sol account
# (copy them from a teammate's working job script if unsure).
#SBATCH -J irac-llm
#SBATCH -p general
#SBATCH -q public
#SBATCH -G a100:1                 # gpt-oss-120b needs an 80 GB GPU (A100-80GB or H100)
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0-12:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
module load mamba/latest
source activate irac                # env with: vllm, sentence-transformers, networkx, leidenalg, igraph, scikit-learn, pandas, requests

PORT=${PORT:-8000}
MODEL=${MODEL:-openai/gpt-oss-120b}
vllm serve "$MODEL" --port "$PORT" --max-model-len 32768 --gpu-memory-utilization 0.92 > vllm-$SLURM_JOB_ID.log 2>&1 &
VLLM=$!
trap 'kill $VLLM' EXIT
until curl -s "http://localhost:$PORT/v1/models" >/dev/null; do sleep 10; done
echo "vLLM up: $MODEL"

export LLM_API_BASE="http://localhost:$PORT/v1"
export GEN_MODEL="$MODEL" JUDGE_MODEL="$MODEL"
export WORKERS=${WORKERS:-32}       # vLLM batches concurrent requests; raise if the GPU is underused
cd "$SLURM_SUBMIT_DIR"
python llm.py                        # connectivity check
bash sol/run_fit_pipeline.sh
