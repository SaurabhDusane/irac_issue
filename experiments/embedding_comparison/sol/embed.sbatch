#!/bin/bash
# Sol: GPU embeddings (Qwen3-Embedding 4B/8B, issue-only and issue+facts).
# CHECK partition/QOS/GPU names for your account before submitting.
#SBATCH -J irac-embed
#SBATCH -p general
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-06:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
module load mamba/latest
source activate irac
cd "$SLURM_SUBMIT_DIR"
for k in ${KEYS:-qwen3-8b qwen3-8b+facts qwen3-4b+facts qwen3-4b+lp}; do
  python embed.py "$k"
done
python embed.py "${TEST_KEY:-qwen3-8b}" test     # test issues for step 5 (issue text only)
python analyze.py gemini-001 qwen3-4b ${KEYS:-qwen3-8b qwen3-8b+facts qwen3-4b+facts qwen3-4b+lp}
