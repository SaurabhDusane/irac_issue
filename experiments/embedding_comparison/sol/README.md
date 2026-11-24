# Running the pipeline on Sol

Everything that needs a GPU or many LLM calls runs here; graph building and
reports can run anywhere. All scripts talk to an OpenAI-compatible endpoint
configured by env vars (see `../llm.py`), so the same code runs against the
ASU API or a local vLLM server.

## One-time setup

```bash
git clone -b claude/affectionate-dirac-z4xscw <repo-url> irac_issue
cd irac_issue/experiments/embedding_comparison
module load mamba/latest
mamba create -n irac python=3.11 -y && source activate irac
pip install vllm sentence-transformers networkx leidenalg igraph scikit-learn scipy pandas requests
```

The Gemini embeddings are committed as `data/emb_gemini-001*.f16.npy`, so no
re-embedding is needed. The train_val split is `splits/train_val_cases.txt`
(placeholder: 700 random cases; replace with the team's list via
`python make_split.py --from <file>` and rebuild the graph).

## Jobs

| job | what | rough time |
|---|---|---|
| `sbatch sol/llm_pipeline.sbatch` | vLLM + gpt-oss-120b; rules on train_fit, split/merge clean-up, validation, test | several hours, resumable |
| `sbatch sol/embed.sbatch` | Qwen3-Embedding 8B/4B, issue-only and issue+facts, plus no-LLM metrics | ~1 h |

Check the `#SBATCH` partition / QOS / GPU lines first; they are typical Sol
values, not verified for your account.

## Without Sol

Same pipeline on the ASU API with cheaper models (once the budget allows):

```bash
GEN_MODEL=openai/gpt5_mini JUDGE_MODEL=aws/claude5_5_haiku bash sol/run_fit_pipeline.sh
```

## Outputs

- `out/graph_gemini-001_0.9v2_fit/rules.jsonl` – final rules (+ `rules_review.md`)
- `out/graph_gemini-001_0.9v2_fit/val/val_summary.json` – validation: rule match (top-1, hit@3, P/R/F1 at tau) and outcome (right in substance / exact) vs nearest precedent
- `out/graph_gemini-001_0.9v2_fit/eval/summary.json` – test applicability
