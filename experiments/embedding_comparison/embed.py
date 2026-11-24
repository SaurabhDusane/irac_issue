"""Embed the legal_issue text of every train issue with one model.
usage: python embed.py <model_key>[+variant] [train|test]
variants: lp (issue + principle), facts (issue + material facts), irac (issue +
facts + principle + rule, no conclusion), rule (if/then + predicates), def
(rule definition). No suffix = issue only."""
import os, sys, time
import numpy as np
from load_data import load_test, load_train

MODELS = {
    "bge-base":   dict(name="BAAI/bge-base-en-v1.5",   prompt=None),
    "bge-large":  dict(name="BAAI/bge-large-en-v1.5",  prompt=None),
    # Qwen3 is instruction-aware; for symmetric issue-issue similarity the same
    # instruction is applied to every text.
    "qwen3-0.6b": dict(name="Qwen/Qwen3-Embedding-0.6B",
                       prompt="Instruct: Given a legal issue decided by a court, retrieve other issues that raise the same legal question\nQuery: "),
    # served by the ASU AIML API (OpenAI-compatible /v1/embeddings); no prompt
    "gemini-001": dict(name="gcp-deepmind/gemini-embedding-001", prompt=None, api=True),
    "gemini-2":   dict(name="gcp-deepmind/gemini-embedding-2", prompt=None, api=True),
    # Voyage AI API; needs VOYAGE_API_KEY (or a proxy that injects it)
    "voyage-law-2": dict(name="voyage-law-2", prompt=None, api=True,
                         url="https://api.voyageai.com/v1/embeddings", batch=64),
    "qwen3-4b":   dict(name="Qwen/Qwen3-Embedding-4B",
                       prompt="Instruct: Given a legal issue decided by a court, retrieve other issues that raise the same legal question\nQuery: "),
    "qwen3-8b":   dict(name="Qwen/Qwen3-Embedding-8B",          # GPU (Sol) recommended
                       prompt="Instruct: Given a legal issue decided by a court, retrieve other issues that raise the same legal question\nQuery: "),
}

key = sys.argv[1]
base, _, variant = key.partition("+")
cfg = MODELS[base]
os.makedirs("out", exist_ok=True)
split = sys.argv[2] if len(sys.argv) > 2 else "train"
df = load_train() if split == "train" else load_test()
suffix = "" if split == "train" else "_test"
df[["node_id", "doc_id", "issue_no"]].to_csv(f"out/nodes{suffix}.csv", index=False)
# input variants (train only; test has legal_issue only)
VARIANTS = {
    "": lambda d: d.legal_issue,
    "lp": lambda d: d.legal_issue + "\n\n" + d.legal_principle,
    "facts": lambda d: d.legal_issue + "\n\nFacts: " + d.facts,
    "irac": lambda d: d.legal_issue + "\n\nFacts: " + d.facts + "\n\nPrinciple: " + d.legal_principle + "\n\nRule: " + d.rule_text,
    "rule": lambda d: d.rule_text,
    "def": lambda d: d.definition.where(d.definition.str.strip() != "", d.legal_issue),  # 2 issues lack one
}
texts = VARIANTS[variant](df).tolist()
t = time.time()


if cfg.get("api"):
    from llm import embed
    kw = {k: cfg[k] for k in ("url", "batch") if k in cfg}
    if "voyageai" in cfg.get("url", ""):
        kw["headers"] = {"Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}"} if os.environ.get("VOYAGE_API_KEY") else {}
    E = embed(texts, cfg["name"], **kw)
else:
    import torch
    from sentence_transformers import SentenceTransformer
    gpu = torch.cuda.is_available()
    torch.set_num_threads(os.cpu_count())
    m = SentenceTransformer(cfg["name"], device="cuda" if gpu else "cpu",
                            model_kwargs={"torch_dtype": torch.bfloat16} if gpu else {})
    E = m.encode(texts, batch_size=int(os.environ.get("EMB_BATCH", 64 if gpu else 32)), prompt=cfg["prompt"],
                 normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
np.save(f"out/emb_{key}{suffix}.npy", E.astype(np.float32))
print(f"{key}{suffix}: {E.shape} in {(time.time()-t)/60:.1f} min", flush=True)
