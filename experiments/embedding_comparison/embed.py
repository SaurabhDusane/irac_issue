"""Embed the legal_issue text of every train issue with one model.
usage: python embed.py <model_key>[+lp]
A "+lp" suffix embeds "legal_issue\n\nlegal_principle" instead of the issue alone."""
import os, sys, time
import numpy as np, torch
from sentence_transformers import SentenceTransformer
from load_data import load_train

MODELS = {
    "bge-base":   dict(name="BAAI/bge-base-en-v1.5",   prompt=None),
    "bge-large":  dict(name="BAAI/bge-large-en-v1.5",  prompt=None),
    # Qwen3 is instruction-aware; for symmetric issue-issue similarity the same
    # instruction is applied to every text.
    "qwen3-0.6b": dict(name="Qwen/Qwen3-Embedding-0.6B",
                       prompt="Instruct: Given a legal issue decided by a court, retrieve other issues that raise the same legal question\nQuery: "),
    "qwen3-4b":   dict(name="Qwen/Qwen3-Embedding-4B",
                       prompt="Instruct: Given a legal issue decided by a court, retrieve other issues that raise the same legal question\nQuery: "),
}

key = sys.argv[1]; with_lp = key.endswith("+lp")
cfg = MODELS[key.removesuffix("+lp")]
os.makedirs("out", exist_ok=True)
df = load_train()
df[["node_id", "doc_id", "issue_no"]].to_csv("out/nodes.csv", index=False)
torch.set_num_threads(os.cpu_count())
m = SentenceTransformer(cfg["name"], device="cpu")
t = time.time()
texts = (df.legal_issue + "\n\n" + df.legal_principle).tolist() if with_lp else df.legal_issue.tolist()
E = m.encode(texts, batch_size=32, prompt=cfg["prompt"],
             normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
np.save(f"out/emb_{key}.npy", E.astype(np.float32))
print(f"{key}: {E.shape} in {(time.time()-t)/60:.1f} min", flush=True)
