"""Embed the legal_issue text of every train issue with one model.
usage: python embed.py <model_key>[+lp]
A "+lp" suffix embeds "legal_issue\n\nlegal_principle" instead of the issue alone."""
import os, sys, time
import numpy as np
from load_data import load_train

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
}

key = sys.argv[1]; with_lp = key.endswith("+lp")
cfg = MODELS[key.removesuffix("+lp")]
os.makedirs("out", exist_ok=True)
df = load_train()
df[["node_id", "doc_id", "issue_no"]].to_csv("out/nodes.csv", index=False)
texts = (df.legal_issue + "\n\n" + df.legal_principle).tolist() if with_lp else df.legal_issue.tolist()
t = time.time()


def api_embed(texts, model, batch=100, workers=4,
              url="https://api-main.aiml.asu.edu/v1/embeddings"):   # ASU auth injected by proxy
    import requests
    from concurrent.futures import ThreadPoolExecutor
    headers = {"Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}"} \
        if "voyageai" in url and os.environ.get("VOYAGE_API_KEY") else {}

    def call(chunk):
        for attempt in range(6):
            try:
                r = requests.post(url, json=dict(model=model, input=chunk), headers=headers, timeout=300)
                d = r.json()["data"]
                return [x["embedding"] for x in sorted(d, key=lambda x: x["index"])]
            except Exception:
                time.sleep(2 ** attempt)
        raise RuntimeError("embedding API failed")
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(workers) as ex:
        E = np.array([v for part in ex.map(call, chunks) for v in part], dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


if cfg.get("api"):
    E = api_embed(texts, cfg["name"], **{k: cfg[k] for k in ("url", "batch") if k in cfg})
else:
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(os.cpu_count())
    m = SentenceTransformer(cfg["name"], device="cpu")
    E = m.encode(texts, batch_size=32, prompt=cfg["prompt"],
                 normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
np.save(f"out/emb_{key}.npy", E.astype(np.float32))
print(f"{key}: {E.shape} in {(time.time()-t)/60:.1f} min", flush=True)
