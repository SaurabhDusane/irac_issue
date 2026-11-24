"""One place for every LLM / embedding call, so the same scripts run against
the ASU AIML API or an OpenAI-compatible server on Sol (e.g. vLLM).

Environment variables (all optional):
  LLM_API_BASE   default https://api-main.aiml.asu.edu/v1   (Sol: http://localhost:8000/v1)
  LLM_API_KEY    sent as "Authorization: Bearer ..." when set (the ASU key is
                 injected by the proxy in the cloud session, so it is unset there)
  GEN_MODEL      model that writes rules / predictions   default aws/claude5_5_sonnet
  JUDGE_MODEL    model that judges                        default aws/claude5_5_sonnet
  WORKERS        parallel requests per script             default 6
  EMB_API_BASE   embeddings endpoint base                  default = LLM_API_BASE
"""
import os, sys, time

import requests

API_BASE = os.environ.get("LLM_API_BASE", "https://api-main.aiml.asu.edu/v1").rstrip("/")
EMB_BASE = os.environ.get("EMB_API_BASE", API_BASE).rstrip("/")
GEN_MODEL = os.environ.get("GEN_MODEL", "aws/claude5_5_sonnet")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "aws/claude5_5_sonnet")
WORKERS = int(os.environ.get("WORKERS", "6"))
_KEY = os.environ.get("LLM_API_KEY")
HEADERS = {"Authorization": f"Bearer {_KEY}"} if _KEY else {}


def stop_if_budget_exhausted(d):
    """The ASU project has a monthly budget; when it is spent every call fails.
    Stop the whole run (results so far are kept and the next run resumes)."""
    msg = (d.get("error") or {}).get("message", "") if isinstance(d, dict) else ""
    if "budget" in msg:
        print(f"STOPPING: API budget exhausted: {msg[:160]}", flush=True)
        os._exit(3)


def chat(messages, model=None, max_tokens=8000, temperature=0, timeout=600):
    """Return the assistant text. Raises on transport / format errors (callers retry)."""
    r = requests.post(f"{API_BASE}/chat/completions", headers=HEADERS, timeout=timeout,
                      json=dict(model=model or GEN_MODEL, max_tokens=max_tokens,
                                temperature=temperature, messages=messages))
    d = r.json()
    stop_if_budget_exhausted(d)
    content = d["choices"][0]["message"].get("content")
    if not content:
        raise ValueError("empty content (reasoning may have used the whole max_tokens budget)")
    return content


def embed(texts, model, batch=100, workers=4, url=None, headers=None):
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    url = url or f"{EMB_BASE}/embeddings"
    hdr = headers if headers is not None else HEADERS

    def call(chunk):
        for attempt in range(6):
            try:
                d = requests.post(url, json=dict(model=model, input=chunk), headers=hdr, timeout=300).json()
                stop_if_budget_exhausted(d)
                return [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]
            except SystemExit:
                raise
            except Exception:
                time.sleep(2 ** attempt)
        raise RuntimeError("embedding API failed")
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    with ThreadPoolExecutor(workers) as ex:
        E = np.array([v for part in ex.map(call, chunks) for v in part], dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


if __name__ == "__main__":       # quick connectivity check: python llm.py
    print(API_BASE, GEN_MODEL, JUDGE_MODEL)
    print(chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=512))
