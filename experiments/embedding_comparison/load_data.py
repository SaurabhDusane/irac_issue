"""Load train issues into a flat table (one row per embedded issue)."""
import glob, json, os
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "irac issues")

def load_train():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "train", "*.json"))):
        d = json.load(open(f))
        for k, iss in enumerate(d.get("issues") or [], start=1):
            if not iss.get("rule"):          # 2 issues have no rule; README excludes them
                continue
            rows.append(dict(
                node_id=f"{d['doc_id']}_{k}", doc_id=d["doc_id"], issue_no=k,
                legal_issue=iss["legal_issue"],
                legal_principle=iss.get("legal_principle", ""),
                conclusion=iss.get("conclusion", ""),
                predicates=" ".join((iss.get("predicates") or {}).keys()),
                facts=" ".join(iss.get("material_facts") or []),
                rule_text=("IF " + iss["rule"].get("if", "") + " THEN " + iss["rule"].get("then", "") + "\n"
                           + "\n".join(f"{k}: {v}" for k, v in (iss.get("predicates") or {}).items())),
                definition=iss.get("definition") or "",
            ))
    return pd.DataFrame(rows)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_emb(key):
    """Embeddings for <key> (e.g. "gemini-001", "gemini-001_test"): out/emb_<key>.npy,
    else the committed float16 copy data/emb_<key>.f16.npy (so Sol needs only a git clone)."""
    import numpy as np
    for p in (os.path.join(HERE, "out", f"emb_{key}.npy"), os.path.join(HERE, "data", f"emb_{key}.f16.npy")):
        if os.path.exists(p):
            return np.load(p).astype(np.float32)
    raise FileNotFoundError(f"no embeddings for {key}: run embed.py {key.removesuffix('_test')} first")


def node_ids():
    """Train node ids in embedding row order."""
    return load_train().node_id.tolist()


def load_test():
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "test", "*.json"))):
        d = json.load(open(f))
        for k, iss in enumerate(d.get("issues") or [], start=1):
            rows.append(dict(node_id=f"{d['doc_id']}_{k}", doc_id=d["doc_id"], issue_no=k,
                             legal_issue=iss["legal_issue"], legal_principle=""))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_train(); print(len(df), df.doc_id.nunique())
