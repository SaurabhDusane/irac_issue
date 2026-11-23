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
            ))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load_train(); print(len(df), df.doc_id.nunique())
