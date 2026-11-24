"""Step 2: choose the similarity threshold for one embedding model.

For thresholds t in a range it reports edge count, isolated share and Louvain
modularity, and for each score band [t, t+step) an LLM judges a random sample
of pairs ("same legal question, coverable by one rule?") to give band precision.

usage: python threshold_sweep.py <model_key> [lo=0.86] [hi=0.95] [step=0.01] [pairs_per_band=40]
"""
import json, random, re, sys
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import requests
from networkx.algorithms.community import louvain_communities, modularity

from load_data import load_train

from llm import JUDGE_MODEL as JUDGE, WORKERS, chat
from load_data import load_emb
PROMPT = """Two legal issues from Indian Supreme Court judgments:

ISSUE A:
{a}

ISSUE B:
{b}

Do A and B raise the SAME legal question, such that one generalised legal rule could cover both — not merely the same topic or statute?
Answer with exactly one word: YES or NO."""


def ask(a, b):
    for _ in range(4):
        try:
            m = re.match(r"(YES|NO)", chat([{"role": "user", "content": PROMPT.format(a=a, b=b)}], model=JUDGE, max_tokens=2048, timeout=180).strip().upper())
            if m:
                return m.group(1) == "YES"
        except Exception:
            pass
    return None


def all_pairs_above(E, floor, block=2048):
    n = len(E); I, J, S = [], [], []
    for a in range(0, n, block):
        s = E[a:a + block] @ E.T
        rows = np.arange(a, min(a + block, n))
        ii, jj = np.nonzero((s >= floor) & (np.arange(n)[None, :] > rows[:, None]))
        I.append(rows[ii]); J.append(jj); S.append(s[ii, jj])
    return np.concatenate(I), np.concatenate(J), np.concatenate(S)


def main(key, lo=0.86, hi=0.95, step=0.01, per_band=40):
    df = load_train()
    E = load_emb(key)
    I, J, S = all_pairs_above(E, lo)
    doc = df.doc_id.values
    cross = doc[I] != doc[J]
    rng = random.Random(0)
    rows = []
    for t in np.round(np.arange(lo, hi + 1e-9, step), 3):
        keep = S >= t
        G = nx.Graph(); G.add_nodes_from(range(len(df)))
        G.add_weighted_edges_from(zip(I[keep].tolist(), J[keep].tolist(), S[keep].tolist()))
        comms = louvain_communities(G, weight="weight", seed=42)
        band = np.nonzero((S >= t) & (S < t + step) & cross)[0]
        sample = rng.sample(band.tolist(), min(per_band, len(band)))
        with ThreadPoolExecutor(WORKERS) as ex:
            votes = [v for v in ex.map(lambda k: ask(df.legal_issue[I[k]], df.legal_issue[J[k]]), sample) if v is not None]
        r = dict(threshold=float(t), edges=int(keep.sum()), isolated=float(np.mean([d == 0 for _, d in G.degree()])),
                 communities_gt1=sum(len(c) > 1 for c in comms),
                 modularity=float(modularity(G, comms, weight="weight")) if keep.any() else 0.0,
                 band=f"[{t:.2f},{t + step:.2f})", band_pairs=int(len(band)),
                 band_precision=float(np.mean(votes)) if votes else None, judged=len(votes),
                 examples=[dict(score=float(S[k]), a=df.legal_issue[I[k]], b=df.legal_issue[J[k]]) for k in sample[:3]])
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "examples"}), flush=True)
    json.dump(rows, open(f"out/threshold_sweep_{key}.json", "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(float(x) for x in a[1:4]), *(int(x) for x in a[4:5]))
