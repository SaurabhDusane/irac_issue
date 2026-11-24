"""Compare embedding models without ground-truth community labels.

For each model it reports:
  * statute agreement@k: share of an issue's nearest cross-case neighbours that
    cite a statute the issue itself cites (in legal_principle). Statutes are a
    noisy weak label, but one the embedding never sees directly when the issue
    text does not name the statute ("unnamed" column).
  * graph + Louvain stats at a matched edge budget (same number of edges for
    every model, since raw cosine scales differ between models), with statute
    purity of the resulting communities / subcommunities.
  * a markdown dump of random subcommunities for manual review.

usage: python analyze.py bge-base bge-large qwen3-0.6b
"""
import json, os, random, re, sys
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from load_data import load_emb, load_train

K = 10                  # neighbours kept per node for the agreement metrics
EDGE_BUDGET = 34_063    # bge-large @ 0.78 in the team's graph
SUB_MIN = 30            # communities larger than this get a second Louvain pass
SEED = 42
OUT = "out"

# --------------------------------------------------------------- weak labels
ACT_RE = re.compile(
    r"((?:[A-Z][\w'\-.&,]*\s+|\([^)]{1,40}\)\s+|of\s+|and\s+){0,10}"
    r"(?:Act|Code|Rules|Regulations|Ordinance))\b")
CONST_RE = re.compile(r"\bArticles?\s+\d+[A-Z]?(?:\(\w+\))*\s+of\s+the\s+Constitution|\bConstitution of India\b")
PROC = [(re.compile(r"Code of Civil Procedure|\bC\.?P\.?C\b"), "civil procedure code"),
        (re.compile(r"Code of Criminal Procedure|\bCr\.?\s?P\.?C\b"), "criminal procedure code")]
DROP = {"the", "indian", "of", "and", "section", "sections", "under", "article", "rule",
        "order", "clause", "sub", "a", "an", "in", "by", "1", "this"}


def norm_act(s):
    words = [w for w in re.sub(r"[^a-z ]", " ", s.lower().replace("income-tax", "incometax")
                               .replace("income tax", "incometax")).split() if w not in DROP]
    while words and words[0] in DROP:
        words = words[1:]
    return " ".join(words) if len(words) >= 2 else None   # "act" alone is useless


def acts_of(text):
    acts = {a for m in ACT_RE.finditer(text) if (a := norm_act(m.group(1)))}
    if CONST_RE.search(text):
        acts.add("constitution")
    for pat, lab in PROC:
        if pat.search(text):
            acts.add(lab)
    return acts


# --------------------------------------------------------------- similarity
def topk_and_edges(E, doc, k=K, block=2048):
    """Top-k cross-case neighbours per node, plus every pair's score above a
    loose floor (for picking the matched-budget threshold)."""
    n = len(E)
    nbr = np.zeros((n, k), dtype=np.int32); nsim = np.zeros((n, k), dtype=np.float32)
    cand = []  # (i, j, s) with i<j, s >= floor; floor tuned so budget fits
    floor = None
    for a in range(0, n, block):
        S = E[a:a + block] @ E.T
        rows = np.arange(a, min(a + block, n))
        S_nb = S.copy()
        S_nb[doc[rows][:, None] == doc[None, :]] = -1     # drop self + same-case
        idx = np.argpartition(-S_nb, k, axis=1)[:, :k]
        val = np.take_along_axis(S_nb, idx, 1)
        o = np.argsort(-val, 1)
        nbr[rows] = np.take_along_axis(idx, o, 1); nsim[rows] = np.take_along_axis(val, o, 1)
        # upper-triangle candidates (graph keeps same-case edges, like the team's)
        S[np.arange(len(rows)), rows] = -1
        tri = np.arange(n)[None, :] > rows[:, None]
        if floor is None:   # first block: pick a floor that keeps ~10x the budget
            sample = S[tri]
            q = 1 - 10 * EDGE_BUDGET / (n * (n - 1) / 2)
            floor = float(np.quantile(sample, max(q, 0)))
        ii, jj = np.nonzero(tri & (S >= floor))
        cand.append((rows[ii], jj, S[ii, jj]))
    ci = np.concatenate([c[0] for c in cand]); cj = np.concatenate([c[1] for c in cand])
    cs = np.concatenate([c[2] for c in cand])
    return nbr, nsim, (ci, cj, cs)


def louvain_two_level(G):
    comms = louvain_communities(G, weight="weight", resolution=1.0, seed=SEED)
    comm_of, sub_of = {}, {}
    for c_id, c in enumerate(sorted(comms, key=len, reverse=True)):
        for v in c:
            comm_of[v] = c_id
        if len(c) > SUB_MIN:
            subs = louvain_communities(G.subgraph(c), weight="weight", seed=SEED)
        else:
            subs = [c]
        for s_id, s in enumerate(sorted(subs, key=len, reverse=True)):
            for v in s:
                sub_of[v] = f"{c_id}.{s_id}"
    return comms, comm_of, sub_of


def purity(labels_by_cluster, min_labeled=3):
    """Mean (weighted by labeled size) share of members citing the cluster's
    most common statute, over clusters with >= min_labeled labeled members."""
    num = den = 0
    for members in labels_by_cluster.values():
        lab = [a for a in members if a]
        if len(lab) < min_labeled:
            continue
        top = Counter(x for s in lab for x in s).most_common(1)[0][0]
        num += sum(top in s for s in lab); den += len(lab)
    return num / den if den else float("nan")


def main(keys):
    df = load_train()
    df["acts"] = df.legal_principle.map(acts_of)
    issue_l = df.legal_issue.str.lower().str.replace("income-tax", "incometax").str.replace("income tax", "incometax")
    # an act is "unnamed" in the issue if none of its distinctive words appear there
    df["unnamed"] = [bool(a) and not any(all(w in t for w in act.split()[:-1] or [act]) for act in a)
                     for a, t in zip(df.acts, issue_l)]
    doc = pd.factorize(df.doc_id)[0]
    acts = df.acts.values
    print(f"issues with a statute label: {df.acts.astype(bool).mean():.1%}; "
          f"label not named in issue text: {df.unnamed.mean():.1%}")

    results, parts, nbrs = {}, {}, {}
    random.seed(SEED)
    for key in keys:
        E = load_emb(key)
        nbr, nsim, (ci, cj, cs) = topk_and_edges(E, doc)
        nbrs[key] = nbr
        np.save(f"{OUT}/nbr_{key}.npy", nbr); np.save(f"{OUT}/nsim_{key}.npy", nsim)

        r = {}
        for k in (1, 5, 10):
            for subset, mask in (("all", df.acts.astype(bool).values), ("unnamed", df.unnamed.values)):
                hits = [np.mean([bool(acts[i] & acts[j]) for j in nbr[i, :k]]) for i in np.nonzero(mask)[0]]
                r[f"statute@{k}_{subset}"] = float(np.mean(hits))

        # matched edge budget
        order = np.argsort(-cs)[:EDGE_BUDGET]
        thr = float(cs[order[-1]])
        G = nx.Graph(); G.add_nodes_from(range(len(df)))
        G.add_weighted_edges_from(zip(ci[order].tolist(), cj[order].tolist(), cs[order].tolist()))
        comms, comm_of, sub_of = louvain_two_level(G)
        nonsingle = [c for c in comms if len(c) > 1]
        r.update(threshold_for_budget=thr, isolated=float(np.mean([G.degree(v) == 0 for v in G])),
                 modularity=modularity(G, comms, weight="weight"),
                 communities_gt1=len(nonsingle),
                 largest_comm_share=max(map(len, comms)) / len(df),
                 subcommunities_gt1=sum(1 for v in Counter(sub_of.values()).values() if v > 1))
        bysub = {}
        for v, s in sub_of.items():
            if G.degree(v):
                bysub.setdefault(s, []).append(acts[v])
        bycomm = {}
        for v, c in comm_of.items():
            if G.degree(v):
                bycomm.setdefault(c, []).append(acts[v])
        r["comm_statute_purity"] = purity(bycomm); r["sub_statute_purity"] = purity(bysub)
        # NMI on non-isolated nodes with exactly one statute
        m = [v for v in G if G.degree(v) and len(acts[v]) == 1]
        r["sub_statute_nmi"] = normalized_mutual_info_score([next(iter(acts[v])) for v in m], [sub_of[v] for v in m])
        if key.startswith("bge-large"):
            r["edges_at_0.78"] = int((cs >= 0.78).sum())
        results[key] = r
        parts[key] = (comm_of, sub_of, G)
        print(key, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)

    # cross-model agreement
    agree = {}
    for a in keys:
        for b in keys:
            if a < b:
                agree[f"{a} vs {b}"] = dict(
                    sub_ARI=adjusted_rand_score([parts[a][1][v] for v in range(len(df))],
                                                [parts[b][1][v] for v in range(len(df))]),
                    top1_same=float(np.mean(nbrs[a][:, 0] == nbrs[b][:, 0])))
    json.dump(dict(per_model=results, cross_model=agree), open(f"{OUT}/metrics.json", "w"), indent=2)
    print(json.dumps(agree, indent=1))

    # random subcommunities for manual review
    with open(f"{OUT}/random_subcommunities.md", "w") as f:
        f.write("# Random subcommunities per model (manual review)\n\n"
                f"Matched edge budget of {EDGE_BUDGET:,} edges per model. Up to 8 members shown per subcommunity.\n\n")
        for key in keys:
            comm_of, sub_of, G = parts[key]
            groups = {}
            for v, s in sub_of.items():
                groups.setdefault(s, []).append(v)
            eligible = sorted(s for s, mem in groups.items() if 4 <= len(mem) <= 60)
            rng = random.Random(SEED)
            f.write(f"## {key}\n\n")
            for s in rng.sample(eligible, min(5, len(eligible))):
                mem = groups[s]
                f.write(f"### subcommunity {s} ({len(mem)} issues)\n\n")
                for v in rng.sample(mem, min(8, len(mem))):
                    f.write(f"- `{df.node_id[v]}` {df.legal_issue[v]}\n")
                f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
