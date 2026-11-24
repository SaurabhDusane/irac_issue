"""Clustering robustness (no LLM calls).

1. Louvain vs Leiden on the 0.90 graph, 5 seeds each: modularity, number of
   communities, and seed stability = mean pairwise ARI between seeds (on
   non-isolated issues).
2. Subcommunity resolution sweep (1, 2, 3) inside communities > 30 issues for
   both algorithms: number / size of subcommunities, statute purity and NMI
   (weak labels from legal_principle), and seed stability.

usage: python robustness.py <model_key> [threshold=0.90]
"""
import itertools, json, sys

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from analyze import acts_of, purity
from load_data import load_train
from threshold_sweep import all_pairs_above

SEEDS = [0, 1, 2, 3, 4]
SUB_MIN = 30


def labels_from(comms, n):
    lab = np.full(n, -1)
    for k, c in enumerate(comms):
        lab[list(c)] = k
    return lab


def run(alg, G, H, seed, res=1.0):
    """G: networkx graph, H: same graph in igraph (vertex ids = node ids)."""
    if alg == "louvain":
        return louvain_communities(G, weight="weight", resolution=res, seed=seed)
    p = la.find_partition(H, la.RBConfigurationVertexPartition, weights="weight",
                          resolution_parameter=res, seed=seed, n_iterations=-1)
    return [set(c) for c in p]


def to_igraph(G):
    H = ig.Graph(n=G.number_of_nodes(), edges=list(G.edges()))
    H.es["weight"] = [d["weight"] for _, _, d in G.edges(data=True)]
    return H


def stability(labelings, mask):
    return float(np.mean([adjusted_rand_score(a[mask], b[mask]) for a, b in itertools.combinations(labelings, 2)]))


def main(key, thr=0.90):
    df = load_train()
    acts = df.legal_principle.map(acts_of).values
    E = np.load(f"out/emb_{key}.npy")
    I, J, S = all_pairs_above(E, thr)
    n = len(df)
    core_nodes = sorted(set(I.tolist()) | set(J.tolist()))
    remap = {v: k for k, v in enumerate(core_nodes)}
    G = nx.Graph(); G.add_nodes_from(range(len(core_nodes)))
    G.add_weighted_edges_from((remap[i], remap[j], float(s)) for i, j, s in zip(I, J, S))
    H = to_igraph(G)
    m = len(core_nodes)
    core_acts = acts[core_nodes]
    mask = np.ones(m, bool)
    out = dict(model=key, threshold=thr, clustered_issues=m, edges=G.number_of_edges())

    # 1. top level
    top = {}
    for alg in ("louvain", "leiden"):
        runs = [run(alg, G, H, s) for s in SEEDS]
        labs = [labels_from(c, m) for c in runs]
        top[alg] = dict(modularity=float(np.mean([modularity(G, c, weight="weight") for c in runs])),
                        communities=float(np.mean([len(c) for c in runs])),
                        largest=float(np.mean([max(map(len, c)) for c in runs])),
                        seed_ARI=stability(labs, mask))
        top[alg]["_labels"] = labs
        print(alg, {k: round(v, 4) for k, v in top[alg].items() if not k.startswith("_")}, flush=True)
    out["top_level"] = {a: {k: v for k, v in d.items() if not k.startswith("_")} for a, d in top.items()}
    out["top_level"]["louvain_vs_leiden_ARI"] = float(np.mean(
        [adjusted_rand_score(a, b) for a in top["louvain"]["_labels"] for b in top["leiden"]["_labels"]]))

    # 2. subcommunity resolution sweep (on seed-0 top-level partition of each algorithm)
    sweep = []
    for alg in ("louvain", "leiden"):
        base = top[alg]["_labels"][0]
        for res in (1.0, 2.0, 3.0):
            labs = []
            for seed in SEEDS[:3]:
                sub = np.empty(m, dtype=object)
                for c in np.unique(base):
                    nodes = np.nonzero(base == c)[0]
                    if len(nodes) <= SUB_MIN:
                        sub[nodes] = f"{c}.0"; continue
                    Gs = G.subgraph(nodes.tolist())
                    rel = {v: k for k, v in enumerate(nodes.tolist())}
                    Gr = nx.relabel_nodes(Gs, rel)
                    parts = run(alg, Gr, to_igraph(Gr), seed, res)
                    for k, p in enumerate(parts):
                        sub[nodes[list(p)]] = f"{c}.{k}"
                labs.append(sub)
            sub = labs[0]
            sizes = np.unique(sub, return_counts=True)[1]
            by = {}
            for v, s in enumerate(sub):
                by.setdefault(s, []).append(core_acts[v])
            single = [v for v in range(m) if len(core_acts[v]) == 1]
            row = dict(algorithm=alg, resolution=res, subcommunities=int(len(sizes)),
                       subcommunities_ge3=int((sizes >= 3).sum()), median_size=float(np.median(sizes)),
                       max_size=int(sizes.max()), issues_in_ge3=int(sizes[sizes >= 3].sum()),
                       statute_purity=purity(by),
                       statute_nmi=float(normalized_mutual_info_score([next(iter(core_acts[v])) for v in single],
                                                                      [sub[v] for v in single])),
                       seed_ARI=float(np.mean([adjusted_rand_score(a.astype(str), b.astype(str))
                                               for a, b in itertools.combinations(labs, 2)])))
            sweep.append(row)
            print(row, flush=True)
    out["subcommunity_resolution_sweep"] = sweep
    json.dump(out, open(f"results/robustness_{key}_{thr}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], *(float(x) for x in sys.argv[2:3]))
