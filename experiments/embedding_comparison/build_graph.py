"""Steps 2-3: build the final issue graph at a chosen threshold and run
two-level Louvain (communities, then subcommunities inside large communities).

Outputs (in out/graph_<model>_<thr>/):
  assignments.csv     node_id, doc_id, community, subcommunity, degree (-1 = residual)
  subcommunities.jsonl one line per subcommunity (>= 2 issues) with its members'
                       issue, principle, conclusion and rule: the input for step 4
  summary.json         graph and partition statistics

usage: python build_graph.py <model_key> <threshold> [sub_min=30] [resolution=1.0]
         [--leiden] [--sub-res R] [--fit]
  --leiden     Leiden (leidenalg, RBConfiguration) instead of Louvain at both levels
  --sub-res R  resolution for the subcommunity level (default = resolution)
  --fit        build on train_fit only: cases in splits/train_val_cases.txt get no
               edges and no cluster (written to out/graph_<model>_<thr>_fit/)
"""
import json, os, sys
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

from load_data import DATA
from threshold_sweep import all_pairs_above
from load_data import load_emb, load_train

SEED = 42


def communities(G, resolution, leiden):
    if not leiden:
        return louvain_communities(G, weight="weight", resolution=resolution, seed=SEED)
    import igraph as ig
    import leidenalg as la
    nodes = list(G.nodes()); pos = {v: k for k, v in enumerate(nodes)}
    H = ig.Graph(n=len(nodes), edges=[(pos[a], pos[b]) for a, b in G.edges()])
    H.es["weight"] = [d["weight"] for _, _, d in G.edges(data=True)]
    p = la.find_partition(H, la.RBConfigurationVertexPartition, weights="weight",
                          resolution_parameter=resolution, seed=SEED, n_iterations=-1)
    return [{nodes[k] for k in c} for c in p]


def main(key, thr, sub_min=30, resolution=1.0, leiden=False, sub_res=None, fit=False):
    sub_res = resolution if sub_res is None else sub_res
    df = load_train()
    E = load_emb(key)
    I, J, S = all_pairs_above(E, thr)
    is_val = np.zeros(len(df), bool)
    if fit:
        from make_split import val_cases
        is_val = df.doc_id.isin(val_cases()).values
        keep = ~(is_val[I] | is_val[J])
        I, J, S = I[keep], J[keep], S[keep]
    G = nx.Graph(); G.add_nodes_from(range(len(df)))
    G.add_weighted_edges_from(zip(I.tolist(), J.tolist(), S.tolist()))
    core = G.subgraph([v for v in G if G.degree(v) > 0]).copy()

    comms = sorted(communities(core, resolution, leiden), key=len, reverse=True)
    comm = np.full(len(df), -1); sub = np.array(["-1"] * len(df), dtype=object)
    sub_mod = []
    for c_id, c in enumerate(comms):
        comm[list(c)] = c_id
        if len(c) > sub_min:
            H = core.subgraph(c)
            subs = sorted(communities(H, sub_res, leiden), key=len, reverse=True)
            sub_mod.append(modularity(H, subs, weight="weight"))
        else:
            subs = [c]
        for s_id, s in enumerate(subs):
            sub[list(s)] = f"{c_id}.{s_id}"

    out = f"out/graph_{key}_{thr}" + ("_fit" if fit else "")
    os.makedirs(out, exist_ok=True)
    asg = df[["node_id", "doc_id"]].assign(split=np.where(is_val, "val", "fit"), community=comm, subcommunity=sub,
                                           degree=[G.degree(v) for v in range(len(df))])
    asg.to_csv(f"{out}/assignments.csv", index=False)

    # full IRAC records for step 4
    raw = {}
    for d in set(df.doc_id):
        raw[d] = json.load(open(os.path.join(DATA, "train", f"{d}.json")))["issues"]
    sizes = Counter(sub[sub != "-1"])
    with open(f"{out}/subcommunities.jsonl", "w") as f:
        for s, n in sorted(sizes.items(), key=lambda x: -x[1]):
            if n < 2:
                continue
            members = []
            for v in np.nonzero(sub == s)[0]:
                iss = raw[df.doc_id[v]][df.issue_no[v] - 1]
                members.append(dict(node_id=df.node_id[v], legal_issue=iss["legal_issue"],
                                    material_facts=iss.get("material_facts"),
                                    legal_principle=iss.get("legal_principle"), conclusion=iss.get("conclusion"),
                                    rule=iss.get("rule"), predicates=iss.get("predicates")))
            f.write(json.dumps(dict(subcommunity=s, community=int(s.split(".")[0]), size=n, members=members)) + "\n")

    csz = [len(c) for c in comms]
    ssz = list(sizes.values())
    n_fit = int((~is_val).sum())
    summary = dict(model=key, threshold=thr, algorithm="leiden" if leiden else "louvain",
                   resolution=resolution, sub_resolution=sub_res, train_fit_only=fit,
                   nodes=n_fit, held_out_val_issues=int(is_val.sum()), edges=G.number_of_edges(),
                   subcommunities_ge3=sum(n >= 3 for n in Counter(sub[sub != "-1"]).values()),
                   issues_in_subcommunities_ge3=int(sum(n for n in Counter(sub[sub != "-1"]).values() if n >= 3)),
                   residual_isolated=int(n_fit - core.number_of_nodes()),
                   residual_share=1 - core.number_of_nodes() / n_fit,
                   modularity=modularity(core, comms, weight="weight"),
                   communities=len(comms), community_size_median=float(np.median(csz)), community_size_max=max(csz),
                   communities_split=len(sub_mod), mean_sub_modularity=float(np.mean(sub_mod)) if sub_mod else None,
                   subcommunities_ge2=sum(n >= 2 for n in ssz), subcommunity_size_median=float(np.median(ssz)),
                   subcommunity_size_max=max(ssz),
                   issues_in_subcommunities_ge5=int(sum(n for n in ssz if n >= 5)))
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    argv = sys.argv[1:]
    leiden = "--leiden" in argv; fit = "--fit" in argv
    sub_res = None
    if "--sub-res" in argv:
        k = argv.index("--sub-res"); sub_res = float(argv[k + 1]); del argv[k:k + 2]
    a = [x for x in argv if not x.startswith("--")]
    main(a[0], float(a[1]), *(int(x) for x in a[2:3]), *(float(x) for x in a[3:4]),
         leiden=leiden, sub_res=sub_res, fit=fit)
