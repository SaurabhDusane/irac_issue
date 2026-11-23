"""Steps 2-3: build the final issue graph at a chosen threshold and run
two-level Louvain (communities, then subcommunities inside large communities).

Outputs (in out/graph_<model>_<thr>/):
  assignments.csv     node_id, doc_id, community, subcommunity, degree (-1 = residual)
  subcommunities.jsonl one line per subcommunity (>= 2 issues) with its members'
                       issue, principle, conclusion and rule: the input for step 4
  summary.json         graph and partition statistics

usage: python build_graph.py <model_key> <threshold> [sub_min=30] [resolution=1.0]
"""
import json, os, sys
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

from load_data import DATA
from threshold_sweep import all_pairs_above
from load_data import load_train

SEED = 42


def main(key, thr, sub_min=30, resolution=1.0):
    df = load_train()
    E = np.load(f"out/emb_{key}.npy")
    I, J, S = all_pairs_above(E, thr)
    G = nx.Graph(); G.add_nodes_from(range(len(df)))
    G.add_weighted_edges_from(zip(I.tolist(), J.tolist(), S.tolist()))
    core = G.subgraph([v for v in G if G.degree(v) > 0]).copy()

    comms = sorted(louvain_communities(core, weight="weight", resolution=resolution, seed=SEED), key=len, reverse=True)
    comm = np.full(len(df), -1); sub = np.array(["-1"] * len(df), dtype=object)
    sub_mod = []
    for c_id, c in enumerate(comms):
        comm[list(c)] = c_id
        if len(c) > sub_min:
            H = core.subgraph(c)
            subs = sorted(louvain_communities(H, weight="weight", resolution=resolution, seed=SEED), key=len, reverse=True)
            sub_mod.append(modularity(H, subs, weight="weight"))
        else:
            subs = [c]
        for s_id, s in enumerate(subs):
            sub[list(s)] = f"{c_id}.{s_id}"

    out = f"out/graph_{key}_{thr}"
    os.makedirs(out, exist_ok=True)
    asg = df[["node_id", "doc_id"]].assign(community=comm, subcommunity=sub,
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
    summary = dict(model=key, threshold=thr, nodes=len(df), edges=G.number_of_edges(),
                   residual_isolated=int(len(df) - core.number_of_nodes()),
                   residual_share=1 - core.number_of_nodes() / len(df),
                   modularity=modularity(core, comms, weight="weight"),
                   communities=len(comms), community_size_median=float(np.median(csz)), community_size_max=max(csz),
                   communities_split=len(sub_mod), mean_sub_modularity=float(np.mean(sub_mod)) if sub_mod else None,
                   subcommunities_ge2=sum(n >= 2 for n in ssz), subcommunity_size_median=float(np.median(ssz)),
                   subcommunity_size_max=max(ssz),
                   issues_in_subcommunities_ge5=int(sum(n for n in ssz if n >= 5)))
    json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]), *(int(x) for x in a[2:3]), *(float(x) for x in a[3:4]))
