"""Rule clean-up -> a v2 rule set next to v1 (v1 is left untouched).

  python refine.py split <v1_dir> <v2_dir>
      keep high-coherence v1 rules; re-partition every medium/low-coherence
      cluster by legal point (LLM; clusters > 25 are first pre-split with
      higher-resolution Louvain so the LLM sees every member); add all
      2-issue subcommunities; generate rules for every new cluster (>= 2).
  python refine.py merge <v2_dir> [min_sim=0.88] [max_pairs=1000]
      find duplicate rules (relation SAME, via rule_conflicts.py), merge them
      with union-find, regenerate one rule per merged cluster.

Cluster ids: unchanged v1 ids for kept clusters, "<v1 id>/<k>" for split
groups, "M<k>" for merged clusters.
"""
import json, os, re, subprocess, sys

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities

from generalize import call, parse

PY = sys.executable
MAX_SHOWN = 25
EDGE_THR = 0.90

PARTITION = """Below are legal issues from different Supreme Court of India judgments that an algorithm grouped together.
Group them by the legal point they decide: two issues belong in the same group only if one generalised rule could govern both (same legal question; consistent or reconcilable outcomes). Same topic or same statute is not enough.

Every node_id must appear in exactly one group. A group may have a single member.

{members}

Return ONLY JSON: {{"groups": [{{"legal_point": "short phrase", "members": ["node_id", ...]}}]}}"""


def short(m):
    return (f"[{m['node_id']}]\nIssue: {m['legal_issue']}\n"
            f"Principle: {(m.get('legal_principle') or '')[:500]}\nConclusion: {(m.get('conclusion') or '')[:300]}")


def llm_partition(members):
    ids = [m["node_id"] for m in members]
    for _ in range(3):
        try:
            out = parse(call([{"role": "user", "content": PARTITION.format(members="\n\n".join(map(short, members)))}]))
            seen, groups = set(), []
            for g in out["groups"]:
                g2 = [x for x in g["members"] if x in ids and x not in seen]
                seen |= set(g2)
                if g2:
                    groups.append(g2)
            groups += [[x] for x in ids if x not in seen]          # anything the model dropped
            return groups
        except Exception:
            pass
    return [ids]                                                    # give up: keep as is


def presplit(members, E, idx_of):
    """Split a large cluster into pieces of <= MAX_SHOWN with Louvain at rising resolution."""
    todo, done = [members], []
    while todo:
        ms = todo.pop()
        if len(ms) <= MAX_SHOWN:
            done.append(ms); continue
        V = E[[idx_of[m["node_id"]] for m in ms]]
        S = V @ V.T
        G = nx.Graph(); G.add_nodes_from(range(len(ms)))
        ii, jj = np.nonzero(np.triu(S, 1) >= EDGE_THR)
        G.add_weighted_edges_from(zip(ii.tolist(), jj.tolist(), S[ii, jj].tolist()))
        parts = None
        for res in (2.0, 4.0, 8.0):
            p = louvain_communities(G, weight="weight", resolution=res, seed=42)
            if len(p) > 1:
                parts = p; break
        if parts is None:                                           # cannot split: cut by centroid order
            c = V.mean(0); order = np.argsort(-(V @ c))
            parts = [set(order[i:i + MAX_SHOWN].tolist()) for i in range(0, len(ms), MAX_SHOWN)]
        todo += [[ms[i] for i in sorted(p)] for p in parts]
    return done


def write_assignments(v2, clusters, n_nodes_csv="out/nodes.csv"):
    nodes = pd.read_csv(n_nodes_csv, dtype=str)
    sub = {m["node_id"]: c["subcommunity"] for c in clusters for m in c["members"]}
    nodes["subcommunity"] = nodes.node_id.map(sub).fillna("-1")
    nodes[["node_id", "doc_id", "subcommunity"]].to_csv(f"{v2}/assignments.csv", index=False)


def split(v1, v2):
    from concurrent.futures import ThreadPoolExecutor
    key = os.path.basename(v1.rstrip("/")).split("_")[1]
    E = np.load(f"out/emb_{key}.npy")
    idx_of = {n: i for i, n in enumerate(pd.read_csv("out/nodes.csv", dtype=str).node_id)}
    os.makedirs(v2, exist_ok=True)
    subs = {c["subcommunity"]: c for c in map(json.loads, open(f"{v1}/subcommunities.jsonl"))}
    rules = {r["subcommunity"]: r for r in map(json.loads, open(f"{v1}/rules.jsonl")) if "error" not in r}

    keep = [s for s, r in rules.items() if r["coherence"] == "high"]
    targets = [s for s, r in rules.items() if r["coherence"] != "high"]
    pairs = [s for s, c in subs.items() if c["size"] == 2]
    print(f"keep {len(keep)} high-coherence rules; re-partition {len(targets)} clusters; add {len(pairs)} pairs", flush=True)

    def do(s):
        groups = []
        for chunk in presplit(subs[s]["members"], E, idx_of):
            if len(chunk) == 1:
                groups.append([chunk[0]["node_id"]]); continue
            groups += llm_partition(chunk)
        return s, groups

    with ThreadPoolExecutor(6) as ex:
        parts = list(ex.map(do, targets))

    clusters, log = [], []
    for s in keep:
        clusters.append(dict(subcommunity=s, community=subs[s]["community"], size=subs[s]["size"],
                             members=subs[s]["members"], origin="v1-high"))
    for s, groups in parts:
        by_id = {m["node_id"]: m for m in subs[s]["members"]}
        multi = sorted([g for g in groups if len(g) >= 2], key=len, reverse=True)
        log.append(dict(v1=s, size=subs[s]["size"], groups=[len(g) for g in groups],
                        singletons=sum(len(g) == 1 for g in groups)))
        for k, g in enumerate(multi):
            clusters.append(dict(subcommunity=f"{s}/{k}", community=subs[s]["community"], size=len(g),
                                 members=[by_id[x] for x in g], origin="split"))
    for s in pairs:
        clusters.append(dict(subcommunity=s, community=subs[s]["community"], size=2,
                             members=subs[s]["members"], origin="pair"))

    with open(f"{v2}/subcommunities.jsonl", "w") as f:
        for c in clusters:
            f.write(json.dumps(c) + "\n")
    json.dump(log, open(f"{v2}/split_log.json", "w"), indent=1)
    write_assignments(v2, clusters)
    with open(f"{v2}/rules.jsonl", "w") as f:                       # reuse high-coherence v1 rules
        for s in keep:
            f.write(json.dumps(rules[s]) + "\n")
    n_split = sum(c["origin"] == "split" for c in clusters)
    print(json.dumps(dict(clusters=len(clusters), kept=len(keep), from_splits=n_split, pairs=len(pairs),
                          split_into_mean=float(np.mean([len([g for g in l["groups"] if g >= 2]) for l in log])),
                          issues_dropped_to_singletons=int(sum(l["singletons"] for l in log)))), flush=True)
    subprocess.run([PY, "generalize.py", v2, "2"], check=True)


def merge(v2, min_sim=0.88, max_pairs=1000):
    subprocess.run([PY, "rule_conflicts.py", v2, str(min_sim), str(max_pairs)], check=True)
    rc = pd.read_csv(f"{v2}/eval/rule_conflicts.csv", dtype={"a": str, "b": str})
    same = rc[rc.relation == "SAME"]
    G = nx.Graph(); G.add_edges_from(zip(same.a, same.b))
    groups = [sorted(c) for c in nx.connected_components(G)]
    clusters = {c["subcommunity"]: c for c in map(json.loads, open(f"{v2}/subcommunities.jsonl"))}
    rules = [r for r in map(json.loads, open(f"{v2}/rules.jsonl"))]
    merged_ids = set()
    new = []
    for k, g in enumerate(groups):
        members = [m for s in g for m in clusters[s]["members"]]
        new.append(dict(subcommunity=f"M{k}", community=clusters[g[0]]["community"], size=len(members),
                        members=members, origin="merged", merged_from=g))
        merged_ids |= set(g)
    out = [c for s, c in clusters.items() if s not in merged_ids] + new
    with open(f"{v2}/subcommunities.jsonl", "w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    with open(f"{v2}/rules.jsonl", "w") as f:
        for r in rules:
            if r["subcommunity"] not in merged_ids:
                f.write(json.dumps(r) + "\n")
    write_assignments(v2, out)
    json.dump(dict(same_pairs=len(same), merged_groups=len(groups), rules_merged=len(merged_ids),
                   merged_group_sizes=[len(g) for g in groups]), open(f"{v2}/merge_log.json", "w"), indent=1)
    print(f"merged {len(merged_ids)} rules into {len(groups)}", flush=True)
    subprocess.run([PY, "generalize.py", v2, "2"], check=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "split":
        split(a[1], a[2])
    else:
        merge(a[1], *(float(x) for x in a[2:3]), *(int(x) for x in a[3:4]))
