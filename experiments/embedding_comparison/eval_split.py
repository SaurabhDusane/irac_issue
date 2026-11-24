"""No-new-API re-analysis of step 5.

1. Temporal tuning: choose the abstain threshold tau on 2023 test issues only
   (max F1, and the highest-accuracy tau), then report 2024 and 2025 with that
   fixed tau -- removes the "tuned on test" problem. Reuses existing judge labels.
2. Test coverage: share of test issues whose nearest rule centroid / nearest
   clustered train issue is within the graph threshold (0.90), and within other
   cut-offs, so it is comparable with the coverage numbers others reported.

usage: python eval_split.py <graph_dir>
"""
import json, os, sys

import numpy as np
import pandas as pd

from evaluate import prf
from load_data import load_test


def main(graph_dir):
    key = os.path.basename(graph_dir.rstrip("/")).split("_")[1]
    pairs = pd.read_csv(f"{graph_dir}/eval/pairs.csv", dtype={"subcommunity": str})
    top1 = pairs[pairs["rank"] == 1].copy()
    top1["gold"] = top1.label == "APPLIES"
    taus = np.round(np.arange(0.80, 0.96, 0.005), 3)

    dev = top1[top1.year == 2023]
    sweep = pd.DataFrame([dict(tau=t, **prf(dev.sim >= t, dev.gold)) for t in taus])
    tau_f1 = float(sweep.loc[sweep.f1.idxmax(), "tau"])
    tau_acc = float(sweep.loc[sweep.accuracy.idxmax(), "tau"])
    res = dict(tuned_on="2023", tau_maxF1=tau_f1, tau_maxAcc=tau_acc, dev_at_tau_maxF1=prf(dev.sim >= tau_f1, dev.gold))
    for name, t in (("tau_maxF1", tau_f1), ("tau_maxAcc", tau_acc), ("tau_0.90", 0.90)):
        for y in (2024, 2025):
            g = top1[top1.year == y]
            res[f"{y}@{name}"] = prf(g.sim >= t, g.gold)
        g = top1[top1.year.isin([2024, 2025])]
        res[f"2024-25@{name}"] = prf(g.sim >= t, g.gold)

    # coverage
    test = load_test()
    T = np.load(f"out/emb_{key}_test.npy")
    E = np.load(f"out/emb_{key}.npy")
    asg = pd.read_csv(f"{graph_dir}/assignments.csv", dtype={"subcommunity": str})
    rules = [json.loads(l) for l in open(f"{graph_dir}/rules.jsonl")]
    rules = [r for r in rules if "error" not in r]
    C = []
    for r in rules:
        v = E[asg.subcommunity.values == r["subcommunity"]].mean(0); C.append(v / np.linalg.norm(v))
    C = np.array(C)
    clustered = E[asg.subcommunity.values != "-1"]
    near_rule = (T @ C.T).max(1)
    near_issue = np.concatenate([(T[i:i + 256] @ clustered.T).max(1) for i in range(0, len(T), 256)])
    near_any = np.concatenate([(T[i:i + 256] @ E.T).max(1) for i in range(0, len(T), 256)])
    cov = {}
    for t in (0.85, 0.88, 0.90, 0.92):
        cov[str(t)] = dict(rule_centroid=float((near_rule >= t).mean()),
                           clustered_train_issue=float((near_issue >= t).mean()),
                           any_train_issue=float((near_any >= t).mean()))
    res["test_coverage"] = cov
    res["test_nearest_similarity_median"] = dict(rule_centroid=float(np.median(near_rule)),
                                                 clustered_train_issue=float(np.median(near_issue)),
                                                 any_train_issue=float(np.median(near_any)))
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/step5_temporal_tuning_and_coverage.json", "w"), indent=1)
    sweep.to_csv("results/step5_tau_sweep_2023.csv", index=False)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
