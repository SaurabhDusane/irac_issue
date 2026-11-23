"""Find generalised rules that conflict: near-identical legal questions with
different outcomes. Candidate pairs are subcommunities in DIFFERENT clusters
whose centroids are close; an LLM classifies each pair and, using the years of
the member judgments, says whether the conflict looks like a doctrinal shift
(later cases departing from earlier ones), a fact-based distinction, or a
genuine inconsistency.

usage: python rule_conflicts.py <graph_dir> [min_sim=0.88] [max_pairs=400]
"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from generalize import call, parse
from load_data import DATA

PROMPT = """Two generalised legal rules were learned from different groups of Supreme Court of India judgments.

RULE A (from judgments decided in {ya}):
Question: {qa}
Rule: IF {ia} THEN {ta}
Statement: {da}

RULE B (from judgments decided in {yb}):
Question: {qb}
Rule: IF {ib} THEN {tb}
Statement: {db}

Classify the relationship:
- SAME = same question, consistent outcome (the rules should be merged)
- CONFLICT = same or overlapping question, and on some set of facts the rules would give different outcomes
- REFINEMENT = one rule is a narrower case or exception of the other, consistent
- DIFFERENT = different legal questions

If CONFLICT, also say which explanation fits best:
- DOCTRINAL_SHIFT = the later judgments depart from the earlier ones
- DISTINCTION = the difference is explained by a fact or statutory difference the rules do not state
- INCONSISTENT = no apparent explanation

Return ONLY JSON: {{"relation": "...", "explanation": "... or null", "note": "one sentence"}}"""


def main(graph_dir, min_sim=0.88, max_pairs=400):
    key = os.path.basename(graph_dir.rstrip("/")).split("_")[1]
    rules = [r for r in map(json.loads, open(f"{graph_dir}/rules.jsonl")) if "error" not in r]
    asg = pd.read_csv(f"{graph_dir}/assignments.csv", dtype={"subcommunity": str, "doc_id": str})
    idx = pd.read_csv(os.path.join(DATA, "index.csv"), dtype={"doc_id": str})
    year = dict(zip(idx.doc_id, idx.year))
    E = np.load(f"out/emb_{key}.npy")
    C, years = [], []
    for r in rules:
        m = np.nonzero(asg.subcommunity.values == r["subcommunity"])[0]
        v = E[m].mean(0); C.append(v / np.linalg.norm(v))
        ys = sorted(year[d] for d in asg.doc_id.values[m])
        years.append(f"{ys[0]}-{ys[-1]}" if ys[0] != ys[-1] else str(ys[0]))
    C = np.array(C)
    S = C @ C.T
    iu = np.triu_indices(len(rules), 1)
    cand = [(i, j, S[i, j]) for i, j in zip(*iu) if S[i, j] >= min_sim]
    cand = sorted(cand, key=lambda x: -x[2])[:max_pairs]
    print(f"{len(rules)} rules, {len(cand)} candidate pairs (centroid sim >= {min_sim})", flush=True)

    def one(c):
        i, j, s = c
        a, b = rules[i], rules[j]
        p = PROMPT.format(ya=years[i], qa=a["legal_question"], ia=a["rule"]["if"], ta=a["rule"]["then"], da=a["definition"],
                          yb=years[j], qb=b["legal_question"], ib=b["rule"]["if"], tb=b["rule"]["then"], db=b["definition"])
        for _ in range(3):
            try:
                out = parse(call([{"role": "user", "content": p}]))
                return dict(a=a["subcommunity"], b=b["subcommunity"], sim=float(s), years_a=years[i], years_b=years[j],
                            question_a=a["legal_question"], question_b=b["legal_question"], **out)
            except Exception:
                pass
        return dict(a=a["subcommunity"], b=b["subcommunity"], sim=float(s), relation=None)

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, cand))
    df = pd.DataFrame(rows)
    os.makedirs(f"{graph_dir}/eval", exist_ok=True)
    df.to_csv(f"{graph_dir}/eval/rule_conflicts.csv", index=False)
    summ = dict(rules=len(rules), candidate_pairs=len(df), relation=df.relation.value_counts().to_dict(),
                conflict_explanation=df[df.relation == "CONFLICT"].explanation.value_counts().to_dict())
    json.dump(summ, open(f"{graph_dir}/eval/rule_conflicts_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(float(x) for x in a[1:2]), *(int(x) for x in a[2:3]))
