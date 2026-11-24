"""Step 5: evaluate the generalised rules on the held-out test issues.

The test set only has legal_issue text (no conclusions), so this measures
rule APPLICABILITY: for each test issue we retrieve the k nearest rules (cosine
between the test issue and each rule's subcommunity centroid, same embedding
model), and an LLM judge decides whether each retrieved rule governs the test
issue's legal question (APPLIES / PARTIAL / NO).

Metrics (overall and by test year):
  precision@1, precision@k       share of retrieved rules judged APPLIES
  hit@k (coverage)               share of test issues with >= 1 applicable rule in top k
  routing accuracy/precision/recall/F1 at similarity threshold tau:
      predict "a rule applies" when top-1 similarity >= tau; gold = judge says
      the top-1 rule APPLIES. Swept over tau to pick an abstain threshold.
Outcome accuracy (does the rule predict the test conclusion?) needs test IRAC
conclusions, which are not in this repo.

usage: python evaluate.py <graph_dir> [k=3] [limit=0 (all test issues)]
"""
import json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from generalize import stop_if_budget_exhausted
from load_data import DATA, load_test

API = "https://api-main.aiml.asu.edu/v1/chat/completions"   # auth injected by proxy
JUDGE = "aws/claude5_5_sonnet"
PROMPT = """A legal issue from a new Supreme Court of India judgment:

TEST ISSUE:
{issue}

A generalised rule learned from earlier judgments:

Legal question: {question}
Rule: IF {cond} THEN {then}
Predicates:
{preds}
Plain statement: {definition}
Exceptions/limits: {exceptions}

Does this rule govern the legal question raised by the test issue — i.e. would a court deciding the test issue need to apply this rule (or decide whether its conditions are met)?
APPLIES = the rule addresses the same legal question.
PARTIAL = the rule addresses a closely related or component question, but not the core one.
NO = different legal question (same topic or statute is not enough).
Answer with exactly one word: APPLIES, PARTIAL or NO."""


def ask(prompt):
    for _ in range(4):
        try:
            r = requests.post(API, json=dict(model=JUDGE, max_tokens=1024, temperature=0,
                                             messages=[{"role": "user", "content": prompt}]), timeout=180)
            d = r.json()
            stop_if_budget_exhausted(d)
            m = re.match(r"(APPLIES|PARTIAL|NO)", d["choices"][0]["message"]["content"].strip().upper())
            if m:
                return m.group(1)
        except Exception:
            pass
    return None


def rule_prompt(issue, r):
    return PROMPT.format(issue=issue, question=r["legal_question"], cond=r["rule"]["if"], then=r["rule"]["then"],
                         preds="\n".join(f"- {k}: {v}" for k, v in r["predicates"].items()),
                         definition=r["definition"], exceptions="; ".join(r.get("exceptions_or_limits") or []) or "none")


def prf(pred, gold):
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    tp = (pred & gold).sum(); fp = (pred & ~gold).sum(); fn = (~pred & gold).sum()
    p = tp / (tp + fp) if tp + fp else float("nan"); r = tp / (tp + fn) if tp + fn else float("nan")
    return dict(accuracy=float((pred == gold).mean()), precision=float(p), recall=float(r),
                f1=float(2 * p * r / (p + r)) if p + r else float("nan"), predicted_share=float(pred.mean()))


def metrics(df, k):
    top1 = df[df["rank"] == 1]
    per_issue = df.groupby("test_node").agg(hit=("label", lambda s: (s == "APPLIES").any()),
                                            hit_or_partial=("label", lambda s: s.isin(["APPLIES", "PARTIAL"]).any()))
    return dict(n_issues=int(top1.shape[0]),
                precision_at_1=float((top1.label == "APPLIES").mean()),
                precision_at_1_incl_partial=float(top1.label.isin(["APPLIES", "PARTIAL"]).mean()),
                **{f"precision_at_{k}": float((df.label == "APPLIES").mean()),
                   f"hit_at_{k}": float(per_issue.hit.mean()),
                   f"hit_at_{k}_incl_partial": float(per_issue.hit_or_partial.mean())})


def main(graph_dir, k=3, limit=0):
    key = os.path.basename(graph_dir.rstrip("/")).split("_")[1]
    rules = [r for r in map(json.loads, open(f"{graph_dir}/rules.jsonl")) if "error" not in r]
    asg = pd.read_csv(f"{graph_dir}/assignments.csv", dtype={"subcommunity": str})
    E = np.load(f"out/emb_{key}.npy")
    C = []
    for r in rules:
        v = E[np.nonzero(asg.subcommunity.values == r["subcommunity"])[0]].mean(0)
        C.append(v / np.linalg.norm(v))
    C = np.array(C)

    test = load_test()
    T = np.load(f"out/emb_{key}_test.npy")
    idx = pd.read_csv(os.path.join(DATA, "index.csv"), dtype={"doc_id": str})
    test["year"] = test.doc_id.map(dict(zip(idx.doc_id, idx.year)))
    sel = list(range(len(test)))
    if limit:
        sel = sorted(random.Random(0).sample(sel, limit))

    S = T[sel] @ C.T
    top = np.argsort(-S, 1)[:, :k]
    out = f"{graph_dir}/eval"
    os.makedirs(out, exist_ok=True)
    cache_path = f"{out}/judgments.jsonl"
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            j = json.loads(line); cache[(j["test_node"], j["subcommunity"])] = j["label"]
    jobs = []
    for row, i in enumerate(sel):
        for rank, ri in enumerate(top[row], start=1):
            jobs.append(dict(test_node=test.node_id[i], year=int(test.year[i]), rank=rank,
                             subcommunity=rules[ri]["subcommunity"], rule_size=rules[ri]["size"],
                             sim=float(S[row, ri]), _issue=test.legal_issue[i], _ri=int(ri)))
    todo = [j for j in jobs if (j["test_node"], j["subcommunity"]) not in cache]
    print(f"{len(rules)} rules; {len(sel)} test issues x k={k}: {len(jobs)} pairs, {len(todo)} to judge", flush=True)
    lock = threading.Lock()

    def work(j):
        lab = ask(rule_prompt(j["_issue"], rules[j["_ri"]]))
        if lab:
            with lock, open(cache_path, "a") as f:
                f.write(json.dumps(dict(test_node=j["test_node"], subcommunity=j["subcommunity"], label=lab)) + "\n")
                cache[(j["test_node"], j["subcommunity"])] = lab
                if len(cache) % 200 == 0:
                    print(f"judged {len(cache)}", flush=True)

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(work, todo))

    df = pd.DataFrame([{k2: v for k2, v in j.items() if not k2.startswith("_")} for j in jobs])
    df["label"] = [cache.get((a, b)) for a, b in zip(df.test_node, df.subcommunity)]
    df = df.dropna(subset=["label"])
    df.to_csv(f"{out}/pairs{'_limit' + str(limit) if limit else ''}.csv", index=False)

    summary = dict(model=key, rules=len(rules), k=k, overall=metrics(df, k),
                   by_year={int(y): metrics(g, k) for y, g in df.groupby("year")})
    top1 = df[df["rank"] == 1]
    gold = (top1.label == "APPLIES").values
    summary["routing"] = [dict(tau=round(float(t), 3), **prf(top1.sim.values >= t, gold))
                          for t in np.arange(0.80, 0.96, 0.01)]
    summary["routing_note"] = ("predict 'a rule applies' when top-1 similarity >= tau; "
                               "gold = judge labels the top-1 rule APPLIES")
    json.dump(summary, open(f"{out}/summary{'_limit' + str(limit) if limit else ''}.json", "w"), indent=1)
    print(json.dumps({k2: v for k2, v in summary.items() if k2 != "routing"}, indent=1))
    print("tau  acc   prec  rec   f1    answered")
    for r in summary["routing"]:
        print(f"{r['tau']:.2f} {r['accuracy']:.3f} {r['precision']:.3f} {r['recall']:.3f} {r['f1']:.3f} {r['predicted_share']:.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:3]))
