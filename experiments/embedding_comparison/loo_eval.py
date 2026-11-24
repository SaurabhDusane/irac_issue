"""Leave-one-out OUTCOME accuracy on train (the test set has no conclusions).

For a random sample of subcommunities (4..25 members, so the generator sees
every member):
  1. hold out one member, generate the rule from the others (same prompt as step 4)
  2. PREDICT the held-out case's outcome from its material facts + the cluster's
     legal question, given
       (a) the generalised rule                      -> "rule"
       (b) the single most similar other train issue  -> "nearest precedent" baseline
           (its issue, principle and conclusion; cross-case, by embedding)
     The predictor never sees the held-out issue text, principle or conclusion
     (train issue texts are phrased as holdings and would leak the answer).
  3. GRADE each prediction against the real conclusion (MATCH / PARTIAL / MISMATCH)
     with a separate call that does not know which method produced it.

usage: python loo_eval.py <graph_dir> [n_clusters=150]
"""
import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from generalize import call, generalize, parse
from llm import WORKERS
from load_data import load_emb, load_train

PREDICT = """You are predicting how the Supreme Court of India will decide a legal issue.

LEGAL QUESTION:
{question}

MATERIAL FACTS OF THE NEW CASE:
{facts}

{guidance_title}:
{guidance}

Using only the guidance above and the facts, predict the court's conclusion on this legal question.
Return ONLY JSON: {{"applies": true/false (does the guidance govern this question?), "predicted_conclusion": "one or two sentences stating the outcome on this issue and the decisive reason"}}"""

GRADE = """Compare a predicted conclusion with the court's actual conclusion on the same legal issue.

PREDICTED:
{pred}

ACTUAL:
{actual}

Does the prediction reach the same outcome on the issue (who wins on this point / whether the claim, plea or exemption succeeds)? Reasoning may differ.
MATCH = same outcome. PARTIAL = same outcome on part of the issue, or right direction but materially different relief. MISMATCH = different outcome.
Answer with exactly one word: MATCH, PARTIAL or MISMATCH."""


def rule_text(r):
    preds = "\n".join(f"- {k}: {v}" for k, v in r["predicates"].items())
    exc = "; ".join(r.get("exceptions_or_limits") or []) or "none"
    return f"IF {r['rule']['if']} THEN {r['rule']['then']}\nPredicates:\n{preds}\nStatement: {r['definition']}\nExceptions/limits: {exc}"


def precedent_text(m):
    return f"Issue: {m['legal_issue']}\nPrinciple: {m.get('legal_principle')}\nConclusion: {m.get('conclusion')}"


def predict(question, facts, title, guidance):
    for _ in range(3):
        try:
            out = parse(call([{"role": "user", "content": PREDICT.format(question=question, facts=facts,
                                                                          guidance_title=title, guidance=guidance)}]))
            return out
        except Exception:
            pass
    return None


def grade(pred, actual):
    for _ in range(3):
        try:
            m = re.match(r"(MATCH|PARTIAL|MISMATCH)", call([{"role": "user", "content": GRADE.format(pred=pred, actual=actual)}]).strip().upper())
            if m:
                return m.group(1)
        except Exception:
            pass
    return None


def main(graph_dir, n=150):
    key = os.path.basename(graph_dir.rstrip("/")).split("_")[1]
    E = load_emb(key)
    nodes = load_train()[["node_id", "doc_id"]]
    idx_of = {x: i for i, x in enumerate(nodes.node_id)}
    doc = nodes.doc_id.values
    clusters = [c for c in map(json.loads, open(f"{graph_dir}/subcommunities.jsonl")) if 4 <= c["size"] <= 25]
    rng = random.Random(1)
    clusters = rng.sample(clusters, min(n, len(clusters)))
    held = [rng.randrange(c["size"]) for c in clusters]

    # full member records by node id (for the nearest-precedent baseline)
    rec = {m["node_id"]: m for c in map(json.loads, open(f"{graph_dir}/subcommunities.jsonl")) for m in c["members"]}

    def nearest_precedent(node):
        i = idx_of[node]
        s = E @ E[i]
        s[doc == doc[i]] = -1                      # other cases only
        for j in np.argsort(-s):
            if nodes.node_id[j] in rec:            # need a full IRAC record
                return rec[nodes.node_id[j]], float(s[j])
            if s[j] < 0.8:
                break
        return None, None

    def one(args):
        c, h = args
        target = c["members"][h]
        rest = dict(c, members=[m for k, m in enumerate(c["members"]) if k != h], size=c["size"] - 1)
        r = generalize(rest, E, idx_of, 25)
        if "error" in r:
            return dict(subcommunity=c["subcommunity"], held_out=target["node_id"], error=r["error"])
        facts = "\n".join(f"- {f}" for f in target.get("material_facts") or [])
        row = dict(subcommunity=c["subcommunity"], size=c["size"], held_out=target["node_id"],
                   coherence=r["coherence"], actual=target["conclusion"])
        p = predict(r["legal_question"], facts, "GENERALISED RULE FROM EARLIER CASES", rule_text(r))
        row.update(rule_applies=p and p.get("applies"), rule_pred=p and p.get("predicted_conclusion"))
        prec, sim = nearest_precedent(target["node_id"])
        if prec:
            q = predict(r["legal_question"], facts, "MOST SIMILAR EARLIER DECISION", precedent_text(prec))
            row.update(nn_node=prec["node_id"], nn_sim=sim, nn_applies=q and q.get("applies"),
                       nn_pred=q and q.get("predicted_conclusion"))
        row["rule_grade"] = grade(row["rule_pred"], row["actual"]) if row.get("rule_pred") else None
        row["nn_grade"] = grade(row["nn_pred"], row["actual"]) if row.get("nn_pred") else None
        return row

    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one, zip(clusters, held)))
    df = pd.DataFrame(rows)
    os.makedirs(f"{graph_dir}/eval", exist_ok=True)
    df.to_csv(f"{graph_dir}/eval/loo_outcomes.csv", index=False)

    ok = df.dropna(subset=["rule_grade", "nn_grade"])
    def acc(col, sub=ok):
        return dict(match=float((sub[col] == "MATCH").mean()),
                    match_or_partial=float(sub[col].isin(["MATCH", "PARTIAL"]).mean()))
    summary = dict(n_clusters=len(df), n_scored=len(ok), errors=int(df.get("error", pd.Series()).notna().sum()),
                   rule=acc("rule_grade"), nearest_precedent=acc("nn_grade"),
                   rule_by_coherence={k: dict(n=len(g), **acc("rule_grade", g)) for k, g in ok.groupby("coherence")},
                   both_match=float(((ok.rule_grade == "MATCH") & (ok.nn_grade == "MATCH")).mean()),
                   rule_only=float(((ok.rule_grade == "MATCH") & (ok.nn_grade != "MATCH")).mean()),
                   nn_only=float(((ok.rule_grade != "MATCH") & (ok.nn_grade == "MATCH")).mean()),
                   rule_said_not_applicable=float((ok.rule_applies == False).mean()))
    json.dump(summary, open(f"{graph_dir}/eval/loo_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:2]))
