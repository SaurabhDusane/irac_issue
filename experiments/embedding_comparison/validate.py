"""Score finished rules on the held-out train_val issues (never used to build
the graph, clusters or rules). Each val issue has a real IRAC record, so we can
check against real answers. Two things are measured:

A. RULE MATCH (comparable to the 10%-holdout "does the top retrieved rule test
   the same legal point" metric): retrieve the top-k rules for every val issue
   (cosine between the issue and each rule's train_fit cluster centroid) and a
   judge compares each rule with the issue's OWN gold rule + principle:
   SAME_POINT / RELATED / DIFFERENT.
   -> top-1 accuracy, hit@k, and answer-or-abstain precision/recall/F1 at tau.

B. OUTCOME (comparable to "right in substance / exact match"): for val issues
   whose top-1 rule clears tau ("assigned"), predict the outcome from the
   issue's material facts + the rule (the predictor never sees the issue's
   principle or conclusion), then a judge compares it with the real conclusion:
   EXACT (same outcome and same decisive reason) / SUBSTANCE (same outcome,
   reason differs or is incomplete) / WRONG.
   -> right in substance = EXACT + SUBSTANCE; exact = EXACT.
   With --baseline the same predictor is also given the single most similar
   train_fit decision instead of the rule.

Judgements are cached in <graph_dir>/val/*.jsonl, so an interrupted run resumes.

usage: python validate.py <graph_dir> [--tau 0.90] [--k 3] [--baseline] [--limit N]
"""
import json, os, random, re, sys, threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from evaluate import prf
from llm import GEN_MODEL, JUDGE_MODEL, WORKERS, chat
from load_data import DATA, load_emb, load_train
from loo_eval import precedent_text, rule_text
from generalize import parse

MATCH = """A held-out legal issue with the rule the court actually applied:

ISSUE: {issue}
COURT'S PRINCIPLE: {principle}
COURT'S RULE: IF {gif} THEN {gthen}

A generalised rule retrieved from other cases:
Question: {question}
Rule: IF {rif} THEN {rthen}
Statement: {definition}

Does the retrieved rule test the SAME legal point as the court's rule (same statute or doctrine and the same legal test)?
SAME_POINT = same legal point; RELATED = same area or a component/neighbouring point; DIFFERENT = different legal point.
Answer with exactly one word: SAME_POINT, RELATED or DIFFERENT."""

PREDICT = """You are predicting how the Supreme Court of India decided a legal issue.

LEGAL QUESTION:
{question}

MATERIAL FACTS:
{facts}

{title}:
{guidance}

Using only the guidance and the facts, predict the court's conclusion on this question.
Return ONLY JSON: {{"applies": true/false, "predicted_conclusion": "one or two sentences: the outcome on this issue and the decisive reason"}}"""

GRADE = """Compare a predicted conclusion with the court's actual conclusion on the same issue.

PREDICTED:
{pred}

ACTUAL:
{actual}

EXACT = same outcome on the issue and the same decisive legal reason.
SUBSTANCE = same outcome (who wins on this point / whether the claim, plea or exemption succeeds), but the reason differs or is incomplete.
WRONG = different outcome.
Answer with exactly one word: EXACT, SUBSTANCE or WRONG."""


class Cache:
    def __init__(self, path):
        self.path, self.d, self.lock = path, {}, threading.Lock()
        if os.path.exists(path):
            for l in open(path):
                j = json.loads(l); self.d[j["key"]] = j["value"]

    def get(self, key, fn):
        if key in self.d:
            return self.d[key]
        v = fn()
        if v is not None:
            with self.lock, open(self.path, "a") as f:
                f.write(json.dumps(dict(key=key, value=v)) + "\n"); self.d[key] = v
        return v


def ask_label(prompt, labels, model):
    pat = re.compile("(" + "|".join(labels) + ")")
    for _ in range(4):
        try:
            m = pat.search(chat([{"role": "user", "content": prompt}], model=model, max_tokens=2048, timeout=180).upper())
            if m:
                return m.group(1)
        except Exception:
            pass
    return None


def ask_json(prompt, model):
    for _ in range(3):
        try:
            return parse(chat([{"role": "user", "content": prompt}], model=model, max_tokens=4096, timeout=300))
        except Exception:
            pass
    return None


def main(argv):
    graph_dir = argv[0]
    opt = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d
    tau, k, limit, baseline = opt("--tau", 0.90), opt("--k", 3), opt("--limit", 0), "--baseline" in argv
    key = os.path.basename(graph_dir.rstrip("/")).split("_")[1]
    df = load_train()
    E = load_emb(key)
    asg = pd.read_csv(f"{graph_dir}/assignments.csv", dtype={"subcommunity": str, "doc_id": str})
    if "split" not in asg:
        sys.exit("assignments.csv has no split column: build the graph with build_graph.py --fit")
    is_val = (asg.split == "val").values
    rules = [r for r in map(json.loads, open(f"{graph_dir}/rules.jsonl")) if "error" not in r]
    C = []
    for r in rules:
        m = (asg.subcommunity.values == r["subcommunity"]) & ~is_val
        v = E[m].mean(0); C.append(v / np.linalg.norm(v))
    C = np.array(C)

    raw = {}
    def rec(i):
        d = df.doc_id[i]
        if d not in raw:
            raw[d] = json.load(open(os.path.join(DATA, "train", f"{d}.json")))["issues"]
        return raw[d][df.issue_no[i] - 1]

    val_idx = np.nonzero(is_val)[0]
    if limit:
        val_idx = np.array(sorted(random.Random(0).sample(val_idx.tolist(), limit)))
    S = E[val_idx] @ C.T
    top = np.argsort(-S, 1)[:, :k]
    out = f"{graph_dir}/val"; os.makedirs(out, exist_ok=True)
    mcache, pcache, gcache = Cache(f"{out}/match.jsonl"), Cache(f"{out}/predict.jsonl"), Cache(f"{out}/grade.jsonl")
    print(f"{len(rules)} rules; {len(val_idx)} val issues; k={k}; tau={tau}; gen={GEN_MODEL}; judge={JUDGE_MODEL}", flush=True)

    # A. rule match for top-k
    jobs = [(row, rank) for row in range(len(val_idx)) for rank in range(k)]

    def do_match(job):
        row, rank = job
        i, r = val_idx[row], rules[top[row, rank]]
        g = rec(i)
        p = MATCH.format(issue=g["legal_issue"], principle=g.get("legal_principle"), gif=g["rule"]["if"], gthen=g["rule"]["then"],
                         question=r["legal_question"], rif=r["rule"]["if"], rthen=r["rule"]["then"], definition=r["definition"])
        return mcache.get(f"{df.node_id[i]}|{r['subcommunity']}",
                          lambda: ask_label(p, ["SAME_POINT", "RELATED", "DIFFERENT"], JUDGE_MODEL))

    with ThreadPoolExecutor(WORKERS) as ex:
        labels = list(ex.map(do_match, jobs))
    lab = np.array(labels, dtype=object).reshape(len(val_idx), k)

    # B. outcome for assigned issues
    assigned = np.nonzero(S[np.arange(len(val_idx)), top[:, 0]] >= tau)[0]
    fit_idx = np.nonzero(~is_val)[0]
    doc = df.doc_id.values

    def do_outcome(row):
        i = val_idx[row]; g = rec(i); r = rules[top[row, 0]]
        facts = "\n".join(f"- {x}" for x in g.get("material_facts") or [])
        res = {}
        guides = [("rule", "GENERALISED RULE FROM EARLIER CASES", rule_text(r))]
        if baseline:
            s = E[fit_idx] @ E[i]; s[doc[fit_idx] == doc[i]] = -1
            j = fit_idx[int(np.argmax(s))]
            guides.append(("precedent", "MOST SIMILAR EARLIER DECISION", precedent_text(rec(j))))
        for name, title, guide in guides:
            pred = pcache.get(f"{df.node_id[i]}|{name}|{r['subcommunity']}",
                              lambda: ask_json(PREDICT.format(question=r["legal_question"], facts=facts, title=title, guidance=guide), GEN_MODEL))
            if not pred:
                continue
            grade = gcache.get(f"{df.node_id[i]}|{name}|{r['subcommunity']}",
                               lambda: ask_label(GRADE.format(pred=pred.get("predicted_conclusion"), actual=g.get("conclusion")),
                                                 ["EXACT", "SUBSTANCE", "WRONG"], JUDGE_MODEL))
            res[name] = dict(applies=pred.get("applies"), pred=pred.get("predicted_conclusion"), grade=grade)
        return row, res

    with ThreadPoolExecutor(WORKERS) as ex:
        outcomes = dict(ex.map(do_outcome, assigned))

    # ---- tables
    rows = []
    for row in range(len(val_idx)):
        i = val_idx[row]; r = rules[top[row, 0]]
        o = outcomes.get(row, {})
        rows.append(dict(node_id=df.node_id[i], top1_rule=r["subcommunity"], top1_sim=float(S[row, top[row, 0]]),
                         top1_coherence=r.get("coherence"), assigned=row in outcomes,
                         **{f"match_{n + 1}": lab[row, n] for n in range(k)},
                         rule_grade=o.get("rule", {}).get("grade"), precedent_grade=o.get("precedent", {}).get("grade"),
                         actual=rec(i).get("conclusion"), rule_pred=o.get("rule", {}).get("pred")))
    t = pd.DataFrame(rows)
    t.to_csv(f"{out}/val_results.csv", index=False)

    same1 = t.match_1 == "SAME_POINT"
    hitk = t[[f"match_{n + 1}" for n in range(k)]].eq("SAME_POINT").any(axis=1)
    a = t[t.assigned]
    def outcome_stats(col):
        g = a[col].dropna()
        return dict(n=int(len(g)), right_in_substance=float(g.isin(["EXACT", "SUBSTANCE"]).mean()) if len(g) else None,
                    exact=float((g == "EXACT").mean()) if len(g) else None)
    summary = dict(
        graph=graph_dir, rules=len(rules), val_issues=len(t), tau=tau, k=k, gen_model=GEN_MODEL, judge_model=JUDGE_MODEL,
        assigned=int(t.assigned.sum()), assigned_share=float(t.assigned.mean()),
        rule_match=dict(top1_same_point=float(same1.mean()),
                        top1_same_or_related=float(t.match_1.isin(["SAME_POINT", "RELATED"]).mean()),
                        **{f"hit_at_{k}": float(hitk.mean())},
                        at_tau=prf(t.assigned.values, same1.values),
                        assigned_precision=float(same1[t.assigned].mean()) if t.assigned.any() else None),
        outcome_rule=outcome_stats("rule_grade"),
        outcome_precedent=outcome_stats("precedent_grade") if baseline else None,
        by_coherence={c: dict(n=int(len(g)), top1_same_point=float((g.match_1 == "SAME_POINT").mean()),
                              right_in_substance=float(g.rule_grade.dropna().isin(["EXACT", "SUBSTANCE"]).mean())
                              if g.rule_grade.notna().any() else None)
                      for c, g in t.groupby("top1_coherence")},
        routing=[dict(tau=round(float(x), 3), **prf(t.top1_sim.values >= x, same1.values)) for x in np.arange(0.84, 0.95, 0.01)])
    json.dump(summary, open(f"{out}/val_summary.json", "w"), indent=1)
    print(json.dumps({k2: v for k2, v in summary.items() if k2 != "routing"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
