"""Robustness of the step-5 labels: re-judge a random sample of (test issue,
rule) pairs with a second LLM from a different provider and report agreement
(raw and Cohen's kappa, 3-class and APPLIES-vs-rest). Also writes a CSV of
50 pairs + 25 rules for human spot-checking.

usage: python judge_agreement.py <graph_dir> [n=300] [judge2=gcp-deepmind/geminipro3_1]
"""
import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from sklearn.metrics import cohen_kappa_score

import evaluate
from load_data import load_test


def ask(model, prompt):
    for _ in range(4):
        try:
            r = requests.post(evaluate.API, json=dict(model=model, max_tokens=2048, temperature=0,
                                                      messages=[{"role": "user", "content": prompt}]), timeout=180)
            m = re.search(r"(APPLIES|PARTIAL|NO)", r.json()["choices"][0]["message"]["content"].strip().upper())
            if m:
                return m.group(1)
        except Exception:
            pass
    return None


def main(graph_dir, n=300, judge2="gcp-deepmind/geminipro3_1"):
    rules = {r["subcommunity"]: r for r in map(json.loads, open(f"{graph_dir}/rules.jsonl")) if "error" not in r}
    pairs = pd.read_csv(f"{graph_dir}/eval/pairs.csv", dtype={"subcommunity": str})
    test = load_test().set_index("node_id")
    sample = pairs.sample(min(n, len(pairs)), random_state=0).reset_index(drop=True)
    prompts = [evaluate.rule_prompt(test.legal_issue[t], rules[s]) for t, s in zip(sample.test_node, sample.subcommunity)]
    with ThreadPoolExecutor(8) as ex:
        sample["label2"] = list(ex.map(lambda p: ask(judge2, p), prompts))
    s = sample.dropna(subset=["label2"])
    a, b = s.label, s.label2
    res = dict(judge1=evaluate.JUDGE, judge2=judge2, n=len(s),
               agreement_3class=float((a == b).mean()), kappa_3class=float(cohen_kappa_score(a, b)),
               agreement_applies=float(((a == "APPLIES") == (b == "APPLIES")).mean()),
               kappa_applies=float(cohen_kappa_score(a == "APPLIES", b == "APPLIES")),
               judge1_applies_rate=float((a == "APPLIES").mean()), judge2_applies_rate=float((b == "APPLIES").mean()),
               confusion=pd.crosstab(a, b).to_dict())
    os.makedirs(f"{graph_dir}/eval", exist_ok=True)
    s.to_csv(f"{graph_dir}/eval/judge_agreement_pairs.csv", index=False)
    json.dump(res, open(f"{graph_dir}/eval/judge_agreement.json", "w"), indent=1)
    print(json.dumps(res, indent=1))

    # human spot-check pack
    hp = s.sample(min(50, len(s)), random_state=1)
    rows = []
    for _, r in hp.iterrows():
        ru = rules[r.subcommunity]
        rows.append(dict(kind="step5_pair", test_node=r.test_node, test_issue=test.legal_issue[r.test_node],
                         subcommunity=r.subcommunity, rule_question=ru["legal_question"], rule_statement=ru["definition"],
                         rule_if=ru["rule"]["if"], rule_then=ru["rule"]["then"], sim=round(r.sim, 3),
                         judge1=r.label, judge2=r.label2, human_label="", human_notes=""))
    for sc in random.Random(2).sample(sorted(rules), min(25, len(rules))):
        ru = rules[sc]
        rows.append(dict(kind="step4_rule", subcommunity=sc, rule_question=ru["legal_question"],
                         rule_statement=ru["definition"], rule_if=ru["rule"]["if"], rule_then=ru["rule"]["then"],
                         coherence=ru["coherence"], size=ru["size"],
                         typical_outcome=ru["recurring_patterns"].get("typical_outcome"),
                         human_label="(faithful / partly / unfaithful)", human_notes=""))
    pd.DataFrame(rows).to_csv(f"{graph_dir}/eval/human_review_pack.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:2]), *a[2:3])
