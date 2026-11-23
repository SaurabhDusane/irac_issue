"""Blind pairwise LLM judgement of nearest neighbours.

For random issues where model A and model B pick a different top-1 cross-case
neighbour, an LLM sees the issue plus both neighbours (order randomised, model
names hidden) and says which neighbour raises the same legal question.

usage: python judge.py <model_a> <model_b> [n=200]
"""
import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from load_data import load_train

API = "https://api-main.aiml.asu.edu/v1/chat/completions"   # auth injected by proxy
JUDGE = "aws/claude5_5_sonnet"
PROMPT = """You are helping evaluate legal-issue clustering for Indian Supreme Court judgments.

TARGET issue:
{t}

CANDIDATE 1:
{c1}

CANDIDATE 2:
{c2}

Which candidate raises the SAME legal question as the target (same legal point, so one generalised rule could cover both) — not merely the same topic or similar wording?
Answer with exactly one word: 1, 2, BOTH or NEITHER."""


def ask(t, c1, c2):
    for _ in range(4):
        try:
            r = requests.post(API, json=dict(model=JUDGE, max_tokens=5, temperature=0,
                messages=[{"role": "user", "content": PROMPT.format(t=t, c1=c1, c2=c2)}]), timeout=120)
            out = r.json()["choices"][0]["message"]["content"].strip().upper()
            m = re.match(r"(BOTH|NEITHER|1|2)", out)
            if m:
                return m.group(1)
        except Exception:
            pass
    return "ERROR"


def main(a, b, n=200):
    df = load_train()
    na, nb = np.load(f"out/nbr_{a}.npy")[:, 0], np.load(f"out/nbr_{b}.npy")[:, 0]
    diff = np.nonzero(na != nb)[0]
    rng = random.Random(0)
    sample = rng.sample(diff.tolist(), min(n, len(diff)))
    jobs = []
    for i in sample:
        flip = rng.random() < 0.5
        c = [na[i], nb[i]][::-1] if flip else [na[i], nb[i]]
        jobs.append((i, flip, c))
    with ThreadPoolExecutor(8) as ex:
        answers = list(ex.map(lambda j: ask(df.legal_issue[j[0]], df.legal_issue[j[2][0]], df.legal_issue[j[2][1]]), jobs))
    rows, tally = [], {a: 0, b: 0, "both": 0, "neither": 0, "error": 0}
    for (i, flip, c), ans in zip(jobs, answers):
        if ans in ("1", "2"):
            first_is_a = not flip
            win = a if (ans == "1") == first_is_a else b
        else:
            win = ans.lower()
        tally[win] += 1
        rows.append(dict(issue=df.node_id[i], winner=win, target=df.legal_issue[i],
                         **{f"nbr_{a}": df.legal_issue[na[i]], f"nbr_{b}": df.legal_issue[nb[i]]}))
    res = dict(a=a, b=b, judge=JUDGE, top1_disagreement_rate=len(diff) / len(df), n=len(jobs), tally=tally)
    json.dump(dict(summary=res, rows=rows), open(f"out/judge_{a}_vs_{b}.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:]))
