"""Step 4: one generalised rule + recurring-patterns summary per subcommunity.

Reads out/graph_<model>_<thr>/subcommunities.jsonl (from build_graph.py) and
writes rules.jsonl + rules_review.md next to it.

The rule uses the same format as the train IRAC records ("if"/"then" over
snake_case predicates, each predicate defined in plain language, plus a
one-sentence "definition"), so step 5 can apply it to test issues the same way.

usage: python generalize.py <graph_dir> [min_size=3] [limit=0 (all)] [max_members=25]
"""
import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

API = "https://api-main.aiml.asu.edu/v1/chat/completions"   # auth injected by proxy
MODEL = "aws/claude5_5_sonnet"

SYSTEM = """You are a legal analyst generalising holdings of the Supreme Court of India.

You receive a cluster of legal issues from different judgments. Each member gives the issue as decided, the material facts, the legal principle, the conclusion and the case-specific rule (an if/then over named predicates).

Your job is to write ONE generalised rule that captures the legal point the members share, and to summarise the recurring patterns across them.

Requirements for the generalised rule:
- It must be general: no party names, places, dates, amounts or other case-specific details. It should apply to a new case with the same legal point.
- It must be faithful: state only what the members' principles and conclusions support. Do not import doctrine that none of the members state.
- Use the same format as the members' rules: "if" is a boolean expression over snake_case predicates using and, or, not and parentheses; "then" is a single snake_case outcome predicate. Define EVERY predicate used in "if" and "then" in "predicates", each as one plain-language sentence a reader could check against the facts of a new case.
- Prefer the smallest set of conditions that every covered member satisfies. If the members reach opposite outcomes, find the condition that separates them and put it in the rule (or in exceptions_or_limits) rather than ignoring it.
- If the members do not share one legal point, still write the rule for the largest coherent group, list the rest as outliers, set coherence to "low" and say how the cluster should be split.

Return ONLY a JSON object, no prose, with exactly these keys:
{
  "legal_question": "the shared legal question, phrased as a question",
  "rule": {"if": "...", "then": "..."},
  "predicates": {"name": "definition", ...},
  "definition": "the generalised rule in one or two plain sentences",
  "exceptions_or_limits": ["conditions under which the rule does not apply or the outcome differs"],
  "covered_members": ["node_id", ...],
  "outlier_members": [{"node_id": "...", "reason": "..."}],
  "recurring_patterns": {
    "common_material_facts": ["fact patterns that recur across members"],
    "statutes_and_provisions": ["provisions relied on, as cited by the members"],
    "typical_outcome": "how these issues are usually resolved, with rough counts",
    "variations": ["how members differ on facts or reasoning"]
  },
  "coherence": "high | medium | low",
  "split_suggestion": "how to split the cluster, or null"
}"""


def member_block(m):
    r = m.get("rule") or {}
    facts = "; ".join(m.get("material_facts") or [])
    return (f"[{m['node_id']}]\n"
            f"Issue: {m['legal_issue']}\n"
            f"Material facts: {facts}\n"
            f"Legal principle: {m.get('legal_principle')}\n"
            f"Conclusion: {m.get('conclusion')}\n"
            f"Rule: IF {r.get('if')} THEN {r.get('then')}")


IDENT = re.compile(r"\b[a-z][a-z0-9_]*\b")


def validate(out, member_ids):
    errs = []
    for k in ("legal_question", "rule", "predicates", "definition", "covered_members", "recurring_patterns", "coherence"):
        if k not in out:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    names = set(IDENT.findall(out["rule"].get("if", ""))) - {"and", "or", "not"}
    names |= {out["rule"].get("then", "")}
    undefined = sorted(n for n in names if n not in out["predicates"])
    if undefined:
        errs.append(f"predicates used but not defined: {undefined}")
    unknown = [n for n in out["covered_members"] if n not in member_ids]
    if unknown:
        errs.append(f"covered_members not in cluster: {unknown}")
    return errs


def call(messages):
    r = requests.post(API, json=dict(model=MODEL, max_tokens=8000, temperature=0, messages=messages), timeout=600)
    return r.json()["choices"][0]["message"]["content"]


def parse(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


def pick_members(members, emb, idx_of, k):
    """Keep the k members closest to the cluster centroid (most typical)."""
    if len(members) <= k:
        return members
    V = emb[[idx_of[m["node_id"]] for m in members]]
    c = V.mean(0); c /= np.linalg.norm(c)
    order = np.argsort(-(V @ c))[:k]
    return [members[i] for i in sorted(order)]


def generalize(cluster, emb, idx_of, max_members):
    shown = pick_members(cluster["members"], emb, idx_of, max_members)
    ids = {m["node_id"] for m in shown}
    user = (f"Cluster {cluster['subcommunity']} has {cluster['size']} issues"
            + (f"; the {len(shown)} most typical are shown" if len(shown) < cluster["size"] else "")
            + ".\n\n" + "\n\n".join(member_block(m) for m in shown))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    last_err = None
    for attempt in range(3):
        try:
            text = call(messages)
            out = parse(text)
            errs = validate(out, ids)
            if not errs:
                return dict(subcommunity=cluster["subcommunity"], size=cluster["size"], shown=len(shown),
                            attempts=attempt + 1, **out)
            last_err = errs
            messages = messages + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": "Fix these problems and return the full JSON again: " + "; ".join(errs)}]
        except Exception as e:  # network / JSON errors: retry from scratch
            last_err = [repr(e)[:200]]
    return dict(subcommunity=cluster["subcommunity"], size=cluster["size"], error=last_err)


def review_md(results, path):
    with open(path, "w") as f:
        f.write("# Generalised rules (step 4) — review\n\n")
        for r in results:
            f.write(f"## {r['subcommunity']} ({r['size']} issues)\n\n")
            if "error" in r:
                f.write(f"**failed:** {r['error']}\n\n"); continue
            f.write(f"**Question:** {r['legal_question']}\n\n**Rule:** IF `{r['rule']['if']}` THEN `{r['rule']['then']}`\n\n"
                    f"**Definition:** {r['definition']}\n\n")
            for k, v in r["predicates"].items():
                f.write(f"- `{k}`: {v}\n")
            p = r["recurring_patterns"]
            f.write(f"\n**Coherence:** {r['coherence']} · covered {len(r['covered_members'])}/{r['shown']} shown"
                    f" · outliers {len(r.get('outlier_members') or [])}\n\n"
                    f"**Typical outcome:** {p.get('typical_outcome')}\n\n"
                    f"**Recurring facts:** {'; '.join(p.get('common_material_facts') or [])}\n\n"
                    f"**Provisions:** {'; '.join(p.get('statutes_and_provisions') or [])}\n\n")
            if r.get("exceptions_or_limits"):
                f.write("**Exceptions/limits:** " + "; ".join(r["exceptions_or_limits"]) + "\n\n")
            if r.get("split_suggestion"):
                f.write(f"**Split suggestion:** {r['split_suggestion']}\n\n")


def main(graph_dir, min_size=3, limit=0, max_members=25):
    key = os.path.basename(graph_dir.rstrip("/")).split("_")[1]
    emb = np.load(f"out/emb_{key}.npy")
    nodes = pd.read_csv("out/nodes.csv", dtype=str)
    idx_of = {n: i for i, n in enumerate(nodes.node_id)}
    clusters = [c for c in map(json.loads, open(f"{graph_dir}/subcommunities.jsonl")) if c["size"] >= min_size]
    if limit:
        clusters = random.Random(0).sample(clusters, min(limit, len(clusters)))
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda c: generalize(c, emb, idx_of, max_members), clusters))
    tag = f"_pilot{limit}" if limit else ""
    with open(f"{graph_dir}/rules{tag}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    review_md(results, f"{graph_dir}/rules{tag}_review.md")
    ok = [r for r in results if "error" not in r]
    print(json.dumps(dict(clusters=len(results), ok=len(ok),
                          coherence={c: sum(r["coherence"] == c for r in ok) for c in ("high", "medium", "low")},
                          mean_covered_share=float(np.mean([len(r["covered_members"]) / r["shown"] for r in ok])) if ok else None)))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:4]))
