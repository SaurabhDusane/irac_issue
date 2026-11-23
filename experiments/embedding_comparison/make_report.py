"""Collect every available result into results/REPORT.md (re-run any time;
sections whose inputs do not exist yet are marked pending)."""
import glob, json, os

import pandas as pd
from scipy.stats import binomtest

G = "out/graph_gemini-001_0.9"
L = []
w = L.append


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.1f}%"


w("# Experiment 2 — results report\n")
w("Pipeline: embed issues → similarity graph → two-level Louvain → generalised rules per subcommunity → evaluation on held-out test issues.\n")

# ---------------------------------------------------------------- step 1
w("## Step 1 — embedding model choice\n")
m = load("out/metrics.json")
if m:
    w("Weak-label metrics at a matched edge budget (34,063 edges each). `*` = embedding includes legal_principle, "
      "which is where the statute labels come from, so its statute scores are inflated.\n")
    w("| model | statute@1 (statute not named in issue) | statute@10 (not named) | subcommunity statute purity | modularity | isolated |")
    w("|---|---|---|---|---|---|")
    for k, r in m["per_model"].items():
        w(f"| {k}{'*' if '+lp' in k else ''} | {r['statute@1_unnamed']:.3f} | {r['statute@10_unnamed']:.3f} | "
          f"{r['sub_statute_purity']:.3f} | {r['modularity']:.3f} | {pct(r['isolated'])} |")
    w("")
js = sorted(glob.glob("out/judge_*.json"))
if js:
    w("Blind LLM judge (Claude Sonnet): for issues where two models pick a different nearest cross-case neighbour, "
      "which neighbour raises the same legal question? (200 per pair, order randomised)\n")
    w("| A vs B | A wins | B wins | both | neither | sign-test p |")
    w("|---|---|---|---|---|---|")
    for f in js:
        s = json.load(open(f))["summary"]; t = s["tally"]
        a, b = t[s["a"]], t[s["b"]]
        if t.get("error", 0) > 20:
            continue
        w(f"| {s['a']} vs {s['b']} | {a} | {b} | {t['both']} | {t['neither']} | {binomtest(a, a + b).pvalue:.3g} |")
    w("")

# ---------------------------------------------------------------- step 2
sw = load("out/threshold_sweep_gemini-001.json")
w("## Step 2 — threshold (gemini-embedding-001)\n")
if sw:
    w("Band precision = share of 40 random cross-case pairs in the score band judged to raise the same legal question.\n")
    w("| threshold | edges | isolated | communities | modularity | band precision |")
    w("|---|---|---|---|---|---|")
    for r in sw:
        w(f"| {r['threshold']:.2f} | {r['edges']:,} | {pct(r['isolated'])} | {r['communities_gt1']} | "
          f"{r['modularity']:.3f} | {pct(r['band_precision'])} |")
    w("\nChosen: **0.90** — the knee where band precision jumps from 45% to 78%.\n")

# ---------------------------------------------------------------- step 3
s3 = load(f"{G}/summary.json")
w("## Step 3 — graph and Louvain\n")
if s3:
    w(f"- {s3['nodes']:,} issues, {s3['edges']:,} edges; residual (unmatched) issues: {s3['residual_isolated']:,} ({pct(s3['residual_share'])})")
    w(f"- {s3['communities']:,} communities (modularity {s3['modularity']:.3f}); {s3['communities_split']} large communities split again "
      f"(mean sub-modularity {s3['mean_sub_modularity']:.3f})")
    w(f"- {s3['subcommunities_ge2']:,} subcommunities with ≥2 issues; {s3['issues_in_subcommunities_ge5']:,} issues sit in subcommunities of ≥5\n")

# ---------------------------------------------------------------- step 4
w("## Step 4 — generalised rules\n")
if os.path.exists(f"{G}/rules.jsonl"):
    rs = [json.loads(l) for l in open(f"{G}/rules.jsonl")]
    ok = [r for r in rs if "error" not in r]
    covered = sum(len(r["covered_members"]) for r in ok)
    coh = pd.Series([r["coherence"] for r in ok]).value_counts()
    w(f"- {len(ok)} rules generated for subcommunities with ≥3 issues ({len(rs) - len(ok)} failed)")
    w(f"- coherence: high {coh.get('high', 0)}, medium {coh.get('medium', 0)}, low {coh.get('low', 0)}")
    w(f"- mean share of shown members the rule covers: {pct(sum(len(r['covered_members']) / r['shown'] for r in ok) / max(len(ok), 1))}; "
      f"outliers flagged: {sum(len(r.get('outlier_members') or []) for r in ok)}")
    w(f"- each rule carries a recurring-patterns summary (common facts, provisions, typical outcome, variations)\n")
else:
    w("_pending_\n")

# ---------------------------------------------------------------- step 5
w("## Step 5 — evaluation on test issues (rule applicability)\n")
e = load(f"{G}/eval/summary.json")
if e:
    o = e["overall"]; k = e["k"]
    w(f"{o['n_issues']} test issues, top-{k} rules each, judged APPLIES / PARTIAL / NO.\n")
    w("| split | n | precision@1 | precision@1 incl. partial | precision@3 | hit@3 | hit@3 incl. partial |")
    w("|---|---|---|---|---|---|---|")
    for name, r in [("all", o)] + [(str(y), v) for y, v in e["by_year"].items()]:
        w(f"| {name} | {r['n_issues']} | {pct(r['precision_at_1'])} | {pct(r['precision_at_1_incl_partial'])} | "
          f"{pct(r[f'precision_at_{k}'])} | {pct(r[f'hit_at_{k}'])} | {pct(r[f'hit_at_{k}_incl_partial'])} |")
    w("\nAnswer-or-abstain at top-1 similarity τ (gold = top-1 rule judged APPLIES):\n")
    w("| τ | accuracy | precision | recall | F1 | answered |")
    w("|---|---|---|---|---|---|")
    for r in e["routing"]:
        w(f"| {r['tau']:.2f} | {pct(r['accuracy'])} | {pct(r['precision'])} | {pct(r['recall'])} | {pct(r['f1'])} | {pct(r['predicted_share'])} |")
    w("")
else:
    w("_pending_\n")

ag = load(f"{G}/eval/judge_agreement.json")
w("### Judge robustness\n")
if ag:
    w(f"Second judge {ag['judge2']} on {ag['n']} pairs: 3-class agreement {pct(ag['agreement_3class'])} "
      f"(κ = {ag['kappa_3class']:.2f}); APPLIES-vs-rest agreement {pct(ag['agreement_applies'])} (κ = {ag['kappa_applies']:.2f}). "
      f"APPLIES rate: {pct(ag['judge1_applies_rate'])} vs {pct(ag['judge2_applies_rate'])}. "
      f"Human spot-check sheet: `eval/human_review_pack.csv`.\n")
else:
    w("_pending_\n")

# ---------------------------------------------------------------- extras
w("## Outcome accuracy — leave-one-out on train\n")
lo = load(f"{G}/eval/loo_summary.json")
if lo:
    w("Hold out one member, build the rule from the rest, predict the held-out outcome from its material facts only, "
      "grade against the real conclusion. Baseline: same predictor given the single most similar other train decision.\n")
    w("| method | outcome MATCH | MATCH or PARTIAL |")
    w("|---|---|---|")
    w(f"| generalised rule | {pct(lo['rule']['match'])} | {pct(lo['rule']['match_or_partial'])} |")
    w(f"| nearest precedent | {pct(lo['nearest_precedent']['match'])} | {pct(lo['nearest_precedent']['match_or_partial'])} |")
    w(f"\nn = {lo['n_scored']}. Only the rule is right: {pct(lo['rule_only'])}; only the precedent: {pct(lo['nn_only'])}; both: {pct(lo['both_match'])}.")
    w("By rule coherence: " + "; ".join(f"{c}: {pct(v['match'])} (n={v['n']})" for c, v in lo["rule_by_coherence"].items()) + "\n")
else:
    w("_pending_\n")

w("## Rule conflicts\n")
rc = load(f"{G}/eval/rule_conflicts_summary.json")
if rc:
    w(f"{rc['candidate_pairs']} close rule pairs from different clusters checked. Relations: "
      + ", ".join(f"{k} {v}" for k, v in rc["relation"].items()) + ".")
    if rc["conflict_explanation"]:
        w("Conflicts explained as: " + ", ".join(f"{k} {v}" for k, v in rc["conflict_explanation"].items()) + ".")
    w("Details: `eval/rule_conflicts.csv`.\n")
else:
    w("_pending_\n")

os.makedirs("results", exist_ok=True)
open("results/REPORT.md", "w").write("\n".join(L))
print("\n".join(L))
