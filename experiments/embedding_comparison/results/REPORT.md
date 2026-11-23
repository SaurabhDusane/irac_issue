# Experiment 2 — results report

Pipeline: embed issues → similarity graph → two-level Louvain → generalised rules per subcommunity → evaluation on held-out test issues.

## Step 1 — embedding model choice

Weak-label metrics at a matched edge budget (34,063 edges each). `*` = embedding includes legal_principle, which is where the statute labels come from, so its statute scores are inflated.

| model | statute@1 (statute not named in issue) | statute@10 (not named) | subcommunity statute purity | modularity | isolated |
|---|---|---|---|---|---|
| bge-base | 0.342 | 0.248 | 0.617 | 0.853 | 49.3% |
| bge-large | 0.335 | 0.245 | 0.597 | 0.889 | 43.8% |
| qwen3-0.6b | 0.378 | 0.287 | 0.756 | 0.873 | 55.2% |
| qwen3-0.6b+lp* | 0.540 | 0.423 | 0.824 | 0.921 | 52.3% |
| gemini-001 | 0.371 | 0.276 | 0.677 | 0.872 | 48.9% |
| gemini-001+lp* | 0.535 | 0.421 | 0.800 | 0.929 | 47.6% |

Blind LLM judge (Claude Sonnet): for issues where two models pick a different nearest cross-case neighbour, which neighbour raises the same legal question? (200 per pair, order randomised)

| A vs B | A wins | B wins | both | neither | sign-test p |
|---|---|---|---|---|---|
| gemini-001+lp vs gemini-001 | 58 | 51 | 26 | 65 | 0.566 |
| gemini-001 vs bge-base | 91 | 28 | 16 | 65 | 5.8e-09 |
| gemini-001 vs bge-large | 91 | 29 | 18 | 62 | 1.22e-08 |
| qwen3-0.6b+lp vs gemini-001 | 55 | 62 | 25 | 58 | 0.579 |
| qwen3-0.6b vs bge-base | 64 | 47 | 13 | 76 | 0.128 |
| qwen3-0.6b vs gemini-001 | 40 | 90 | 16 | 54 | 1.37e-05 |

## Step 2 — threshold (gemini-embedding-001)

Band precision = share of 40 random cross-case pairs in the score band judged to raise the same legal question.

| threshold | edges | isolated | communities | modularity | band precision |
|---|---|---|---|---|---|
| 0.86 | 369,174 | 4.5% | 101 | 0.684 | 27.5% |
| 0.87 | 197,818 | 10.7% | 260 | 0.741 | 22.5% |
| 0.88 | 109,129 | 21.2% | 557 | 0.800 | 40.0% |
| 0.89 | 61,154 | 34.6% | 918 | 0.846 | 45.0% |
| 0.90 | 34,532 | 48.6% | 1254 | 0.872 | 77.5% |
| 0.91 | 19,536 | 62.0% | 1280 | 0.889 | 80.0% |
| 0.92 | 10,982 | 73.2% | 1172 | 0.897 | 85.0% |
| 0.93 | 6,125 | 81.5% | 976 | 0.910 | 92.5% |
| 0.94 | 3,178 | 88.1% | 707 | 0.924 | 97.5% |
| 0.95 | 1,481 | 92.9% | 463 | 0.944 | 97.5% |

Chosen: **0.90** — the knee where band precision jumps from 45% to 78%.

## Step 3 — graph and Louvain

- 18,968 issues, 34,533 edges; residual (unmatched) issues: 9,227 (48.6%)
- 1,254 communities (modularity 0.872); 41 large communities split again (mean sub-modularity 0.576)
- 1,563 subcommunities with ≥2 issues; 7,083 issues sit in subcommunities of ≥5

## Step 4 — generalised rules

- 760 rules generated for subcommunities with ≥3 issues (0 failed)
- coherence: high 445, medium 300, low 15
- mean share of shown members the rule covers: 92.4%; outliers flagged: 785
- each rule carries a recurring-patterns summary (common facts, provisions, typical outcome, variations)

## Step 5 — evaluation on test issues (rule applicability)

_pending_

### Judge robustness

_pending_

## Outcome accuracy — leave-one-out on train

Hold out one member, build the rule from the rest, predict the held-out outcome from its material facts only, grade against the real conclusion. Baseline: same predictor given the single most similar other train decision.

| method | outcome MATCH | MATCH or PARTIAL |
|---|---|---|
| generalised rule | 83.3% | 97.3% |
| nearest precedent | 84.0% | 97.3% |

n = 150. Only the rule is right: 7.3%; only the precedent: 8.0%; both: 76.0%.
By rule coherence: high: 91.2% (n=80); low: 66.7% (n=3); medium: 74.6% (n=67)

## Rule conflicts

_pending_
