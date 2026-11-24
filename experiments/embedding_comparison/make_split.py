"""Hold out train_val cases (by CASE, so all issues of a case stay together).

  python make_split.py                      # PLACEHOLDER: 700 random train cases (seed 42)
  python make_split.py --from <file>        # use the team's list (one doc_id per line, or a CSV with a doc_id column)

Writes splits/train_val_cases.txt (doc ids) and splits/split_summary.json.
Everything downstream (build_graph --exclude, generalize, refine) only sees
train_fit; validate.py is the only script that reads train_val.
"""
import json, os, random, sys

import pandas as pd

from load_data import load_train

N_VAL = 700
SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")


def main(argv):
    df = load_train()
    cases = sorted(df.doc_id.unique())
    if "--from" in argv:
        src = argv[argv.index("--from") + 1]
        if src.endswith(".csv"):
            val = pd.read_csv(src, dtype=str).doc_id.str.strip().tolist()
        else:
            val = [l.strip() for l in open(src) if l.strip()]
        unknown = sorted(set(val) - set(cases))
        if unknown:
            print(f"warning: {len(unknown)} ids are not train cases with issues, ignored (e.g. {unknown[:3]})")
        val = sorted(set(val) & set(cases))
        source = f"team list ({os.path.basename(src)})"
    else:
        val = sorted(random.Random(SEED).sample(cases, N_VAL))
        source = f"PLACEHOLDER random {N_VAL} cases, seed {SEED}"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "train_val_cases.txt"), "w").write("\n".join(val) + "\n")
    is_val = df.doc_id.isin(set(val))
    summary = dict(source=source, train_fit_cases=len(cases) - len(val), train_val_cases=len(val),
                   train_fit_issues=int((~is_val).sum()), train_val_issues=int(is_val.sum()))
    json.dump(summary, open(os.path.join(OUT, "split_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


def val_cases():
    p = os.path.join(OUT, "train_val_cases.txt")
    return set(l.strip() for l in open(p) if l.strip())


if __name__ == "__main__":
    main(sys.argv[1:])
