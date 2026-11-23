# IRAC records (train) and issues (test)

## Stats

| | Cases | Issues | Issues per case (mean / median / max) |
|---|---|---|---|
| Train | 6,668 | 18,970 | 2.84 / 3 / 16 |
| Test | 286 | 1,155 | 4.04 / 4 / 13 |

- Train: 1 case has no issue; 2 issues have an IRAC but no rule (816126 issue 3, 738221 issue 1), so 18,968 issues are embedded, clustered and become rules.
- Test by year: 2023 102 cases, 386 issues; 2024 99 cases, 449 issues; 2025 85 cases, 320 issues.

## index.csv

One row per case (6,954: 6,668 train, 286 test): `doc_id`, `split` (train / test), `year`, `case_name` and `date` (from the first line of the judgment text), `issues` (number of issues in the file), `file` (path inside this folder) and `drive_path` (the judgment PDF in the Google Drive dataset).
