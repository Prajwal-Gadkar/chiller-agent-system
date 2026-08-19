"""
Pulls two 7-day chiller trend windows and runs the Data Validation Agent on
each, to see side by side whether one window looks suspiciously cleaner (or
dirtier) than the other. Read-only against the database.

NOTE: the original plan was to compare a confirmed-real week against a
"future" week from 2026-09/2026-10. Checking coverage showed that range has
zero rows for chiller sensors specifically — that data belongs to other
(non-chiller) machines in trendseriesmeterdata. The last week with real
chiller sensor data is 2026-07-02 to 2026-07-08 (chiller data tops out at
2026-07-08), so that's used as the second window instead.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pull_trend_data import build_wide_trend_df
from agents.data_validation import validate

WINDOWS = [
    ("confirmed_real", "2026-06-01", "2026-06-07"),
    ("last_available", "2026-07-02", "2026-07-08"),
]


def main():
    reports = {}

    for label, start, end in WINDOWS:
        print(f"Pulling window '{label}': {start} to {end} ...")
        wide, _ = build_wide_trend_df(start, end)
        if wide.empty:
            print(f"  No data returned for window '{label}'.")
            reports[label] = pd.DataFrame(columns=["column", "pct_flagged"])
            continue
        _, report_df = validate(wide)
        reports[label] = report_df[["column", "pct_flagged"]]
        print(f"  {len(wide)} rows, {wide['machineId'].nunique()} chillers.")

    labels = [label for label, _, _ in WINDOWS]
    merged = reports[labels[0]].rename(columns={"pct_flagged": f"pct_flagged_{labels[0]}"})
    for label in labels[1:]:
        merged = merged.merge(
            reports[label].rename(columns={"pct_flagged": f"pct_flagged_{label}"}),
            on="column",
            how="outer",
        )

    merged = merged.sort_values(f"pct_flagged_{labels[0]}", ascending=False).reset_index(drop=True)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 140)
    print("\nFlagged %% per column, side by side:")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
