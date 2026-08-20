"""
Data Validation Agent — sanity-bound gate on every reading before any other
agent sees it. Read-only: never drops or mutates rows, only annotates them.

Per CLAUDE.md's confirmed corruption findings: Flow, KW, and Fan_speed sensors
can jump from a sane range to six-figure impossible values, corruption varies
by sensor and by chiller, and status/flag columns are continuous in [0, 1]
rather than strict binaries. This agent applies a generous, physically-
plausible bound per column (inferred from column-name keywords) and flags
readings outside it — it does not decide what to do with flagged readings,
that's left to the consuming step.
"""

import numpy as np
import pandas as pd

METADATA_COLUMNS = {"machineId", "timestamp", "status", "Criticality"}

# (keywords to match in the column name, lowercase, min, max, rule label)
# First matching rule wins, so more specific keywords must come first.
BOUND_RULES = [
    (["status", "alarm", "fault", "trip"], 0.0, 1.0, "duty_cycle_status"),
    (["percent", "%", "deviation", "load", "performance"], -10.0, 110.0, "percentage"),
    (["hours", "runhours"], 0.0, 200000.0, "cumulative_hours"),
    (["temperature", "temp", "wet bulb", "cwet"], -20.0, 60.0, "temperature"),
    (["pressure", "press"], -50.0, 500.0, "pressure"),
    (["flow"], -10.0, 5000.0, "flow"),
    (["speed"], -10.0, 110.0, "speed_pct"),
    (["kw", "power", "ikw"], -10.0, 5000.0, "power"),
    (["setpoint"], -20.0, 60.0, "setpoint_temperature"),
]

STATISTICAL_FALLBACK_SIGMA = 6


def infer_bounds(column_name):
    """Return (min, max, rule_label) for a column name, or a statistical fallback."""
    name_lower = column_name.lower()
    # Replace 'compressor' with 'comp' to prevent 'press' inside 'compressor' from matching pressure rule
    name_check = name_lower.replace("compressor", "comp")
    for keywords, lo, hi, label in BOUND_RULES:
        if any(kw in name_check for kw in keywords):
            return lo, hi, label
    return None, None, "statistical_fallback"


def validate(df):
    """
    Apply per-column physical sanity bounds to a wide-format DataFrame.

    Returns (flagged_df, report_df):
      - flagged_df: a copy of df with one extra "<column>_flagged" boolean
        column per validated sensor column. Original values are untouched.
      - report_df: one row per validated column with the bound rule used,
        the range applied, and the % of readings flagged.
    """
    flagged_df = df.copy()
    report_rows = []
    new_cols = {}

    value_columns = [
        c
        for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in value_columns:
        series = df[col]
        lo, hi, rule = infer_bounds(col)

        if rule == "statistical_fallback":
            mean = series.mean()
            std = series.std()
            lo = mean - STATISTICAL_FALLBACK_SIGMA * std
            hi = mean + STATISTICAL_FALLBACK_SIGMA * std

        flagged = ((series < lo) | (series > hi)).fillna(False)
        new_cols[f"{col}_flagged"] = flagged.values

        n_total = series.notna().sum()
        n_flagged = int(flagged.sum())
        pct_flagged = (n_flagged / n_total * 100) if n_total > 0 else np.nan

        report_rows.append(
            {
                "column": col,
                "rule": rule,
                "bound_min": lo,
                "bound_max": hi,
                "n_total": int(n_total),
                "n_flagged": n_flagged,
                "pct_flagged": pct_flagged,
            }
        )

    if new_cols:
        flagged_cols_df = pd.DataFrame(new_cols, index=df.index)
        flagged_df = pd.concat([df.copy(), flagged_cols_df], axis=1)

    report_df = pd.DataFrame(report_rows).sort_values("pct_flagged", ascending=False).reset_index(drop=True)
    return flagged_df, report_df




def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the data validation agent on a wide trend CSV.")
    parser.add_argument("csv_path", help="Path to a wide-format trend CSV (e.g. data/trend_wide.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    _, report_df = validate(df)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
