"""
Diagnostic Script for Forecast Agent Failures.
Analyzes worst-performing chillers (2762, 2824, 2830) to identify regime shifts,
near-zero test power readings causing MAPE blowups, and Pearson correlation shifts.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add repo root to path for agent imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate

TARGET_CHILLERS = [2762, 2824, 2830]


def diagnose_chiller(df_wide, df_types, m_id):
    """Diagnose train vs test window distribution, near-zero power readings, and Pearson r."""
    chiller_df = df_wide[df_wide["machineId"] == m_id].copy()
    flagged_df, _ = validate(chiller_df)

    flow_col = "Flow ValueY"
    power_col = "KW ValueY"
    flow_flag = f"{flow_col}_flagged"
    power_flag = f"{power_col}_flagged"

    # Valid mask
    mask = (
        flagged_df[flow_col].notna() &
        flagged_df[power_col].notna()
    )
    if flow_flag in flagged_df.columns:
        mask &= (~flagged_df[flow_flag])
    if power_flag in flagged_df.columns:
        mask &= (~flagged_df[power_flag])

    clean_df = flagged_df[mask].sort_values("timestamp").reset_index(drop=True)

    if len(clean_df) < 50:
        print(f"Chiller {m_id}: Insufficient clean data ({len(clean_df)} rows)")
        return

    # Chronological 80/20 split
    split_idx = int(len(clean_df) * 0.8)
    train_df = clean_df.iloc[:split_idx]
    test_df = clean_df.iloc[split_idx:]

    # Extract series
    train_flow, test_flow = train_df[flow_col], test_df[flow_col]
    train_power, test_power = train_df[power_col], test_df[power_col]

    # Pearson correlation r
    r_train = train_flow.corr(train_power)
    r_test = test_flow.corr(test_power)
    r_overall = clean_df[flow_col].corr(clean_df[power_col])

    # Near-zero power check (near zero or < 5th percentile of overall power)
    overall_p5 = clean_df[power_col].quantile(0.05)
    near_zero_test_count = (test_power <= max(1.0, overall_p5)).sum()
    near_zero_pct = (near_zero_test_count / len(test_df)) * 100.0

    print("\n" + "="*85)
    print(f"DIAGNOSTIC REPORT FOR CHILLER {m_id}")
    print("="*85)
    print(f"Total Clean Rows: {len(clean_df)} | Train Rows: {len(train_df)} | Test Rows: {len(test_df)}")
    
    print("\n1. REGIME SHIFT & DISTRIBUTION COMPARISON:")
    stats_df = pd.DataFrame({
        "Metric": ["Flow Mean", "Flow Std", "Flow Min", "Flow Max",
                   "Power Mean", "Power Std", "Power Min", "Power Max"],
        "Train Window": [
            train_flow.mean(), train_flow.std(), train_flow.min(), train_flow.max(),
            train_power.mean(), train_power.std(), train_power.min(), train_power.max()
        ],
        "Test Window": [
            test_flow.mean(), test_flow.std(), test_flow.min(), test_flow.max(),
            test_power.mean(), test_power.std(), test_power.min(), test_power.max()
        ]
    })
    print(stats_df.to_string(index=False))

    print("\n2. MAPE BLOWUP / NEAR-ZERO POWER READINGS:")
    print(f"  Overall Power 5th Percentile: {overall_p5:.3f} KW")
    print(f"  Test Window Power <= {max(1.0, overall_p5):.3f} KW: {near_zero_test_count} / {len(test_df)} rows ({near_zero_pct:.2f}%)")
    print(f"  Explanation: Division by near-zero power in MAPE formula causes 10,000%+ spikes!")

    print("\n3. PEARSON CORRELATION (r) ANALYSIS:")
    print(f"  Train Window Pearson r (Flow vs Power): {r_train:.4f}")
    print(f"  Test Window Pearson r  (Flow vs Power): {r_test:.4f}")
    print(f"  Overall Pearson r       (Flow vs Power): {r_overall:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose forecast failure causes for worst chillers.")
    parser.add_argument(
        "--wide-input",
        default=os.path.join(PROJECT_ROOT, "data", "trend_wide.csv"),
        help="Path to trend_wide.csv"
    )
    parser.add_argument(
        "--types-input",
        default=os.path.join(PROJECT_ROOT, "data", "chiller_types.csv"),
        help="Path to chiller_types.csv"
    )
    args = parser.parse_args()

    df_wide = pd.read_csv(args.wide_input)
    df_types = pd.read_csv(args.types_input)

    for m_id in TARGET_CHILLERS:
        diagnose_chiller(df_wide, df_types, m_id)


if __name__ == "__main__":
    main()
