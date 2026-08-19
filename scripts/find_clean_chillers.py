"""
Identifies the "cleanest" group of chillers (high sensor coverage + low data corruption)
from data/trend_wide.csv and data/chiller_types.csv for initial Forecast and
Anomaly agent modeling. Read-only, no DB writes. Per CLAUDE.md, these chillers
will STILL be modeled individually (never pooled).
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add repo root to path for agent imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate


def analyze_chiller_coverage(df_types):
    """Compute sensor coverage score and key availability (Flow, Power, Temp, Setpoint) per chiller."""
    sensor_cols = [c for c in df_types.columns if c not in {"machineId", "chiller_type", "status", "Criticality"}]

    coverage_info = []
    for idx, row in df_types.iterrows():
        m_id = row["machineId"]
        pop_cols = [c for c in sensor_cols if row[c] == True]
        n_pop = len(pop_cols)

        # Keyword checks
        has_flow = any("flow" in c.lower() for c in pop_cols)
        has_power = any(("kw" in c.lower() or "power" in c.lower()) for c in pop_cols)
        has_temp = any(("temp" in c.lower() or "celsius" in c.lower() or "deg" in c.lower()) for c in pop_cols)
        has_setpoint = any(("setpoint" in c.lower() or "sp" in c.lower()) for c in pop_cols)

        # Weighted coverage score:
        # Base count + Flow/Power primary relationship bonus (+20) + Temp/Setpoint bonus (+5 each)
        flow_power_bonus = 20.0 if (has_flow and has_power) else 0.0
        temp_sp_bonus = (5.0 if has_temp else 0.0) + (5.0 if has_setpoint else 0.0)
        
        coverage_score = n_pop + flow_power_bonus + temp_sp_bonus

        coverage_info.append({
            "machineId": m_id,
            "chiller_type": row["chiller_type"],
            "n_columns_populated": n_pop,
            "has_flow": has_flow,
            "has_power": has_power,
            "has_temp": has_temp,
            "has_setpoint": has_setpoint,
            "coverage_score": coverage_score
        })

    return pd.DataFrame(coverage_info)


def analyze_chiller_cleanliness(df_wide):
    """Run data validation agent on each chiller individually and compute average pct_flagged."""
    cleanliness_info = []

    grouped = df_wide.groupby("machineId")
    for m_id, chiller_df in grouped:
        _, report_df = validate(chiller_df)

        if not report_df.empty:
            valid_reports = report_df[report_df["n_total"] > 0]
            avg_pct_flagged = valid_reports["pct_flagged"].mean() if not valid_reports.empty else 100.0
        else:
            avg_pct_flagged = 100.0

        cleanliness_info.append({
            "machineId": m_id,
            "avg_pct_flagged": avg_pct_flagged
        })

    return pd.DataFrame(cleanliness_info)


def rank_chillers(coverage_df, cleanliness_df):
    """Combine coverage score and cleanliness score into a combined rank."""
    merged = pd.merge(coverage_df, cleanliness_df, on="machineId")

    # Min-max normalization for composite score computation
    min_cov, max_cov = merged["coverage_score"].min(), merged["coverage_score"].max()
    cov_norm = (merged["coverage_score"] - min_cov) / (max_cov - min_cov) if max_cov > min_cov else 1.0

    # Cleanliness score: lower avg_pct_flagged is better
    clean_score = 100.0 - merged["avg_pct_flagged"]
    min_clean, max_clean = clean_score.min(), clean_score.max()
    clean_norm = (clean_score - min_clean) / (max_clean - min_clean) if max_clean > min_clean else 1.0

    # Flow + Power requirement bonus (+100) ensures chillers with Flow->Power relationship rank first
    both_flow_power_bonus = np.where(merged["has_flow"] & merged["has_power"], 100.0, 0.0)

    # Combined composite score: 50% normalized coverage + 50% normalized cleanliness + Flow/Power prerequisite bonus
    merged["composite_score"] = (0.5 * cov_norm + 0.5 * clean_norm) * 100.0 + both_flow_power_bonus

    # Sort descending by composite score, then ascending by avg_pct_flagged
    merged = merged.sort_values(by=["composite_score", "avg_pct_flagged"], ascending=[False, True]).reset_index(drop=True)
    merged["combined_rank"] = merged.index + 1

    return merged


def main():
    parser = argparse.ArgumentParser(description="Find cleanest group of chillers based on coverage and low data corruption.")
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
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data", "clean_chiller_group.csv"),
        help="Path to save clean_chiller_group.csv"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of top clean chillers to select for clean group (default 12)"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.wide_input} and {args.types_input}...")
    df_wide = pd.read_csv(args.wide_input)
    df_types = pd.read_csv(args.types_input)

    print("Computing coverage scores...")
    coverage_df = analyze_chiller_coverage(df_types)

    print("Computing cleanliness scores via Data Validation Agent...")
    cleanliness_df = analyze_chiller_cleanliness(df_wide)

    print("Ranking chillers by combined coverage & cleanliness...")
    ranked_df = rank_chillers(coverage_df, cleanliness_df)

    # Select display columns
    display_cols = [
        "combined_rank", "machineId", "chiller_type", "n_columns_populated",
        "avg_pct_flagged", "has_flow", "has_power", "has_temp", "has_setpoint"
    ]
    
    print("\n" + "="*80)
    print("ALL CHILLERS RANKING SUMMARY (Top 25 shown)")
    print("="*80)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(ranked_df[display_cols].head(25).to_string(index=False))

    top_clean_group = ranked_df.head(args.top_n)

    print("\n" + "="*80)
    print(f"RECOMMENDED TOP {args.top_n} CLEAN CHILLER GROUP (For Initial Forecast/Anomaly Modeling)")
    print("="*80)
    print(top_clean_group[display_cols].to_string(index=False))

    # Save output CSV containing clean group
    save_cols = [
        "machineId", "combined_rank", "chiller_type", "n_columns_populated",
        "avg_pct_flagged", "has_flow", "has_power"
    ]
    top_clean_group[save_cols].to_csv(args.output, index=False)
    print(f"\nSaved clean chiller group ({len(top_clean_group)} chillers) to {args.output}")


if __name__ == "__main__":
    main()
