"""
Forecast Agent — Flow -> Power prediction per chiller.
Trained and validated individually per chiller on the clean group (data/clean_chiller_group.csv).

Per CLAUDE.md:
- NEVER pool chillers — each chiller gets its own independent model.
- Uses controllable feature (Flow) to predict outcome target (Power/KW).
- Uses chronological 80/20 train/test split + 5-fold cross-validation.
- Classifies chillers as Tier 1 eligible if test R2 >= 0.5.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# Add repo root to path for agent imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate


def identify_sensor_columns(df_chiller, m_id, df_types):
    """
    Dynamically identify the appropriate Flow and Power columns for a given machineId
    from chiller_types.csv or the wide dataframe columns.
    """
    row = df_types[df_types["machineId"] == m_id].iloc[0]
    pop_cols = [c for c in df_types.columns if c not in {"machineId", "chiller_type", "status", "Criticality"} and row[c] == True]
    
    # Flow candidates
    flow_candidates = [c for c in pop_cols if "flow" in c.lower()]
    # Power candidates (exclude Energy_Consumption, KWH, IKW)
    power_candidates = [
        c for c in pop_cols 
        if ("kw" in c.lower() or "power" in c.lower()) 
        and "energy" not in c.lower() 
        and "kwh" not in c.lower()
        and "ikw" not in c.lower()
    ]
    
    # Priority defaults
    flow_col = "Flow ValueY" if "Flow ValueY" in flow_candidates else (flow_candidates[0] if flow_candidates else None)
    power_col = "KW ValueY" if "KW ValueY" in power_candidates else (power_candidates[0] if power_candidates else None)
    
    return flow_col, power_col


def train_and_evaluate_chiller(chiller_df, m_id, chiller_type, flow_col, power_col):
    """
    Train and evaluate a Flow -> Power Linear Regression model for a single chiller.
    Returns metrics dictionary.
    """
    if not flow_col or not power_col:
        return {
            "machineId": m_id,
            "chiller_type": chiller_type,
            "flow_column": str(flow_col),
            "power_column": str(power_col),
            "test_R2": np.nan,
            "cv_R2_mean": np.nan,
            "cv_R2_std": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "n_train_rows": 0,
            "n_test_rows": 0,
            "tier_1_eligible": False,
            "status": "Missing Flow or Power Column"
        }

    # Run Data Validation Agent gate to annotate out-of-bounds readings
    flagged_df, _ = validate(chiller_df)
    
    flow_flag_col = f"{flow_col}_flagged"
    power_flag_col = f"{power_col}_flagged"

    # Filter to non-null, non-flagged readings only
    valid_mask = (
        flagged_df[flow_col].notna() &
        flagged_df[power_col].notna()
    )
    
    if flow_flag_col in flagged_df.columns:
        valid_mask &= (~flagged_df[flow_flag_col])
    if power_flag_col in flagged_df.columns:
        valid_mask &= (~flagged_df[power_flag_col])

    clean_df = flagged_df[valid_mask].sort_values("timestamp").reset_index(drop=True)

    # Require minimum sample size
    if len(clean_df) < 50:
        return {
            "machineId": m_id,
            "chiller_type": chiller_type,
            "flow_column": flow_col,
            "power_column": power_col,
            "test_R2": np.nan,
            "cv_R2_mean": np.nan,
            "cv_R2_std": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "n_train_rows": 0,
            "n_test_rows": 0,
            "tier_1_eligible": False,
            "status": f"Insufficient Clean Rows ({len(clean_df)})"
        }

    # Extract X (Flow) and y (Power)
    X = clean_df[[flow_col]].values
    y = clean_df[power_col].values

    # Chronological Train/Test Split (80% train, 20% test)
    split_idx = int(len(clean_df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5-Fold Cross-Validation on Training Portion
    kf = KFold(n_splits=5, shuffle=False)
    cv_model = LinearRegression()
    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=kf, scoring="r2")
    cv_R2_mean = np.mean(cv_scores)
    cv_R2_std = np.std(cv_scores)

    # Fit final model on full training set
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = model.predict(X_test)
    test_R2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # MAPE (clip tiny y_test denominators to prevent division by zero spikes)
    y_test_safe = np.maximum(np.abs(y_test), 1e-3)
    mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100.0

    # Reliability Tier Classification (Threshold: test_R2 >= 0.5)
    tier_1_eligible = bool(test_R2 >= 0.5)

    return {
        "machineId": m_id,
        "chiller_type": chiller_type,
        "flow_column": flow_col,
        "power_column": power_col,
        "test_R2": test_R2,
        "cv_R2_mean": cv_R2_mean,
        "cv_R2_std": cv_R2_std,
        "RMSE": rmse,
        "MAPE": mape,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
        "tier_1_eligible": tier_1_eligible,
        "status": "Validated"
    }


def main():
    parser = argparse.ArgumentParser(description="Forecast Agent: Flow -> Power prediction per chiller.")
    parser.add_argument(
        "--wide-input",
        default=os.path.join(PROJECT_ROOT, "data", "trend_wide.csv"),
        help="Path to trend_wide.csv"
    )
    parser.add_argument(
        "--clean-group",
        default=os.path.join(PROJECT_ROOT, "data", "clean_chiller_group.csv"),
        help="Path to clean_chiller_group.csv"
    )
    parser.add_argument(
        "--types-input",
        default=os.path.join(PROJECT_ROOT, "data", "chiller_types.csv"),
        help="Path to chiller_types.csv"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "data", "forecast_agent_results.csv"),
        help="Path to save forecast_agent_results.csv"
    )
    args = parser.parse_args()

    print(f"Loading datasets...")
    df_wide = pd.read_csv(args.wide_input)
    df_clean_group = pd.read_csv(args.clean_group)
    df_types = pd.read_csv(args.types_input)

    clean_ids = df_clean_group["machineId"].tolist()
    print(f"Running Forecast Agent on {len(clean_ids)} clean group chillers...")

    results = []
    for m_id in clean_ids:
        chiller_df = df_wide[df_wide["machineId"] == m_id].copy()
        c_type = df_clean_group[df_clean_group["machineId"] == m_id]["chiller_type"].values[0]
        
        flow_col, power_col = identify_sensor_columns(chiller_df, m_id, df_types)
        res = train_and_evaluate_chiller(chiller_df, m_id, c_type, flow_col, power_col)
        results.append(res)

    results_df = pd.DataFrame(results)

    # Format results display table
    display_df = results_df[[
        "machineId", "chiller_type", "test_R2", "cv_R2_mean", "cv_R2_std",
        "RMSE", "MAPE", "n_train_rows", "n_test_rows", "tier_1_eligible"
    ]].copy()

    print("\n" + "="*95)
    print("FORECAST AGENT (FLOW -> POWER) MODEL RESULTS PER CHILLER")
    print("="*95)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(display_df.to_string(index=False))

    tier_1_count = results_df["tier_1_eligible"].sum()
    print("\n" + "="*95)
    print(f"RELIABILITY CASCADE SUMMARY: {tier_1_count}/{len(results_df)} Chillers Cleared Tier 1 Eligibility (Test R² >= 0.5)")
    print("="*95)

    results_df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
