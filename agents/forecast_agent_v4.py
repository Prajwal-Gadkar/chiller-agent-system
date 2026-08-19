"""
Forecast Agent v4 — Single-Regime (2026-01-01 Onward) Evaluation.

Restricts evaluation strictly to the current operating regime (2026-01-01 onward)
to eliminate false regime-shift cross-boundary errors.

Evaluates Thermal Load (Flow * Delta_T -> Power) Linear Regression model:
1. Running rows only (Power > max(5.0, 5th percentile)).
2. 5-fold TimeSeriesSplit CV within the 2026 window.
3. Evaluates on Chiller 1657 (priority target) and all 12 clean group chillers.

Strict Rules:
- No DB mutations, no hardcoded secrets.
- Per-chiller evaluated individually (never pooled).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

# Add repo root to path for agent imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate

TREND_WIDE_CSV = os.path.join(PROJECT_ROOT, "data", "trend_wide.csv")
CLEAN_GROUP_CSV = os.path.join(PROJECT_ROOT, "data", "clean_chiller_group.csv")
CHILLER_1657_CSV = os.path.join(PROJECT_ROOT, "data", "chiller_1657_full_history.csv")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "data", "forecast_agent_v4_results.csv")


def identify_chiller_sensor_columns(m_df):
    """
    Find actual sensor column names for a specific machineId dataframe by inspecting non-null columns.
    """
    populated_cols = m_df.columns[m_df.notna().any()].tolist()

    flow_col = None
    power_col = None
    ret_col = None
    leave_col = None

    for c in populated_cols:
        c_lower = c.lower()
        if "flow" in c_lower and flow_col is None:
            flow_col = c
        elif ("kw" in c_lower or "power" in c_lower) and power_col is None and "running_kw" not in c_lower and "energy" not in c_lower:
            power_col = c
        elif ("evaporator_inlet" in c_lower or "chw return" in c_lower or "inlet_temperature" in c_lower) and ret_col is None:
            ret_col = c
        elif ("evaporator_outlet" in c_lower or "chw leave" in c_lower or "outlet_temperature" in c_lower) and leave_col is None:
            leave_col = c

    return flow_col, power_col, ret_col, leave_col


def evaluate_chiller_v4(chiller_df, m_id, chiller_type):
    """
    Evaluates Forecast Agent v4 on a single chiller's dataframe (filtered >= 2026-01-01).
    """
    flow_col, power_col, ret_col, leave_col = identify_chiller_sensor_columns(chiller_df)

    if not all([flow_col, power_col, ret_col, leave_col]):
        return {
            "machineId": m_id,
            "chiller_type": chiller_type,
            "flow_col": flow_col,
            "power_col": power_col,
            "ret_col": ret_col,
            "leave_col": leave_col,
            "test_R2": np.nan,
            "cv_R2_mean": np.nan,
            "cv_R2_std": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "n_clean_rows": 0,
            "running_pct": 0.0,
            "tier_1_eligible": False,
            "status": "Missing required sensor columns"
        }

    # Subset ONLY relevant columns for this chiller before validation
    relevant_cols = ["machineId", "timestamp", flow_col, power_col, ret_col, leave_col]
    sub_df = chiller_df[relevant_cols].dropna(subset=[flow_col, power_col, ret_col, leave_col]).copy()

    if sub_df.empty:
        return {
            "machineId": m_id,
            "chiller_type": chiller_type,
            "flow_col": flow_col,
            "power_col": power_col,
            "ret_col": ret_col,
            "leave_col": leave_col,
            "test_R2": np.nan,
            "cv_R2_mean": np.nan,
            "cv_R2_std": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "n_clean_rows": 0,
            "running_pct": 0.0,
            "tier_1_eligible": False,
            "status": "No valid data rows"
        }

    # Step 1: Data Validation Gate
    flagged_df, report_df = validate(sub_df)

    cols_to_check = [flow_col, power_col, ret_col, leave_col]
    valid_mask = pd.Series(True, index=flagged_df.index)
    for c in cols_to_check:
        flag_col = f"{c}_flagged"
        if flag_col in flagged_df.columns:
            valid_mask &= (~flagged_df[flag_col])

    clean_df = flagged_df[valid_mask].sort_values("timestamp").reset_index(drop=True)
    n_clean = len(clean_df)

    if n_clean < 50:
        return {
            "machineId": m_id,
            "chiller_type": chiller_type,
            "flow_col": flow_col,
            "power_col": power_col,
            "ret_col": ret_col,
            "leave_col": leave_col,
            "test_R2": np.nan,
            "cv_R2_mean": np.nan,
            "cv_R2_std": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "n_clean_rows": n_clean,
            "running_pct": 0.0,
            "tier_1_eligible": False,
            "status": "Insufficient clean data (<50 rows)"
        }

    # Step 2: Active Running Segmentation (Power > max(5.0, 5th percentile))
    p5_power = clean_df[power_col].quantile(0.05)
    power_cutoff = max(5.0, p5_power)
    is_running = clean_df[power_col] > power_cutoff
    running_df = clean_df[is_running].copy().reset_index(drop=True)

    running_pct = (len(running_df) / n_clean) * 100.0 if n_clean > 0 else 0.0

    if len(running_df) < 50:
        return {
            "machineId": m_id,
            "chiller_type": chiller_type,
            "flow_col": flow_col,
            "power_col": power_col,
            "ret_col": ret_col,
            "leave_col": leave_col,
            "test_R2": np.nan,
            "cv_R2_mean": np.nan,
            "cv_R2_std": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "n_clean_rows": n_clean,
            "running_pct": running_pct,
            "tier_1_eligible": False,
            "status": "Insufficient running data (<50 rows)"
        }

    # Feature Engineering
    running_df["Delta_T"] = running_df[ret_col] - running_df[leave_col]
    running_df["Thermal_Load"] = running_df[flow_col] * running_df["Delta_T"]

    feature_cols = [flow_col, "Delta_T", "Thermal_Load"]
    X = running_df[feature_cols].values
    y = running_df[power_col].values

    # Step 3: 5-Fold TimeSeriesSplit CV within the 2026 regime
    tscv = TimeSeriesSplit(n_splits=5)
    cv_r2_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = LinearRegression()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        cv_r2_scores.append(r2_score(y_te, y_pred))

    # Overall 80/20 chronological split inside the 2026 window
    split_idx = int(len(running_df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    test_r2 = float(r2_score(y_test, y_pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    y_test_safe = np.maximum(np.abs(y_test), 1.0)
    mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test_safe)) * 100.0)

    cv_r2_mean = float(np.mean(cv_r2_scores))
    cv_r2_std = float(np.std(cv_r2_scores))

    tier_1_eligible = test_r2 >= 0.5

    return {
        "machineId": m_id,
        "chiller_type": chiller_type,
        "flow_col": flow_col,
        "power_col": power_col,
        "ret_col": ret_col,
        "leave_col": leave_col,
        "test_R2": test_r2,
        "cv_R2_mean": cv_r2_mean,
        "cv_R2_std": cv_r2_std,
        "RMSE": rmse,
        "MAPE": mape,
        "n_clean_rows": n_clean,
        "running_pct": running_pct,
        "tier_1_eligible": tier_1_eligible,
        "status": "Validated"
    }


def main():
    print("="*95)
    print("FORECAST AGENT V4 — SINGLE-REGIME (2026-01-01 ONWARD) EVALUATION")
    print("="*95)

    results = []

    # 1. Evaluate Chiller 1657 (Priority long-history target)
    if os.path.exists(CHILLER_1657_CSV):
        print("\nLoading Chiller 1657 full history and filtering to >= 2026-01-01...")
        df_1657 = pd.read_csv(CHILLER_1657_CSV)
        df_1657["timestamp"] = pd.to_datetime(df_1657["timestamp"])
        df_1657_2026 = df_1657[df_1657["timestamp"] >= "2026-01-01 00:00:00"].copy()

        res_1657 = evaluate_chiller_v4(df_1657_2026, 1657, "type_2")
        results.append(res_1657)
        print(f"  Chiller 1657 -> Test R2: {res_1657['test_R2']:.4f} | CV R2 Mean: {res_1657['cv_R2_mean']:.4f} | RMSE: {res_1657['RMSE']:.2f} | MAPE: {res_1657['MAPE']:.2f}% | Tier 1: {res_1657['tier_1_eligible']}")

    # 2. Evaluate all 12 clean group chillers from trend_wide.csv
    if os.path.exists(TREND_WIDE_CSV) and os.path.exists(CLEAN_GROUP_CSV):
        print("\nLoading trend_wide.csv and clean_chiller_group.csv...")
        trend_df = pd.read_csv(TREND_WIDE_CSV)
        trend_df["timestamp"] = pd.to_datetime(trend_df["timestamp"])
        trend_2026 = trend_df[trend_df["timestamp"] >= "2026-01-01 00:00:00"].copy()

        clean_group = pd.read_csv(CLEAN_GROUP_CSV)

        for _, row in clean_group.iterrows():
            m_id = int(row["machineId"])
            c_type = row["chiller_type"]

            m_df = trend_2026[trend_2026["machineId"] == m_id].copy()
            if m_df.empty:
                print(f"  No 2026 data found for machineId {m_id} in trend_wide.csv")
                continue

            res = evaluate_chiller_v4(m_df, m_id, c_type)
            results.append(res)
            print(f"  Chiller {m_id:4d} ({c_type}) -> Test R2: {res['test_R2']:.4f} | RMSE: {res['RMSE']:.2f} | MAPE: {res['MAPE']:.2f}% | Tier 1: {res['tier_1_eligible']}")

    res_df = pd.DataFrame(results)

    print("\n" + "="*115)
    print("FORECAST AGENT V4 SUMMARY TABLE (2026 SINGLE-REGIME THERMAL LOAD -> POWER)")
    print("="*115)

    display_cols = ["machineId", "chiller_type", "n_clean_rows", "running_pct", "test_R2", "cv_R2_mean", "RMSE", "MAPE", "tier_1_eligible"]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(res_df[display_cols].to_string(index=False))

    n_tier_1 = res_df["tier_1_eligible"].sum()
    total_validated = len(res_df)
    print("\n" + "="*115)
    print(f"RELIABILITY CASCADE SUMMARY: {n_tier_1}/{total_validated} Chillers Cleared Tier 1 Eligibility (Test R2 >= 0.5)")
    print("="*115)

    res_df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved results to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
