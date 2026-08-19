"""
Forecast Agent v3 — Multi-Year Validation on Chiller 1657 (1,166 Days, 2023-2026).
Evaluates Linear Regression vs RandomForestRegressor across 10-fold TimeSeriesSplit,
segmented on active running hours with physics-based Thermal Load features.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

# Add repo root to path for agent imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate


def analyze_quarterly_correlations(running_df, flow_col, power_col):
    """Compute quarterly Pearson correlation between Flow/Thermal_Load and Power across 3.2 years."""
    df = running_df.copy()
    df["quarter"] = df["timestamp"].dt.to_period("Q").astype(str)

    quarterly_stats = []
    for q_name, group in df.groupby("quarter"):
        if len(group) >= 20:
            r_flow_power = group[flow_col].corr(group[power_col])
            r_thermal_power = group["Thermal_Load"].corr(group[power_col])
            quarterly_stats.append({
                "Quarter": q_name,
                "Running Rows": len(group),
                "Pearson_r (Flow vs Power)": r_flow_power,
                "Pearson_r (ThermalLoad vs Power)": r_thermal_power,
                "Power Mean (KW)": group[power_col].mean(),
                "Power Max (KW)": group[power_col].max()
            })

    return pd.DataFrame(quarterly_stats)


def run_time_series_cv(running_df, flow_col, power_col, n_splits=10):
    """Run 10-fold expanding window TimeSeriesSplit comparing Linear Regression vs RandomForest."""
    feature_cols = [flow_col, "Delta_T", "Thermal_Load"]
    X = running_df[feature_cols].values
    y = running_df[power_col].values
    timestamps = running_df["timestamp"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    fold_idx = 1

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        test_ts_start = pd.to_datetime(timestamps[test_idx[0]]).strftime("%Y-%m-%d")
        test_ts_end = pd.to_datetime(timestamps[test_idx[-1]]).strftime("%Y-%m-%d")
        date_range_str = f"{test_ts_start} -> {test_ts_end}"

        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        y_test_safe = np.maximum(np.abs(y_test), 1.0)
        lr_mape = np.mean(np.abs((y_test - lr_pred) / y_test_safe)) * 100.0

        # 2. Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test_safe)) * 100.0

        fold_results.append({
            "Fold": fold_idx,
            "Test Period": date_range_str,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "LR_Test_R2": lr_r2,
            "LR_RMSE": lr_rmse,
            "LR_MAPE": lr_mape,
            "RF_Test_R2": rf_r2,
            "RF_RMSE": rf_rmse,
            "RF_MAPE": rf_mape
        })

        fold_idx += 1

    return pd.DataFrame(fold_results)


def main():
    parser = argparse.ArgumentParser(description="Forecast Agent v3: Multi-year validation on Chiller 1657.")
    parser.add_argument(
        "--input-csv",
        default=os.path.join(PROJECT_ROOT, "data", "chiller_1657_full_history.csv"),
        help="Path to chiller_1657_full_history.csv"
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(PROJECT_ROOT, "data", "forecast_v3_chiller_1657_results.csv"),
        help="Path to save forecast_v3_chiller_1657_results.csv"
    )
    args = parser.parse_args()

    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    flow_col = "Flow ValueY"
    power_col = "KW ValueY"
    inlet_col = "inlet_temperature ValueY"
    outlet_col = "Outlet_temperature ValueY"

    # Step 1: Data Validation Gate
    print("\n1. Running Data Validation Agent gate...")
    flagged_df, report_df = validate(df)

    cols_to_check = [flow_col, power_col, inlet_col, outlet_col]
    valid_mask = pd.Series(True, index=flagged_df.index)
    for c in cols_to_check:
        valid_mask &= flagged_df[c].notna()
        flag_col = f"{c}_flagged"
        if flag_col in flagged_df.columns:
            valid_mask &= (~flagged_df[flag_col])

    clean_df = flagged_df[valid_mask].sort_values("timestamp").reset_index(drop=True)
    print(f"  Total Raw Rows: {len(df):,} | Clean Unflagged Rows: {len(clean_df):,}")

    # Step 2: ON/OFF Segmentation
    print("\n2. ON/OFF Segmentation...")
    p5_power = clean_df[power_col].quantile(0.05)
    power_cutoff = max(5.0, p5_power)
    is_running = clean_df[power_col] > power_cutoff

    running_count = is_running.sum()
    idle_count = (~is_running).sum()
    running_pct = (running_count / len(clean_df)) * 100.0
    idle_pct = (idle_count / len(clean_df)) * 100.0

    print(f"  Power Threshold (Cutoff): {power_cutoff:.2f} KW")
    print(f"  Active Running Rows (KW > {power_cutoff:.1f}): {running_count:,} ({running_pct:.2f}%)")
    print(f"  Idle/Off Rows       (KW <= {power_cutoff:.1f}): {idle_count:,} ({idle_pct:.2f}%)")

    # Step 3 & 4: Active Running Dataset + Feature Engineering
    running_df = clean_df[is_running].sort_values("timestamp").copy().reset_index(drop=True)
    running_df["Delta_T"] = running_df[inlet_col] - running_df[outlet_col]
    running_df["Thermal_Load"] = running_df[flow_col] * running_df["Delta_T"]

    # Step 7: Quarterly Pearson Correlation Analysis across 3.2 years
    print("\n" + "="*95)
    print("QUARTERLY PEARSON CORRELATION STABILITY ANALYSIS (2023 - 2026)")
    print("="*95)
    quarterly_df = analyze_quarterly_correlations(running_df, flow_col, power_col)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(quarterly_df.to_string(index=False))

    # Step 5 & 6: 10-Fold TimeSeriesSplit Model Comparison
    print("\n" + "="*95)
    print("10-FOLD BLOCKED TIME-SERIES CROSS-VALIDATION (LR vs RANDOM FOREST)")
    print("="*95)
    cv_results_df = run_time_series_cv(running_df, flow_col, power_col, n_splits=10)
    print(cv_results_df.to_string(index=False))

    # Overall Summary
    lr_mean_r2 = cv_results_df["LR_Test_R2"].mean()
    rf_mean_r2 = cv_results_df["RF_Test_R2"].mean()
    lr_mean_rmse = cv_results_df["LR_RMSE"].mean()
    rf_mean_rmse = cv_results_df["RF_RMSE"].mean()
    lr_mean_mape = cv_results_df["LR_MAPE"].mean()
    rf_mean_mape = cv_results_df["RF_MAPE"].mean()

    tier_1_lr_eligible = lr_mean_r2 >= 0.5
    tier_1_rf_eligible = rf_mean_r2 >= 0.5

    print("\n" + "="*95)
    print("MULTI-YEAR VALIDATION SUMMARY — CHILLER 1657 (1,166 DAYS)")
    print("="*95)
    print(f"Linear Regression  -> Mean Test R²: {lr_mean_r2:.4f} | Mean RMSE: {lr_mean_rmse:.2f} | Mean MAPE: {lr_mean_mape:.2f}%")
    print(f"Random Forest (d=5)-> Mean Test R²: {rf_mean_r2:.4f} | Mean RMSE: {rf_mean_rmse:.2f} | Mean MAPE: {rf_mean_mape:.2f}%")
    print(f"\nTIER 1 ELIGIBILITY ASSESSMENT (Threshold: Mean Test R² >= 0.5):")
    print(f"  Linear Regression Tier 1 Eligible? {'YES' if tier_1_lr_eligible else 'NO (Needs Fallback)'}")
    print(f"  Random Forest     Tier 1 Eligible? {'YES' if tier_1_rf_eligible else 'NO (Needs Fallback)'}")

    cv_results_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved detailed 10-fold cross-validation results to {args.output_csv}")


if __name__ == "__main__":
    main()
