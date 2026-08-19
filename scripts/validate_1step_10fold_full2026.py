"""
1-Step-Ahead Forecasting Validation (10-Fold TimeSeriesSplit CV over Full 6-Month 2026 History).

Features at t-1: Thermal Load[t-1] + Compressor Load[t-1]
Target at t: Power[t] (KW)
Dataset: 2026-01-01 to 2026-07-08 (~18,000 timesteps per chiller)
Chillers: 2821, 2828, 2831
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate

TARGET_CHILLERS = [2821, 2828, 2831]


def evaluate_1step_10fold_for_chiller(m_id):
    csv_path = os.path.join(PROJECT_ROOT, "data", f"chiller_{m_id}_2026_full_history.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    flow_col = "Flow ValueY"
    power_col = "KW ValueY"
    ret_col = "Evaporator_Inlet_Temp"
    leave_col = "Evaporator_Outlet_Temp"

    sub_cols = ["machineId", "timestamp", flow_col, power_col, ret_col, leave_col, "Compressor_1_Load", "Compressor_2_Load"]
    sub_df = df[sub_cols].dropna().copy()

    # Step 1: Data Validation Gate
    flagged_df, report_df = validate(sub_df)

    valid_mask = pd.Series(True, index=flagged_df.index)
    for c in [flow_col, power_col, ret_col, leave_col]:
        if f"{c}_flagged" in flagged_df.columns:
            valid_mask &= (~flagged_df[f"{c}_flagged"])

    clean_df = flagged_df[valid_mask].sort_values("timestamp").reset_index(drop=True)

    # Step 2: Running Rows Only (Power > 10.0 KW)
    running_df = clean_df[clean_df[power_col] > 10.0].copy().reset_index(drop=True)

    # Feature Engineering
    running_df["Delta_T"] = running_df[ret_col] - running_df[leave_col]
    running_df["Thermal_Load"] = running_df[flow_col] * running_df["Delta_T"]
    running_df["Comp_1_Load"] = running_df["Compressor_1_Load"]
    running_df["Comp_2_Load"] = running_df["Compressor_2_Load"]
    running_df["Total_Comp_Load"] = running_df["Comp_1_Load"] + running_df["Comp_2_Load"]

    feature_cols = [flow_col, "Delta_T", "Thermal_Load", "Comp_1_Load", "Comp_2_Load", "Total_Comp_Load"]

    # Step 3: Shift features back by 1 timestep (t-1 -> t)
    df_lagged = running_df.copy()
    for col in feature_cols:
        df_lagged[f"{col}_lag1"] = df_lagged[col].shift(1)

    df_lagged = df_lagged.dropna().reset_index(drop=True)

    lag_feature_cols = [f"{col}_lag1" for col in feature_cols]
    X_lag = df_lagged[lag_feature_cols].values
    y_lag = df_lagged[power_col].values
    timestamps = df_lagged["timestamp"].values

    # Step 4: 10-Fold TimeSeriesSplit CV over full ~6-month window
    tscv = TimeSeriesSplit(n_splits=10)
    fold_records = []

    fold_idx = 1
    for train_idx, test_idx in tscv.split(X_lag):
        X_tr, X_te = X_lag[train_idx], X_lag[test_idx]
        y_tr, y_te = y_lag[train_idx], y_lag[test_idx]

        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)

        r2 = float(r2_score(y_te, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mape = float(np.mean(np.abs((y_te - y_pred) / np.maximum(y_te, 1.0))) * 100.0)

        t_start = pd.to_datetime(timestamps[test_idx[0]]).strftime("%Y-%m-%d")
        t_end = pd.to_datetime(timestamps[test_idx[-1]]).strftime("%Y-%m-%d")

        fold_records.append({
            "machineId": m_id,
            "Fold": fold_idx,
            "Test Period": f"{t_start} -> {t_end}",
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "R2": r2,
            "RMSE": rmse,
            "MAPE": mape
        })

        fold_idx += 1

    fold_df = pd.DataFrame(fold_records)
    mean_cv_r2 = fold_df["R2"].mean()
    std_cv_r2 = fold_df["R2"].std()
    mean_rmse = fold_df["RMSE"].mean()
    mean_mape = fold_df["MAPE"].mean()

    return fold_df, {
        "machineId": m_id,
        "n_total_running_rows": len(df_lagged),
        "mean_cv_R2": mean_cv_r2,
        "std_cv_R2": std_cv_r2,
        "mean_RMSE": mean_rmse,
        "mean_MAPE": mean_mape
    }


def main():
    all_folds = []
    summary_rows = []

    print("="*95)
    print("1-STEP-AHEAD FORECAST VALIDATION (10-FOLD TIME-SERIES SPLIT CV OVER FULL 2026 HISTORY)")
    print("Features: Thermal Load[t-1] + Compressor Load[t-1] -> Power[t]")
    print("="*95)

    for m_id in TARGET_CHILLERS:
        fold_df, summary = evaluate_1step_10fold_for_chiller(m_id)
        if fold_df is not None:
            all_folds.append(fold_df)
            summary_rows.append(summary)

            print(f"\n--- CHILLER {m_id} (10-FOLD PER-FOLD R2 BREAKDOWN) ---")
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 110)
            print(fold_df[["Fold", "Test Period", "n_train", "n_test", "R2", "RMSE", "MAPE"]].to_string(index=False))
            print(f"  --> MEAN CV R2: {summary['mean_cv_R2']:.4f} (Std: {summary['std_cv_R2']:.4f})")

    sum_df = pd.DataFrame(summary_rows)

    print("\n" + "="*95)
    print("SUMMARY OF MEAN 10-FOLD CV METRICS (2026-01-01 TO 2026-07-08)")
    print("="*95)
    for _, r in sum_df.iterrows():
        m_id = int(r["machineId"])
        m_r2 = r["mean_cv_R2"]

        tier_status = "Clears >= 0.5 (Tier 1 Cleared)" if m_r2 >= 0.5 else ("Clears 0.3 - 0.4 Range" if m_r2 >= 0.3 else "Does NOT Clear 0.3")
        print(f"Chiller {m_id} -> Total Running Rows: {int(r['n_total_running_rows']):,} | Mean CV R2: {m_r2:.4f} (Std: {r['std_cv_R2']:.4f}) | Mean RMSE: {r['mean_RMSE']:.2f} | Status: {tier_status}")

    combined_folds_df = pd.concat(all_folds, ignore_index=True)
    out_csv = os.path.join(PROJECT_ROOT, "data", "forecast_1step_10fold_2026_results.csv")
    combined_folds_df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed 10-fold per-fold results to {out_csv}")


if __name__ == "__main__":
    main()
