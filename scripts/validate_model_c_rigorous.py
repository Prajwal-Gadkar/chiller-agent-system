"""
Rigorous Validation Check on Model C (RandomForest / Non-Linear Model) for Chillers 2821, 2828, 2831.

1. Per-fold R2 across 5-fold TimeSeriesSplit CV.
2. Metrics on Bulk Operation (Power > 50th Percentile).
3. Genuine 1-step-ahead forecasting (predict Power at t using features at t-1).
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

TREND_WIDE_CSV = os.path.join(PROJECT_ROOT, "data", "trend_wide.csv")
TARGET_CHILLERS = [2821, 2828, 2831]


def prepare_chiller_data(df, m_id):
    flow_col = "Flow ValueY"
    power_col = "KW ValueY"
    ret_col = "Evaporator_Inlet_Temp"
    leave_col = "Evaporator_Outlet_Temp"

    sub_df = df[["machineId", "timestamp", flow_col, power_col, ret_col, leave_col, "Compressor_1_Load", "Compressor_2_Load"]].dropna().copy()
    flagged_df, report_df = validate(sub_df)

    valid_mask = pd.Series(True, index=flagged_df.index)
    for c in [flow_col, power_col, ret_col, leave_col]:
        if f"{c}_flagged" in flagged_df.columns:
            valid_mask &= (~flagged_df[f"{c}_flagged"])

    clean_df = flagged_df[valid_mask].sort_values("timestamp").reset_index(drop=True)
    running_df = clean_df[clean_df[power_col] > 10.0].copy().reset_index(drop=True)

    running_df["Delta_T"] = running_df[ret_col] - running_df[leave_col]
    running_df["Thermal_Load"] = running_df[flow_col] * running_df["Delta_T"]
    running_df["Comp_1_Load"] = running_df["Compressor_1_Load"]
    running_df["Comp_2_Load"] = running_df["Compressor_2_Load"]
    running_df["Total_Comp_Load"] = running_df["Comp_1_Load"] + running_df["Comp_2_Load"]

    return running_df, flow_col, power_col


def check_1_per_fold_cv(running_df, flow_col, power_col):
    """Check 1: 5-Fold TimeSeriesSplit CV Per-Fold Breakdown."""
    feature_cols = [flow_col, "Delta_T", "Thermal_Load", "Comp_1_Load", "Comp_2_Load", "Total_Comp_Load"]
    X = running_df[feature_cols].values
    y = running_df[power_col].values
    timestamps = running_df["timestamp"].values

    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []

    fold_idx = 1
    for tr, te in tscv.split(X):
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X[tr], y[tr])
        y_pred = rf.predict(X[te])

        r2 = float(r2_score(y[te], y_pred))
        rmse = float(np.sqrt(mean_squared_error(y[te], y_pred)))
        mape = float(np.mean(np.abs((y[te] - y_pred) / np.maximum(y[te], 1.0))) * 100.0)

        t_start = pd.to_datetime(timestamps[te[0]]).strftime("%m-%d %H:%M")
        t_end = pd.to_datetime(timestamps[te[-1]]).strftime("%m-%d %H:%M")

        fold_results.append({
            "Fold": fold_idx,
            "Test Period": f"{t_start} -> {t_end}",
            "R2": r2,
            "RMSE": rmse,
            "MAPE": mape
        })
        fold_idx += 1

    return pd.DataFrame(fold_results)


def check_2_bulk_operation(running_df, flow_col, power_col):
    """Check 2: Compute metrics ONLY on rows where Power > 50th percentile (bulk peak load)."""
    p50_kw = running_df[power_col].median()
    bulk_df = running_df[running_df[power_col] > p50_kw].copy().reset_index(drop=True)

    feature_cols = [flow_col, "Delta_T", "Thermal_Load", "Comp_1_Load", "Comp_2_Load", "Total_Comp_Load"]
    X = bulk_df[feature_cols].values
    y = bulk_df[power_col].values

    split_idx = int(len(bulk_df) * 0.8)
    X_tr, X_te = X[:split_idx], X[split_idx:]
    y_tr, y_te = y[:split_idx], y[split_idx:]

    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    r2 = float(r2_score(y_te, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mape = float(np.mean(np.abs((y_te - y_pred) / np.maximum(y_te, 1.0))) * 100.0)

    # Also run 5-fold CV on bulk operation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for tr, te in tscv.split(X):
        m = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        m.fit(X[tr], y[tr])
        cv_scores.append(r2_score(y[te], m.predict(X[te])))

    return {
        "p50_kw": p50_kw,
        "n_bulk_rows": len(bulk_df),
        "bulk_Test_R2": r2,
        "bulk_CV_R2_mean": float(np.mean(cv_scores)),
        "bulk_RMSE": rmse,
        "bulk_MAPE": mape
    }


def check_3_lagged_forecast(running_df, flow_col, power_col):
    """Check 3: Shift features back by 1 timestep (use features at t-1 to forecast Power at t)."""
    df_lagged = running_df.copy()

    feature_cols = [flow_col, "Delta_T", "Thermal_Load", "Comp_1_Load", "Comp_2_Load", "Total_Comp_Load"]

    # Shift feature columns by 1 step
    for col in feature_cols:
        df_lagged[f"{col}_lag1"] = df_lagged[col].shift(1)

    df_lagged = df_lagged.dropna().reset_index(drop=True)

    lag_feature_cols = [f"{col}_lag1" for col in feature_cols]
    X_lag = df_lagged[lag_feature_cols].values
    y_lag = df_lagged[power_col].values

    # 5-fold TimeSeriesSplit CV for 1-step-ahead forecast
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for tr, te in tscv.split(X_lag):
        m = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        m.fit(X_lag[tr], y_lag[tr])
        cv_scores.append(r2_score(y_lag[te], m.predict(X_lag[te])))

    # 80/20 Chronological Split
    split_idx = int(len(df_lagged) * 0.8)
    X_tr, X_te = X_lag[:split_idx], X_lag[split_idx:]
    y_tr, y_te = y_lag[:split_idx], y_lag[split_idx:]

    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    r2 = float(r2_score(y_te, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mape = float(np.mean(np.abs((y_te - y_pred) / np.maximum(y_te, 1.0))) * 100.0)

    return {
        "lag1_Test_R2": r2,
        "lag1_CV_R2_mean": float(np.mean(cv_scores)),
        "lag1_RMSE": rmse,
        "lag1_MAPE": mape
    }


def main():
    print("Loading trend_wide.csv...")
    df = pd.read_csv(TREND_WIDE_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    summary_rows = []

    for m_id in TARGET_CHILLERS:
        m_df = df[df["machineId"] == m_id].copy()
        if m_df.empty:
            continue

        print("\n" + "="*95)
        print(f"RIGOROUS VALIDATION OF MODEL C FOR CHILLER {m_id}")
        print("="*95)

        running_df, flow_col, power_col = prepare_chiller_data(m_df, m_id)

        # Check 1
        cv_df = check_1_per_fold_cv(running_df, flow_col, power_col)
        print(f"\n1. PER-FOLD 5-FOLD TIMESERIES SPLIT CV BREAKDOWN (CHILLER {m_id}):")
        print(cv_df.to_string(index=False))
        mean_cv_r2 = cv_df["R2"].mean()

        # Check 2
        bulk_res = check_2_bulk_operation(running_df, flow_col, power_col)
        print(f"\n2. BULK OPERATION CHECK (POWER > 50TH PERCENTILE = {bulk_res['p50_kw']:.1f} KW):")
        print(f"   Bulk Rows: {bulk_res['n_bulk_rows']} | Bulk Test R2: {bulk_res['bulk_Test_R2']:.4f} | Bulk CV R2 Mean: {bulk_res['bulk_CV_R2_mean']:.4f} | Bulk RMSE: {bulk_res['bulk_RMSE']:.2f} | Bulk MAPE: {bulk_res['bulk_MAPE']:.2f}%")

        # Check 3
        lag1_res = check_3_lagged_forecast(running_df, flow_col, power_col)
        print(f"\n3. GENUINE 1-STEP-AHEAD FORECASTING (FEATURES AT t-1 -> POWER AT t):")
        print(f"   Lag-1 Test R2: {lag1_res['lag1_Test_R2']:.4f} | Lag-1 CV R2 Mean: {lag1_res['lag1_CV_R2_mean']:.4f} | Lag-1 RMSE: {lag1_res['lag1_RMSE']:.2f} | Lag-1 MAPE: {lag1_res['lag1_MAPE']:.2f}%")

        summary_rows.append({
            "machineId": m_id,
            "Full_CV_R2_Mean": mean_cv_r2,
            "Fold_1_R2": cv_df.loc[0, "R2"],
            "Fold_2_R2": cv_df.loc[1, "R2"],
            "Fold_3_R2": cv_df.loc[2, "R2"],
            "Fold_4_R2": cv_df.loc[3, "R2"],
            "Fold_5_R2": cv_df.loc[4, "R2"],
            "P50_KW": bulk_res["p50_kw"],
            "Bulk_Test_R2": bulk_res["bulk_Test_R2"],
            "Bulk_CV_R2_Mean": bulk_res["bulk_CV_R2_mean"],
            "Lag1_Forecast_Test_R2": lag1_res["lag1_Test_R2"],
            "Lag1_Forecast_CV_R2_Mean": lag1_res["lag1_CV_R2_mean"]
        })

    res_df = pd.DataFrame(summary_rows)

    print("\n" + "="*115)
    print("SUMMARY COMPARISON TABLE ACROSS ALL 3 VALIDATION CHECKS")
    print("="*115)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(res_df.to_string(index=False))

    output_csv = os.path.join(PROJECT_ROOT, "data", "model_c_rigorous_validation_results.csv")
    res_df.to_csv(output_csv, index=False)
    print(f"\nSaved detailed validation results to {output_csv}")


if __name__ == "__main__":
    main()
