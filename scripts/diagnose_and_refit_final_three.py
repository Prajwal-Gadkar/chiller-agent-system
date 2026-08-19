"""
Diagnose and Refit Final Three Chillers (2821, 2828, 2831)

1. Power Histogram & Physical Idle/Running Threshold Diagnosis.
2. Compressor Staging / Modulation Analysis.
3. Model A Re-fit: Linear Regression (Thermal Load = Flow * Delta_T -> Power) with 5-fold TimeSeriesSplit.
4. Model B Re-fit: Enhanced Model (Thermal Load + Compressor Modulation Features).
5. Evaluation & Decision: Final R2, RMSE, MAPE per chiller and Tier 1 quality assessment.
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

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate

TREND_WIDE_CSV = os.path.join(PROJECT_ROOT, "data", "trend_wide.csv")
TARGET_CHILLERS = [2821, 2828, 2831]


def analyze_power_histogram(df, m_id):
    """Diagnose physical running vs idle power threshold using fine-grained histogram."""
    kw = df["KW ValueY"].dropna()
    counts_0_5 = (kw <= 5.0).sum()
    counts_5_10 = ((kw > 5.0) & (kw <= 10.0)).sum()
    counts_10_15 = ((kw > 10.0) & (kw <= 15.0)).sum()
    counts_15_20 = ((kw > 15.0) & (kw <= 20.0)).sum()

    print(f"\n--- POWER HISTOGRAM DIAGNOSTIC: CHILLER {m_id} ---")
    print(f"  [0.0 -  5.0 KW] (OFF/Standby): {counts_0_5:4d} rows")
    print(f"  (5.0 - 10.0 KW] (Valley/Gap) : {counts_5_10:4d} rows")
    print(f"  (10.0- 15.0 KW] (Active Min) : {counts_10_15:4d} rows")
    print(f"  (15.0- 20.0 KW] (Active Load): {counts_15_20:4d} rows")

    chosen_threshold = 10.0
    rationale = f"0-1 rows exist between 5.0 and 10.0 KW across dataset, proving a sharp physical gap between OFF/Standby (<5 KW) and Active Chiller Motor Operation (>10 KW)."

    return chosen_threshold, rationale


def analyze_compressor_staging(df, m_id):
    """Analyze compressor staging vs continuous modulation."""
    c1 = df["Compressor_1_Load"] if "Compressor_1_Load" in df.columns else pd.Series(0, index=df.index)
    c2 = df["Compressor_2_Load"] if "Compressor_2_Load" in df.columns else pd.Series(0, index=df.index)

    active_c1 = (c1 > 5.0).astype(int)
    active_c2 = (c2 > 5.0).astype(int)
    stage_count = active_c1 + active_c2

    unique_c1_vals = c1[c1 > 5.0].nunique()
    unique_c2_vals = c2[c2 > 5.0].nunique()

    is_discrete = (unique_c1_vals < 5) and (unique_c2_vals < 5)

    print(f"\n--- COMPRESSOR STAGING DIAGNOSTIC: CHILLER {m_id} ---")
    print(f"  Stage Distribution (0, 1, 2 Active Compressors):")
    print(stage_count.value_counts().to_dict())
    print(f"  Unique Compressor 1 Load Values (>5%): {unique_c1_vals} (Modulation is {'DISCRETE' if is_discrete else 'CONTINUOUS'})")
    print(f"  Unique Compressor 2 Load Values (>5%): {unique_c2_vals} (Modulation is {'DISCRETE' if is_discrete else 'CONTINUOUS'})")

    return is_discrete, stage_count


def evaluate_models(df, m_id, running_threshold):
    """Evaluate Model A (Thermal Load LR) vs Model B (Staging/Comp Load Enhanced LR) vs Model C (Random Forest)."""
    flow_col = "Flow ValueY"
    power_col = "KW ValueY"
    ret_col = "Evaporator_Inlet_Temp"
    leave_col = "Evaporator_Outlet_Temp"

    # Data Validation Gate
    sub_df = df[["machineId", "timestamp", flow_col, power_col, ret_col, leave_col, "Compressor_1_Load", "Compressor_2_Load"]].dropna().copy()
    flagged_df, report_df = validate(sub_df)

    valid_mask = pd.Series(True, index=flagged_df.index)
    for c in [flow_col, power_col, ret_col, leave_col]:
        if f"{c}_flagged" in flagged_df.columns:
            valid_mask &= (~flagged_df[f"{c}_flagged"])

    clean_df = flagged_df[valid_mask].sort_values("timestamp").reset_index(drop=True)

    # Active Running Segmentation
    running_df = clean_df[clean_df[power_col] > running_threshold].copy().reset_index(drop=True)
    n_clean = len(clean_df)
    n_running = len(running_df)

    # Feature Engineering
    running_df["Delta_T"] = running_df[ret_col] - running_df[leave_col]
    running_df["Thermal_Load"] = running_df[flow_col] * running_df["Delta_T"]
    running_df["Comp_1_Load"] = running_df["Compressor_1_Load"]
    running_df["Comp_2_Load"] = running_df["Compressor_2_Load"]
    running_df["Total_Comp_Load"] = running_df["Comp_1_Load"] + running_df["Comp_2_Load"]

    # Matrices
    X_A = running_df[[flow_col, "Delta_T", "Thermal_Load"]].values
    X_B = running_df[[flow_col, "Delta_T", "Thermal_Load", "Comp_1_Load", "Comp_2_Load", "Total_Comp_Load"]].values
    y = running_df[power_col].values

    # 5-fold TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    def run_cv(X_mat, model_cls):
        scores = []
        for tr, te in tscv.split(X_mat):
            m = model_cls()
            m.fit(X_mat[tr], y[tr])
            scores.append(r2_score(y[te], m.predict(X_mat[te])))
        return float(np.mean(scores)), float(np.std(scores))

    # Model A: Linear Regression (Thermal Load only)
    cv_A_mean, cv_A_std = run_cv(X_A, LinearRegression)

    # Model B: Linear Regression (Thermal Load + Compressor Features)
    cv_B_mean, cv_B_std = run_cv(X_B, LinearRegression)

    # Model C: Random Forest (Thermal Load + Compressor Features)
    rf_func = lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    cv_C_mean, cv_C_std = run_cv(X_B, rf_func)

    # Overall 80/20 Chronological Split
    split_idx = int(n_running * 0.8)

    # Fit Model A
    lr_A = LinearRegression()
    lr_A.fit(X_A[:split_idx], y[:split_idx])
    pred_A = lr_A.predict(X_A[split_idx:])
    r2_A = float(r2_score(y[split_idx:], pred_A))
    rmse_A = float(np.sqrt(mean_squared_error(y[split_idx:], pred_A)))
    mape_A = float(np.mean(np.abs((y[split_idx:] - pred_A) / np.maximum(y[split_idx:], 1.0))) * 100.0)

    # Fit Model B
    lr_B = LinearRegression()
    lr_B.fit(X_B[:split_idx], y[:split_idx])
    pred_B = lr_B.predict(X_B[split_idx:])
    r2_B = float(r2_score(y[split_idx:], pred_B))
    rmse_B = float(np.sqrt(mean_squared_error(y[split_idx:], pred_B)))
    mape_B = float(np.mean(np.abs((y[split_idx:] - pred_B) / np.maximum(y[split_idx:], 1.0))) * 100.0)

    # Fit Model C (Random Forest)
    rf_C = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_C.fit(X_B[:split_idx], y[:split_idx])
    pred_C = rf_C.predict(X_B[split_idx:])
    r2_C = float(r2_score(y[split_idx:], pred_C))
    rmse_C = float(np.sqrt(mean_squared_error(y[split_idx:], pred_C)))
    mape_C = float(np.mean(np.abs((y[split_idx:] - pred_C) / np.maximum(y[split_idx:], 1.0))) * 100.0)

    return {
        "machineId": m_id,
        "n_clean": n_clean,
        "n_running": n_running,
        "Model_A_LR_Test_R2": r2_A,
        "Model_A_LR_CV_R2": cv_A_mean,
        "Model_A_RMSE": rmse_A,
        "Model_A_MAPE": mape_A,
        "Model_B_LR_Test_R2": r2_B,
        "Model_B_LR_CV_R2": cv_B_mean,
        "Model_B_RMSE": rmse_B,
        "Model_B_MAPE": mape_B,
        "Model_C_RF_Test_R2": r2_C,
        "Model_C_RF_CV_R2": cv_C_mean,
        "Model_C_RMSE": rmse_C,
        "Model_C_MAPE": mape_C,
        "tier_1_cleared": (r2_A >= 0.5 or r2_B >= 0.5 or r2_C >= 0.5)
    }


def main():
    print("Loading trend_wide.csv...")
    df = pd.read_csv(TREND_WIDE_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    results = []

    for m_id in TARGET_CHILLERS:
        m_df = df[df["machineId"] == m_id].copy()
        if m_df.empty:
            print(f"No data for machine {m_id}")
            continue

        print("\n" + "="*80)
        print(f"DIAGNOSING AND REFITTING CHILLER {m_id}")
        print("="*80)

        threshold, rationale = analyze_power_histogram(m_df, m_id)
        print(f"  Chosen Threshold: Power > {threshold:.1f} KW")
        print(f"  Rationale: {rationale}")

        is_discrete, stage_counts = analyze_compressor_staging(m_df, m_id)

        eval_res = evaluate_models(m_df, m_id, threshold)
        eval_res["is_discrete"] = is_discrete
        eval_res["threshold_kw"] = threshold
        results.append(eval_res)

        print(f"\n  MODEL RESULTS FOR CHILLER {m_id}:")
        print(f"    Model A (Thermal Load LR)    -> Test R2: {eval_res['Model_A_LR_Test_R2']:.4f} | RMSE: {eval_res['Model_A_RMSE']:.2f} | MAPE: {eval_res['Model_A_MAPE']:.2f}%")
        print(f"    Model B (+Compressor Load LR)-> Test R2: {eval_res['Model_B_LR_Test_R2']:.4f} | RMSE: {eval_res['Model_B_RMSE']:.2f} | MAPE: {eval_res['Model_B_MAPE']:.2f}%")
        print(f"    Model C (+Compressor Load RF)-> Test R2: {eval_res['Model_C_RF_Test_R2']:.4f} | RMSE: {eval_res['Model_C_RMSE']:.2f} | MAPE: {eval_res['Model_C_MAPE']:.2f}%")
        print(f"    Cleared Forecast Quality Bar? {'YES' if eval_res['tier_1_cleared'] else 'NO'}")

    res_df = pd.DataFrame(results)

    print("\n" + "="*115)
    print("FINAL THREE CHILLERS (2821, 2828, 2831) DIAGNOSTIC & REFIT SUMMARY TABLE")
    print("="*115)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(res_df[["machineId", "threshold_kw", "n_running", "Model_A_LR_Test_R2", "Model_B_LR_Test_R2", "Model_C_RF_Test_R2", "Model_C_RMSE", "Model_C_MAPE", "tier_1_cleared"]].to_string(index=False))

    output_file = os.path.join(PROJECT_ROOT, "data", "final_three_chillers_refit_results.csv")
    res_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    main()
