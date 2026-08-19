"""
Fleet-Wide Response Model Evaluation (KW = f(Flow, InletTemp, OutletTemp, DeltaT, Thermal_Load))

Evaluates Random 5-Fold Cross-Validation across all 29 clean chillers on the universal column set.
Reports R2, RMSE, and MAPE per chiller sorted best to worst.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate

TREND_WIDE_PATH = os.path.join(PROJECT_ROOT, "data", "trend_wide.csv")
CLEAN_M_IDS = [1657, 1658, 1659, 1661, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761]

# Long history files if available
HISTORY_FILES = {
    1657: os.path.join(PROJECT_ROOT, "data", "chiller_1657_full_history.csv"),
    1660: os.path.join(PROJECT_ROOT, "data", "chiller_1660_full_history.csv"),
    1661: os.path.join(PROJECT_ROOT, "data", "chiller_1661_full_history.csv")
}


def evaluate_response_model_for_chiller(m_id, df_trend_wide):
    h_file = HISTORY_FILES.get(m_id)
    if h_file and os.path.exists(h_file):
        m_df = pd.read_csv(h_file)
        source = "Full Multi-Year History"
    else:
        m_df = df_trend_wide[df_trend_wide["machineId"] == m_id].copy()
        source = "trend_wide.csv (June 2026)"

    m_df["timestamp"] = pd.to_datetime(m_df["timestamp"])

    flow_col = "Flow ValueY"
    power_col = "KW ValueY"
    inlet_col = "inlet_temperature ValueY" if "inlet_temperature ValueY" in m_df.columns else "Evaporator_Inlet_Temp"
    outlet_col = "Outlet_temperature ValueY" if "Outlet_temperature ValueY" in m_df.columns else "Evaporator_Outlet_Temp"

    req_cols = ["machineId", "timestamp", flow_col, power_col, inlet_col, outlet_col]
    sub_df = m_df[req_cols].dropna().copy()

    if len(sub_df) < 50:
        return None

    # Step 1: Data Validation Gate
    flagged_df, report_df = validate(sub_df)

    valid_mask = pd.Series(True, index=flagged_df.index)
    for c in [flow_col, power_col, inlet_col, outlet_col]:
        if f"{c}_flagged" in flagged_df.columns:
            valid_mask &= (~flagged_df[f"{c}_flagged"])

    clean_df = flagged_df[valid_mask].sort_values("timestamp").reset_index(drop=True)

    # Step 2: Active Running Rows Only (KW > 10.0)
    running_df = clean_df[clean_df[power_col] > 10.0].copy().reset_index(drop=True)

    if len(running_df) < 50:
        return None

    # Feature Engineering (No Ambient - Pure Response Model: Flow, Inlet, Outlet, DeltaT, Thermal_Load)
    running_df["DeltaT"] = running_df[inlet_col] - running_df[outlet_col]
    running_df["Thermal_Load"] = running_df[flow_col] * running_df["DeltaT"]

    feature_cols = [flow_col, inlet_col, outlet_col, "DeltaT", "Thermal_Load"]
    X = running_df[feature_cols].values
    y = running_df[power_col].values

    # Step 3: Random 5-Fold Cross Validation (shuffle=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def eval_model(name, model_inst):
        r2_scores, rmse_scores, mape_scores = [], [], []

        for tr, te in kf.split(X):
            m = model_inst()
            m.fit(X[tr], y[tr])
            preds = m.predict(X[te])

            r2_scores.append(r2_score(y[te], preds))
            rmse_scores.append(np.sqrt(mean_squared_error(y[te], preds)))
            mape_scores.append(np.mean(np.abs((y[te] - preds) / np.maximum(y[te], 1.0))) * 100.0)

        return {
            "model_name": name,
            "mean_R2": float(np.mean(r2_scores)),
            "std_R2": float(np.std(r2_scores)),
            "mean_RMSE": float(np.mean(rmse_scores)),
            "mean_MAPE": float(np.mean(mape_scores))
        }

    lr_res = eval_model("Linear Regression", LinearRegression)
    rf_func = lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_res = eval_model("Random Forest (d=5)", rf_func)

    return {
        "machineId": m_id,
        "source": source,
        "n_running_rows": len(running_df),
        "LR_R2": lr_res["mean_R2"],
        "LR_RMSE": lr_res["mean_RMSE"],
        "RF_R2": rf_res["mean_R2"],
        "RF_RMSE": rf_res["mean_RMSE"],
        "RF_MAPE": rf_res["mean_MAPE"]
    }


def main():
    print("="*110)
    print("FLEET-WIDE RESPONSE MODEL EVALUATION: KW = f(Flow, InletTemp, OutletTemp, DeltaT, Thermal_Load)")
    print("Random 5-Fold Cross-Validation Across 29 Clean Chillers")
    print("="*110)

    if not os.path.exists(TREND_WIDE_PATH):
        print(f"Error: {TREND_WIDE_PATH} not found.")
        return

    df_tw = pd.read_csv(TREND_WIDE_PATH)

    results = []
    for m_id in CLEAN_M_IDS:
        res = evaluate_response_model_for_chiller(m_id, df_tw)
        if res is not None:
            results.append(res)

    res_df = pd.DataFrame(results)

    # Sort best to worst by Random Forest R2
    res_df = res_df.sort_values("RF_R2", ascending=False).reset_index(drop=True)

    print("\n" + "="*120)
    print("FLEET RESPONSE MODEL RESULTS TABLE (SORTED BEST TO WORST BY RANDOM FOREST R2)")
    print("="*120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 130)
    print(res_df[["machineId", "n_running_rows", "LR_R2", "LR_RMSE", "RF_R2", "RF_RMSE", "RF_MAPE", "source"]].to_string(index=False))

    n_total = len(res_df)
    n_r2_70 = len(res_df[res_df["RF_R2"] >= 0.70])
    n_r2_60 = len(res_df[res_df["RF_R2"] >= 0.60])
    n_r2_50 = len(res_df[res_df["RF_R2"] >= 0.50])

    print("\n" + "="*90)
    print(f"FLEET ELIGIBILITY SUMMARY ({n_total} CLEAN CHILLERS EVALUATED):")
    print(f"  --> Chillers Clearing R2 >= 0.70 (Tier 1 Benchmark):  {n_r2_70} / {n_total} ({n_r2_70/n_total*100:.1f}%)")
    print(f"  --> Chillers Clearing R2 >= 0.60 (Strong Fit):         {n_r2_60} / {n_total} ({n_r2_60/n_total*100:.1f}%)")
    print(f"  --> Chillers Clearing R2 >= 0.50 (Defensible Fit):     {n_r2_50} / {n_total} ({n_r2_50/n_total*100:.1f}%)")
    print("="*90)

    out_csv = os.path.join(PROJECT_ROOT, "data", "fleet_clean_chillers_response_model_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved full fleet response model results to {out_csv}")


if __name__ == "__main__":
    main()
