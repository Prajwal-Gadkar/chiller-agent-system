"""
Rigorous Validation Checks on Fleet Response Model (KW = f(Flow, InletTemp, OutletTemp, DeltaT, Thermal_Load))

Check 1: Restrict Chillers 1657 and 1661 to 2026-01-01 onward ONLY (Single Regime Window).
Check 2: Bulk Operation Metrics (R2, RMSE, MAPE) computed on KW > 50th Percentile across ALL 29 Clean Chillers.
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

HISTORY_FILES = {
    1657: os.path.join(PROJECT_ROOT, "data", "chiller_1657_full_history.csv"),
    1660: os.path.join(PROJECT_ROOT, "data", "chiller_1660_full_history.csv"),
    1661: os.path.join(PROJECT_ROOT, "data", "chiller_1661_full_history.csv")
}


def run_check1_restricted_2026_history():
    print("="*110)
    print("CHECK 1: RESTRICT CHILLERS 1657 & 1661 TO 2026-01-01 ONWARD ONLY (SINGLE REGIME)")
    print("Evaluating Random 5-Fold CV on 2026-only data vs Full Multi-Year History...")
    print("="*110)

    check1_records = []

    for m_id in [1657, 1661]:
        h_file = HISTORY_FILES[m_id]
        if not os.path.exists(h_file):
            continue

        full_df = pd.read_csv(h_file)
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])

        flow_col = "Flow ValueY"
        power_col = "KW ValueY"
        inlet_col = "inlet_temperature ValueY" if "inlet_temperature ValueY" in full_df.columns else "Evaporator_Inlet_Temp"
        outlet_col = "Outlet_temperature ValueY" if "Outlet_temperature ValueY" in full_df.columns else "Evaporator_Outlet_Temp"

        req_cols = ["machineId", "timestamp", flow_col, power_col, inlet_col, outlet_col]

        def eval_window(sub_m_df, label):
            s_df = sub_m_df[req_cols].dropna().copy()
            flagged_df, report_df = validate(s_df)

            valid_mask = pd.Series(True, index=flagged_df.index)
            for c in [flow_col, power_col, inlet_col, outlet_col]:
                if f"{c}_flagged" in flagged_df.columns:
                    valid_mask &= (~flagged_df[f"{c}_flagged"])

            clean_df = flagged_df[valid_mask].sort_values("timestamp").reset_index(drop=True)
            running_df = clean_df[clean_df[power_col] > 10.0].copy().reset_index(drop=True)

            running_df["DeltaT"] = running_df[inlet_col] - running_df[outlet_col]
            running_df["Thermal_Load"] = running_df[flow_col] * running_df["DeltaT"]

            feature_cols = [flow_col, inlet_col, outlet_col, "DeltaT", "Thermal_Load"]
            X = running_df[feature_cols].values
            y = running_df[power_col].values

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_s, rmse_s, mape_s = [], [], []

            for tr, te in kf.split(X):
                rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                rf.fit(X[tr], y[tr])
                preds = rf.predict(X[te])

                r2_s.append(r2_score(y[te], preds))
                rmse_s.append(np.sqrt(mean_squared_error(y[te], preds)))
                mape_s.append(np.mean(np.abs((y[te] - preds) / np.maximum(y[te], 1.0))) * 100.0)

            return {
                "machineId": m_id,
                "Window": label,
                "n_running_rows": len(running_df),
                "R2": float(np.mean(r2_s)),
                "std_R2": float(np.std(r2_s)),
                "RMSE": float(np.mean(rmse_s)),
                "MAPE": float(np.mean(mape_s))
            }

        # 1. Full Multi-Year History
        res_full = eval_window(full_df, "Full Multi-Year History (2023-2026)")
        # 2. 2026-01-01 Onward ONLY
        df_2026 = full_df[full_df["timestamp"] >= "2026-01-01"].copy()
        res_2026 = eval_window(df_2026, "2026-01-01 Onward ONLY (Single Regime)")

        check1_records.append(res_full)
        check1_records.append(res_2026)

    c1_df = pd.DataFrame(check1_records)
    print(c1_df.to_string(index=False))
    return c1_df


def run_check2_bulk_operation_all_chillers(df_trend_wide):
    print("\n\n" + "="*110)
    print("CHECK 2: BULK OPERATION METRICS (R2, RMSE, MAPE COMPUTED ON KW > 50TH PERCENTILE) ACROSS ALL 29 CLEAN CHILLERS")
    print("Evaluating 2026 dataset for all 29 clean chillers...")
    print("="*110)

    check2_records = []

    for m_id in CLEAN_M_IDS:
        h_file = HISTORY_FILES.get(m_id)
        if h_file and os.path.exists(h_file):
            m_df = pd.read_csv(h_file)
            m_df["timestamp"] = pd.to_datetime(m_df["timestamp"])
            # Restrict to 2026-01-01 onward for 1657/1661 so all chillers share the same 2026 window
            m_df = m_df[m_df["timestamp"] >= "2026-01-01"].copy()
            source = "2026 Full History"
        else:
            m_df = df_trend_wide[df_trend_wide["machineId"] == m_id].copy()
            m_df["timestamp"] = pd.to_datetime(m_df["timestamp"])
            source = "trend_wide.csv (June 2026)"

        flow_col = "Flow ValueY"
        power_col = "KW ValueY"
        inlet_col = "inlet_temperature ValueY" if "inlet_temperature ValueY" in m_df.columns else "Evaporator_Inlet_Temp"
        outlet_col = "Outlet_temperature ValueY" if "Outlet_temperature ValueY" in m_df.columns else "Evaporator_Outlet_Temp"

        req_cols = ["machineId", "timestamp", flow_col, power_col, inlet_col, outlet_col]
        sub_df = m_df[req_cols].dropna().copy()

        if len(sub_df) < 50:
            continue

        flagged_df, report_df = validate(sub_df)

        valid_mask = pd.Series(True, index=flagged_df.index)
        for c in [flow_col, power_col, inlet_col, outlet_col]:
            if f"{c}_flagged" in flagged_df.columns:
                valid_mask &= (~flagged_df[f"{c}_flagged"])

        clean_df = flagged_df[valid_mask].sort_values("timestamp").reset_index(drop=True)
        running_df = clean_df[clean_df[power_col] > 10.0].copy().reset_index(drop=True)

        if len(running_df) < 50:
            continue

        p50 = float(running_df[power_col].quantile(0.50))

        running_df["DeltaT"] = running_df[inlet_col] - running_df[outlet_col]
        running_df["Thermal_Load"] = running_df[flow_col] * running_df["DeltaT"]

        feature_cols = [flow_col, inlet_col, outlet_col, "DeltaT", "Thermal_Load"]
        X = running_df[feature_cols].values
        y = running_df[power_col].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        y_true_all = []
        y_pred_all = []

        for tr, te in kf.split(X):
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X[tr], y[tr])
            preds = rf.predict(X[te])

            y_true_all.extend(y[te])
            y_pred_all.extend(preds)

        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)

        # 1. Full Running Range (KW > 10.0)
        full_r2 = float(r2_score(y_true_arr, y_pred_arr))
        full_rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        full_mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / np.maximum(y_true_arr, 1.0))) * 100.0)

        # 2. Bulk Operation Range (KW > P50)
        bulk_mask = (y_true_arr > p50)
        if np.sum(bulk_mask) > 10:
            y_true_bulk = y_true_arr[bulk_mask]
            y_pred_bulk = y_pred_arr[bulk_mask]

            bulk_r2 = float(r2_score(y_true_bulk, y_pred_bulk))
            bulk_rmse = float(np.sqrt(mean_squared_error(y_true_bulk, y_pred_bulk)))
            bulk_mape = float(np.mean(np.abs((y_true_bulk - y_pred_bulk) / np.maximum(y_true_bulk, 1.0))) * 100.0)
        else:
            bulk_r2, bulk_rmse, bulk_mape = np.nan, np.nan, np.nan

        check2_records.append({
            "machineId": m_id,
            "n_running": len(running_df),
            "P50_KW": p50,
            "Full_R2": full_r2,
            "Full_RMSE": full_rmse,
            "Full_MAPE": full_mape,
            "Bulk_R2 (>P50)": bulk_r2,
            "Bulk_RMSE": bulk_rmse,
            "Bulk_MAPE": bulk_mape,
            "source": source
        })

    c2_df = pd.DataFrame(check2_records)
    c2_df = c2_df.sort_values("Bulk_R2 (>P50)", ascending=False).reset_index(drop=True)

    print("\n" + "="*130)
    print("ALL 29 CLEAN CHILLERS — BULK OPERATION METRICS (SORTED BEST TO WORST BY BULK R2 (>P50))")
    print("="*130)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(c2_df[["machineId", "n_running", "P50_KW", "Full_R2", "Full_MAPE", "Bulk_R2 (>P50)", "Bulk_RMSE", "Bulk_MAPE", "source"]].to_string(index=False))

    n_tot = len(c2_df)
    n_bulk_clear_50 = len(c2_df[c2_df["Bulk_R2 (>P50)"] >= 0.50])
    n_bulk_clear_60 = len(c2_df[c2_df["Bulk_R2 (>P50)"] >= 0.60])

    print("\n" + "="*95)
    print(f"BULK OPERATION SUMMARY ({n_tot} CHILLERS):")
    print(f"  --> Chillers Clearing Bulk R2 >= 0.60 (Strong Bulk Fit):     {n_bulk_clear_60} / {n_tot} ({n_bulk_clear_60/n_tot*100:.1f}%)")
    print(f"  --> Chillers Clearing Bulk R2 >= 0.50 (Defensible Bulk Fit): {n_bulk_clear_50} / {n_tot} ({n_bulk_clear_50/n_tot*100:.1f}%)")
    print("="*95)

    out_csv = os.path.join(PROJECT_ROOT, "data", "fleet_response_model_rigorous_results.csv")
    c2_df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed rigorous validation results to {out_csv}")


def main():
    run_check1_restricted_2026_history()
    if os.path.exists(TREND_WIDE_PATH):
        df_tw = pd.read_csv(TREND_WIDE_PATH)
        run_check2_bulk_operation_all_chillers(df_tw)


if __name__ == "__main__":
    main()
