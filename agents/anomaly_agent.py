"""
Anomaly Agent — Validated Physical Response Model & Residual Anomaly Detection

This agent implements the validated physical response model:
    KW = f(Flow, InletTemp, OutletTemp, DeltaT, Thermal_Load, [CompressorLoad])
using RandomForestRegressor.

Core principles enforced:
1. Per-chiller modeling: Models are fitted and saved strictly per chiller (never pooled).
2. Regime Boundary: Training and evaluation occur strictly within a single regime
   (default: post-2026-01-01 window, respecting the hard 2026-01-01 regime shift).
3. Random 5-Fold CV: Evaluates physical response fit using 5-fold random K-Fold CV.
4. Anomaly Detection: Computes actual vs predicted KW residuals, converts them to z-scores
   (z = (residual - mean) / std), and flags readings with |z| > 3.0 as anomalous.
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.data_validation import validate

MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "anomaly_models")


def find_sensor_column(df: pd.DataFrame, sensor_type: str) -> Optional[str]:
    """Find the best populated column for a given sensor type (flow, kw, inlet, outlet, comp)."""
    exact_candidates = {
        "flow": ["Flow ValueY", "CHW FLOW RATE (m3/h)", "Flow"],
        "kw": ["KW ValueY", "CH-1, POWER CONSUMPTION (KW)", "Running_KW_Active_Power ValueY", "KW", "power"],
        "inlet": ["inlet_temperature ValueY", "Evaporator_Inlet_Temp", "inlet_temperature", "CHW RETURN TEMPERATURE (DEG C)"],
        "outlet": ["Outlet_temperature ValueY", "Evaporator_Outlet_Temp", "Outlet_temperature", "CHW LEAVE TEMPERATURE (DEG C)"],
        "comp": ["Compressor_1_Load ValueY", "Compressor_1_Load", "CompressorLoad", "CompLoad"]
    }

    candidates = exact_candidates.get(sensor_type, [])

    # Step 1: Exact candidate match first
    for cand in candidates:
        for col in df.columns:
            col_lower = col.lower()
            if "commit" in col_lower or "committed" in col_lower:
                continue
            if sensor_type == "kw" and ("ikw" in col_lower or "kwh" in col_lower):
                continue
            if sensor_type in ["inlet", "outlet"] and "condenser" in col_lower:
                continue
            if col_lower == cand.lower() and df[col].notna().sum() > 30:
                return col

    # Step 2: Substring match fallback
    for col in df.columns:
        col_lower = col.lower()
        if "commit" in col_lower or "committed" in col_lower:
            continue
        if sensor_type == "kw" and ("ikw" in col_lower or "kwh" in col_lower):
            continue
        if sensor_type in ["inlet", "outlet"] and "condenser" in col_lower:
            continue
        if any(c.lower() in col_lower for c in candidates):
            if df[col].notna().sum() > 30:
                return col

    return None


def extract_response_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, str]]:
    """
    Identify and extract response model features and target from a chiller DataFrame.

    Returns:
        (X_df, y_series, feature_names, col_map)
    """
    flow_col = find_sensor_column(df, "flow")
    power_col = find_sensor_column(df, "kw")
    inlet_col = find_sensor_column(df, "inlet")
    outlet_col = find_sensor_column(df, "outlet")
    comp_col = find_sensor_column(df, "comp")

    if not flow_col or not power_col or not inlet_col or not outlet_col:
        missing = []
        if not flow_col: missing.append("Flow")
        if not power_col: missing.append("Power (KW)")
        if not inlet_col: missing.append("Inlet Temp")
        if not outlet_col: missing.append("Outlet Temp")
        raise ValueError(f"Missing required response model features: {missing}")

    work_df = df.copy()

    # Compute derived physical features
    work_df["DeltaT"] = work_df[inlet_col] - work_df[outlet_col]
    work_df["Thermal_Load"] = work_df[flow_col] * work_df["DeltaT"]

    feature_cols = [flow_col, inlet_col, outlet_col, "DeltaT", "Thermal_Load"]
    if comp_col and comp_col in work_df.columns and work_df[comp_col].notna().sum() > 30:
        feature_cols.append(comp_col)

    col_map = {
        "flow": flow_col,
        "power": power_col,
        "inlet": inlet_col,
        "outlet": outlet_col,
        "comp": comp_col
    }

    return work_df[feature_cols], work_df[power_col], feature_cols, col_map



class AnomalyAgent:
    """
    Per-chiller physical response model and residual anomaly detector.
    """

    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = []
        self.col_map: Dict[str, str] = {}
        self.residual_mean: float = 0.0
        self.residual_std: float = 1.0
        self.cv_metrics: Dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        regime_start: str = "2026-01-01",
        n_splits: int = 5,
        n_estimators: int = 100,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Fit response model on clean running data within a single regime and evaluate 5-fold CV.

        Args:
            df: Raw wide trend DataFrame for this chiller.
            regime_start: Start date for regime window (default '2026-01-01').
            n_splits: Number of CV folds.
            n_estimators: Trees in RandomForest.
            random_state: Seed.

        Returns:
            Dictionary of cross-validation metrics (R2, RMSE, MAE).
        """
        df_copy = df.copy()
        if "timestamp" in df_copy.columns:
            df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
            if regime_start:
                df_copy = df_copy[df_copy["timestamp"] >= regime_start].copy()

        if len(df_copy) < 30:
            raise ValueError(f"Insufficient data for machine {self.machine_id} in regime >= {regime_start} (count={len(df_copy)})")

        # Step 1: Data validation filter
        flagged_df, _ = validate(df_copy)
        X_raw, y_raw, feat_cols, col_map = extract_response_features(flagged_df)

        self.feature_names = feat_cols
        self.col_map = col_map

        # Filter valid unflagged rows
        valid_mask = pd.Series(True, index=flagged_df.index)
        for c in [col_map["flow"], col_map["power"], col_map["inlet"], col_map["outlet"]]:
            if f"{c}_flagged" in flagged_df.columns:
                valid_mask &= (~flagged_df[f"{c}_flagged"])

        clean_df = flagged_df[valid_mask].copy()

        # Running filter (chiller active, KW > 10.0)
        power_col = col_map["power"]
        running_mask = clean_df[power_col] > 10.0
        running_df = clean_df[running_mask].copy().reset_index(drop=True)

        if len(running_df) < 30:
            raise ValueError(f"Too few clean running samples for machine {self.machine_id} (count={len(running_df)})")

        X_feat, y_target, _, _ = extract_response_features(running_df)
        X = X_feat.values
        y = y_target.values

        # 5-fold random K-Fold CV
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        r2_list, rmse_list, mape_list = [], [], []

        for train_idx, test_idx in kf.split(X):
            rf_fold = RandomForestRegressor(n_estimators=n_estimators, max_depth=8, random_state=random_state)
            rf_fold.fit(X[train_idx], y[train_idx])
            preds = rf_fold.predict(X[test_idx])

            r2_list.append(r2_score(y[test_idx], preds))
            rmse_list.append(np.sqrt(mean_squared_error(y[test_idx], preds)))
            mape_list.append(np.mean(np.abs((y[test_idx] - preds) / np.maximum(y[test_idx], 1.0))) * 100.0)

        self.cv_metrics = {
            "R2": float(np.mean(r2_list)),
            "R2_std": float(np.std(r2_list)),
            "RMSE": float(np.mean(rmse_list)),
            "MAPE": float(np.mean(mape_list)),
            "n_samples": len(running_df)
        }

        # Fit full model on all clean running data
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=8, random_state=random_state)
        self.model.fit(X, y)

        # Compute residual statistics
        train_preds = self.model.predict(X)
        residuals = y - train_preds
        self.residual_mean = float(np.mean(residuals))
        self.residual_std = float(np.std(residuals))
        if self.residual_std < 1e-6:
            self.residual_std = 1.0

        return self.cv_metrics

    def detect_anomalies(self, df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Evaluate new/test readings, calculate actual vs predicted KW residuals and z-scores,
        and flag |z| > z_threshold as anomalous.

        Returns DataFrame with extra columns:
            - predicted_KW
            - residual_KW
            - z_score
            - is_anomalous
        """
        if self.model is None:
            raise ValueError(f"Model for machine {self.machine_id} has not been fitted or loaded yet.")

        df_out = df.copy()

        # Extract features
        work_df = df_out.copy()
        flow_col = self.col_map["flow"]
        inlet_col = self.col_map["inlet"]
        outlet_col = self.col_map["outlet"]
        power_col = self.col_map["power"]

        work_df["DeltaT"] = work_df[inlet_col] - work_df[outlet_col]
        work_df["Thermal_Load"] = work_df[flow_col] * work_df["DeltaT"]

        X_input = work_df[self.feature_names].values

        # Handle NaNs safely
        valid_rows = ~np.isnan(X_input).any(axis=1)

        predicted_kw = np.full(len(df_out), np.nan)
        residuals = np.full(len(df_out), np.nan)
        z_scores = np.full(len(df_out), np.nan)
        is_anomalous = np.full(len(df_out), False)

        if np.any(valid_rows):
            preds = self.model.predict(X_input[valid_rows])
            actuals = work_df.loc[valid_rows, power_col].values

            res = actuals - preds
            zs = (res - self.residual_mean) / self.residual_std
            anom = np.abs(zs) > z_threshold

            predicted_kw[valid_rows] = preds
            residuals[valid_rows] = res
            z_scores[valid_rows] = zs
            is_anomalous[valid_rows] = anom

        df_out["predicted_KW"] = predicted_kw
        df_out["residual_KW"] = residuals
        df_out["z_score"] = z_scores
        df_out["is_anomalous"] = is_anomalous

        return df_out

    def save(self, save_dir: str = MODEL_SAVE_DIR) -> str:
        """Save fitted model and metadata to disk."""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"chiller_{self.machine_id}_response_model.pkl")
        payload = {
            "machine_id": self.machine_id,
            "model": self.model,
            "feature_names": self.feature_names,
            "col_map": self.col_map,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "cv_metrics": self.cv_metrics
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)
        return save_path

    @classmethod
    def load(cls, machine_id: int, save_dir: str = MODEL_SAVE_DIR) -> "AnomalyAgent":
        """Load fitted model and metadata from disk."""
        load_path = os.path.join(save_dir, f"chiller_{machine_id}_response_model.pkl")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No saved model found for machine {machine_id} at {load_path}")
        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        agent = cls(machine_id=payload["machine_id"])
        agent.model = payload["model"]
        agent.feature_names = payload["feature_names"]
        agent.col_map = payload["col_map"]
        agent.residual_mean = payload["residual_mean"]
        agent.residual_std = payload["residual_std"]
        agent.cv_metrics = payload["cv_metrics"]
        return agent


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Anomaly Agent response models for clean fleet chillers.")
    parser.add_argument("--trend-wide", default=os.path.join(PROJECT_ROOT, "data", "trend_wide.csv"), help="Path to wide trend CSV")
    parser.add_argument("--clean-list", default=os.path.join(PROJECT_ROOT, "data", "inlet_outlet_fleet_corruption_check.csv"), help="Path to clean chillers CSV")
    parser.add_argument("--save-dir", default=MODEL_SAVE_DIR, help="Directory to save trained models")
    parser.add_argument("--regime-start", default="2026-01-01", help="Hard regime boundary start date (default '2026-01-01')")
    args = parser.parse_args()

    print("=" * 100, flush=True)
    print("ANOMALY AGENT — FIT PHYSICAL RESPONSE MODELS & DETECT RESIDUAL ANOMALIES", flush=True)
    print(f"Regime boundary: >= {args.regime_start}", flush=True)
    print("=" * 100, flush=True)

    if not os.path.exists(args.trend_wide):
        print(f"Error: {args.trend_wide} not found.", flush=True)
        sys.exit(1)

    df_wide = pd.read_csv(args.trend_wide)
    clean_df = pd.read_csv(args.clean_list)
    clean_m_ids = clean_df["machineId"].unique().tolist()

    print(f"Loaded {len(df_wide)} rows across fleet. Training models for {len(clean_m_ids)} clean pool chillers...", flush=True)

    summary_records = []

    for m_id in clean_m_ids:
        # Check if individual full history CSV exists first (for long-history chillers 1657, 1660, 1661)
        hist_path = os.path.join(PROJECT_ROOT, "data", f"chiller_{m_id}_full_history.csv")
        if os.path.exists(hist_path):
            m_df = pd.read_csv(hist_path)
            source = f"chiller_{m_id}_full_history.csv"
        else:
            m_df = df_wide[df_wide["machineId"] == m_id].copy()
            source = "trend_wide.csv"

        if len(m_df) == 0:
            continue

        agent = AnomalyAgent(machine_id=m_id)
        try:
            cv_metrics = agent.fit(m_df, regime_start=args.regime_start)
            save_path = agent.save(save_dir=args.save_dir)

            # Test anomaly detection on fitted data
            detected_df = agent.detect_anomalies(m_df)
            n_anom = int(detected_df["is_anomalous"].sum())
            pct_anom = (n_anom / len(detected_df)) * 100.0 if len(detected_df) > 0 else 0.0

            summary_records.append({
                "machineId": m_id,
                "R2_CV": cv_metrics["R2"],
                "RMSE_CV": cv_metrics["RMSE"],
                "MAPE_CV": cv_metrics["MAPE"],
                "n_samples": cv_metrics["n_samples"],
                "res_mean": agent.residual_mean,
                "res_std": agent.residual_std,
                "n_anom_flagged": n_anom,
                "pct_anom_flagged": pct_anom,
                "source": source
            })
            print(f"Chiller {m_id:4d} | R2 CV: {cv_metrics['R2']:.4f} | RMSE: {cv_metrics['RMSE']:.2f} | Anomalies: {n_anom:3d} ({pct_anom:.2f}%) | Model saved", flush=True)
        except Exception as e:
            print(f"Chiller {m_id:4d} | Skipped: {e}", flush=True)

    summary_df = pd.DataFrame(summary_records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("R2_CV", ascending=False).reset_index(drop=True)
        print("\n" + "=" * 100, flush=True)
        print("SUMMARY TABLE — ANOMALY AGENT RESPONSE MODEL CROSS-VALIDATION & RESIDUALS", flush=True)
        print("=" * 100, flush=True)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(summary_df.to_string(index=False), flush=True)

        out_summary_path = os.path.join(PROJECT_ROOT, "data", "anomaly_agent_fit_summary.csv")
        summary_df.to_csv(out_summary_path, index=False)
        print(f"\nSaved Anomaly Agent training summary to {out_summary_path}", flush=True)


if __name__ == "__main__":
    main()

