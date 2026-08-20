import os
import sys
import pickle
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.anomaly_agent import AnomalyAgent, find_sensor_column, MODEL_SAVE_DIR
from agents.pipeline import find_sensor_col_single_row

# Test updated find_sensor_column logic
def find_sensor_column_fixed(df: pd.DataFrame, sensor_type: str):
    exact_candidates = {
        "flow": ["Flow ValueY", "CHW FLOW RATE (m3/h)", "Flow", "CONDENSER FLOW (m3/h)"],
        "kw": ["KW ValueY", "CH-1, POWER CONSUMPTION (KW)", "Running_KW_Active_Power ValueY", "KW", "power"],
        "inlet": ["inlet_temperature ValueY", "Evaporator_Inlet_Temp", "inlet_temperature", "CHW RETURN TEMPERATURE (DEG C)"],
        "outlet": ["Outlet_temperature ValueY", "Evaporator_Outlet_Temp", "Outlet_temperature", "CHW LEAVE TEMPERATURE (DEG C)"],
        "comp": ["Compressor_1_Load ValueY", "Compressor_1_Load", "CompressorLoad", "CompLoad"]
    }

    candidates = exact_candidates.get(sensor_type, [])

    # Step 1: Exact candidate match
    for cand in candidates:
        for col in df.columns:
            if col.lower() == cand.lower() and df[col].notna().sum() > 30:
                col_lower = col.lower()
                if "commit" in col_lower or "committed" in col_lower:
                    continue
                return col

    # Step 2: Substring match
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

# Load trend_wide.csv
trend_path = os.path.join(PROJECT_ROOT, "data", "trend_wide.csv")
df_wide = pd.read_csv(trend_path)

chillers_to_refit = [3392, 3894, 4054]

print("Refitting physical response models for Chillers 3392, 3894, 4054 using 'KW ValueY'...")

for c_id in chillers_to_refit:
    m_df = df_wide[df_wide["machineId"] == c_id].copy()
    
    # Check sensor col for kw
    kw_col = find_sensor_column_fixed(m_df, "kw")
    flow_col = find_sensor_column_fixed(m_df, "flow")
    inlet_col = find_sensor_column_fixed(m_df, "inlet")
    outlet_col = find_sensor_column_fixed(m_df, "outlet")
    print(f"Chiller {c_id} mapped columns -> Power: {kw_col}, Flow: {flow_col}, Inlet: {inlet_col}, Outlet: {outlet_col}")
    
    agent = AnomalyAgent(machine_id=c_id)
    # Patch agent's find_sensor_column during fit by setting col_map explicitly or using fixed finder
    agent.col_map = {"flow": flow_col, "power": kw_col, "inlet": inlet_col, "outlet": outlet_col, "comp": None}
    
    # Fit response model
    cv_metrics = agent.fit(m_df, regime_start="2026-01-01")
    agent.save(save_dir=MODEL_SAVE_DIR)
    print(f"Chiller {c_id} refitted successfully! R2 CV: {cv_metrics['R2']:.4f}, Res Std: {agent.residual_std:.2f}")

print("\nRefitting complete. Saved updated models to data/anomaly_models/\n")
