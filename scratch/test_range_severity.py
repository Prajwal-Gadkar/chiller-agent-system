import os
import sys
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.pipeline import build_pipeline_graph, PipelineState, get_anomaly_agent

def run_direct_tests():
    print("=" * 100, flush=True)
    print("DIRECT PIPELINE TEST HARNESS FOR SAFE_RANGE_KW & RANGE_SEVERITY", flush=True)
    print("=" * 100, flush=True)
    
    app = build_pipeline_graph()

    # 1. Test Chiller 2825
    print("\n--- 1. Testing Chiller 2825 ---", flush=True)
    agent_2825 = get_anomaly_agent(2825)
    print(f"Chiller 2825 Residual Std: {agent_2825.residual_std:.2f} kW", flush=True)

    power_col = agent_2825.col_map["power"]
    flow_col = agent_2825.col_map["flow"]
    inlet_col = agent_2825.col_map["inlet"]
    outlet_col = agent_2825.col_map["outlet"]
    comp_col = agent_2825.col_map.get("comp", "Compressor_1_Load")

    # We will test normal (|z|<=2), elevated (2<|z|<=3), and critical (|z|>3) readings
    # Predict benchmark first:
    base_df = pd.DataFrame([{flow_col: 200.0, inlet_col: 12.0, outlet_col: 7.0, comp_col: 50.0}])
    # Add DeltaT and Thermal_Load
    base_df["DeltaT"] = base_df[inlet_col] - base_df[outlet_col]
    base_df["Thermal_Load"] = base_df[flow_col] * base_df["DeltaT"]
    pred_base = agent_2825.model.predict(base_df[agent_2825.feature_names].values)[0]
    std_2825 = agent_2825.residual_std

    test_readings_2825 = [
        # Normal (|z| <= 2.0)
        {"label": "Normal Reading", power_col: pred_base + 0.5 * std_2825, flow_col: 200.0, inlet_col: 12.0, outlet_col: 7.0, comp_col: 50.0},
        # Elevated (2.0 < |z| <= 3.0)
        {"label": "Elevated Reading (2.5σ)", power_col: pred_base + 2.5 * std_2825, flow_col: 200.0, inlet_col: 12.0, outlet_col: 7.0, comp_col: 50.0},
        # Critical (|z| > 3.0)
        {"label": "Critical Reading (4.0σ Spike)", power_col: pred_base + 4.0 * std_2825, flow_col: 200.0, inlet_col: 12.0, outlet_col: 7.0, comp_col: 50.0},
    ]

    for idx, item in enumerate(test_readings_2825, 1):
        raw = {k: v for k, v in item.items() if k != "label"}
        state = {
            "chiller_id": 2825,
            "timestamp": f"2026-08-19 12:{idx*15:02d}:00",
            "raw_reading": raw,
            "validation_result": {},
            "anomaly_result": {},
            "insight_text": None
        }
        res = app.invoke(state)
        anom = res["anomaly_result"]
        print(f"[{item['label']}] Actual: {anom['actual_kw']:.1f} kW | Pred: {anom['predicted_kw']:.1f} kW | Range: {anom['safe_range_kw']} kW | Sev: {anom['range_severity'].upper()} | Z: {anom['z_score']:.2f}", flush=True)
        print(f"   Insight Text: \"{res['insight_text']}\"", flush=True)

    # 2. Test Chiller 4054 (Aliased asset Chiller-111)
    print("\n--- 2. Testing Chiller 4054 (Aliased Asset) ---", flush=True)
    agent_4054 = get_anomaly_agent(4054)
    print(f"Chiller 4054 Residual Std: {agent_4054.residual_std:.2f} kW", flush=True)

    p_col_4054 = agent_4054.col_map["power"]
    f_col_4054 = agent_4054.col_map["flow"]
    i_col_4054 = agent_4054.col_map["inlet"]
    o_col_4054 = agent_4054.col_map["outlet"]

    base_4054 = pd.DataFrame([{f_col_4054: 210.0, i_col_4054: 10.5, o_col_4054: 7.2}])
    base_4054["DeltaT"] = base_4054[i_col_4054] - base_4054[o_col_4054]
    base_4054["Thermal_Load"] = base_4054[f_col_4054] * base_4054["DeltaT"]
    pred_4054 = agent_4054.model.predict(base_4054[agent_4054.feature_names].values)[0]
    std_4054 = agent_4054.residual_std

    test_readings_4054 = [
        # Normal (|z| <= 2.0)
        {"label": "Normal Reading", p_col_4054: pred_4054 + 0.3 * std_4054, f_col_4054: 210.0, i_col_4054: 10.5, o_col_4054: 7.2},
        # Elevated (2.0 < |z| <= 3.0)
        {"label": "Elevated Reading (2.4σ)", p_col_4054: pred_4054 + 2.4 * std_4054, f_col_4054: 210.0, i_col_4054: 10.5, o_col_4054: 7.2},
        # Critical (|z| > 3.0)
        {"label": "Critical Reading (5.0σ Spike)", p_col_4054: pred_4054 + 5.0 * std_4054, f_col_4054: 210.0, i_col_4054: 10.5, o_col_4054: 7.2},
    ]

    for idx, item in enumerate(test_readings_4054, 1):
        raw = {k: v for k, v in item.items() if k != "label"}
        state = {
            "chiller_id": 4054,
            "timestamp": f"2026-08-19 14:{idx*15:02d}:00",
            "raw_reading": raw,
            "validation_result": {},
            "anomaly_result": {},
            "insight_text": None
        }
        res = app.invoke(state)
        anom = res["anomaly_result"]
        print(f"[{item['label']}] Actual: {anom['actual_kw']:.1f} kW | Pred: {anom['predicted_kw']:.1f} kW | Range: {anom['safe_range_kw']} kW | Sev: {anom['range_severity'].upper()} | Z: {anom['z_score']:.2f}", flush=True)
        print(f"   Insight Text: \"{res['insight_text']}\"", flush=True)

if __name__ == "__main__":
    run_direct_tests()
