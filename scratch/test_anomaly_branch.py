import os
import sys
import json

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.pipeline import build_pipeline_graph, PipelineState

pipeline_app = build_pipeline_graph()

# Inject an anomaly reading for Chiller 4054 (e.g. actual power spike to 250 kW when expected is ~88 kW)
anomaly_reading = {
    "DELTA T (DEG C)": 2.85,
    "CONDENSER FLOW (m3/h)": 275.5,
    "Outlet_Setpoint_Celsius": 7.0,
    "Outlet_temperature ValueY": 7.34,
    "inlet_temperature ValueY": 10.19,
    "Flow ValueY": 210.33,
    "KW ValueY": 250.0, # Actual power spike (well above expected ~88 kW)
    "Compressor1_RunHours": 6236.0,
    "machineId": 4054,
    "timestamp": "2026-08-19 12:00:00"
}

initial_state: PipelineState = {
    "chiller_id": 4054,
    "timestamp": "2026-08-19 12:00:00",
    "raw_reading": anomaly_reading,
    "validation_result": {},
    "anomaly_result": {},
    "insight_text": None
}

final_state = pipeline_app.invoke(initial_state)

print("=" * 100)
print("TESTING ANOMALY BRANCH ROUTING END-TO-END:")
print("=" * 100)
print(f"Chiller ID   : {final_state['chiller_id']}")
print(f"Timestamp    : {final_state['timestamp']}")
print(f"Actual KW    : {final_state['anomaly_result'].get('actual_kw'):.2f} kW")
print(f"Predicted KW : {final_state['anomaly_result'].get('predicted_kw'):.2f} kW")
print(f"Z-Score      : {final_state['anomaly_result'].get('z_score'):.2f}")
print(f"Is Anomaly   : {final_state['anomaly_result'].get('is_anomaly')}")
print(f"Insight Text : {final_state['insight_text']}")
print("\nFull State Output:\n")
print(json.dumps(final_state, indent=2))
