import os
import sys
import json
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.pipeline import build_pipeline_graph, fetch_recent_readings_from_db, PipelineState

app = build_pipeline_graph()
readings = fetch_recent_readings_from_db([4054], limit_per_chiller=20)

target_item = None
for item in readings:
    if "2026-07-31 09:30:00" in item["timestamp"]:
        target_item = item
        break

if target_item is None and readings:
    target_item = readings[0]

initial_state: PipelineState = {
    "chiller_id": target_item["chiller_id"],
    "timestamp": target_item["timestamp"],
    "raw_reading": target_item["raw_reading"],
    "validation_result": {},
    "anomaly_result": {},
    "insight_text": None
}

final_state = app.invoke(initial_state)

print("=" * 100)
print(f"CHILLER {final_state['chiller_id']} READING AT {final_state['timestamp']} EVALUATION:")
print("=" * 100)
print(f"Actual KW (KW ValueY) : {final_state['anomaly_result'].get('actual_kw'):.2f} kW")
print(f"Predicted KW          : {final_state['anomaly_result'].get('predicted_kw'):.2f} kW")
print(f"Residual              : {final_state['anomaly_result'].get('actual_kw') - final_state['anomaly_result'].get('predicted_kw'):.2f} kW")
print(f"Z-Score               : {final_state['anomaly_result'].get('z_score'):.2f}")
print(f"Is Anomaly (|z| > 3)  : {final_state['anomaly_result'].get('is_anomaly')}")
print(f"Insight Text          : {final_state['insight_text']}")
print("=" * 100)
