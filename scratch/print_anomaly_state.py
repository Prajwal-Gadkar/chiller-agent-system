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
readings = fetch_recent_readings_from_db([4054, 2828, 2821], limit_per_chiller=20)

anomaly_states = []

for item in readings:
    initial_state: PipelineState = {
        "chiller_id": item["chiller_id"],
        "timestamp": item["timestamp"],
        "raw_reading": item["raw_reading"],
        "validation_result": {},
        "anomaly_result": {},
        "insight_text": None
    }
    final_state = app.invoke(initial_state)
    if final_state["anomaly_result"].get("is_anomaly", False):
        anomaly_states.append(final_state)

print("=" * 100)
print(f"FLAGGED ANOMALIES FOUND: {len(anomaly_states)}")
print("=" * 100)

for idx, state in enumerate(anomaly_states, 1):
    print(f"\n--- ANOMALY STATE #{idx} ---")
    print(f"Chiller ID      : {state['chiller_id']}")
    print(f"Timestamp       : {state['timestamp']}")
    print(f"Actual KW       : {state['anomaly_result'].get('actual_kw')}")
    print(f"Predicted KW    : {state['anomaly_result'].get('predicted_kw')}")
    print(f"Z-Score         : {state['anomaly_result'].get('z_score')}")
    print(f"Insight Text    : {state['insight_text']}")
    print(f"\nValidation Result:\n{json.dumps(state['validation_result'], indent=2)}")
    print(f"\nAnomaly Result   :\n{json.dumps(state['anomaly_result'], indent=2)}")
    print(f"\nRaw Reading      :\n{json.dumps(state['raw_reading'], indent=2)}")
    print("=" * 100)
