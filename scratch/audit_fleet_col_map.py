import os
import sys
import pickle
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.anomaly_agent import AnomalyAgent, MODEL_SAVE_DIR

model_files = [f for f in os.listdir(MODEL_SAVE_DIR) if f.endswith(".pkl")]
print(f"Auditing {len(model_files)} saved anomaly models in {MODEL_SAVE_DIR}...\n")

committed_used = []
col_map_summary = []

for f in sorted(model_files):
    path = os.path.join(MODEL_SAVE_DIR, f)
    with open(path, "rb") as fp:
        payload = pickle.load(fp)
        
    m_id = payload.get("machine_id")
    col_map = payload.get("col_map", {})
    features = payload.get("feature_names", [])
    power_col = col_map.get("power")
    
    if power_col and ("commit" in power_col.lower() or "committed" in power_col.lower()):
        committed_used.append((m_id, power_col, f))
        
    col_map_summary.append({
        "machine_id": m_id,
        "power_col": power_col,
        "flow_col": col_map.get("flow"),
        "inlet_col": col_map.get("inlet"),
        "outlet_col": col_map.get("outlet"),
        "file": f
    })

df_summary = pd.DataFrame(col_map_summary)
print("=== AUDIT SUMMARY ===")
print(f"Total Models Audited       : {len(model_files)}")
print(f"Models Using Committed Col : {len(committed_used)}")

if committed_used:
    print("\nModels improperly using 'COMMITED' columns as power target:")
    for m_id, pcol, filename in committed_used:
        print(f"   Chiller {m_id:4d} | Power Col: {pcol} | File: {filename}")
else:
    print("\nNo models used committed columns.")

print("\nDistinct Power Columns across all 52 models:")
print(df_summary["power_col"].value_counts())
