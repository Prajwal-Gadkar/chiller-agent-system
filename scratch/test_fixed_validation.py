import os
import sys
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.pipeline import fetch_recent_readings_from_db
import agents.data_validation as dv

# Test refined BOUND_RULES where "hours" comes before "pressure" and "compressor" is stripped when checking "press"
BOUND_RULES_FIXED = [
    (["status", "alarm", "fault", "trip"], 0.0, 1.0, "duty_cycle_status"),
    (["percent", "%", "deviation", "load", "performance"], -10.0, 110.0, "percentage"),
    (["temperature", "temp", "wet bulb", "cwet"], -20.0, 60.0, "temperature"),
    (["hours", "runhours"], 0.0, 200000.0, "cumulative_hours"),
    (["speed"], -10.0, 110.0, "speed_pct"),
    (["kw", "power", "ikw"], -10.0, 5000.0, "power"),
    (["pressure", "press"], -50.0, 500.0, "pressure"),
    (["flow"], -10.0, 5000.0, "flow"),
    (["setpoint"], -20.0, 60.0, "setpoint_temperature"),
]

def infer_bounds_fixed(column_name):
    name_lower = column_name.lower()
    # Strip 'compressor' so 'press' inside 'compressor' doesn't accidentally trigger pressure rule
    name_cleaned = name_lower.replace("compressor", "comp")
    for keywords, lo, hi, label in BOUND_RULES_FIXED:
        if any(kw in name_cleaned for kw in keywords):
            return lo, hi, label
    return None, None, "statistical_fallback"

chiller_ids = [4054, 2828, 2821]
readings = fetch_recent_readings_from_db(chiller_ids, limit_per_chiller=20)

print(f"Testing fixed infer_bounds across {len(readings)} readings...\n")

failed_count = 0
for item in readings:
    c_id = item["chiller_id"]
    ts = item["timestamp"]
    raw_reading = item["raw_reading"]
    df = pd.DataFrame([raw_reading])
    
    # Run validation with fixed infer_bounds logic
    flagged_cols = []
    for col in df.columns:
        if col in dv.METADATA_COLUMNS or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col]
        lo, hi, rule = infer_bounds_fixed(col)
        if rule != "statistical_fallback":
            flagged = ((series < lo) | (series > hi)).fillna(False)
            if bool(flagged.iloc[0]):
                flagged_cols.append((col, series.iloc[0], rule, lo, hi))
                
    if flagged_cols:
        failed_count += 1
        print(f"FAILED Reading #{failed_count} | Chiller {c_id} at {ts}:")
        for col, val, rule, lo, hi in flagged_cols:
            print(f"   Col: {col} = {val} | Rule: {rule} [{lo}, {hi}]")
            
print(f"\nResult with Fixed Rules: {failed_count} failed out of {len(readings)} (Valid: {len(readings) - failed_count}/{len(readings)})")
