import os
import sys
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.pipeline import fetch_recent_readings_from_db
from agents.data_validation import validate, infer_bounds

chiller_ids = [4054, 2828, 2821]
readings = fetch_recent_readings_from_db(chiller_ids, limit_per_chiller=20)

print(f"Inspecting {len(readings)} readings...\n")

failed_count = 0

for item in readings:
    c_id = item["chiller_id"]
    ts = item["timestamp"]
    raw_reading = item["raw_reading"]
    
    df_single = pd.DataFrame([raw_reading])
    flagged_df, report_df = validate(df_single)
    
    flagged_cols = []
    for c in flagged_df.columns:
        if c.endswith("_flagged") and bool(flagged_df[c].iloc[0]):
            col_name = c[:-8]
            val = raw_reading.get(col_name)
            lo, hi, rule = infer_bounds(col_name)
            flagged_cols.append({
                "column": col_name,
                "value": val,
                "value_type": type(val).__name__,
                "bound_min": lo,
                "bound_max": hi,
                "rule": rule
            })
            
    if flagged_cols:
        failed_count += 1
        print(f"FAILED Reading #{failed_count:02d} | Chiller {c_id} at {ts}:")
        for f in flagged_cols:
            val = f['value']
            lo = f['bound_min']
            hi = f['bound_max']
            is_below = (val < lo) if (lo is not None and val is not None) else False
            is_above = (val > hi) if (hi is not None and val is not None) else False
            print(f"   Column      : {f['column']}")
            print(f"   Raw Value   : {val} (type: {f['value_type']})")
            print(f"   Applied Rule: {f['rule']} [min: {lo}, max: {hi}]")
            print(f"   Reason      : below min ({is_below}) or above max ({is_above})")
        print("-" * 80)

print(f"\nTotal Failed Readings: {failed_count} / {len(readings)}")
