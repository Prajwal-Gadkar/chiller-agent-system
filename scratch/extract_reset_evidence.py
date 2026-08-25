import os
import json
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

APPDB_HOST = os.getenv("APPDB_HOST", "localhost")
APPDB_PORT = os.getenv("APPDB_PORT", "5432")
APPDB_NAME = os.getenv("APPDB_NAME", "Persistent_AppDb")
APPDB_USER = os.getenv("APPDB_USER", "postgres")
APPDB_PASSWORD = os.getenv("APPDB_PASSWORD", "admin")

TIMESCALE_HOST = os.getenv("TIMESCALE_HOST", "localhost")
TIMESCALE_PORT = os.getenv("TIMESCALE_PORT", "5432")
TIMESCALE_NAME = os.getenv("TIMESCALE_NAME", "Persistent_Timescale")
TIMESCALE_USER = os.getenv("TIMESCALE_USER", "postgres")
TIMESCALE_PASSWORD = os.getenv("TIMESCALE_PASSWORD", "admin")

def get_appdb():
    return psycopg2.connect(host=APPDB_HOST, port=APPDB_PORT, dbname=APPDB_NAME, user=APPDB_USER, password=APPDB_PASSWORD)

def get_timescale():
    return psycopg2.connect(host=TIMESCALE_HOST, port=TIMESCALE_PORT, dbname=TIMESCALE_NAME, user=TIMESCALE_USER, password=TIMESCALE_PASSWORD)

def search_chiller_resets():
    app_conn = get_appdb()
    cur = app_conn.cursor()
    cur.execute("""
        SELECT me."MachineId", me."Id", me."SeriesDescription"
        FROM machine m
        JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
        WHERE m."machineId" IN (1657, 1658, 1659, 2761, 2827) AND m."machineType" = 'Chiller';
    """)
    sensors = cur.fetchall()
    app_conn.close()
    
    ts_conn = get_timescale()
    ts_cur = ts_conn.cursor()
    
    evidence_list = []
    
    for m_id, s_id, desc in sensors:
        print(f"Scanning Machine {m_id} | Sensor {s_id} ({desc})...", flush=True)
        ts_cur.execute("""
            SELECT "timestamp" AT TIME ZONE 'Asia/Calcutta' AS ts, value
            FROM trendseriesmeterdata
            WHERE machineexplorerid = %s
            ORDER BY "timestamp";
        """, (s_id,))
        rows = ts_cur.fetchall()
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["timestamp", "value"])
        vals = df["value"].values
        ts_list = df["timestamp"].astype(str).tolist()
        
        # Look for sequences where value moves consistently up/down, crosses bound, then drops/jumps back into normal range
        for i in range(len(vals) - 1):
            curr_v = vals[i]
            next_v = vals[i+1]
            
            # Case 1: Temp > 45°C (or > 60°C physical bound) dropping back to < 30°C
            if "temp" in desc.lower() or "inlet" in desc.lower() or "outlet" in desc.lower():
                if curr_v > 45.0 and next_v < 30.0:
                    # Find ramp start (go backward to find where it started ramping)
                    ramp_start_idx = max(0, i - 10)
                    for j in range(i, max(0, i - 100), -1):
                        if vals[j] < 35.0:
                            ramp_start_idx = j
                            break
                    
                    ev = {
                        "machine_id": int(m_id),
                        "sensor_id": int(s_id),
                        "column": desc,
                        "ramp_start_time": ts_list[ramp_start_idx],
                        "ramp_start_val": float(vals[ramp_start_idx]),
                        "peak_time": ts_list[i],
                        "peak_val": float(curr_v),
                        "reset_time": ts_list[i+1],
                        "reset_val": float(next_v),
                        "single_step_drop": float(curr_v - next_v)
                    }
                    evidence_list.append(ev)
                    print(f"  FOUND EVIDENCE: {ev}", flush=True)

            # Case 2: Energy/KW/RunHours resetting from peak value to 0 or normal
            elif "energy" in desc.lower() or "kw" in desc.lower() or "hours" in desc.lower():
                if curr_v > 500.0 and next_v < 50.0:
                    ev = {
                        "machine_id": int(m_id),
                        "sensor_id": int(s_id),
                        "column": desc,
                        "peak_time": ts_list[i],
                        "peak_val": float(curr_v),
                        "reset_time": ts_list[i+1],
                        "reset_val": float(next_v),
                        "single_step_drop": float(curr_v - next_v)
                    }
                    evidence_list.append(ev)
                    print(f"  FOUND COUNTER RESET EVIDENCE: {ev}", flush=True)

    ts_conn.close()
    
    # Save evidence to data/reset_evidence.csv and data/reset_evidence.json
    out_json = "data/reset_evidence.json"
    with open(out_json, "w") as f:
        json.dump(evidence_list, f, indent=2)
    print(f"Saved {len(evidence_list)} evidence entries to {out_json}")
    
    if evidence_list:
        out_csv = "data/reset_evidence.csv"
        pd.DataFrame(evidence_list).to_csv(out_csv, index=False)
        print(f"Saved evidence CSV to {out_csv}")

if __name__ == "__main__":
    search_chiller_resets()
