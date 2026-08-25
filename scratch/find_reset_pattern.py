import os
import sys
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

def inspect_machine_pre2026(m_id):
    conn = get_appdb()
    cur = conn.cursor()
    cur.execute("""
        SELECT me."Id", me."SeriesDescription"
        FROM machine m
        JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
        WHERE m."machineId" = %s AND m."machineType" = 'Chiller';
    """, (m_id,))
    sensors = cur.fetchall()
    conn.close()
    
    sensor_map = {s[0]: s[1] for s in sensors}
    sensor_ids = tuple(sensor_map.keys())
    
    if not sensor_ids:
        print(f"Machine {m_id}: No sensors found.")
        return
        
    ts_conn = get_timescale()
    cur = ts_conn.cursor()
    
    # Check date range pre-2026
    cur.execute("""
        SELECT MIN("timestamp" AT TIME ZONE 'Asia/Calcutta'), MAX("timestamp" AT TIME ZONE 'Asia/Calcutta'), COUNT(*)
        FROM trendseriesmeterdata
        WHERE machineexplorerid IN %s AND "timestamp" < '2026-01-01 00:00:00+05:30'::timestamptz;
    """, (sensor_ids,))
    min_ts, max_ts, count = cur.fetchone()
    print(f"\nMachine {m_id} pre-2026 DB range: {min_ts} to {max_ts}, total rows: {count:,}")
    
    if count == 0:
        ts_conn.close()
        return

    # Pull pre-2026 data for temperature or runhours or power columns
    cur.execute("""
        SELECT "timestamp" AT TIME ZONE 'Asia/Calcutta' AS ts, machineexplorerid, value
        FROM trendseriesmeterdata
        WHERE machineexplorerid IN %s AND "timestamp" < '2026-01-01 00:00:00+05:30'::timestamptz
        ORDER BY "timestamp";
    """, (sensor_ids,))
    rows = cur.fetchall()
    ts_conn.close()
    
    df = pd.DataFrame(rows, columns=["timestamp", "sensor_id", "value"])
    df["series"] = df["sensor_id"].map(sensor_map)
    
    print(f"Machine {m_id} unique series pre-2026:", df["series"].unique().tolist())
    
    # Check each series for monotonic ramp-then-reset or out-of-bound values
    for s_name, group in df.groupby("series"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        vals = group["value"].values
        ts_vals = group["timestamp"].values
        
        v_min, v_max = vals.min(), vals.max()
        print(f"  Machine {m_id} | {s_name}: min={v_min}, max={v_max}, count={len(vals)}")
        
        # Check for values exceeding typical bounds (e.g. > 60 for temp, > 5000 for flow/power)
        # or monotonic ramp followed by sharp drop
        # Let's inspect diffs
        diffs = np.diff(vals)
        # Look for sustained positive steps followed by a large negative drop
        pos_steps = np.sum(diffs > 0)
        neg_steps = np.sum(diffs < 0)
        
        # Find max single drop
        if len(diffs) > 0:
            max_drop_idx = np.argmin(diffs)
            max_drop_val = diffs[max_drop_idx]
            if max_drop_val < -50: # significant drop
                before_val = vals[max_drop_idx]
                after_val = vals[max_drop_idx + 1]
                t_before = ts_vals[max_drop_idx]
                t_after = ts_vals[max_drop_idx + 1]
                print(f"    --> RAMP/RESET CANDIDATE in {s_name}: drop of {max_drop_val:.2f} at index {max_drop_idx}")
                print(f"        Before ({t_before}): {before_val:.2f}, After ({t_after}): {after_val:.2f}")

if __name__ == "__main__":
    for m in [1657, 1658, 1659]:
        inspect_machine_pre2026(m)
