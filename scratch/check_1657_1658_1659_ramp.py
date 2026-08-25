import os
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

def check_long_chillers():
    app_conn = get_appdb()
    cur = app_conn.cursor()
    cur.execute("""
        SELECT me."MachineId", me."Id", me."SeriesDescription"
        FROM machine m
        JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
        WHERE m."machineId" IN (1657, 1658, 1659) AND m."machineType" = 'Chiller';
    """)
    sensors = cur.fetchall()
    app_conn.close()
    
    ts_conn = get_timescale()
    ts_cur = ts_conn.cursor()
    
    for m_id, s_id, desc in sensors:
        ts_cur.execute("""
            SELECT "timestamp" AT TIME ZONE 'Asia/Calcutta' AS ts, value
            FROM trendseriesmeterdata
            WHERE machineexplorerid = %s AND "timestamp" < '2026-01-01 00:00:00+05:30'::timestamptz
            ORDER BY "timestamp";
        """, (s_id,))
        rows = ts_cur.fetchall()
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["timestamp", "value"])
        vals = df["value"].values
        diffs = np.diff(vals)
        if len(diffs) == 0:
            continue
        
        # Look for monotonic ramp: consecutive positive diffs over sustained run crossing bound, then single negative drop
        min_v, max_v = vals.min(), vals.max()
        print(f"Machine {m_id} | Sensor {s_id} ({desc}) pre-2026: min={min_v}, max={max_v}, rows={len(df)}")
        
        # Find any drops where val was abnormally high or ramped
        # For instance, diffs where negative drop > 20% of max_v or drop > 15
        drops = np.where(diffs < -15)[0]
        for d in drops:
            val_before = vals[d]
            val_after = vals[d+1]
            t_before = df.loc[d, "timestamp"]
            t_after = df.loc[d+1, "timestamp"]
            print(f"  --> Large Drop: idx {d} ({t_before}): {val_before:.2f} -> ({t_after}): {val_after:.2f}")

    ts_conn.close()

if __name__ == "__main__":
    check_long_chillers()
