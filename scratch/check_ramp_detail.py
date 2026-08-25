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

def get_timescale():
    return psycopg2.connect(host=TIMESCALE_HOST, port=TIMESCALE_PORT, dbname=TIMESCALE_NAME, user=TIMESCALE_USER, password=TIMESCALE_PASSWORD)

def inspect_sensor(sensor_id, machine_id, series_desc):
    ts_conn = get_timescale()
    cur = ts_conn.cursor()
    cur.execute("""
        SELECT "timestamp" AT TIME ZONE 'Asia/Calcutta' AS ts, value
        FROM trendseriesmeterdata
        WHERE machineexplorerid = %s
        ORDER BY "timestamp";
    """, (sensor_id,))
    rows = cur.fetchall()
    ts_conn.close()
    
    df = pd.DataFrame(rows, columns=["timestamp", "value"])
    print(f"\nMachine {machine_id} | Sensor {sensor_id} ({series_desc}) total rows: {len(df)}")
    
    # Check max values and look for ramp reset
    df_out = df[df["value"] > 40.0]
    print("Rows with value > 40°C:", len(df_out))
    if not df_out.empty:
        print("Sample out of bound rows:")
        print(df_out.head(10))
        
        # Find where it drops back down to normal (< 30) after being > 40
        idx_high = df_out.index
        for i in idx_high:
            if i + 1 < len(df):
                next_val = df.loc[i+1, "value"]
                if next_val < 30.0:
                    t_high = df.loc[i, "timestamp"]
                    val_high = df.loc[i, "value"]
                    t_next = df.loc[i+1, "timestamp"]
                    print(f"FOUND RAMP RESET RESET POINT: at {t_high} val={val_high} -> at {t_next} val={next_val}")

if __name__ == "__main__":
    # Check Machine 2761 sensors 21301 and 21300
    inspect_sensor(21301, 2761, "inlet_temperature ValueY")
    inspect_sensor(21300, 2761, "Outlet_temperature ValueY")
