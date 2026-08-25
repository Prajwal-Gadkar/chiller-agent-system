import os
import psycopg2
import pandas as pd
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

def find_reset_in_db():
    app_conn = get_appdb()
    cur = app_conn.cursor()
    cur.execute("""
        SELECT m."machineId", me."Id", me."SeriesDescription"
        FROM machine m
        JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
        WHERE m."machineType" = 'Chiller';
    """)
    sensors = cur.fetchall()
    app_conn.close()
    
    sensor_df = pd.DataFrame(sensors, columns=["machineId", "sensor_id", "series_description"])
    print(f"Total chiller sensors: {len(sensor_df)}")
    
    ts_conn = get_timescale()
    ts_cur = ts_conn.cursor()
    
    # Query max/min per sensor in pre-2026 data
    print("Checking max/min values per sensor in pre-2026 data...")
    ts_cur.execute("""
        SELECT machineexplorerid, MIN(value), MAX(value), COUNT(*)
        FROM trendseriesmeterdata
        WHERE "timestamp" < '2026-01-01 00:00:00+05:30'::timestamptz
        GROUP BY machineexplorerid;
    """)
    stats = ts_cur.fetchall()
    
    stats_df = pd.DataFrame(stats, columns=["sensor_id", "min_val", "max_val", "count"])
    merged = stats_df.merge(sensor_df, on="sensor_id", how="inner")
    
    print("\nSensors with high max values pre-2026:")
    high_vals = merged[merged["max_val"] > 60].sort_values("max_val", ascending=False)
    for _, row in high_vals.iterrows():
        print(f"Machine {row['machineId']} | Sensor {row['sensor_id']} ({row['series_description']}): min={row['min_val']}, max={row['max_val']}, count={row['count']}")
        
    ts_conn.close()
    return merged

if __name__ == "__main__":
    find_reset_in_db()
