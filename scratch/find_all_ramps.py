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

def find_temperature_ramps():
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
    
    ts_conn = get_timescale()
    ts_cur = ts_conn.cursor()
    
    print("Checking max/min across entire DB per sensor...")
    ts_cur.execute("""
        SELECT machineexplorerid, MIN(value), MAX(value), COUNT(*)
        FROM trendseriesmeterdata
        GROUP BY machineexplorerid;
    """)
    stats = ts_cur.fetchall()
    stats_df = pd.DataFrame(stats, columns=["sensor_id", "min_val", "max_val", "count"])
    merged = stats_df.merge(sensor_df, on="sensor_id", how="inner")
    
    print("\nSensors with max_val > 60:")
    high_temp = merged[merged["max_val"] > 60].sort_values("max_val", ascending=False)
    for _, row in high_temp.iterrows():
        print(f"Machine {row['machineId']} | Sensor {row['sensor_id']} ({row['series_description']}): min={row['min_val']}, max={row['max_val']}, count={row['count']}")

if __name__ == "__main__":
    find_temperature_ramps()
