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

def get_timescale():
    return psycopg2.connect(host=TIMESCALE_HOST, port=TIMESCALE_PORT, dbname=TIMESCALE_NAME, user=TIMESCALE_USER, password=TIMESCALE_PASSWORD)

def check_reset_fast():
    ts_conn = get_timescale()
    cur = ts_conn.cursor()
    
    # Query sensors with temperature/flow/KW ramping or out-of-bound behavior
    # Specifically looking for pre-2026 or post-2026 reset events
    cur.execute("""
        SELECT me."MachineId", t.machineexplorerid, me."SeriesDescription",
               MIN(t.value), MAX(t.value), COUNT(*)
        FROM trendseriesmeterdata t
        JOIN "MachineExplorer" me ON t.machineexplorerid = me."Id"
        WHERE t.value > 60.0 AND me."SeriesDescription" ILIKE '%temp%'
        GROUP BY me."MachineId", t.machineexplorerid, me."SeriesDescription";
    """)
    rows = cur.fetchall()
    print("Temperature sensors exceeding 60C:")
    for r in rows:
        print(f"Machine {r[0]} | Sensor {r[1]} ({r[2]}): min_out_of_bound={r[3]}, max={r[4]}, count_above_60={r[5]}")
        
    ts_conn.close()

if __name__ == "__main__":
    check_reset_fast()
