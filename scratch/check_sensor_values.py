import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn_ts = psycopg2.connect(
    host=os.environ["TIMESCALE_HOST"],
    port=os.environ["TIMESCALE_PORT"],
    dbname=os.environ["TIMESCALE_NAME"],
    user=os.environ["TIMESCALE_USER"],
    password=os.environ["TIMESCALE_PASSWORD"],
)

# Pull sensor values for 3392 (KW=24504), 3894 (KW=24713), 4054 (KW=24945)
query = """
    SELECT machineexplorerid, "timestamp", value
    FROM trendseriesmeterdata
    WHERE machineexplorerid IN (24504, 24713, 24945)
      AND "timestamp" >= '2026-05-01'
    LIMIT 20;
"""
df = pd.read_sql_query(query, conn_ts)
print("=== SENSOR READINGS SAMPLE FOR 3392 (24504), 3894 (24713), 4054 (24945) ===")
print(df.to_string())

# Check total row count per sensor id
query_counts = """
    SELECT machineexplorerid, COUNT(*) as row_count, AVG(value) as avg_val, MIN(value) as min_val, MAX(value) as max_val
    FROM trendseriesmeterdata
    WHERE machineexplorerid IN (24504, 24713, 24945, 24503, 24712, 24944)
      AND "timestamp" >= '2026-05-01'
    GROUP BY machineexplorerid;
"""
df_counts = pd.read_sql_query(query_counts, conn_ts)
print("\n=== AGGREGATES PER SENSOR ID ===")
print(df_counts.to_string())

conn_ts.close()
