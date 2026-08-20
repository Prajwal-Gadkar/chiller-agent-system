import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn_app = psycopg2.connect(
    host=os.environ["APPDB_HOST"],
    port=os.environ["APPDB_PORT"],
    dbname=os.environ["APPDB_NAME"],
    user=os.environ["APPDB_USER"],
    password=os.environ["APPDB_PASSWORD"],
)

query_m = """
    SELECT *
    FROM machine
    WHERE "machineId" IN (3392, 3894, 4054);
"""
df_m = pd.read_sql_query(query_m, conn_app)
print("=== MACHINE MASTER ROWS FOR 3392, 3894, 4054 ===")
print(df_m.to_string())

conn_app.close()
