import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.environ["APPDB_HOST"],
    port=os.environ["APPDB_PORT"],
    dbname=os.environ["APPDB_NAME"],
    user=os.environ["APPDB_USER"],
    password=os.environ["APPDB_PASSWORD"],
)

query = """
    SELECT
        m."machineId",
        me."Id" AS "MachineExplorerId",
        me."SeriesDescription"
    FROM machine m
    JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
    WHERE m."machineType" = 'Chiller'
    ORDER BY m."machineId", me."Id";
"""
df = pd.read_sql_query(query, conn)
conn.close()

all_me_mids = set(df["machineId"].unique())
valid_series_mids = set(df.dropna(subset=["SeriesDescription"])["machineId"].unique())

missing_4 = sorted(list(all_me_mids - valid_series_mids))
print(f"Total distinct machines in MachineExplorer: {len(all_me_mids)}")
print(f"Machines with non-null SeriesDescription: {len(valid_series_mids)}")
print(f"The 4 missing chillers with NULL SeriesDescription: {missing_4}")

print("\nDetails of NULL SeriesDescription rows:")
print(df[df["machineId"].isin(missing_4)].to_string(index=False))
