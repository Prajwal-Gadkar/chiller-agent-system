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

query_all = 'SELECT DISTINCT "machineId", "status", "Criticality" FROM machine WHERE "machineType" = \'Chiller\' ORDER BY "machineId";'
df_all = pd.read_sql_query(query_all, conn)

query_me = 'SELECT DISTINCT m."machineId" FROM machine m JOIN "MachineExplorer" me ON m."machineId" = me."MachineId" WHERE m."machineType" = \'Chiller\';'
df_me = pd.read_sql_query(query_me, conn)
conn.close()

all_ids = set(df_all["machineId"])
me_ids = set(df_me["machineId"])
missing = sorted(list(all_ids - me_ids))

print(f"Total Chiller assets in machine table: {len(all_ids)}")
print(f"Chillers with MachineExplorer sensor mappings: {len(me_ids)}")
print(f"Missing {len(missing)} chillers with zero sensor rows in MachineExplorer: {missing}")

if missing:
    print("\nDetails of missing chillers:")
    print(df_all[df_all["machineId"].isin(missing)].to_string(index=False))
