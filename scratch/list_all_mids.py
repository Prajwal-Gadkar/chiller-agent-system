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

cur = conn.cursor()
cur.execute('SELECT DISTINCT "machineId" FROM machine WHERE "machineType" = \'Chiller\' ORDER BY "machineId";')
m_ids = [r[0] for r in cur.fetchall()]
conn.close()

print(f"Total chillers in AppDb: {len(m_ids)}")
print(m_ids)
