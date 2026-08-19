"""
Read-only exploration script: lists every asset in AppDb where
machine.machineType = 'Chiller', joined with MachineExplorer on MachineId.

Does not write to the database. Connection details are loaded from
environment variables (.env), never hardcoded.
"""

import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

QUERY = """
    SELECT
        m."machineId",
        m."machineType",
        m."status",
        m."Criticality",
        me."Id" AS "MachineExplorerId",
        me."SeriesDescription"
    FROM machine m
    JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
    WHERE m."machineType" = 'Chiller'
    ORDER BY m."machineId", me."Id";
"""


def get_connection():
    return psycopg2.connect(
        host=os.environ["APPDB_HOST"],
        port=os.environ["APPDB_PORT"],
        dbname=os.environ["APPDB_NAME"],
        user=os.environ["APPDB_USER"],
        password=os.environ["APPDB_PASSWORD"],
    )


def main():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(QUERY)
            rows = cur.fetchall()
    finally:
        conn.close()

    print(f"Total chiller sensor rows: {len(rows)}")

    distinct_machines = {row["machineId"] for row in rows}
    print(f"Distinct chiller assets: {len(distinct_machines)}")

    print("\nSample rows:")
    for row in rows[:10]:
        print(dict(row))


if __name__ == "__main__":
    main()
