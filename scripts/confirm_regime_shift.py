"""
Confirm Fleet-Wide Regime Shift (Pre-2026 vs Post-2026-01-01)
Queries Power (KW) readings for multiple candidate chillers across the fleet to
determine if the ~10x power scale drop at 2026-01-01 is fleet-wide.
"""

import os
import sys
import dotenv
import psycopg2
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

dotenv.load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

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

# Candidate chillers to check
CANDIDATE_CHILLERS = [1657, 1660, 1661, 1658, 1659, 2762, 2825, 2824]


def get_appdb_conn():
    return psycopg2.connect(host=APPDB_HOST, port=APPDB_PORT, dbname=APPDB_NAME, user=APPDB_USER, password=APPDB_PASSWORD)


def get_timescale_conn():
    return psycopg2.connect(host=TIMESCALE_HOST, port=TIMESCALE_PORT, dbname=TIMESCALE_NAME, user=TIMESCALE_USER, password=TIMESCALE_PASSWORD)


def get_power_sensor_info():
    conn = get_appdb_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT m."machineId", me."Id" AS "MachineExplorerId", me."SeriesDescription"
            FROM machine m
            JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
            WHERE m."machineType" = 'Chiller'
              AND (me."SeriesDescription" ILIKE '%KW%' OR me."SeriesDescription" ILIKE '%Power%');
        """)
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=["machineId", "MachineExplorerId", "SeriesDescription"])
    finally:
        conn.close()


def analyze_chiller_power_regimes():
    power_sensors = get_power_sensor_info()
    ts_conn = get_timescale_conn()

    results = []

    try:
        cur = ts_conn.cursor()
        for m_id in CANDIDATE_CHILLERS:
            m_sensors = power_sensors[power_sensors["machineId"] == m_id]
            if m_sensors.empty:
                continue

            sensor_ids = tuple(int(x) for x in m_sensors["MachineExplorerId"].unique())

            # Query pre-2026 Power
            cur.execute("""
                SELECT COUNT(*), AVG(value), MAX(value)
                FROM trendseriesmeterdata
                WHERE machineexplorerid IN %s
                  AND "timestamp" < '2026-01-01 00:00:00+05:30'
                  AND value > 5.0;
            """, (sensor_ids,))
            pre_count, pre_avg, pre_max = cur.fetchone()

            # Query post-2026-01-01 Power
            cur.execute("""
                SELECT COUNT(*), AVG(value), MAX(value)
                FROM trendseriesmeterdata
                WHERE machineexplorerid IN %s
                  AND "timestamp" >= '2026-01-01 00:00:00+05:30'
                  AND value > 5.0;
            """, (sensor_ids,))
            post_count, post_avg, post_max = cur.fetchone()

            pre_avg_str = f"{pre_avg:.1f} KW" if pre_avg is not None else "N/A"
            pre_max_str = f"{pre_max:.1f} KW" if pre_max is not None else "N/A"
            post_avg_str = f"{post_avg:.1f} KW" if post_avg is not None else "N/A"
            post_max_str = f"{post_max:.1f} KW" if post_max is not None else "N/A"

            scale_drop = "N/A"
            if pre_avg and post_avg and post_avg > 0:
                ratio = pre_avg / post_avg
                scale_drop = f"{ratio:.1f}x drop"

            results.append({
                "machineId": m_id,
                "Pre-2026 Running Rows": pre_count,
                "Pre-2026 Mean Power": pre_avg_str,
                "Pre-2026 Max Power": pre_max_str,
                "Post-2026 Running Rows": post_count,
                "Post-2026 Mean Power": post_avg_str,
                "Post-2026 Max Power": post_max_str,
                "Power Scale Shift": scale_drop
            })
    finally:
        ts_conn.close()

    res_df = pd.DataFrame(results)
    print("\n" + "="*110)
    print("FLEET-WIDE POWER REGIME SHIFT CONFIRMATION TABLE (PRE-2026 vs POST-2026-01-01)")
    print("="*110)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(res_df.to_string(index=False))

    return res_df


if __name__ == "__main__":
    analyze_chiller_power_regimes()
