"""
Read-only exploration script: pulls trendseriesmeterdata for every chiller
sensor over a given date range, pivots it into a wide per-chiller time series,
and writes data/trend_wide.csv.

Processes the range in daily chunks (pull -> pivot -> append to CSV) so peak
memory stays bounded regardless of total row count, instead of holding the
whole range in memory at once.

Does not write to the database (SELECT only). Connection details are loaded
from environment variables (.env), never hardcoded.

TIMEZONE NOTE: trendseriesmeterdata."timestamp" is timestamptz. Reading it
back via pd.read_sql_query over a raw (non-SQLAlchemy) psycopg2 connection
silently converts tz-aware values to UTC in the resulting DataFrame — this
was caught in monthly_coverage.py, where it shifted date_trunc('month', ...)
labels back by one month whenever midnight IST fell on the 1st of the month.
Same root cause applies here: it would relabel the "timestamp" column (and
any date-boundary math done on it) into UTC wall-clock instead of the DB
session's local time (Asia/Calcutta). Fixed by converting to local time in
SQL via `AT TIME ZONE` before the value ever reaches pandas, so the column
comes back as a plain naive local timestamp — never tz-aware in Python.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psutil
import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from list_chillers import QUERY as CHILLER_QUERY
from list_chillers import get_connection as get_appdb_connection

SENSOR_ID_CHUNK_SIZE = 200
READINGS_PER_DAY = 96  # 15-minute intervals
DB_TIMEZONE = "Asia/Calcutta"  # matches the Timescale session's `SHOW timezone`

TREND_QUERY = """
    SELECT ("timestamp" AT TIME ZONE %(tz)s) AS "timestamp", machineexplorerid, value
    FROM trendseriesmeterdata
    WHERE machineexplorerid IN %(sensor_ids)s
      AND "timestamp" >= %(start)s
      AND "timestamp" < %(end)s
"""


def get_timescale_connection():
    return psycopg2.connect(
        host=os.environ["TIMESCALE_HOST"],
        port=os.environ["TIMESCALE_PORT"],
        dbname=os.environ["TIMESCALE_NAME"],
        user=os.environ["TIMESCALE_USER"],
        password=os.environ["TIMESCALE_PASSWORD"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Pull chiller trend data into a wide CSV.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=1,
        help="Days per internal pull/pivot/write chunk (default: 1, i.e. daily).",
    )
    return parser.parse_args()


def load_chiller_list():
    conn = get_appdb_connection()
    try:
        chiller_df = pd.read_sql_query(CHILLER_QUERY, conn)
    finally:
        conn.close()
    return chiller_df


def pull_trend_rows(sensor_ids, start, end):
    conn = get_timescale_connection()
    try:
        chunks = []
        for i in range(0, len(sensor_ids), SENSOR_ID_CHUNK_SIZE):
            chunk_ids = tuple(sensor_ids[i : i + SENSOR_ID_CHUNK_SIZE])
            params = {"sensor_ids": chunk_ids, "start": start, "end": end, "tz": DB_TIMEZONE}
            chunk_df = pd.read_sql_query(TREND_QUERY, conn, params=params)
            chunks.append(chunk_df)
    finally:
        conn.close()
    if not chunks:
        return pd.DataFrame(columns=["timestamp", "machineexplorerid", "value"])
    return pd.concat(chunks, ignore_index=True)


def pivot_chunk(raw_df, chiller_df, value_columns, chiller_meta):
    """
    Round -> merge -> pivot -> reindex to a fixed column set -> merge metadata
    -> downcast, for one raw (timestamp, machineexplorerid, value) chunk.

    value_columns is the full set of SeriesDescription values across ALL
    chillers (not just this chunk), so every chunk's output has identical
    columns in identical order — required for safely appending chunks to the
    same CSV.
    """
    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"]).dt.round("15min")

    raw_df = raw_df.merge(
        chiller_df[["MachineExplorerId", "machineId", "SeriesDescription"]],
        left_on="machineexplorerid",
        right_on="MachineExplorerId",
        how="left",
    )

    wide = raw_df.pivot_table(
        index=["machineId", "timestamp"],
        columns="SeriesDescription",
        values="value",
        aggfunc="first",
    )
    wide = wide.reindex(columns=value_columns).reset_index()
    wide.columns.name = None

    wide = wide.merge(chiller_meta, on="machineId", how="left")

    wide["machineId"] = wide["machineId"].astype("int32")
    for col in value_columns:
        wide[col] = wide[col].astype(np.float32)

    return wide[["machineId", "timestamp"] + value_columns + ["status", "Criticality"]]


def daterange_chunks(start_date, end_date_exclusive, chunk_days):
    cur = start_date
    while cur < end_date_exclusive:
        nxt = min(cur + timedelta(days=chunk_days), end_date_exclusive)
        yield cur, nxt
        cur = nxt


def build_wide_trend_df(start_date_str, end_date_str, chunk_days=7):
    """
    Pull + pivot chiller trend data for [start_date_str, end_date_str]
    (inclusive, YYYY-MM-DD) entirely in memory, chunked internally by
    chunk_days. Meant for small/exploratory windows (e.g. a 7-day comparison)
    where holding the result in memory is fine. For a full-month-or-larger
    pull, use main()/stream_pull_to_csv instead, which never holds more than
    one chunk in memory.

    Returns (wide_df, date_range_days). wide_df is empty if no data was
    returned for the window.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date_exclusive = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
    date_range_days = (end_date_exclusive - start_date).days

    chiller_df = load_chiller_list()
    chiller_df = chiller_df.dropna(subset=["SeriesDescription"])
    sensor_ids = chiller_df["MachineExplorerId"].unique().tolist()
    value_columns = sorted(chiller_df["SeriesDescription"].unique())
    chiller_meta = chiller_df[["machineId", "status", "Criticality"]].drop_duplicates("machineId")

    chunks = []
    for chunk_start, chunk_end in daterange_chunks(start_date, end_date_exclusive, chunk_days):
        raw_df = pull_trend_rows(sensor_ids, chunk_start, chunk_end)
        if raw_df.empty:
            continue
        chunks.append(pivot_chunk(raw_df, chiller_df, value_columns, chiller_meta))

    if not chunks:
        return pd.DataFrame(), date_range_days

    return pd.concat(chunks, ignore_index=True), date_range_days


def stream_pull_to_csv(start_date_str, end_date_str, out_path, chunk_days=1):
    """
    Pull + pivot chiller trend data day-by-day (or chunk_days at a time),
    appending each chunk to out_path as it's produced. Never holds more than
    one chunk's worth of rows in memory. Prints progress per chunk.
    """
    process = psutil.Process(os.getpid())
    peak_rss_mb = process.memory_info().rss / 1e6
    t0 = time.perf_counter()

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date_exclusive = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
    date_range_days = (end_date_exclusive - start_date).days

    chiller_df = load_chiller_list()
    chiller_df = chiller_df.dropna(subset=["SeriesDescription"])
    sensor_ids = chiller_df["MachineExplorerId"].unique().tolist()
    value_columns = sorted(chiller_df["SeriesDescription"].unique())
    chiller_meta = chiller_df[["machineId", "status", "Criticality"]].drop_duplicates("machineId")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)  # fresh pull, don't append onto a stale prior run

    total_rows = 0
    machine_ids_seen = set()
    min_ts, max_ts = None, None
    header_written = False

    chunks = list(daterange_chunks(start_date, end_date_exclusive, chunk_days))
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        raw_df = pull_trend_rows(sensor_ids, chunk_start, chunk_end)
        chunk_rows = 0
        if not raw_df.empty:
            wide_chunk = pivot_chunk(raw_df, chiller_df, value_columns, chiller_meta)
            wide_chunk.to_csv(out_path, mode="a", header=not header_written, index=False)
            header_written = True

            chunk_rows = len(wide_chunk)
            total_rows += chunk_rows
            machine_ids_seen.update(wide_chunk["machineId"].unique().tolist())
            chunk_min, chunk_max = wide_chunk["timestamp"].min(), wide_chunk["timestamp"].max()
            min_ts = chunk_min if min_ts is None else min(min_ts, chunk_min)
            max_ts = chunk_max if max_ts is None else max(max_ts, chunk_max)

        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / 1e6)
        print(
            f"[{i}/{len(chunks)}] {chunk_start.date()} -> {chunk_end.date()}: "
            f"{chunk_rows} rows this chunk, {total_rows} running total"
        )

    elapsed = time.perf_counter() - t0
    n_chillers = len(machine_ids_seen)
    theoretical_max = n_chillers * date_range_days * READINGS_PER_DAY

    print(f"\nSaved {out_path}")
    print(f"Total rows: {total_rows}")
    print(f"Distinct chillers covered: {n_chillers}")
    print(f"Date range actually returned: {min_ts} to {max_ts}")
    print(f"Theoretical max (n_chillers * days * 96): {theoretical_max}")
    if theoretical_max:
        print(f"Actual rows as fraction of theoretical max: {total_rows / theoretical_max:.2%}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Peak process RSS memory: {peak_rss_mb:.1f} MB")


def main():
    args = parse_args()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "trend_wide.csv")
    stream_pull_to_csv(args.start_date, args.end_date, out_path, chunk_days=args.chunk_days)


if __name__ == "__main__":
    main()
