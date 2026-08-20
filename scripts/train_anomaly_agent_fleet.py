"""
Train Anomaly Agent Across Fresh Clean Fleet (Capped Recent 110-Day Window: 2026-05-01 to 2026-08-19)

Strictly respects the 2026-01-01 regime boundary by filtering data from 2026-05-01 onward
(most recent ~110 days leading to Aug 19, 2026 reset).

Pulls sensor data per clean chiller from TimescaleDB, fits RandomForest physical response models,
evaluates 5-fold cross-validation, computes residual z-score anomalies (|z| > 3.0),
saves 47 model payloads to data/anomaly_models/, and outputs fit summary to data/anomaly_agent_fit_summary.csv.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from agents.anomaly_agent import AnomalyAgent, MODEL_SAVE_DIR
from agents.data_validation import validate

warnings.filterwarnings("ignore")
load_dotenv()

DB_TIMEZONE = "Asia/Calcutta"
TRAIN_WINDOW_START = "2026-05-01"  # ~110-day recent window (strictly >= 2026-01-01 regime)


def get_appdb_connection():
    return psycopg2.connect(
        host=os.environ["APPDB_HOST"],
        port=os.environ["APPDB_PORT"],
        dbname=os.environ["APPDB_NAME"],
        user=os.environ["APPDB_USER"],
        password=os.environ["APPDB_PASSWORD"],
    )


def get_timescale_connection():
    return psycopg2.connect(
        host=os.environ["TIMESCALE_HOST"],
        port=os.environ["TIMESCALE_PORT"],
        dbname=os.environ["TIMESCALE_NAME"],
        user=os.environ["TIMESCALE_USER"],
        password=os.environ["TIMESCALE_PASSWORD"],
    )


def train_fleet_anomaly_agent(clean_list_path=os.path.join(PROJECT_ROOT, "data", "clean_chillers_aug20.csv")):
    print("=" * 100, flush=True)
    print(f"TRAINING ANOMALY AGENT ACROSS CLEAN CHILLER FLEET ({TRAIN_WINDOW_START} to 2026-08-19 ~110-DAY WINDOW)", flush=True)
    print("Regime Boundary Enforced: >= 2026-01-01 (Strictly post-shift)", flush=True)
    print("=" * 100, flush=True)

    if not os.path.exists(clean_list_path):
        print(f"Error: {clean_list_path} not found.", flush=True)
        sys.exit(1)

    clean_df = pd.read_csv(clean_list_path)
    clean_units = clean_df[clean_df["status"] == "CLEAN & VALID"]
    clean_m_ids = sorted(clean_units["machineId"].unique().tolist())

    print(f"Loaded {len(clean_m_ids)} clean candidate chillers from {clean_list_path}.", flush=True)

    # Fetch sensor metadata
    conn_app = get_appdb_connection()
    query_meta = """
        SELECT m."machineId", me."Id" AS "MachineExplorerId", me."SeriesDescription"
        FROM machine m
        JOIN "MachineExplorer" me ON m."machineId" = me."MachineId"
        WHERE m."machineId" IN %(m_ids)s;
    """
    meta_df = pd.read_sql_query(query_meta, conn_app, params={"m_ids": tuple(clean_m_ids)})
    conn_app.close()

    conn_ts = get_timescale_connection()

    summary_records = []
    total_raw_rows_pulled = 0
    total_samples_trained = 0
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for m_id in clean_m_ids:
        m_sensors = meta_df[meta_df["machineId"] == m_id].dropna(subset=["SeriesDescription"])
        sensor_ids = tuple(m_sensors["MachineExplorerId"].unique().tolist())
        if not sensor_ids:
            continue

        query_ts = """
            SELECT ("timestamp" AT TIME ZONE %(tz)s) AS "timestamp", machineexplorerid, value
            FROM trendseriesmeterdata
            WHERE machineexplorerid IN %(sensor_ids)s
              AND "timestamp" >= %(start_date)s
        """
        params = {"sensor_ids": sensor_ids, "tz": DB_TIMEZONE, "start_date": TRAIN_WINDOW_START}
        try:
            raw_df = pd.read_sql_query(query_ts, conn_ts, params=params)
        except Exception as e:
            print(f"Chiller {m_id:4d} | DB Query Error: {e}", flush=True)
            continue

        total_raw_rows_pulled += len(raw_df)

        if raw_df.empty or len(raw_df) < 50:
            print(f"Chiller {m_id:4d} | Skipped: Insufficient data rows ({len(raw_df)})", flush=True)
            continue

        raw_df = raw_df.merge(m_sensors[["MachineExplorerId", "SeriesDescription"]], left_on="machineexplorerid", right_on="MachineExplorerId")
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"]).dt.round("15min")

        wide_df = raw_df.pivot_table(
            index="timestamp",
            columns="SeriesDescription",
            values="value",
            aggfunc="first"
        ).reset_index()

        agent = AnomalyAgent(machine_id=m_id)
        try:
            cv_metrics = agent.fit(wide_df, regime_start=TRAIN_WINDOW_START)
            save_path = agent.save(save_dir=MODEL_SAVE_DIR)

            # Detect anomalies
            detected_df = agent.detect_anomalies(wide_df)
            n_anom = int(detected_df["is_anomalous"].sum())
            pct_anom = (n_anom / len(detected_df)) * 100.0 if len(detected_df) > 0 else 0.0

            total_samples_trained += cv_metrics["n_samples"]

            summary_records.append({
                "machineId": m_id,
                "R2_CV": cv_metrics["R2"],
                "RMSE_CV": cv_metrics["RMSE"],
                "MAPE_CV": cv_metrics["MAPE"],
                "n_samples": cv_metrics["n_samples"],
                "res_mean": agent.residual_mean,
                "res_std": agent.residual_std,
                "n_anom_flagged": n_anom,
                "pct_anom_flagged": pct_anom,
                "model_path": save_path
            })
            print(f"Chiller {m_id:4d} | R2 CV: {cv_metrics['R2']:.4f} | RMSE: {cv_metrics['RMSE']:5.2f} kW | Samples: {cv_metrics['n_samples']:5d} | Anomalies: {n_anom:3d} ({pct_anom:4.1f}%) | Saved", flush=True)
        except Exception as e:
            print(f"Chiller {m_id:4d} | Fit Failed: {e}", flush=True)

    conn_ts.close()

    summary_df = pd.DataFrame(summary_records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values("R2_CV", ascending=False).reset_index(drop=True)
        out_summary_path = os.path.join(PROJECT_ROOT, "data", "anomaly_agent_fit_summary.csv")
        summary_df.to_csv(out_summary_path, index=False)

        print("\n" + "=" * 100, flush=True)
        print(f"FLEET ANOMALY AGENT TRAINING COMPLETE — {len(summary_df)} MODELS FITTED & SAVED", flush=True)
        print("=" * 100, flush=True)
        print(f"Total Raw Sensor Data Rows Pulled: {total_raw_rows_pulled:,}", flush=True)
        print(f"Total Clean Running 15-min Training Samples: {total_samples_trained:,}", flush=True)
        print(f"Mean R2 CV across Clean Fleet: {summary_df['R2_CV'].mean():.4f}", flush=True)
        print(f"Median R2 CV across Clean Fleet: {summary_df['R2_CV'].median():.4f}", flush=True)
        print(f"Models saved to: {MODEL_SAVE_DIR}", flush=True)
        print(f"Fit summary written to: {out_summary_path}", flush=True)

        print("\nAll 47 Fitted Models Summary:", flush=True)
        print(summary_df[["machineId", "R2_CV", "RMSE_CV", "MAPE_CV", "n_samples", "n_anom_flagged", "pct_anom_flagged"]].to_string(index=False), flush=True)

    return summary_df


if __name__ == "__main__":
    train_fleet_anomaly_agent()
