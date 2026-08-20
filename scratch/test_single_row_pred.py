import os
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from agents.anomaly_agent import AnomalyAgent, MODEL_SAVE_DIR

app_conn = psycopg2.connect(
    host=os.environ["APPDB_HOST"],
    port=os.environ["APPDB_PORT"],
    dbname=os.environ["APPDB_NAME"],
    user=os.environ["APPDB_USER"],
    password=os.environ["APPDB_PASSWORD"]
)

ts_conn = psycopg2.connect(
    host=os.environ["TIMESCALE_HOST"],
    port=os.environ["TIMESCALE_PORT"],
    dbname=os.environ["TIMESCALE_NAME"],
    user=os.environ["TIMESCALE_USER"],
    password=os.environ["TIMESCALE_PASSWORD"]
)

def find_sensor_col_single_row(df: pd.DataFrame, sensor_type: str):
    exact_candidates = {
        "flow": ["Flow ValueY", "CHW FLOW RATE (m3/h)", "Flow", "CONDENSER FLOW (m3/h)"],
        "kw": ["KW ValueY", "CH-1, POWER CONSUMPTION (KW)", "KW", "power", "COMMITED KW"],
        "inlet": ["inlet_temperature ValueY", "Evaporator_Inlet_Temp", "inlet_temperature", "CHW RETURN TEMPERATURE (DEG C)"],
        "outlet": ["Outlet_temperature ValueY", "Evaporator_Outlet_Temp", "Outlet_temperature", "CHW LEAVE TEMPERATURE (DEG C)"],
        "comp": ["Compressor_1_Load ValueY", "Compressor_1_Load", "CompressorLoad", "CompLoad"]
    }
    candidates = exact_candidates.get(sensor_type, [])
    for col in df.columns:
        if any(c.lower() == col.lower() or c.lower() in col.lower() for c in candidates):
            if df[col].notna().any():
                col_lower = col.lower()
                if sensor_type == "kw" and ("ikw" in col_lower or "kwh" in col_lower):
                    continue
                if sensor_type in ["inlet", "outlet"] and "condenser" in col_lower:
                    continue
                return col
    return None

chiller_ids = [4054, 2828, 2821]

for c_id in chiller_ids:
    agent = AnomalyAgent.load(c_id, save_dir=MODEL_SAVE_DIR)
    
    query_sensors = """
        SELECT me."MachineId" as machine_id, me."Id" as sensor_id, me."SeriesDescription" as series_desc
        FROM "MachineExplorer" me
        JOIN machine m ON m."machineId" = me."MachineId"
        WHERE m."machineId" = %s AND me."SeriesDescription" IS NOT NULL
    """
    sensors_df = pd.read_sql_query(query_sensors, app_conn, params=(c_id,))
    sensor_ids = tuple(sensors_df["sensor_id"].tolist())
    
    # Get recent timestamps for sensor_ids[0]
    query_ts = """
        SELECT ("timestamp" AT TIME ZONE 'Asia/Calcutta') AS timestamp
        FROM trendseriesmeterdata
        WHERE machineexplorerid = %s
        ORDER BY timestamp DESC
        LIMIT 5
    """
    ts_df = pd.read_sql_query(query_ts, ts_conn, params=(sensor_ids[0],))
    timestamps = tuple(ts_df["timestamp"].tolist())
    
    query_readings = """
        SELECT ("timestamp" AT TIME ZONE 'Asia/Calcutta') AS timestamp, machineexplorerid, value
        FROM trendseriesmeterdata
        WHERE machineexplorerid IN %s AND ("timestamp" AT TIME ZONE 'Asia/Calcutta') IN %s
    """
    readings_df = pd.read_sql_query(query_readings, ts_conn, params=(sensor_ids, timestamps))
    merged = readings_df.merge(sensors_df, left_on="machineexplorerid", right_on="sensor_id")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"]).dt.round("15min")
    
    for ts in timestamps[:2]:
        ts_df = merged[merged["timestamp"] == ts]
        raw_reading = ts_df.set_index("series_desc")["value"].to_dict()
        df = pd.DataFrame([raw_reading])
        
        # Map sensor columns
        for stype, col_name in agent.col_map.items():
            if col_name and col_name not in df.columns:
                matched_col = find_sensor_col_single_row(df, stype)
                if matched_col and matched_col in df.columns:
                    df[col_name] = df[matched_col]
        
        # Compute derived features
        inlet = agent.col_map.get("inlet")
        outlet = agent.col_map.get("outlet")
        flow = agent.col_map.get("flow")
        if inlet and outlet and inlet in df.columns and outlet in df.columns:
            df["DeltaT"] = df[inlet] - df[outlet]
        if flow and flow in df.columns and "DeltaT" in df.columns:
            df["Thermal_Load"] = df[flow] * df["DeltaT"]
            
        detected = agent.detect_anomalies(df)
        pred = detected["predicted_KW"].iloc[0]
        actual = detected[agent.col_map["power"]].iloc[0] if agent.col_map["power"] in detected.columns else np.nan
        z = detected["z_score"].iloc[0]
        print(f"Chiller {c_id} at {ts} -> Actual: {actual:.2f}, Predicted: {pred:.2f}, Z: {z:.2f}")

app_conn.close()
ts_conn.close()
