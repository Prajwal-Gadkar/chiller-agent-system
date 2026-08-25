"""
sync_knowledge_base.py

Chiller onboarding and live knowledge synchronization script.
Executes whenever a client database/telemetry dataset is processed.
Populates and updates chiller_agent_knowledge.json via knowledge_store.py functions:
  - Registers chillers (with deduplicated canonical assets and aliases)
  - Records parameter observations (meanings, units, running min/max)
  - Updates cluster membership and Layer-2 statistical bounds
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import knowledge_store

KNOWN_PARAMETERS = {
    "inlet_temperature": ("Return chilled water temperature entering evaporator", "degC"),
    "inlet_temperature ValueY": ("Return chilled water temperature entering evaporator", "degC"),
    "Outlet_temperature": ("Supply chilled water temperature leaving evaporator", "degC"),
    "Outlet_temperature ValueY": ("Supply chilled water temperature leaving evaporator", "degC"),
    "Evaporator_Inlet_Temp": ("Evaporator chilled water entering temperature", "degC"),
    "Evaporator_Outlet_Temp": ("Evaporator chilled water leaving temperature", "degC"),
    "Condenser_Inlet_Temp": ("Condenser water supply temperature", "degC"),
    "Condenser_Outlet_Temp": ("Condenser water return temperature", "degC"),
    "KW": ("Compressor active electrical power input", "kW"),
    "KW ValueY": ("Compressor active electrical power input", "kW"),
    "Running_KW_Active_Power ValueY": ("Compressor running active power draw", "kW"),
    "Flow": ("Chilled water flow rate through evaporator", "m3/h"),
    "Flow ValueY": ("Chilled water flow rate through evaporator", "m3/h"),
    "Compressor_1_Load": ("Percentage load demand for compressor circuit 1", "%"),
    "Compressor_1_Load ValueY": ("Percentage load demand for compressor circuit 1", "%"),
    "Compressor_2_Load": ("Percentage load demand for compressor circuit 2", "%"),
    "Compressor_2_Load ValueY": ("Percentage load demand for compressor circuit 2", "%"),
    "Compressor_1_Fan_speed": ("Fan speed percentage for circuit 1 condenser fans", "%"),
    "Circuit_1_Fan_Speed ValueY": ("Fan speed percentage for circuit 1 condenser fans", "%"),
    "Compressor_2_Fan_speed": ("Fan speed percentage for circuit 2 condenser fans", "%"),
    "Circuit_2_Fan_Speed ValueY": ("Fan speed percentage for circuit 2 condenser fans", "%"),
    "Compressor_1_Discharge_press": ("Refrigerant discharge pressure for compressor circuit 1", "bar"),
    "Compressor_1_Suction_press": ("Refrigerant suction pressure for compressor circuit 1", "bar"),
    "Compressor_2_Discharge_press": ("Refrigerant discharge pressure for compressor circuit 2", "bar"),
    "Compressor_2_Suction_press": ("Refrigerant suction pressure for compressor circuit 2", "bar"),
    "Common_Alarm_Status": ("Direct alarm/fault status indicator from controller", "Boolean/Code"),
    "Fault_Status": ("Direct fault status indicator from controller", "Boolean/Code"),
    "On_Off_Status ValueY": ("Unit operating mode and run command status", "Boolean/Code"),
    "Auto_Manual_Status": ("Control mode selection status", "Boolean/Code"),
    "Energy_Consumption ValueY": ("Cumulative active energy consumption", "kWh"),
    "CKT_1_RUN_Hours ValueY": ("Cumulative operating hours for compressor circuit 1", "hours"),
    "CKT_2_RUN_Hours ValueY": ("Cumulative operating hours for compressor circuit 2", "hours"),
    "Compressor1_RunHours": ("Cumulative operating hours for compressor circuit 1", "hours"),
    "Compressor2_RunHours": ("Cumulative operating hours for compressor circuit 2", "hours"),
    "Ambient_Temperature ValueY": ("Ambient air temperature", "degC"),
    "Outlet_Setpoint_Celsius": ("Leaving chilled water temperature target setpoint", "degC"),
    "Current_Temp_SP ValueY": ("Leaving chilled water temperature target setpoint", "degC"),
    "HP_Value_Circuit#_1 ValueY": ("High pressure reading for circuit 1", "psi"),
    "HP_Value_Circuit#_2 ValueY": ("High pressure reading for circuit 2", "psi"),
    "LP_Value_Circuit#_1 ValueY": ("Low pressure reading for circuit 1", "psi"),
    "LP_Value_Circuit#_2 ValueY": ("Low pressure reading for circuit 2", "psi"),
}

METADATA_COLUMNS = {"machineId", "timestamp", "status", "Criticality"}


def load_alias_mapping():
    alias_csv = os.path.join(PROJECT_ROOT, "data", "asset_aliases.csv")
    alias_map = {}
    if os.path.exists(alias_csv):
        df_alias = pd.read_csv(alias_csv)
        for _, row in df_alias.iterrows():
            alias_map[str(row["alias_id"])] = str(row["canonical_id"])
    return alias_map


def sync_knowledge_base():
    print("=" * 80)
    print("STARTING CHILLER ONBOARDING & KNOWLEDGE BASE SYNCHRONIZATION")
    print("=" * 80)

    trend_csv = os.path.join(PROJECT_ROOT, "data", "trend_wide.csv")
    types_csv = os.path.join(PROJECT_ROOT, "data", "chiller_types.csv")

    if not os.path.exists(trend_csv):
        print(f"Error: {trend_csv} not found.")
        return

    df_trend = pd.read_csv(trend_csv)
    alias_map = load_alias_mapping()

    chiller_types_df = pd.read_csv(types_csv) if os.path.exists(types_csv) else pd.DataFrame()
    cluster_map = {}
    if not chiller_types_df.empty:
        for _, row in chiller_types_df.iterrows():
            cluster_map[str(row["machineId"])] = row["chiller_type"]

    # Load store data once for high performance batch processing
    kb_data = knowledge_store.load()
    inventory = kb_data["chiller_inventory"]["entries"]

    # 1. Register Chillers
    all_machines = sorted(df_trend["machineId"].unique())
    print(f"\n1. Registering {len(all_machines)} chillers in live inventory...")

    canonical_111 = "3392"  # canonical asset ID for Chiller-111
    registered_mids = set()

    for mid in all_machines:
        smid = str(mid)
        if smid in ["3392", "3894", "4054"]:
            if canonical_111 in registered_mids:
                continue
            smid = canonical_111
            name = "Chiller-111"
            aliases = ["3392", "3894", "4054"]
            evidence = "Asset deduplication rule: 3392, 3894, 4054 share identical telemetry (Chiller-111)"
        else:
            name = f"Chiller-{smid}"
            aliases = None
            evidence = "Telemetry stream observation"

        chiller_type = cluster_map.get(smid, "unclustered_outlier")
        crit_series = df_trend[df_trend["machineId"] == mid]["Criticality"].dropna() if "Criticality" in df_trend.columns else pd.Series()
        criticality_val = crit_series.iloc[0] if not crit_series.empty else "Medium"

        rec = inventory.get(smid, {
            "name": None, "model_number": None, "manufacturer": None,
            "location": None, "capacity": None, "unit": None, "criticality": None,
            "chiller_type": None, "industry": None, "confidence": None, "evidence": None,
            "aliases": [], "parameters_tracked": {},
        })
        rec.update({
            "name": name,
            "criticality": str(criticality_val),
            "chiller_type": chiller_type,
            "industry": "HVAC / Commercial Real Estate",
            "confidence": 0.852 if chiller_type != "unclustered_outlier" else 0.5,
            "evidence": evidence,
            "aliases": aliases or rec.get("aliases", []),
            "updated": knowledge_store._now()
        })
        inventory[smid] = rec
        registered_mids.add(smid)

    print(f"  Registered {len(registered_mids)} canonical physical chiller entries.")

    # 2. Record Parameter Observations
    sensor_cols = [c for c in df_trend.columns if c not in METADATA_COLUMNS]
    print(f"\n2. Syncing parameter observations for {len(sensor_cols)} parameters...")

    for mid in registered_mids:
        matching_mids = [int(mid)]
        if mid == canonical_111:
            matching_mids = [3392, 3894, 4054]
        
        chiller_data = df_trend[df_trend["machineId"].isin(matching_mids)]
        params_tracked = inventory[mid].setdefault("parameters_tracked", {})

        for col in sensor_cols:
            series = chiller_data[col].dropna()
            if series.empty:
                continue

            v_min = float(series.min())
            v_max = float(series.max())
            reading_cnt = len(series)

            if col in KNOWN_PARAMETERS:
                meaning, unit = KNOWN_PARAMETERS[col]
                sanity_status = "ok"
            else:
                meaning, unit = None, None
                sanity_status = "review_needed"
                
                # Check if unknown pattern is already logged
                lp_list = kb_data.setdefault("learned_patterns", [])
                if not any(p.get("parameter") == col and p.get("machine_id") == mid for p in lp_list):
                    lp_list.append({
                        "id": f"lp-{knowledge_store.uuid.uuid4().hex[:8]}",
                        "discovered": knowledge_store._now(),
                        "pattern": "unknown_parameter",
                        "machine_id": str(mid),
                        "parameter": col,
                        "evidence": f"Observed min={v_min}, max={v_max}, count={reading_cnt}",
                        "description": f"Unrecognized sensor column '{col}' observed for chiller {mid}",
                        "rule": "Flag for engineering review to assign standard parameter taxonomy meaning and unit.",
                        "status": "active",
                        "promoted_on": None,
                    })

            p_entry = params_tracked.get(col, {
                "meaning": None, "unit": None,
                "observed_min": None, "observed_max": None,
                "reading_count": 0, "sanity_status": "unknown",
            })
            if meaning: p_entry["meaning"] = meaning
            if unit: p_entry["unit"] = unit
            p_entry["sanity_status"] = sanity_status
            p_entry["observed_min"] = v_min if p_entry["observed_min"] is None else min(p_entry["observed_min"], v_min)
            p_entry["observed_max"] = v_max if p_entry["observed_max"] is None else max(p_entry["observed_max"], v_max)
            p_entry["reading_count"] += reading_cnt

            params_tracked[col] = p_entry

    # 3. Update Cluster Registry with Layer-2 Statistical Bounds
    print("\n3. Updating Cluster Registry with Layer-2 Statistical Bounds...")
    cluster_registry = kb_data.setdefault("cluster_registry", {}).setdefault("entries", {})

    if not chiller_types_df.empty:
        for ctype, group in chiller_types_df.groupby("chiller_type"):
            member_mids = [str(m) for m in group["machineId"].tolist()]
            mapped_members = []
            for m in member_mids:
                if m in ["3392", "3894", "4054"]:
                    if canonical_111 not in mapped_members:
                        mapped_members.append(canonical_111)
                else:
                    mapped_members.append(m)

            raw_mids = group["machineId"].tolist()
            cluster_trend = df_trend[df_trend["machineId"].isin(raw_mids)]

            stat_bounds = {}
            for col in sensor_cols:
                vals = cluster_trend[col].dropna()
                if not vals.empty:
                    q01 = float(np.percentile(vals, 1))
                    q99 = float(np.percentile(vals, 99))
                    stat_bounds[col] = {"p1": round(q01, 3), "p99": round(q99, 3), "count": len(vals)}

            cluster_registry[str(ctype)] = {
                "members": mapped_members,
                "stat_bounds": stat_bounds,
                "updated": knowledge_store._now(),
            }
            print(f"  Updated cluster '{ctype}' ({len(mapped_members)} members, {len(stat_bounds)} parameter bounds)")

    # Atomic save of the entire updated knowledge store
    knowledge_store.save(kb_data)

    print("\n" + "=" * 80)
    print("KNOWLEDGE BASE SYNCHRONIZATION COMPLETE & SAVED TO DISK")
    print("=" * 80)


if __name__ == "__main__":
    sync_knowledge_base()
