"""
Chiller Multi-Agent System — Interactive Dashboard
Run with: streamlit run app.py
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add repo root to path for agent imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from agents.data_validation import validate

# Streamlit Page Config
st.set_page_config(
    page_title="Chiller Multi-Agent System Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Aesthetics)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #A0AEC0;
        font-size: 0.95rem;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2D3748;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1A202C;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #A0AEC0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3182CE !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_wide_data():
    csv_path = os.path.join(REPO_ROOT, "data", "trend_wide.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def load_chiller_types():
    csv_path = os.path.join(REPO_ROOT, "data", "chiller_types.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


@st.cache_data
def run_validation_cached(df):
    flagged_df, report_df = validate(df)
    return flagged_df, report_df


# Header Section
st.title("❄️ Chiller Multi-Agent System — Overview Dashboard")
st.markdown("Interactive exploration of dataset analytics, instrumentation clustering, data quality validation, and multi-agent architecture.")

# Load Data
df_wide = load_wide_data()
df_types = load_chiller_types()

if df_wide is None or df_types is None:
    st.error("Data files (`data/trend_wide.csv` or `data/chiller_types.csv`) not found. Please run data extraction scripts first.")
    st.stop()

# Key Performance Metrics Top Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df_wide['machineId'].nunique()}</div>
            <div class="metric-label">Active Chillers (June '26)</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df_wide):,}</div>
            <div class="metric-label">Total Time-Series Rows</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    sensor_cols = [c for c in df_wide.columns if c not in {"machineId", "timestamp", "status", "Criticality"}]
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(sensor_cols)}</div>
            <div class="metric-label">Monitored Sensor Columns</div>
        </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df_types['chiller_type'].nunique()}</div>
            <div class="metric-label">Instrumentation Clusters</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Dashboard Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Fleet & Timeline Overview",
    "🧩 Chiller Types Clustering",
    "🛡️ Data Validation Agent",
    "📈 Physics & Sensor Inspector",
    "🤖 Multi-Agent Architecture"
])

# ---------------------------------------------------------
# TAB 1: FLEET & TIMELINE OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.markdown("<div class='section-header'>Fleet Overview & Instrumentation Staging</div>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.markdown("""
        ### Staged Fleet Instrumentation Timeline
        - **2023-04 to 2024-12**: Only **18 sensors** (1 chiller logging).
        - **2025-01 to 2025-09**: **54 sensors** across fleet.
        - **2025-10 to 2025-12**: **90 sensors**.
        - **2026-01 Onward**: **795–971 sensors** — First window with near-full fleet coverage.
        - **Chiller Data Cutoff**: Data ends at **2026-07-08** (later dates in TimescaleDB belong to non-chiller assets).
        """)
        
        # Synthetic timeline visualization based on documented milestones
        timeline_data = pd.DataFrame([
            {"Month": "2023-04", "Sensors": 18, "Active Chillers": 1},
            {"Month": "2024-01", "Sensors": 18, "Active Chillers": 1},
            {"Month": "2024-12", "Sensors": 18, "Active Chillers": 1},
            {"Month": "2025-01", "Sensors": 54, "Active Chillers": 15},
            {"Month": "2025-10", "Sensors": 90, "Active Chillers": 25},
            {"Month": "2026-01", "Sensors": 795, "Active Chillers": 57},
            {"Month": "2026-06", "Sensors": 971, "Active Chillers": 57},
        ])
        
        fig_timeline = px.line(
            timeline_data, x="Month", y="Sensors", markers=True,
            title="Fleet Sensor Expansion Timeline",
            color_discrete_sequence=["#00F2FE"]
        )
        fig_timeline.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
    with col_right:
        st.markdown("""
        ### Database Architecture Summary
        - **AppDb (`Persistent_AppDb`)**: Asset master (`machine` table, `machineType='Chiller'`) & metadata (`MachineExplorer`).
        - **TimescaleDB (`Persistent_Timescale`)**: Time-series readings (`trendseriesmeterdata` hypertable).
        - **Timezone**: `Asia/Calcutta` (UTC+5:30).
        
        ### Key Data Discovery Takeaways
        > [!IMPORTANT]
        > **No Global Pooling**: Models trained on data before Jan 2026 represent only a handful of chillers. Per-chiller modeling is strictly required ($R^2$ collapses from 0.99 to 0.02 if pooled).
        """)
        
        st.info("Dataset June 2026 (`data/trend_wide.csv`) yields 98.28% data completeness of theoretical maximum!")

# ---------------------------------------------------------
# TAB 2: CHILLER TYPES CLUSTERING
# ---------------------------------------------------------
with tab2:
    st.markdown("<div class='section-header'>Chiller Instrumentation Clustering (`data/chiller_types.csv`)</div>", unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns([1, 1.3])
    
    with col_c1:
        st.markdown("### Cluster Distribution")
        cluster_counts = df_types['chiller_type'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster Type', 'Chiller Count']
        
        fig_clusters = px.bar(
            cluster_counts, x='Cluster Type', y='Chiller Count',
            color='Cluster Type',
            text='Chiller Count',
            title="Chillers per Instrumentation Type (Silhouette Selected K=4)",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_clusters.update_layout(template="plotly_dark", height=340)
        st.plotly_chart(fig_clusters, use_container_width=True)
        
        st.markdown("""
        **Cluster Breakdown**:
        - **`type_2`**: 25 chillers (~20 avg populated columns) — Standard full instrumentation.
        - **`type_3`**: 29 chillers (~9 avg populated columns) — Light instrumentation.
        - **`type_1`**: 1 chiller (machineId 2825, ~38 columns) — Heavy instrumentation outlier.
        - **`type_4`**: 2 chillers (machineIds 2833, 2834, ~2 columns) — Barely instrumented outlier.
        """)

    with col_c2:
        st.markdown("### Interactive Chiller Lookup")
        selected_chiller = st.selectbox("Select Machine ID to Inspect:", sorted(df_types['machineId'].unique()))
        
        chiller_row = df_types[df_types['machineId'] == selected_chiller].iloc[0]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Assigned Type", chiller_row['chiller_type'])
        col_m2.metric("Status", str(chiller_row['status']))
        col_m3.metric("Criticality", str(chiller_row['Criticality']))
        
        # Sensor presence boolean columns
        bool_cols = [c for c in df_types.columns if c not in {"machineId", "chiller_type", "status", "Criticality"}]
        populated_sensors = [c for c in bool_cols if chiller_row[c] == True]
        unpopulated_sensors = [c for c in bool_cols if chiller_row[c] == False]
        
        st.write(f"**Populated Sensors ({len(populated_sensors)} columns):**")
        st.caption(", ".join(populated_sensors) if populated_sensors else "None")
        
        st.write(f"**Missing/Absent Sensors ({len(unpopulated_sensors)} columns):**")
        st.caption(", ".join(unpopulated_sensors[:15]) + ("..." if len(unpopulated_sensors) > 15 else ""))

# ---------------------------------------------------------
# TAB 3: DATA VALIDATION AGENT
# ---------------------------------------------------------
with tab3:
    st.markdown("<div class='section-header'>Data Quality & Validation Agent (`agents/data_validation.py`)</div>", unsafe_allow_html=True)
    
    st.markdown("""
    The **Data Validation Agent** applies physical sanity bounds per column based on column keywords (Temperature, Flow, Pressure, Power, Duty-Cycle Status) without dropping or mutating rows in the database.
    """)
    
    with st.spinner("Running Data Validation Agent on dataset..."):
        flagged_df, report_df = run_validation_cached(df_wide)
        
    col_v1, col_v2 = st.columns([1.2, 1])
    
    with col_v1:
        st.markdown("### Highest Flagged Sensor Columns")
        top_flagged = report_df.head(15)
        
        fig_flags = px.bar(
            top_flagged, x='pct_flagged', y='column', orientation='h',
            color='rule',
            title="Top Flagged Columns (% Readings Out of Plausible Bounds)",
            labels={'pct_flagged': '% Flagged', 'column': 'Sensor Column'}
        )
        fig_flags.update_layout(template="plotly_dark", height=420, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_flags, use_container_width=True)

    with col_v2:
        st.markdown("### Applied Physical Bound Rules")
        st.dataframe(
            report_df[['column', 'rule', 'bound_min', 'bound_max', 'n_flagged', 'pct_flagged']],
            use_container_width=True,
            height=380
        )

# ---------------------------------------------------------
# TAB 4: PHYSICS & SENSOR INSPECTOR
# ---------------------------------------------------------
with tab4:
    st.markdown("<div class='section-header'>Physics & Sensor Inspector (Flow → Power Validation)</div>", unsafe_allow_html=True)
    
    chiller_ids = sorted(df_wide['machineId'].unique())
    selected_inspect_id = st.selectbox("Select Chiller to Plot Physics Trends:", chiller_ids, index=0)
    
    df_chiller = df_wide[df_wide['machineId'] == selected_inspect_id].sort_values("timestamp")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("### Flow vs. Power Correlation ($R^2$ Physics Signal)")
        
        # Identify flow and power columns dynamically
        flow_cols = [c for c in df_chiller.columns if "flow" in c.lower() and not c.endswith("_flagged")]
        power_cols = [c for c in df_chiller.columns if ("kw" in c.lower() or "power" in c.lower()) and not c.endswith("_flagged")]
        
        if flow_cols and power_cols:
            flow_col = flow_cols[0]
            power_col = power_cols[0]
            
            fig_scatter = px.scatter(
                df_chiller, x=flow_col, y=power_col,
                opacity=0.6,
                trendline="ols",
                title=f"Chiller {selected_inspect_id}: {flow_col} vs {power_col}",
                color_discrete_sequence=["#4FACFE"]
            )
            fig_scatter.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Selected chiller does not populate both Flow and Power columns.")

    with col_p2:
        st.markdown("### Temperature & Load Time Series")
        temp_cols = [c for c in df_chiller.columns if "temp" in c.lower() and not c.endswith("_flagged")][:3]
        
        if temp_cols:
            fig_temp = px.line(
                df_chiller, x="timestamp", y=temp_cols,
                title=f"Chiller {selected_inspect_id}: Temperature Trends",
                template="plotly_dark"
            )
            fig_temp.update_layout(height=380)
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("No temperature sensor readings populated for this chiller.")

# ---------------------------------------------------------
# TAB 5: MULTI-AGENT ARCHITECTURE
# ---------------------------------------------------------
with tab5:
    st.markdown("<div class='section-header'>Multi-Agent System Architecture & Roadmap</div>", unsafe_allow_html=True)
    
    st.markdown("""
    The overall system architecture follows a 7-agent pipeline built on LangGraph to provide verified, safety-gated chiller optimization.
    """)
    
    st.markdown("""
    | Step | Agent Name | Status | Function |
    | :--- | :--- | :--- | :--- |
    | **1** | **Data Validation Agent** | ✅ Built | Sanity-bound gate on sensor readings; flags corruption without DB mutations |
    | **2** | **Supervisor** | ⏳ Next | Routes chillers by instrumentation type (`type_1`..`type_4`) & reliability tier |
    | **3** | **Forecast Agent** | ⏳ Next | Flow → Power prediction using 3-tier cascade (Per-Chiller → Type-Level → Rolling Avg) |
    | **4** | **Anomaly Agent** | ⏳ Scheduled | Parallel statistical check on chiller efficiency and operational drift |
    | **5** | **Consensus & Skeptic Gate** | ⏳ Scheduled | Validates Forecast & Anomaly agreement; checks for sensor co-corruption |
    | **6** | **Optimization Agent** | ⏳ Scheduled | Drafts setpoint recommendations with mandatory **Human-in-the-Loop (HITL)** safety approval |
    | **7** | **Insight Agent** | ⏳ Scheduled | Translates mathematical reasoning into plain-language explanations |
    """)

st.markdown("---")
st.caption("Chiller Multi-Agent System | InnoShri YntraM Architecture Layer")
