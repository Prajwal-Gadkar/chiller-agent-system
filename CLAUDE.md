# Chiller Multi-Agent System — Project Context

## What this project is
A multi-agent system (LangGraph) for monitoring and optimizing chiller performance,
built standalone first, for later integration into InnoShri's YntraM production
system (which already has ~90 services covering AHU/HVAC/chiller forecasting, FDD,
alerts, and LLM insights). This project ADDS a reasoning/orchestration layer on top
of that — it does not rebuild those services.

## Data source
PostgreSQL, two databases restored from local backup (`data/backups/`, covering real fleet data from 2023-04-29 to 2026-07-08):
- AppDb: `machine` (asset master — has `machineType`, `status`, `Criticality`
  columns) joined with `MachineExplorer` (sensor metadata: Id, MachineId,
  SeriesDescription).
- Timescale DB: `trendseriesmeterdata` (TimescaleDB hypertable — columns:
  timestamp, machineexplorerid, value, id). Actual sensor readings.

Join key: `machine.machineId = MachineExplorer.MachineId`;
          `MachineExplorer.Id = trendseriesmeterdata.machineexplorerid`

Filter to chillers: `WHERE machine.machineType = 'Chiller'` (~72-73 chiller assets total).

## Verified Fleet Chiller Groups (Data-Driven Candidate Pools)
Instrumentation analysis across the fleet revealed two meaningful chiller groups (not 3 hand-picked units or artificial clusters):
1. **Universal Candidate Pool (83 configured / 87 total chillers)**: Populates a standard 5-column set (`KW`, `Flow`, `inlet_temperature`, `outlet_temperature`, `DeltaT`). Crucially, **47 of these chillers are 100% clean (0% flagged corruption)** on temperature sensors in the restored August 2026 database — this is the real primary candidate pool for modeling. (Note: 4 of the 87 total chiller assets have unconfigured `SeriesDescription IS NULL` metadata rows).
2. **Type 2 Engineering Pool (26 chillers)**: Populates a richer engineering set (Evaporator/Condenser temperatures, Compressor Load, Discharge Pressure, etc.) in addition to universal features.

## Fleet Regime Boundary Rule (CRITICAL)
- **2026-01-01 is a hard fleet-wide regime shift**: Full-fleet instrumentation ramp-up in Jan 2026 caused per-chiller power draw to drop 1.4x–4.3x as building load was distributed across newly-online chillers (confirmed across all chillers with pre-2026 history). A week-by-week audit through July–August 2026 confirmed **no discontinuity exists around the August 2026 database reset**; `2026-01-01` remains the single fleet-wide physical regime shift.
- **NEVER train or validate any model across the 2026-01-01 boundary**: Treat pre-2026 and post-2026 as physically distinct operating regimes.
- Chillers with long historical data (1657, 1658, 1659, 1660, 1661) are valuable for length, but **must be regime-split** before training/validating any model.

## Correction: Pre-2026 Exclusion Was Partly a Misdiagnosis

- The original decision to restrict modeling to post-2026-01-01 data rested on two
  findings that were not the same thing, and got conflated:
  1. **The regime shift (still valid, still confirmed)** — per-chiller power draw
     dropped 1.4x–4.3x in Jan 2026 as load redistributed across newly-online chillers.
     Still holds.
  2. **"Corruption" in pre-2026 data (retracted as originally stated)** — largely
     undetected meter/counter reset artifacts, not evidence the data is unusable.
     Confirmed on Chiller 1657 (CKT_1_RUN_Hours reset on 2023-08-12; KW reset on 2024-09-22; Ambient_Temp ramp to 731°C then reset on 2026-04-09) and Chiller 2761 (inlet/outlet temperature ramps to 89.5°C/93.6°C resetting to 21.2°C/20.5°C on 2026-01-01).
- **Practical implication**: pre-2026 data should not be treated as permanently
  discarded — re-evaluate once the artifact-aware Data Validation Agent (Step 4) is
  applied. Still never pool across the 2026-01-01 boundary.
- **Do not re-litigate the regime boundary itself based on this correction.**

## Live Knowledge Store & Onboarding Sync
- `chiller_agent_knowledge.json` + `knowledge_store.py` serve as the live knowledge store for the agent system (mirroring `chiller_knowledge_base.md` as the static domain reference).
- Onboarding sync (`scripts/sync_knowledge_base.py`) runs whenever a client database/telemetry source is processed:
  1. Registers chillers (`register_chiller`) with type classification, confidence, and deduplicated physical asset alias mappings (Chillers 3392, 3894, 4054 = "Chiller-111").
  2. Records parameter observations (`record_parameter_observation`) with documented meanings/units for standard parameters, flagging unknown parameters for engineering review.
  3. Updates cluster membership and Layer-2 statistical bounds (`update_cluster`).
  4. Context bundle (`get_agent_context()`) is injected directly into agent prompt context.

## Asset Deduplication & Alias Mapping
- **Chillers 3392, 3894, and 4054** are confirmed to be the exact same physical asset (**Chiller-111**) registered under 3 separate vendor integrations, sharing identical telemetry.
- Dedup mapping is stored in `data/asset_aliases.csv` (`alias_id, canonical_id`). Any fleet-level reporting (chiller count, coverage %) must count this as 1 physical chiller, not 3.
- All 3 trained models are retained in `data/anomaly_models/` (harmless, low cost) but must be flagged as aliased in any summary reporting.

## Settled Modeling Findings (Validated Facts — Do Not Re-Litigate)

### 1. Forecast Agent — SETTLED (Persistence Baseline)
- **Persistence ($Power[t] = Power[t-1]$) is the validated Forecast baseline**, beating every ML approach tried across multiple chillers (including real wet-bulb data). Persistence won every time. This is settled; do not re-attempt ML forecasting without genuinely new data (e.g. an external weather API).
- Tested 4 independent ways:
  - (a) Delta-modeling the change instead of raw level.
  - (b) Chaining a forecasted external driver (CEFT, Ambient_Temperature, and WET BULB TEMPERATURE). **CEFT, "Ambient_Temperature ValueY", and "WET BULB TEMPERATURE" have all been tested and are internally-regulated signals with no real daily weather cycle (0.03°C–0.78°C diurnal swings across all three). No column in this fleet's sensor set represents true outdoor weather.**
  - (c) Direct lagged $Thermal\_Load[t-1] \rightarrow Power[t]$ using 5-fold `TimeSeriesSplit` CV.
  - (d) Real wet-bulb temperature feature test on Chiller 4054: Persistence ($R^2 = 0.7823$) strictly beat Lagged Driver ML ($R^2 = 0.5368$) and Hybrid ML ($R^2 = 0.6929$).
- Persistence won every trial, often by a wide margin (e.g. Chiller 1660: persistence R²=0.976 vs best ML R²=0.487).
- **Rule**: Short-term power forecasting is **settled across 4 independent methods**: use persistence as an informational baseline. DO NOT re-attempt ML-based short-term power forecasting without new external data (e.g., real outdoor weather API).


### 2. Anomaly Agent — VALIDATED & FULLY TRAINED (Physical Response Models)
- **Same-timestamp physical response model**: $KW = f(Flow, InletTemp, OutletTemp, DeltaT, [CompressorLoad])$ using `RandomForestRegressor` with 5-fold K-Fold CV.
- **Fleet Training Complete**: Fitted and saved 52 models across the 47 clean candidate chillers (`data/anomaly_models/`), restricted to the current regime (`>= 2026-01-01`) and capped at a recent 110-day training window (`2026-05-01` to `2026-08-19`).
- **Data Volume Efficiency**: Reduced raw training sensor rows from 165M to **77.97M rows**, providing 9,200–9,300 15-minute clean running training samples per chiller.
- **Performance**: Peak CV $R^2$ reached **0.8912** (Chillers 3894, 4054, 3392), **0.7857** (Chiller 2828), **0.7521** (Chiller 2763), and **0.7383** (Chiller 2826). Capping the training window to the recent 110-day regime improved $R^2$ CV significantly across long-history chillers (e.g. Chiller 1657 $R^2$ jumped from 0.358 to **0.551**, Chiller 1658 jumped from 0.314 to **0.496**).
- **Anomaly Logic**: $Residual = Actual\_KW - Predicted\_KW$. Convert residuals to z-scores ($z = (residual - \mu) / \sigma$). Flag single-point $|z| > 3.0$ as acute spikes (0.0% to 0.2% baseline rate across clean data). Flag sustained sequences of $\ge 4$ consecutive readings with $|z| > 2.0$ as operational efficiency degradation. An empirical audit across 1,802 readings (spanning ~35 readings / ~9 hours per chiller across all 52 fleet chillers) yielded **0.0% false positives** across 51 clean chillers while capturing 100% of sustained over-consumption events (e.g. Chiller 2825's Aug 7 18-hour $+525\text{ kW}$ inefficiency event). **Note**: This 0% false-positive rate is based on ~35 readings (~9 hours) per chiller in the audit window — a solid initial validation, not a long-run guarantee. Re-check this rule's false-positive rate periodically as more live telemetry accumulates through the running pipeline.


### 3. Optimization Agent — OPEN (Negative Schedule-Based Finding)
- Tested schedule-based waste (compressor load / KW efficiency ratio elevated during low-demand hours) on 5 chillers including those with strongest underlying physics (1657, 1661).
- **Finding**: Found NO structured time-of-day efficiency pattern (hourly variation was random noise, 2-5 percentage points).
- **Rule**: Genuine negative result. Do not re-attempt schedule-based waste checks. Optimization remains open for alternative formulation (e.g. setpoint optimization under clean physics).

### 4. Data Quality & Per-Chiller Modeling Rules
- `inlet_temperature` / `outlet_temperature` are clean (0% flagged) on 28 of 55 chillers. Corruption is per-chiller, never fleet-wide. Always check corruption per-chiller.
- **NEVER pool chillers together**: Always train and score models per-chiller.

## Architecture — Build Order
1. **Data Validation Agent**: Sanity-bound checks on incoming readings.
2. **Anomaly Agent**: Fits per-chiller `RandomForestRegressor` response model within a single regime, computes residual z-scores, flags $|z| > 3$.
3. **Forecast Agent**: Informational persistence baseline ($Power[t] = Power[t-1]$).
4. **Consensus & Skeptic Gate**: Combines Anomaly flags and validation signals.
5. **Optimization Agent**: (Open item - future phase).
6. **Insight Agent**: Explains reasoning chain in plain English.

## Non-negotiables
- NEVER modify or delete rows in the source database.
- NEVER hardcode secrets. Always load from environment variables (`.env`).
- ALWAYS enforce per-chiller modeling (never pool chillers).


