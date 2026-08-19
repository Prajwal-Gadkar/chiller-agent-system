# Chiller Multi-Agent System — Project Context

## What this project is
A multi-agent system (LangGraph) for monitoring and optimizing chiller performance,
built standalone first, for later integration into InnoShri's YntraM production
system (which already has ~90 services covering AHU/HVAC/chiller forecasting, FDD,
alerts, and LLM insights). This project ADDS a reasoning/orchestration layer on top
of that — it does not rebuild those services.

## Data source
PostgreSQL, two databases (connection strings live in .env — NEVER hardcode them):
- AppDb: `machine` (asset master — has `machineType`, `status`, `Criticality`
  columns) joined with `MachineExplorer` (sensor metadata: Id, MachineId,
  SeriesDescription).
- Timescale DB: `trendseriesmeterdata` (TimescaleDB hypertable — columns:
  timestamp, machineexplorerid, value, id). Actual sensor readings, physically
  chunked by day internally (relevant if ever reading the raw pg_dump directly).

Join key: `machine.machineId = MachineExplorer.MachineId`;
          `MachineExplorer.Id = trendseriesmeterdata.machineexplorerid`

Filter to chillers: `WHERE machine.machineType = 'Chiller'`

Known count as of last analysis: ~73 chiller-type assets, ~57 had readings in the
analyzed window (June 2026, the current v1 training window). Instrumentation
clustering on that window (silhouette-selected k, see PROJECT.md) found 2 solid
types plus 2 likely outlier singleton/near-empty groups, not a clean 3:
type_2 (25 chillers, ~20 cols), type_3 (29 chillers, ~9 cols), plus a 1-chiller
"type_1" (~38 cols, unusually well-instrumented) and a 2-chiller "type_4"
(~2 cols, barely instrumented). type_2/type_3 line up with the "two more-
instrumented types" Flow→Power was verified on. types must be discovered fresh
from the data (which columns each chiller actually populates), never
hardcoded, since the fleet may have changed — treat the numbers above as the
last-known snapshot, not a fixed constant.

## CRITICAL data quality findings — trust these, don't re-derive from scratch
- Raw sensor data contains genuine, confirmed corruption: some readings jump from
  a sane range to six-figure impossible values (Flow, KW, Fan_speed sensors were
  directly verified against the raw Postgres backup — this is real, not a
  pipeline bug).
- Corruption is NOT uniform. It varies by sensor and by chiller, and sometimes
  MULTIPLE sensors on the SAME chiller corrupt together at the same moments
  (co-corruption). This can make a corrupted relationship look statistically
  "real" even under cross-validation. Before trusting any strong relationship,
  sanity-check that the target and its driving feature aren't sharing an
  implausible physical scale together (e.g. both reaching hundreds of thousands
  when they should be under a few hundred).
- Status/flag columns (On_Off_Status, Auto_Manual_Status, Fault_Status,
  Common_Alarm_Status) are NOT strict 0/1 binaries — they're continuous
  duty-cycle-style values in [0,1]. Valid-range check should be `0 <= x <= 1`,
  not `x in {0, 1}`. Getting this wrong silently breaks any "% of time" metric.
- NEVER pool different chiller types together in one model — column sets differ
  and mixing them reintroduces massive sparsity.
- NEVER pool different chillers of the SAME type together either. Verified:
  pooled R² ~0.02 (no real signal) vs. per-chiller R² 0.35-0.99 (real signal) on
  identical underlying data. Per-chiller modeling is the single most important
  rule in this project.
- Some chillers show NO reliable model relationship even modeled individually.
  This is a real, expected finding — those chillers need a simpler/rule-based
  fallback, not a forced ML model.
- Flow → Power (KW) is the one relationship independently verified (proper
  train/test split + 5-fold cross-validation, not same-data R²) to generalize
  for the two more-instrumented chiller types. Treat this as the primary
  trusted physical relationship for forecasting.
- `Energy_Consumption` is a cumulative counter — never use as a raw model
  feature (it's circular with power draw, not a real driver).

## Feature selection rule (from the instructor)
Model features must be things that can be CONTROLLED — setpoint, actual
inlet/outlet temperature, flow rate, fan speed, compressor load/staging, on/off
mode — not things that are only OBSERVED outcomes (pressures, internal
evaporator/condenser temps, run hours, fault flags). Apply this filter whenever
selecting inputs for any predictive model.

## Architecture — build in this order
1. **Data Validation Agent** — sanity-bound gate on every reading before anything
   else sees it. Filters at read time only; never mutates the source database.
   Reference real thresholds from appsettings_2.json (OverPumping,
   TowerEfficiency, LowLoadPenalty rules) once available.
2. **Supervisor** — routes based on chiller type + reliability tier (see cascade
   below).
3. **Forecast Agent** — Flow→Power prediction using the 3-tier reliability
   cascade.
4. **Anomaly Agent** — independent statistical check, runs parallel to Forecast.
5. **Consensus & Skeptic Gate** — Forecast and Anomaly must agree, AND the
   result must pass a co-corruption sanity check, before anything escalates.
6. **Optimization Agent** — drafts a recommendation, self-simulates the
   predicted effect before presenting it. Any real setpoint change requires
   human approval (HITL) — this is a physical-safety boundary, not optional.
7. **Insight Agent** — explains the reasoning chain in plain language, not just
   a verdict.

## Reliability cascade (core modeling pattern — mirrors InnoShri's own
FDDPerformanceProcessing service, confirmed as a real production pattern)
- Tier 1: per-chiller model (only if this specific chiller cleared a genuine
  cross-validated reliability check)
- Tier 2: type-level model (fallback for chillers that didn't clear Tier 1)
- Tier 3: time-series / rolling-average, no ML (last-resort fallback)

## Non-negotiables
- NEVER modify or delete rows in the source database. All filtering happens at
  read time, in application code, never as a database mutation.
- NEVER hardcode API keys, database passwords, or any secret. Always load from
  environment variables via `.env` (gitignored). If a reference file contains a
  real secret, treat it as sensitive — never echo it into generated code, logs,
  or documentation.
- Any new model's accuracy must be validated with a genuine train/test split or
  cross-validation before being trusted. Same-data R² (train and score on
  identical rows) is not acceptable evidence of anything.
- Before treating a strong model result as real, check it isn't the
  co-corruption pattern described above.
