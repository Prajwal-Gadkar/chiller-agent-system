# Chiller Multi-Agent System — Progress Log

This file is a handoff summary for continuing this project in a new chat.
Background/rules for the project itself live in `CLAUDE.md` at the repo root
(read that first) — this file tracks what has actually been *done*, what
broke along the way, and what's next.

## Setup completed

- Two Postgres databases, connection details in `.env` (gitignored — see
  `.env.example` for the shape). `AppDb` = `Persistent_AppDb`, Timescale =
  `Persistent_Timescale`, both `localhost:5432`, user `postgres`.
- Confirmed DB session timezone is `Asia/Calcutta` (UTC+5:30) — this matters,
  see the timezone bug below.

## Scripts/agents built so far

| File | Purpose |
|---|---|
| `scripts/list_chillers.py` | `machine` ⋈ `MachineExplorer` filtered to `machineType='Chiller'`. Read-only. Base query reused by every other script. |
| `scripts/pull_trend_data.py` | Pulls `trendseriesmeterdata` for all chiller sensors over a date range, rounds timestamps to nearest 15 min, pivots to wide format (`[machineId, timestamp] × SeriesDescription`), joins back `status`/`Criticality` metadata after pivoting, downcasts to float32/int32. Processes **daily chunks internally** (pull → pivot → append to CSV) so memory stays bounded on large pulls — never holds more than one day in memory. Exposes `build_wide_trend_df()` (in-memory, for small exploratory windows) and `stream_pull_to_csv()` (chunked, for real pulls). CLI: `--start-date`, `--end-date`, `--chunk-days`. |
| `scripts/find_date_range.py` | Single aggregate query: earliest/latest timestamp + total row count across all chiller sensors. Runs in seconds. |
| `scripts/monthly_coverage.py` | Single aggregate query grouped by month: row count + distinct sensor count per month. Used to see how fleet instrumentation ramped up over time. |
| `agents/data_validation.py` | **First real agent.** Takes a wide DataFrame, applies a physical sanity-bound check per column (bounds inferred from column-name keywords: temperature, pressure, flow, power/kw, speed, status/fault/alarm → [0,1], hours, setpoint; unmatched columns fall back to a 6σ statistical bound). Never drops/mutates data — adds a `<col>_flagged` boolean companion column per sensor column. Returns a report DataFrame (column, rule, bounds, n_total, n_flagged, pct_flagged). |
| `scripts/compare_corruption_windows.py` | Pulls two 7-day windows and runs the validation agent's report on both, side by side, to see if corruption is uniform over time or concentrated. |
| `scripts/cluster_chiller_types.py` | Reads `data/trend_wide.csv`, computes per-`machineId` non-null coverage per sensor column, binarizes to populated/absent (>5% non-null), clusters chillers on that binary pattern with `AgglomerativeClustering`, tries k=2/3/4 and picks the best by silhouette score. Prints per-cluster size + defining columns, saves `data/chiller_types.csv` (machineId, chiller_type, status, Criticality, one bool column per sensor). Read-only, file-based only — no DB access needed since the sensor list is already baked into the wide CSV's columns. |

## Key data findings (confirmed, trust these)

- **~72–73 chiller assets total, but instrumentation ramped up in stages** —
  this is bigger than what CLAUDE.md originally documented and worth
  internalizing before picking any training window:
  - 2023-04 → 2024-12: only **18 sensors** (effectively 1 chiller) logging.
  - 2025-01 → 2025-09: **54 sensors**.
  - 2025-10 → 2025-12: **90 sensors**.
  - 2026-01 onward: **795–971 sensors** — this is the first window with
    near-full-fleet coverage. Chiller data **stops at 2026-07-08**; nothing
    after that belongs to chillers (see error #3 below).
  - Practical implication: **any model trained on data before 2026-01 is
    only representative of a handful of chillers, not the fleet.**
- Full June 2026 pull (official v1 training dataset, `data/trend_wide.csv`):
  161,340 rows, 57 chillers, 98.28% of theoretical max (57 × 30 days × 96
  fifteen-minute intervals) — a sane, stable fraction.
- Corruption confirmed **non-uniform over time**, not just non-uniform across
  sensors/chillers as CLAUDE.md already noted: comparing the week of
  2026-06-01 vs. 2026-07-02, temperature/fan-speed/load columns went from
  20–33% flagged to 0% flagged. Same sensors, same chillers, different week.
- Two columns worth a second look, not yet resolved:
  - `Compressor1_RunHours` / `Compressor2_RunHours` flag high (61–70%) in
    *both* windows — possibly a bound-calibration issue (cumulative run-hours
    can legitimately be large) rather than real corruption.
  - `IT_LOAD`, `TOTAL_FACILITY_LOAD`, `CH-1, POWER CONSUMPTION (KWH)` flag
    **100% in both windows** — almost certainly the default power bound
    (-10 to 5000) being too narrow for facility-level aggregates, not real
    corruption. Left as-is pending your call on the right scale/units for
    these columns.
- **Chiller instrumentation types, clustered from `data/trend_wide.csv`**
  (`data/chiller_types.csv`): silhouette score picked **k=4**, not a clean 3:
  - `type_2`: 25 chillers, ~20.2 columns populated on average — matches
    CLAUDE.md's "~22 columns" type.
  - `type_3`: 29 chillers, ~9.0 columns populated — close to CLAUDE.md's
    "~13 columns" type, a bit lower.
  - `type_1`: **1 chiller** (machineId 2825), ~38 columns populated — far
    more instrumented than anything else, looks like an outlier rather than
    a real third type.
  - `type_4`: **2 chillers** (machineIds 2833, 2834), ~2 columns populated —
    almost nothing logs, looks like broken/barely-instrumented units, not a
    real fourth type.
  - Silhouette scores across k=2/3/4 were all close (0.826 → 0.839 → 0.852);
    the bump at k=4 looks driven mainly by isolating the single most-
    instrumented chiller into its own cluster, a common artifact when one
    point is a genuine outlier rather than the head of a real cluster.
  - `type_2`/`type_3` look like the real "two more-instrumented types" that
    Flow→Power was verified on in CLAUDE.md. **Open question, not yet
    decided:** whether to treat `type_1`/`type_4` as flagged outliers
    (e.g. route straight to Tier 3 rolling-average — too few members for a
    type-level model) rather than legitimate types, before this feeds the
    Supervisor's routing logic.

## Errors encountered and how they were resolved

1. **Initial DB auth failure.** First connection attempt (before getting real
   credentials) failed with `password authentication failed for user
   "postgres"` — the `reference/envSettings.json` password was stale for this
   local instance. Resolved once you provided the working `Persistent_AppDb`
   / `Persistent_Timescale` / password `admin` credentials.
2. **First test pull looked broken (only 1 chiller returned).** A 2024-06-01
   to 2024-06-07 test pull returned only 1 chiller / 672 rows. Investigated
   rather than assumed-buggy: confirmed via direct query that only 18 of
   1,137 sensors (all on one chiller) had *any* data logged that week — real
   sparsity, not a bug. Led directly to the staged-instrumentation finding
   above.
3. **Wrong assumption about "future" data.** Assumed the table's max
   timestamp (2026-10-04) meant chiller data extended that far, and asked to
   pull a comparison window from 2026-09/2026-10. Checked first and found
   that range has **zero rows for chiller sensors** — it belongs entirely to
   non-chiller machines in the same `trendseriesmeterdata` table. Substituted
   the last real chiller week (2026-07-02 to 2026-07-08) instead.
4. **Timezone bug (the significant one).** `pd.read_sql_query` over a raw
   (non-SQLAlchemy) psycopg2 connection **silently converts tz-aware
   timestamptz columns to UTC** in the resulting DataFrame. Since the DB
   session's local time is Asia/Calcutta (UTC+5:30), this shifts any
   date/month boundary computed client-side backward — first caught in
   `monthly_coverage.py`, where every `date_trunc('month', ...)` label was
   off by one month (e.g. real April 2023 data was printed under "March
   2023"). Root-caused by comparing raw `psycopg2.connect().cursor()` output
   (correct, tz-aware with +05:30 offset) against what came back through
   pandas (silently UTC). **Fix pattern: never trust pandas' tz handling on a
   raw DBAPI2 connection — do date/timezone formatting in SQL** (`to_char()`
   for labels, `AT TIME ZONE 'Asia/Calcutta'` for raw timestamp columns)
   before the value reaches pandas. Applied to `monthly_coverage.py`,
   `pull_trend_data.py` (was silently mislabeling the pulled date range by a
   day), and `find_date_range.py` (same bug, hadn't yet produced a visibly
   wrong date by chance, but was fixed anyway for correctness). Re-verified
   all three afterward against raw psycopg2 output — dates now match exactly.
   **Watch for this same pattern in any future script that reads timestamp
   columns back through `pd.read_sql_query` on a raw psycopg2 connection.**

## Data Restoration & Extended History
- **Postgres Backup Restoration**: Restored full local backup to `data/backups/`, recovering complete fleet history from **2023-04-29 to 2026-07-08** across all ~73 chiller assets.
- **2026-01-01 Regime Boundary Discovered**:
  - Full-fleet instrumentation ramp-up in Jan 2026 caused per-chiller power draw to drop **1.4x to 4.3x** across all 7 chillers with pre-2026 history (e.g. Chillers 1657, 1658, 1659, 1660, 1661), as building load was distributed across newly online chillers.
  - Hard rule established: **NEVER train or validate models across the 2026-01-01 boundary**. Pre/post 2026 are distinct physical operating regimes.

## Chiller Selection Pivot & Instrumentation Analysis
- **Two Real Instrumentation Groups Identified**:
  - **Universal Pool (55/72 chillers)**: Populates standard 5-column set (`KW`, `Flow`, `inlet_temperature`, `outlet_temperature`, `DeltaT`).
  - **Type 2 Engineering Pool (26/72 chillers)**: Populates richer engineering set (Evaporator/Condenser temperatures, Compressor Load, Discharge Pressure).
- **28 Clean Chillers Confirmed**:
  - Performed fleet-wide temp sensor corruption audit (`fleet_temp_validation_summary.csv`).
  - **28 chillers have 0% flagged corruption** on `inlet_temperature` / `outlet_temperature` (including 1657, 1658, 1659, 1661, and 2737–2760). This clean universal group forms the real candidate pool.
  - Temperature corruption is NOT fleet-wide; it only affects a specific subset (e.g. 2821, 2828, 2831). Always audit per-chiller.

## Forecast, Anomaly, & Optimization Experiments Summary

| Experiment / Agent | Approach Tested | Outcome / Finding | Status / Decision |
|---|---|---|---|
| **Forecast Agent** | (a) Delta modeling ($Power[t] - Power[t-1]$)<br>(b) External driver chaining (CEFT / Ambient Temp)<br>(c) Direct lagged $Thermal\_Load[t-1] \rightarrow Power[t]$ via 5-fold `TimeSeriesSplit` CV | Persistence ($Power[t] = Power[t-1]$) beat every ML model across all 3 tests by a wide margin (e.g. Chiller 1660: Persistence R²=0.976 vs ML R²=0.487). Ambient_Temp proved to be an internal setpoint, not real outdoor weather. | **SETTLED**: Short-term power forecasting ML is closed. Persistence is the designated informational baseline. |
| **Anomaly Agent** | Physical response model $KW = f(Flow, InletTemp, OutletTemp, DeltaT, [CompLoad])$ via `RandomForestRegressor` with random 5-fold K-Fold CV. | Confirmed strong physical response ($R^2 = 0.31–0.99$). Strongest on long-history chillers 1657/1660/1661 ($R^2 = 0.96–0.99$), usable across 28-clean pool. $z$-scored residuals ($|z| > 3$) reliably flag physical anomalies. | **VALIDATED FOUNDATION**: Ready for production implementation in `agents/anomaly_agent.py`. |
| **Optimization Agent** | Schedule-based efficiency waste (Compressor Load / KW efficiency ratio during low-demand hours). Tested on 5 chillers (incl. 1657, 1661). | NO time-of-day efficiency pattern found (hourly variation was random noise, 2–5 percentage points). | **TESTED & NOT FOUND**: Schedule-based waste check is a confirmed negative result. Optimization remains open. |

## Current State

- **2026-08-20**:
  - Restored fresh databases `Persistent_AppDb_Aug20` (87 chillers) and `Persistent_Timescale_Aug20` (2023-04-29 to 2026-08-19, 95.2M rows).
  - Executed fleet-wide clean scan on 83 configured chillers saved to [`data/clean_chillers_aug20.csv`](file:///c:/Users/Admin/Documents/chiller-agent-system/chiller-agent-system/data/clean_chillers_aug20.csv): **47 chillers** confirmed `CLEAN & VALID` on universal set (0% corruption).
  - Executed week-by-week August 2026 regime shift check (`scripts/check_august_regime_shift.py`) saved to [`data/august_regime_shift_analysis.csv`](file:///c:/Users/Admin/Documents/chiller-agent-system/chiller-agent-system/data/august_regime_shift_analysis.csv): **No discontinuity exists in July/August 2026**; `2026-01-01` remains the single fleet-wide regime boundary.
  - Analyzed Chiller 4054 (`scripts/analyze_chiller_4054_wetbulb.py`) saved to [`data/chiller_4054_wetbulb_analysis.csv`](file:///c:/Users/Admin/Documents/chiller-agent-system/chiller-agent-system/data/chiller_4054_wetbulb_analysis.csv):
    - Diurnal swing: 0.78 °C (internal loop sensor, minimal diurnal swing).
    - Response Model (5-fold random CV): Base $R^2 = 0.9004$ vs With WetBulb $R^2 = 0.9250$ ($\Delta R^2 = +0.0246$).
    - Short-Term Forecast (5-fold TimeSeriesSplit CV): Persistence ($R^2 = 0.7823$) beats Lagged Driver ML ($R^2 = 0.5368$) and Hybrid ML ($R^2 = 0.6929$). Settled rule holds: **Persistence baseline wins for short-term power forecasting**.
  - **Completed Fleet Anomaly Agent Training (`scripts/train_anomaly_agent_fleet.py`)**:
    - Fitted physical response models across the 47 clean candidate chillers (`data/anomaly_models/`), restricted to post-2026-01-01 regime and capped at a recent ~110-day training window (`2026-05-01` to `2026-08-19`).
    - Reduced raw training rows from ~165M to **77.97M rows**, providing 9,200–9,300 clean running 15-min training samples per chiller.
    - Achieved peak CV $R^2$ of **0.8912** (Chillers 3894, 4054, 3392), **0.7857** (Chiller 2828), **0.7521** (Chiller 2763), and **0.7383** (Chiller 2826). Capping the training window to the recent ~110 days significantly improved $R^2$ across long-history chillers (e.g. Chiller 1657 $R^2$ jumped from 0.358 to **0.551**, Chiller 1658 jumped from 0.314 to **0.496**).
    - Generated full fit summary in [`data/anomaly_agent_fit_summary.csv`](file:///c:/Users/Admin/Documents/chiller-agent-system/chiller-agent-system/data/anomaly_agent_fit_summary.csv).

- **2026-08-25 (Data Validation Agent Upgrade & Live Knowledge Store Integration)**:
  - **Empirical Check Completed (Step 1)**: Verified counter and meter reset artifacts in pre-2026 data for long-history chiller 1657 (`CKT_1_RUN_Hours` reset to 0.0 on 2023-08-12; `KW` reset to 0.0 on 2024-09-22; `Ambient_Temperature` ramp to 731°C resetting to 20.95°C on 2026-04-09) and chiller 2761 (`inlet_temperature` ramp to 89.5°C resetting to 21.2°C; `Outlet_temperature` ramp to 93.6°C resetting to 20.5°C on 2026-01-01). Saved evidence to [`data/reset_evidence.csv`](file:///c:/Users/Admin/Documents/chiller-agent-system/chiller-agent-system/data/reset_evidence.csv) and [`data/reset_evidence.json`](file:///c:/Users/Admin/Documents/chiller-agent-system/chiller-agent-system/data/reset_evidence.json).
  - **Live Knowledge Base Integration (`scripts/sync_knowledge_base.py`)**: Built onboarding and sync pipeline that reads telemetry data and registers 57 chillers into `chiller_agent_knowledge.json` via `knowledge_store.py`. Handled asset deduplication (mapping Chillers 3392, 3894, and 4054 under canonical asset `3392` / "Chiller-111" with alias array), populates parameter meanings and units for 56 documented parameters, records unknown parameters as active learned patterns for engineering review, and populates Layer-2 cluster statistical bounds.
  - **Cluster Minimum Size Threshold (`scripts/cluster_chiller_types.py`)**: Implemented 5-chiller minimum size guard. Small clusters (`type_1` with 1 chiller, `type_4` with 2 chillers) route to Tier-3 rolling average fallback and register as `chiller_type="unclustered_outlier"`.
  - **Data Validation Agent Rewrite (`agents/data_validation.py`)**: Added sentinel value detection (`0` flatline duration, `-1`, `9999`, `10000`, `65535`, `-999`), vectorized monotonic ramp-then-reset detection, Layer-2 cluster statistical bounds, cross-parameter correlation checks, and new companion schema (`<col>_artifact_type`, `<col>_fault_window_id`, `<col>_evidence`).
  - **Agent Prompt Wiring (`agents/pipeline.py`)**: Wired live `get_agent_context()` bundle directly into agent prompts and pipeline execution.
  - **Documentation Correction (`CLAUDE.md`)**: Documented pre-2026 misdiagnosis correction (meter resets vs regime boundary) and knowledge-sync pipeline structure.

## Next Steps

1. **Re-evaluate Pre-2026 Data for Response Models**:
   - Apply artifact-aware Data Validation Agent on pre-2026 long-history chiller data (1657, 1658, 1659, 1660, 1661) to test if cleaned pre-2026 data can fit distinct pre-2026 regime models.
2. **Build Supervisor & Consensus Gate (`agents/consensus_gate.py`)**:
   - Combine Anomaly Agent residual z-scores ($|z| > 3.0$) and Data Validation companion flags into unified risk scores.
3. **Build Insight Agent (`agents/insight_agent.py`)**:
   - Synthesize validation metrics and physical anomaly flags into plain-English reasoning summaries.



