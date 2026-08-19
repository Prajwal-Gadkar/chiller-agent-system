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

## Current state

- Architecture step 1 (Data Validation Agent) is built and validated against
  real data, but its bound table is a first pass — power/facility-aggregate
  bounds are known to be miscalibrated (see findings above).
- `data/trend_wide.csv` (June 2026, 161,340 rows) is ready to use as the v1
  training dataset.
- Chiller instrumentation types have been clustered from that data
  (`data/chiller_types.csv`) — see findings above. Result is 2 solid types
  + 2 likely-outlier groups, not a clean 3. **Not yet decided** how the
  Supervisor should treat the 2 outlier groups (type_1: 1 chiller, type_4:
  2 chillers) — this blocks finalizing the Supervisor's routing logic.

## Next steps (per CLAUDE.md's build order)

1. Decide how to treat `type_1`/`type_4` outlier clusters (see open question
   above) before wiring them into Supervisor routing.
2. Decide on the flagged-power-column bound fix (tighten per-column, or
   leave the flag for the consuming step to interpret) — was left open as a
   question, not yet answered.
3. **Supervisor** — routes by chiller type + reliability tier. Not started.
4. **Forecast Agent** — Flow→Power prediction, 3-tier reliability cascade
   (per-chiller → type-level → rolling-average fallback), per-chiller only,
   never pooled. Not started. June 2026 data is the intended training set.
5. **Anomaly Agent** — independent statistical check, parallel to Forecast.
   Not started.
6. **Consensus & Skeptic Gate**, **Optimization Agent** (HITL-gated),
   **Insight Agent** — not started, downstream of the above.
