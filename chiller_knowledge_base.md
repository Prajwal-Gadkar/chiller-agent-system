# Chiller Multi-Agent System — Domain Knowledge Base & Data Validation Agent Design

**Purpose of this file:** This is the static domain-knowledge reference the Data Validation Agent (and other agents in the pipeline) should be grounded on before processing live chiller data — general chiller knowledge plus the Data Validation Agent's design.

Findings specific to our own chiller database (inventory, tracked parameters, real data-quality patterns) are intentionally **not** in this file — those are produced during data preprocessing and belong in the live, agent-editable knowledge store described in [Section 4](#4-the-live-knowledge-store), not in this static reference.

**Last updated:** Reflects the decision to keep the live store as JSON (not Excel) with a `chiller_inventory` registry (merging chiller metadata + per-parameter tracking) plus `register_chiller()`/`record_parameter_observation()` in `knowledge_store.py`. The live store currently ships seeded but empty — the real 103-chiller / 56-parameter dataset is populated via a one-time backfill, not yet run.

---

## Table of Contents

1. [Chiller Domain Knowledge](#1-chiller-domain-knowledge)
   - 1.1 [Use Cases](#11-use-cases)
   - 1.2 [Chiller Types](#12-chiller-types)
   - 1.3 [Industries](#13-industries)
   - 1.4 [Standards & Regulatory Frameworks](#14-standards--regulatory-frameworks)
   - 1.5 [Performance Metrics](#15-performance-metrics)
   - 1.6 [Refrigerant Safety Classification](#16-refrigerant-safety-classification)
   - 1.7 [Generic Operational Parameters to Monitor](#17-generic-operational-parameters-to-monitor)
2. [Data Validation Agent — Design](#2-data-validation-agent--design)
   - 2.1 [The Five Sub-Jobs](#21-the-five-sub-jobs)
   - 2.2 [Chiller Type & Industry Identification](#22-chiller-type--industry-identification)
   - 2.3 [Range Bound Validation (Two Layers)](#23-range-bound-validation-two-layers)
   - 2.4 [Artifact Detection Rules](#24-artifact-detection-rules)
   - 2.5 [Clustering Similar Chillers](#25-clustering-similar-chillers)
   - 2.6 [Suggested LangGraph Subgraph Structure](#26-suggested-langgraph-subgraph-structure)
3. [System Architecture Context](#3-system-architecture-context)
4. [The Live Knowledge Store](#4-the-live-knowledge-store)
   - 4.1 [Why JSON, not Excel](#41-why-json-not-excel)
   - 4.2 [The Two Files](#42-the-two-files)
   - 4.3 [Update Discipline](#43-update-discipline-so-the-store-doesnt-degrade-over-time)

---

## 1. Chiller Domain Knowledge

### 1.1 Use Cases

A chiller's core job: remove heat from a fluid (usually water or water-glycol) and reject that heat elsewhere (air or a water source).

| Category | Sub-Application | Description | Typical Failure Tolerance |
|---|---|---|---|
| Comfort Cooling (HVAC) | Large commercial buildings | Office towers, malls, airports, hospitals — central chilled water plants feeding AHUs/FCUs instead of individual AC units | Moderate |
| Comfort Cooling (HVAC) | District cooling | One large chiller plant supplies chilled water to multiple buildings in a zone | Moderate-High |
| Industrial Process Cooling | Plastics & injection molding | Mold cooling to control cycle time and part quality | Moderate |
| Industrial Process Cooling | Metal working | CNC machines, laser cutting, welding — prevents thermal drift/tool wear | Moderate |
| Industrial Process Cooling | Chemical & pharma processing | Reaction temperature control, distillation, crystallization — often ±0.5°C or tighter | Very Low |
| Industrial Process Cooling | Food & beverage | Brewing fermentation, dairy processing, bottling lines | Low-Moderate |
| Industrial Process Cooling | Printing | Ink temperature and roller cooling | Moderate |
| Industrial Process Cooling | Laser & electronics manufacturing | Cooling laser sources, semiconductor fab tools | Very Low |
| Data Centers | Server room / rack-level cooling | Precision cooling, increasingly critical with GPU/AI compute loads | Extremely Low |
| Medical & Healthcare | Diagnostic & surgical equipment | MRI/CT scanner cooling, surgical suite HVAC, sterilization support | Very Low |
| Power Generation | Turbine inlet air cooling | Boosts gas turbine output in hot weather; auxiliary generator cooling | Low (during peak demand) |
| Specialty / Other | Ice rinks, cold storage, wine cellars, labs, sports facilities | Niche applications with bespoke requirements | Varies |

**Key pattern for agent design:** comfort cooling tolerates more temperature swing and prioritizes energy efficiency; process cooling prioritizes precision and uptime over efficiency. This distinction should drive different setpoint-tolerance and fault-escalation logic per application type.

### 1.2 Chiller Types

**By heat rejection method:**

| Type | Description | Typical COP | Pros | Cons | Best For |
|---|---|---|---|---|---|
| Air-Cooled | Rejects heat directly to ambient air via condenser coils/fans | 2.5–3.5 | Simple install, no water treatment, lower maintenance | Lower efficiency, performance drops with ambient temp, noisier | Smaller buildings, water-scarce sites |
| Water-Cooled | Rejects heat to a condenser water loop → cooling tower | 4–7 | Much higher efficiency, stable performance, quieter, compact | Needs cooling tower + water treatment (Legionella risk), more complex | Large buildings, efficiency-at-scale needs |

**By compressor / refrigeration technology:**

| Type | Description | Typical Capacity | Notes |
|---|---|---|---|
| Reciprocating | Piston-based compressors | Small-mid | Older tech, mostly phased out for new installs |
| Scroll | Two interleaving spiral scrolls | ~15–150 tons | Efficient at partial loads, quiet, reliable |
| Screw | Twin helical rotors, continuous compression | ~70–500+ tons | Good efficiency, tolerant of load fluctuation |
| Centrifugal | Rotating impeller accelerates refrigerant vapor | ~200–10,000+ tons | Highest efficiency at scale; **surge risk** at low load (unstable reverse flow — important failure mode) |
| Absorption | No mechanical compressor — heat + refrigerant/absorbent pair (LiBr-water or ammonia-water) | Varies | Low electrical use, lower overall efficiency (COP often <1), needs heat source |
| Magnetic Bearing (Oil-Free) | Modern centrifugal variant, magnetic levitation bearings | Similar to centrifugal | Very high part-load efficiency, low maintenance (no oil system) |

**By application mode:** Process chillers (tight setpoint control, portable/skid-mounted) vs. Comfort chillers (building-integrated, BMS-tied).

**Type-inference signal for agents:** presence/absence of certain parameters is itself diagnostic — oil pressure absent = likely magnetic bearing; solution concentration/generator temp present = absorption chiller; condenser water temp absent = likely air-cooled.

### 1.3 Industries

| Industry | Typical Applications | Typical Chiller Type(s) | Precision Required | Failure Tolerance | Typical Redundancy |
|---|---|---|---|---|---|
| HVAC / Commercial Real Estate | Offices, malls, hotels, airports, campuses | Centrifugal, screw (large); scroll (small) | Low-Moderate | Moderate | N (occasionally N+1) |
| Pharmaceutical & Biotech | Reaction vessel cooling, cleanroom HVAC, cold storage | Screw, scroll, precision process chillers | Very High (±0.5°C+) | Very Low | N+1 minimum, often 2N |
| Food & Beverage | Brewing, dairy, bottling, cold storage | Screw, scroll (hygienic design) | Moderate | Low-Moderate | N, sometimes N+1 |
| Chemical & Petrochemical | Exothermic reaction control, distillation | Screw, centrifugal (ATEX rating possible) | High | Very Low (safety hazard) | N+1 or higher |
| Plastics & Injection Molding | Mold temp control, extrusion cooling | Portable scroll/screw units | Moderate | Moderate | N (per-machine) |
| Data Centers | Server room / rack-level, GPU/AI liquid cooling | Screw, centrifugal, magnetic bearing | High (thermal stability) | Extremely Low | N+1, 2N, or 2N+1 by tier |
| Healthcare / Hospitals | Comfort cooling, MRI/CT, surgical HVAC | Screw, centrifugal | High | Very Low | N+1 minimum (life-safety) |
| Power Generation | Turbine inlet air chilling | Large centrifugal | Moderate | Low (peak demand) | N, sometimes N+1 |
| Metalworking & Machining | CNC spindle, laser cutting, EDM | Portable/machine-integrated | High | Moderate | N (per-machine) |
| Semiconductor / Electronics Mfg | Photolithography cooling, cleanroom control | Precision process, magnetic bearing | Extremely High (±0.1°C) | Extremely Low | High (comparable to pharma) |
| Marine & Offshore | Ship HVAC, refrigerated cargo | Compact air/seawater-cooled | Moderate | Low-Moderate | N, sometimes N+1 |

**Three variables that determine chiller criticality across all industries:** (1) precision required (temperature tolerance band), (2) consequence of failure (safety/financial/spoilage/compliance), (3) redundancy level needed (N, N+1, 2N). A general "criticality classifier" agent should reason on these three axes rather than treating industries as flat categories.

### 1.4 Standards & Regulatory Frameworks

| Standard / Metric | Body | Type | Description |
|---|---|---|---|
| AHRI 550/590 | AHRI | Standard | The reference standard for rating water-chilling packages in North America; defines standard test conditions for apples-to-apples comparison |
| AHRI 551/591 | AHRI | Standard | Same scope, SI (metric) units |
| ASHRAE 90.1 | ASHRAE | Standard | Energy standard for buildings; minimum COP/IPLV thresholds by chiller type/capacity |
| ASHRAE 15 | ASHRAE | Standard | Safety standard for refrigeration systems (refrigerant class, machine room, ventilation) |
| ASHRAE 34 | ASHRAE | Standard | Refrigerant designation and safety classification (see 1.6) |
| ASHRAE Guideline 3 | ASHRAE | Guideline | Reducing emission of halogenated refrigerants |
| Eurovent Certification | Eurovent | Certification | European equivalent to AHRI; often required in EU tenders |
| ISO 5151 / ISO 15042 | ISO | Standard | Testing and rating of chillers/heat pumps |
| ISO 13256 | ISO | Standard | Water-source heat pump ratings |
| ISO 14001 | ISO | Standard | Environmental management (refrigerant handling programs) |
| UL / cUL | UL (N. America) | Certification | Electrical safety |
| CE Marking / PED | EU | Certification/Directive | EU conformity; Pressure Equipment Directive for pressure vessel design |
| BEE Star Rating | BEE (India) | Labeling | India energy efficiency labeling |
| GB Standards | SAC (China) | Standard | China-specific chiller standards |
| Montreal Protocol | UN/Global | Treaty | Phase-out of ozone-depleting substances (CFCs, HCFCs) |
| Kigali Amendment | UN/Global | Treaty Amendment | Phase-down of HFCs (global warming potential) |
| US EPA Section 608 | US EPA | Regulation | Refrigerant handling/recovery/technician certification |
| US AIM Act | US EPA | Regulation | Domestic implementation of Kigali Amendment |
| EU F-Gas Regulation | EU | Regulation | HFC phase-down driving low-GWP refrigerant adoption |

### 1.5 Performance Metrics

| Metric | Definition / Formula | Typical Values / Notes |
|---|---|---|
| **COP** (Coefficient of Performance) | Cooling Effect (kW) / Power Input (kW), at a single rated condition | Air-cooled: 2.5–3.5, Water-cooled: 4–7, Magnetic bearing: 7–8+ |
| **EER** (Energy Efficiency Ratio) | Btu/h output per watt input. EER = COP × 3.412 | — |
| **kW/Ton** | kW input per ton of cooling. Lower = better | 0.5–0.7 kW/ton for efficient water-cooled centrifugal |
| **IPLV** (Integrated Part Load Value) | Weighted avg efficiency across 100%/75%/50%/25% load (AHRI weighting: 1%/42%/45%/12%) | Most representative real-world efficiency metric |
| **NPLV** (Non-Standard Part Load Value) | Same as IPLV but at site-specific conditions | Used when actual conditions differ from AHRI standard |
| **Standard Rating Conditions** | CHW: 44°F leaving/54°F entering (10°F ΔT); Condenser water: 85°F entering/95°F leaving; Fouling factor 0.0001 h·ft²·°F/Btu | Basis for cross-manufacturer comparison |
| **Approach Temperature** | Temp difference between refrigerant and water at heat exchanger | Rising over time = fouling/maintenance flag |
| **Capacity (Tonnage)** | 1 ton = 12,000 Btu/h = 3.517 kW | Historical basis: heat to melt 1 ton of ice in 24h |

### 1.6 Refrigerant Safety Classification

(ASHRAE 34 / ISO 817 toxicity/flammability grid)

| Class | Description | Examples | Notes |
|---|---|---|---|
| A1 | Lower toxicity, no flame propagation | R-134a, R-410A | Most common historically |
| A2L | Lower toxicity, mildly flammable | R-32, R-1234yf | Emerging generation; different safety/ventilation design needed |
| A3 | Lower toxicity, higher flammability | Propane (R-290), hydrocarbons | Engineered flammability risk |
| B1 | Higher toxicity, no flame propagation | — | Stricter machine-room codes |
| B2 / B2L | Higher toxicity, flammable/mildly flammable | Ammonia (R-717) | Common in industrial refrigeration; strict ventilation codes |
| B3 | Higher toxicity, higher flammability | — | Rare in commercial chillers |

### 1.7 Generic Operational Parameters to Monitor

| Parameter | Why It Matters | Typical Unit |
|---|---|---|
| Chilled water supply/return temp (CHWS/CHWR) | Core control loop; deviation = load/capacity issue | °C / °F |
| Condenser water supply/return temp | Heat rejection performance | °C / °F |
| Delta-T (ΔT) across evaporator | Flow verification; low ΔT = common plant inefficiency | °C / °F |
| Refrigerant suction/discharge pressure | Compressor health, charge level, leak detection | bar / psi |
| Superheat / subcooling | Refrigerant charge diagnostics | °C / °F |
| Compressor amp draw / power input | Load tracking, early fault detection | A / kW |
| Oil pressure (non-mag-bearing) | Lubrication system health | bar / psi |
| Approach temperature (evap & condenser) | Fouling/scaling indicator | °C / °F |
| Vibration | Bearing wear, especially centrifugal/screw | mm/s or g |
| Surge conditions (centrifugal only) | Critical fault — unstable reverse flow | Boolean/event flag |
| Run hours (per compressor) | Maintenance scheduling | hours |
| Fault/Alarm status | Direct fault signal from controller | Boolean/code |
| On/Off & Auto/Manual status | Operating mode context — needed to interpret other readings correctly | Boolean/status code |
| Fan speed (air-cooled/cooling tower) | Heat rejection capacity indicator | RPM |
| Flow rate (chilled & condenser water) | Confirms adequate flow for accurate ΔT-based load calcs | m³/h, L/s, GPM |

---

## 2. Data Validation Agent — Design

### 2.1 The Five Sub-Jobs

The Data Validation Agent bundles five distinct sub-problems. Treat them as sub-steps (or sub-tools) inside the validation node, not one monolithic check:

1. **Chiller identification** — what type/category is this unit
2. **Industry/application context** — where and how it's used
3. **Range bound validation** — is each parameter within physically/operationally sane limits
4. **Artifact detection** — resets, sensor dropouts, wiring faults vs. genuine anomalies
5. **Clustering** — grouping similar chillers so bounds/baselines are meaningful

### 2.2 Chiller Type & Industry Identification

**Two paths, use both:**

- **Metadata-based** (fast, reliable when present): nameplate data — compressor type, refrigerant, rated tonnage, model number. Many real-world fleets have incomplete metadata, so this path alone is often insufficient for a large share of the fleet.
- **Data-driven inference** (needed as the primary path when metadata is sparse): behavioral fingerprinting from the timeseries —
  - Screw vs. centrifugal vs. scroll show different load-vs-efficiency curves (centrifugal has surge risk near low load).
  - Air-cooled vs. water-cooled: check for presence of a condenser water temp parameter, and correlation strength between performance and ambient/outdoor air temp.
  - Reciprocating/older units show more compressor cycling noise in amp draw and discharge pressure.
  - Parameter *presence/absence* is itself a strong signal (see Section 1.2).
- **Ask the client** — fallback only, when both above are ambiguous. Structure as: infer → compute confidence → if below threshold, ask a *targeted* question ("Does this unit have an oil pressure sensor?") rather than "what type is this?"

**Industry context** follows the same three-path logic. It matters because industry determines what "normal" range bounds even are — e.g. a pharma chiller flatlining at a tight setpoint is expected; the same flatline on an HVAC chiller may indicate a stuck control loop.

### 2.3 Range Bound Validation (Two Layers)

**Layer 1 — Physical/absolute bounds** (same for all chillers of a type, from physics/standards): refrigerant pressures bounded by the refrigerant's P-T chart; CHW supply temp can't physically be below freezing (unless glycol) or unrealistically high; these are "impossible value" filters — hard rejects/flags regardless of context.

**Layer 2 — Operational/statistical bounds** (specific to this chiller, from its own history): once chillers are clustered (2.5), compute per-cluster percentile bands (e.g. 1st–99th percentile) as the "expected operating range." This catches drift, fouling, and slow degradation that Layer 1 is too loose to catch.

Keep these separate in the state schema — a value can pass physical bounds but fail statistical bounds, and downstream agents need to know which kind of flag they're looking at.

### 2.4 Artifact Detection Rules

"Spike" is actually at least four different phenomena:

| Pattern | Signature | Meaning |
|---|---|---|
| Sensor reset to zero | Value drops to 0 (or fixed default) instantaneously, correlated parameters unaffected | Power cycle / controller reboot — not a real reading |
| Wire/sensor disconnect | Value flatlines at 0 or a fixed "null" (e.g. -999, 9999, max int) and **stays** there | Hardware fault, not a real reading |
| Counter/register rollover | Value hits a max and wraps to 0 sharply, then climbs normally | Firmware/register behavior, not a physical event |
| Genuine spike (real event) | Value moves sharply AND correlated parameters move consistently with it | Real fault — refrigerant loss, fouling, fire, etc. |

**Key discriminator: cross-parameter correlation, not the single parameter in isolation.** A temp spike accompanied by a coherent move in pressure/current/flow is more likely real; a spike appearing in exactly one parameter while everything else stays flat is almost always an artifact.

**Concrete detection rules:**

- **Known "dead value" codes** — hard-code known sentinel values (0, -1, 9999, 10000, NaN) as instant flags, not statistical ones.
- **Instantaneous jump rate** — compute rate of change between consecutive samples using *actual elapsed time*, not an assumed fixed interval. A jump exceeding the physically possible rate of change for that parameter is a strong artifact signal.
- **Post-reset ramp pattern** — a value that ramps back to normal following an expected startup curve after a reset should be tagged "reset event," not "fault."
- **Duration matters** — a single-sample outlier that self-corrects next sample is usually noise/reset. A sustained deviation across many consecutive samples is much more likely a real, ongoing event and should be tagged as ONE fault window, not one alert per sample.
- **Multi-parameter fire-flag** — for severe events like fire, don't rely on one temp sensor. Compound rule: sustained abnormal temp rise + no corresponding compressor/control action + possible drop in performance elsewhere = escalate immediately, bypass the normal "flag as anomaly and pass downstream" path.

This is really its own mini classification problem: **artifact vs. real anomaly vs. real fault**, each needing different downstream handling (artifact → clean/impute/drop; real anomaly → pass to Anomaly agent; real fault → escalate).

Concrete, dated examples of these patterns as they're actually discovered in live data belong in the **live knowledge store** (Section 4), populated during data preprocessing/validation runs — not in this static reference document.

### 2.5 Clustering Similar Chillers

**Features to cluster on:**
- Static: type, capacity/tonnage, refrigerant, heat rejection method, age/install year, available parameter set (this varies widely fleet-to-fleet and is itself a useful clustering feature)
- Behavioral: typical load factor, daily/seasonal cyclicality strength, average COP/efficiency band, setpoint tightness (variance)

**Approach:** for a manageable fleet size, lightweight k-means or hierarchical clustering on normalized static + behavioral features works well. Cluster assignment is a slow-changing property — cache it, only recompute on new chiller onboarding or major behavior drift, not every cycle. Cluster output feeds directly into Layer 2 statistical bounds (Section 2.3).

### 2.6 Suggested LangGraph Subgraph Structure

```
raw_data_in
   → sentinel_value_check      (hard artifact filter, cheap, first pass)
   → type_identification       (lookup → rule inference → ask client if low confidence)
   → industry_context          (lookup → behavioral inference)
   → cluster_assignment        (lookup cached cluster, or trigger re-cluster if new chiller)
   → bound_check                (physical bounds + cluster-based statistical bounds)
   → artifact_classifier       (reset / disconnect / rollover / real-event, via cross-parameter correlation)
   → severity_router           (normal → pass through; artifact → clean & tag;
                                 real anomaly → flag for Anomaly agent;
                                 critical/fire-pattern → escalate immediately)
   → validated_output + confidence + reasoning trace
```

**Important:** carry a confidence score and a reasoning trace through the whole subgraph, not just a pass/fail flag. The Supervisor node's confidence-aware routing needs this to decide whether to trust the data enough to route it to Forecast/Anomaly, or whether it needs a client clarification loop first.

---

## 3. System Architecture Context

This Data Validation Agent design replaces the "Data validation agent" node in the original pipeline architecture:

```
Data validation agent (sanity bounds, corruption flags)
   ↓
Supervisor (confidence-aware routing) ← Per-chiller memory (quirks, overrides, trust)
   ↓                              ↓
Forecast (predicts a few steps)   Anomaly (deviation from expected)
   ↓                              ↓
        Consensus & skeptic gate (agreement + trust check)
              ↓                        ↓
      Optimization (simulates      Insight / NLG (explains
       before acting)               reasoning, adapts tone)
              ↓                        ↓
              Combined output (dashboard / report)
              ↺ outcomes update per-chiller memory & confidence
```

The richer Data Validation Agent design in Section 2 feeds the Supervisor with not just pass/fail data, but: chiller type/industry classification, cluster assignment, physical + statistical bound results, artifact classification, and a confidence score/reasoning trace — giving the Supervisor much better grounds for its confidence-aware routing decision.

---

## 4. The Live Knowledge Store

This markdown file is the **static** reference — domain knowledge and design decisions that don't change at runtime. It is deliberately separate from the **live, agent-editable knowledge store**, which is a real JSON file the agent reads and writes during actual operation.

### 4.1 Why JSON, not Excel

An earlier version of this project kept the live store as an Excel workbook. That was reconsidered and replaced with JSON for a few concrete reasons:

| Concern | Excel (.xlsx) | JSON |
|---|---|---|
| Write speed | Full workbook (styles, merged cells, formatting) has to be reloaded and rewritten on every update — expensive for frequent small writes | Near-instant; it's just text |
| Nested data | Flat grid — doesn't map cleanly to nested structures like `{machine_id: {type, parameters_tracked: {...}}}` | Native fit for nested records |
| Concurrent/crash safety | File-lock conflicts if open elsewhere; a crash mid-write can corrupt the whole workbook | Atomic write (temp file + rename) — a crash can never leave a half-written file |
| Version control / auditing | Binary format, `git diff` shows nothing useful | Plain text, diffs are readable |
| Read cost for the agent | Needs `openpyxl`/`pandas` to parse first | `json.load()`, or read as raw text |

**Decision:** JSON is the single source of truth the agent reads and writes. If a human-readable spreadsheet view is wanted later (for review, or for showing an instructor), the right pattern is a one-way **export** script that regenerates a formatted `.xlsx` snapshot *from* the JSON on demand — never the reverse, so there's never a risk of the two files drifting out of sync. (Not yet built — noted here as the agreed approach if/when needed.)

For a much larger fleet or heavy concurrent write load, SQLite would be the next step up from JSON (real transactions, queryability). At the current scale (103 chillers, a pattern log in the hundreds not millions of entries), JSON is simpler and sufficient.

### 4.2 The two files

- **`chiller_agent_knowledge.json`** — structured store containing:
  - `static_reference` — a condensed copy of Section 1 of this file (category lists only, for fast in-context lookup without re-reading the whole document)
  - `validation_rules` — sentinel values, physical bound templates, and `artifact_patterns` (starts with generic seed rules; grows via promotion — see 4.3)
  - `chiller_inventory` — **the live equivalent of what used to be separate Excel "Inventory" and "Tracked Parameters" sheets**, merged into one registry keyed by `machine_id`. Each entry holds identifying metadata (name, model number, manufacturer, location, capacity, criticality), the agent's inferred type/industry + confidence, and a `parameters_tracked` map that accumulates per-parameter meaning, unit, observed min/max, and reading count as the agent sees more data — min/max widen over repeated observations rather than being overwritten.
  - `cluster_registry` — populated by the clustering step: cluster membership and per-cluster Layer-2 statistical bounds
  - `learned_patterns` — the real-time "what I've discovered" log
  - **Current state:** the schema is built and tested, but starts empty/seeded — it does not yet contain the 103 chillers or 56 parameters found in the source database. That population is a one-time backfill (looping the known inventory and parameter stats through `register_chiller`/`record_parameter_observation`), after which the same functions get called incrementally as new data flows in.

- **`knowledge_store.py`** — the actual read/write interface. Every write is atomic. Key functions:
  - `register_chiller(machine_id, name=, model_number=, manufacturer=, location=, capacity=, criticality=, chiller_type=, industry=, confidence=, evidence=)` — create or update a chiller's inventory record; only the fields passed are changed, so it's safe to call repeatedly as more is learned about a unit over time.
  - `record_parameter_observation(machine_id, parameter, meaning=, unit=, value=, sanity_status=)` — record that a parameter was seen for a chiller; merges into a running `observed_min`/`observed_max`/`reading_count` rather than overwriting.
  - `get_chiller(machine_id)` / `list_chillers()` — read back inventory records.
  - `add_learned_pattern(pattern, description, rule, machine_id=, parameter=, evidence=)` — append a new discovery, timestamped and ID'd.
  - `promote_pattern(pattern_id)` — move a learned pattern into the stable `validation_rules.artifact_patterns` block.
  - `add_sentinel_value(value)` — register a newly confirmed sentinel/placeholder value.
  - `update_cluster(cluster_id, member_machine_ids, stat_bounds=)` — record clustering output.
  - `get_agent_context(max_learned_patterns=20)` — returns a compact text bundle (static reference summary, sentinel values, stable rules, recent active patterns, inventory summary) sized for direct injection into an agent's prompt.

  These are plain, JSON-serializable functions — each wraps directly as a LangGraph/LangChain tool (an example is included as a comment at the top of the module).

### 4.3 Update discipline, so the store doesn't degrade over time

1. The agent calls `register_chiller(...)` / `record_parameter_observation(...)` as it meets chillers and parameters — never edits `static_reference` directly.
2. The agent calls `add_learned_pattern(...)` when it detects something new during validation — not by editing `validation_rules` directly.
3. A periodic maintenance pass (scheduled, or triggered when a pattern recurs across multiple chillers) calls `promote_pattern(...)` to fold well-established learned patterns into the stable `validation_rules` block, keeping the active `learned_patterns` log short and high-signal.
4. The cluster registry is refreshed on its own cadence (e.g. on new chiller onboarding, or a periodic re-cluster job) rather than per-reading.
