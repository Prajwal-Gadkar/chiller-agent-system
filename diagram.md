Diagram 1: The main per-reading pipeline
Step 0 — Per-chiller memory (the starting context, read before anything else)

Before any reasoning happens, the system pulls up what it already knows about this specific chiller — accumulated, chiller-specific history: sensor corruption profiles (e.g. clean temperature sensors vs corrupted status flags), past technician feedback, and baseline model performance. This per-chiller context informs all downstream evaluation steps.

Step 1 — Data Validation Agent

Input: raw incoming sensor reading(s) for this chiller at a given timestamp.
What it does: applies physical sanity bounds (KW, Flow, Inlet/Outlet Temp, DeltaT, Speed, Status). Tags each column as trustworthy or flagged as implausible. Filtering happens strictly at read time without mutating source databases.
Output: validated reading with per-sensor validity flags.

Step 2 — Supervisor

Input: validated reading + per-chiller metadata/memory.
What it does: orchestrates execution across pipeline agents for this chiller. Verifies regime boundaries (must operate within a single regime, e.g. post-2026-01-01) and passes validated readings to the Anomaly Agent and informational Forecast baseline.
Output: reading dispatched to Anomaly Agent and Forecast baseline.

Step 3a — Forecast Agent (Informational Persistence Baseline)

Input: current and recent power readings for this chiller.
What it does: provides a short-term persistence baseline ($Power[t] = Power[t-1]$). Extensive empirical validation showed persistence consistently outperforms ML approaches (delta modeling, driver chaining, lagged load models). Used for context and reference only.
Output: persistence baseline power estimate.

Step 3b — Anomaly Agent (The Validated Foundation)

Input: current validated physical sensors ($Flow, InletTemp, OutletTemp, DeltaT, [CompLoad]$).
What it does: evaluates a per-chiller `RandomForestRegressor` response model ($KW = f(Flow, InletTemp, OutletTemp, DeltaT, [CompLoad])$), trained strictly within a single regime. Computes actual minus predicted power residual and calculates its z-score ($z = (residual - \mu) / \sigma$). Flags $|z| > 3$ as anomalous.
Output: anomaly flag, residual z-score, and predicted vs actual power delta.

Step 4 — Consensus & Skeptic Gate

Input: Anomaly Agent output, Forecast baseline, and Data Validation flags.
What it does: checks if physical anomalies align across sensors and filters co-corruption artifacts (e.g. simultaneous sensor spikes). Ensures anomalies represent real physical deviations rather than sensor failure.
Output: cleared anomaly report with attached confidence and reasoning.

Step 5a — Optimization Agent (Open Item — Not Yet Built)

Input: cleared anomaly report.
What it does: (Open item / future phase). Initial schedule-based waste hypothesis was tested across 5 chillers and showed no structured time-of-day efficiency pattern (noise). Future optimization will focus on physics-based setpoint tuning once validated.
Output: pending future implementation.

Step 5b — Insight Agent

Input: validation flags, anomaly residuals, and consensus gate findings.
What it does: synthesizes technical results into plain English narratives (e.g., explaining why a power reading was flagged as anomalous relative to flow and delta-T physics).
Output: structured human-readable insight summary.

Step 6 — Final Output

Combines consensus gate findings and Insight Agent narratives into system alerts, dashboard updates, or operational reports.


Step 7 — Feedback loop back to memory

What it does: whatever happened this cycle — was the recommendation accepted or dismissed by a human, did the gate hold or clear, was a sensor flagged as corrupted again — gets written back into per-chiller memory, so the next cycle starts a little smarter than this one did.

Diagram 2: The background layer (runs independently, on its own schedule)
Left half — Self-Monitoring Agent

Trigger: not per-reading — runs on a slow, scheduled cadence (weekly, say), independent of the main pipeline.
What it does: for every chiller, compares what the Forecast Agent predicted over the past week against what actually happened — real accuracy metrics (MAPE, R², RMSE), the same discipline we applied by hand throughout this whole chat, now automated and ongoing rather than something a human has to remember to check.
Output → "Flags chiller for retraining": if a chiller's predictions have drifted meaningfully from reality (maybe a physical change happened — a part was replaced, a setpoint policy changed), this agent flags it. That flag feeds back into Diagram 1 — specifically, it lowers that chiller's confidence level at the Supervisor step until it's been retrained and re-verified, exactly the same reliability-gate logic, just re-triggered by drift instead of an initial data check.

Right half — Paired chiller negotiation

Trigger: whenever Chiller A's Optimization Agent (running inside its own copy of Diagram 1) wants to act, and it knows Chiller B is its designated backup/pair.
What it does: before committing, A's Optimization Agent checks in with B's Optimization Agent — is B also about to act on something related? If both are reacting to the same underlying event (say, a shared electrical fault) and both independently decide to change their setpoints, they could end up working against each other. The negotiation step catches that overlap and coordinates a single combined decision instead of two conflicting ones.
Output: a single, agreed-upon action (or an explicit decision that only one of the pair needs to act), which then flows back down into each chiller's own Step 5a