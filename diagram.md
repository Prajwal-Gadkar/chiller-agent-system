Diagram 1: The main per-reading pipeline
Step 0 — Per-chiller memory (the starting context, read before anything else)

Before any reasoning happens, the system pulls up what it already knows about this specific chiller — not generic knowledge, but accumulated, chiller-specific history: which sensors on this unit tend to corrupt (we found this varies wildly — some chillers had clean On_Off_Status, others had 15%+ corruption in Common_Alarm_Status), what a technician previously told the system about a past alert ("checked, was a false alarm"), and a running confidence score for how much to trust this chiller's own model versus falling back to the type-level one. This isn't a separate "step" in the flow so much as a lookup that colors everything downstream — think of it as the system's institutional memory for this one piece of equipment.

Step 1 — Data Validation Agent

Input: the raw incoming sensor reading(s) for this chiller at this timestamp.
What it does: applies the sanity-bound check we built in Section 5c — is KW, Flow, or any other reading within a physically plausible range for this kind of equipment? This is the gate that catches the exact problem we spent hours chasing: a Compressor_2_Fan_speed reading of 977,735, or a KW reading in the hundreds of thousands. Crucially, it checks readings individually, not just in aggregate — a single corrupted value shouldn't taint an otherwise-clean reading cycle.
Output: each incoming value gets tagged either "trustworthy" or "flagged as implausible," and that tag travels with the data into every downstream step. Nothing gets silently dropped — everything downstream knows what it's working with.

Step 2 — Supervisor

Input: the validated (and tagged) reading, plus whatever per-chiller memory said about this unit's trust level.
What it does: decides how to route this reading through the rest of the pipeline. This is where the reliability-gate logic from Section 5 lives — if this chiller is one of the ones that cleared our cross-validated reliability bar (real R² of 0.5–0.99), the Supervisor routes it toward the full analytical path with confidence. If this chiller is more like Type 1 (6 of 8 chillers had no reliable relationship), the Supervisor still routes it through the same boxes, but flags the whole request as "low confidence" — meaning downstream agents should apply more caution, and the Consensus & Skeptic Gate (Step 4) should hold a much higher bar before letting anything through.
Output: the reading, tagged with a routing confidence level, dispatched in parallel to both Forecast and Anomaly.

Step 3a — Forecast Agent (runs in parallel with 3b)

Input: current and recent Flow, plus other controllable features (from the instructor's "predict from things you can control" guidance).
What it does: predicts expected power draw a few intervals ahead — not just "what should power be right now," but "what should it be in the near future given how flow is trending." This is a direct extension of the one relationship we proved generalizes under real cross-validation (not the false 0.998 we caught and threw out) — Flow genuinely drives Power for Types 2 and 3.
Output: a predicted power value (with a confidence interval, not just a point estimate) for the next few intervals.

Step 3b — Anomaly Agent (runs in parallel with 3a)

Input: the current actual reading, plus this chiller's historical baseline behavior.
What it does: independently checks whether the current reading is statistically unusual — this is a different question from Forecast's "what will happen next." Anomaly asks "is what's happening right now already outside normal bounds," using a method resistant to the outlier corruption we've documented (something like Isolation Forest, per the real InnoShri AnomalyDetetcionService pattern).
Output: an anomaly flag with its own independent confidence score.

Step 4 — Consensus & Skeptic Gate

Input: Forecast's prediction, Anomaly's flag, and the validation tags from Step 1.
What it does two things, both must pass:

Consensus check — do Forecast and Anomaly actually agree something's worth escalating? If Forecast says "power is tracking exactly as flow predicts" but Anomaly says "this looks unusual," that disagreement itself is informative — it likely means the anomaly isn't really about power/flow physics at all (maybe it's a different sensor acting up), so the gate holds rather than passing a shaky signal forward.
Skeptic check — even if they agree, re-verify: is this the co-corruption pattern from Section 5c (multiple sensors on the same chiller spiking together, which fooled us once already)? Is this chiller currently low-confidence per the Supervisor's routing? Did a technician already dismiss this exact pattern before (checked against memory)?

Output: either "cleared — proceed to action" or "held — not enough confidence to act," with the reasoning for that decision attached (this reasoning is what Insight will later explain in plain English).

Step 5a — Optimization Agent (runs only if Step 4 cleared)

Input: the cleared signal, plus context on what kind of situation this is (efficiency drift, incipient fault, etc.).
What it does: drafts a specific recommendation (e.g. "flow rate is elevated relative to load — check for a stuck valve," or a proactive setpoint suggestion). Before finalizing it, it runs a quick internal What-If simulation on its own suggestion — does the predicted outcome of this specific recommendation actually look like an improvement, checked against the same Forecast relationship, before presenting it as advice.
Output: a vetted recommendation, plus (for real setpoint changes) a request for human approval before execution.

Step 5b — Insight Agent (runs in parallel with 5a)

Input: everything that happened in Steps 1–4 (and 5a's recommendation, if there is one).
What it does: synthesizes the whole reasoning chain into plain English — not just "power is high," but "flow rose 20%, forecast expected power to rise proportionally, actual power matched that, so this tracks known physics rather than a fault" (or the opposite, if the gate held). It also checks alert history for this chiller — the 5th alert this week reads differently (more urgent, less repetitive) than the 1st.
Output: a human-readable narrative, tagged with appropriate urgency.

Step 6 — Combined output

Input: Optimization's recommendation (if any) and Insight's narrative.
What it does: assembles both into whatever the actual deliverable is — dashboard update, report entry, or (in a fuller build) an actual ticket via equipment_health_router.

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