"""
knowledge_store.py

Real read/write interface for the live chiller agent knowledge store
(chiller_agent_knowledge.json). This is the module the Data Validation
Agent (and periodic maintenance jobs) actually call at runtime to read
context and record what it learns.

Design notes:
- Every write is atomic (write to a temp file, then replace) so a crash
  mid-write can never corrupt the store.
- Every function is a plain, synchronous Python function with simple
  JSON-serializable arguments/return values, so each one can be wrapped
  directly as a LangGraph / LangChain tool, e.g.:

    from langchain_core.tools import tool

    @tool
    def record_learned_pattern(pattern: str, description: str, rule: str,
                                machine_id: str = None, evidence: str = None) -> str:
        \"\"\"Record a newly discovered data-quality pattern in the live
        chiller knowledge store.\"\"\"
        entry = add_learned_pattern(pattern=pattern, description=description,
                                     rule=rule, machine_id=machine_id, evidence=evidence)
        return f"Recorded pattern {entry['id']}"

- Nothing here talks to a database directly. The Data Validation Agent's
  preprocessing step is expected to call these functions with whatever it
  found; this module's only job is keeping the knowledge store consistent.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

DEFAULT_STORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "chiller_agent_knowledge.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Core load / save
# ---------------------------------------------------------------------------

def load(path: str = DEFAULT_STORE_PATH) -> dict:
    """Load the knowledge store from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(data: dict, path: str = DEFAULT_STORE_PATH) -> None:
    """Atomically write the knowledge store back to disk.

    Writes to a temp file in the same directory, then os.replace()s it
    over the target — this can never leave a half-written JSON file on
    disk even if the process dies mid-write.
    """
    data["last_updated"] = _now()
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, prefix=".kb_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Learned patterns (the agent's real-time "what I've discovered" log)
# ---------------------------------------------------------------------------

def add_learned_pattern(
    pattern: str,
    description: str,
    rule: str,
    machine_id: Optional[str] = None,
    parameter: Optional[str] = None,
    evidence: Optional[str] = None,
    path: str = DEFAULT_STORE_PATH,
) -> dict:
    """Append a newly discovered data-quality / behavioral pattern.

    Call this the moment the Data Validation Agent notices something new
    and worth remembering — a fault signature, a sentinel value, a
    fleet-specific quirk — during preprocessing or live validation.

    Returns the stored entry (including its generated id and timestamp).
    """
    data = load(path)
    entry = {
        "id": f"lp-{uuid.uuid4().hex[:8]}",
        "discovered": _now(),
        "pattern": pattern,
        "machine_id": machine_id,
        "parameter": parameter,
        "evidence": evidence,
        "description": description,
        "rule": rule,
        "status": "active",
        "promoted_on": None,
    }
    data.setdefault("learned_patterns", []).append(entry)
    save(data, path)
    return entry


def list_learned_patterns(
    status: Optional[str] = "active", path: str = DEFAULT_STORE_PATH
) -> list[dict]:
    """List learned patterns, optionally filtered by status
    ('active' | 'promoted' | None for all)."""
    data = load(path)
    patterns = data.get("learned_patterns", [])
    if status is None:
        return patterns
    return [p for p in patterns if p.get("status") == status]


def promote_pattern(pattern_id: str, path: str = DEFAULT_STORE_PATH) -> dict:
    """Move a learned pattern from the active log into the stable
    validation_rules.artifact_patterns block, and mark it promoted.

    Use this once a pattern has recurred enough (e.g. across multiple
    chillers, or confirmed by a human/maintenance job) to be trusted as
    a standing rule rather than a one-off observation.
    """
    data = load(path)
    patterns = data.get("learned_patterns", [])
    target = None
    for p in patterns:
        if p["id"] == pattern_id:
            target = p
            break
    if target is None:
        raise KeyError(f"No learned pattern with id {pattern_id!r}")

    target["status"] = "promoted"
    target["promoted_on"] = _now()

    stable_entry = {
        "id": pattern_id,
        "pattern": target["pattern"],
        "description": target["description"],
        "rule": target["rule"],
        "status": "promoted",
        "promoted_on": target["promoted_on"],
    }
    data["validation_rules"]["artifact_patterns"]["patterns"].append(stable_entry)
    save(data, path)
    return stable_entry


# ---------------------------------------------------------------------------
# Sentinel values
# ---------------------------------------------------------------------------

def add_sentinel_value(value: float, path: str = DEFAULT_STORE_PATH) -> list:
    """Register a newly confirmed sentinel/placeholder value (e.g. a
    round-number ceiling found on a specific data source)."""
    data = load(path)
    values = data["validation_rules"]["sentinel_values"]["values"]
    if value not in values:
        values.append(value)
        save(data, path)
    return values


# ---------------------------------------------------------------------------
# Chiller inventory (live equivalent of the old Excel "Inventory" +
# "Tracked Parameters" sheets — grows as the agent meets new chillers
# and new parameters on existing chillers)
# ---------------------------------------------------------------------------

def register_chiller(
    machine_id: str,
    name: Optional[str] = None,
    model_number: Optional[str] = None,
    manufacturer: Optional[str] = None,
    location: Optional[str] = None,
    capacity: Optional[str] = None,
    unit: Optional[str] = None,
    criticality: Optional[str] = None,
    chiller_type: Optional[str] = None,
    industry: Optional[str] = None,
    confidence: Optional[float] = None,
    evidence: Optional[str] = None,
    path: str = DEFAULT_STORE_PATH,
) -> dict:
    """Create or update a chiller's inventory record.

    Safe to call repeatedly — only the fields you pass are updated; any
    field left as None keeps its previously stored value (a new record
    starts with None for anything not supplied). This is what the agent
    calls the moment it meets a chiller for the first time, and again
    whenever it learns more about one it already knows (e.g. a model
    number that was blank in the source database gets filled in later).
    """
    data = load(path)
    entries = data["chiller_inventory"]["entries"]
    mid = str(machine_id)
    existing = entries.get(mid, {
        "name": None, "model_number": None, "manufacturer": None,
        "location": None, "capacity": None, "unit": None, "criticality": None,
        "chiller_type": None, "industry": None, "confidence": None, "evidence": None,
        "parameters_tracked": {},
    })

    updates = {
        "name": name, "model_number": model_number, "manufacturer": manufacturer,
        "location": location, "capacity": capacity, "unit": unit, "criticality": criticality,
        "chiller_type": chiller_type, "industry": industry,
        "confidence": confidence, "evidence": evidence,
    }
    for k, v in updates.items():
        if v is not None:
            existing[k] = v
    existing["updated"] = _now()

    entries[mid] = existing
    save(data, path)
    return existing


def record_parameter_observation(
    machine_id: str,
    parameter: str,
    meaning: Optional[str] = None,
    unit: Optional[str] = None,
    value: Optional[float] = None,
    sanity_status: Optional[str] = None,
    path: str = DEFAULT_STORE_PATH,
) -> dict:
    """Record that a parameter was observed for a chiller, updating its
    running min/max/count rather than overwriting — so repeated calls
    build up the observed range over time instead of losing history.

    Call this from preprocessing/validation whenever a new parameter is
    seen for a chiller (first time), or periodically with a fresh value
    to keep the observed range current.
    """
    data = load(path)
    entries = data["chiller_inventory"]["entries"]
    mid = str(machine_id)
    if mid not in entries:
        raise KeyError(f"Chiller {mid!r} not registered yet — call register_chiller() first")

    params = entries[mid].setdefault("parameters_tracked", {})
    existing = params.get(parameter, {
        "meaning": None, "unit": None,
        "observed_min": None, "observed_max": None,
        "reading_count": 0, "sanity_status": "unknown",
    })

    if meaning is not None:
        existing["meaning"] = meaning
    if unit is not None:
        existing["unit"] = unit
    if sanity_status is not None:
        existing["sanity_status"] = sanity_status
    if value is not None:
        existing["observed_min"] = value if existing["observed_min"] is None else min(existing["observed_min"], value)
        existing["observed_max"] = value if existing["observed_max"] is None else max(existing["observed_max"], value)
        existing["reading_count"] += 1

    params[parameter] = existing
    entries[mid]["updated"] = _now()
    save(data, path)
    return existing


def get_chiller(machine_id: str, path: str = DEFAULT_STORE_PATH) -> Optional[dict]:
    """Fetch a single chiller's full inventory record, or None if unknown."""
    data = load(path)
    return data["chiller_inventory"]["entries"].get(str(machine_id))


def list_chillers(path: str = DEFAULT_STORE_PATH) -> dict:
    """Return the full chiller_inventory entries dict."""
    return load(path)["chiller_inventory"]["entries"]


def update_cluster(
    cluster_id: str,
    member_machine_ids: list[str],
    stat_bounds: Optional[dict] = None,
    path: str = DEFAULT_STORE_PATH,
) -> dict:
    """Record (or update) a cluster's membership and its Layer-2
    statistical bounds, as produced by the clustering step."""
    data = load(path)
    entry = {
        "members": member_machine_ids,
        "stat_bounds": stat_bounds or {},
        "updated": _now(),
    }
    data["cluster_registry"]["entries"][str(cluster_id)] = entry
    save(data, path)
    return entry


# ---------------------------------------------------------------------------
# Agent-facing context bundle
# ---------------------------------------------------------------------------

def get_agent_context(max_learned_patterns: int = 20, path: str = DEFAULT_STORE_PATH) -> str:
    """Produce a compact, human/LLM-readable text bundle of the current
    live knowledge store, suitable for injecting straight into an
    agent's prompt context. Keeps token usage bounded by only including
    the most recent active learned patterns.
    """
    data = load(path)
    lines = []
    lines.append(f"# Chiller Agent Knowledge (live store, last updated {data.get('last_updated')})")

    ref = data.get("static_reference", {})
    lines.append("\n## Static reference summary")
    lines.append(f"- Chiller types tracked: {ref.get('chiller_types')}")
    lines.append(f"- Industries tracked: {ref.get('industries')}")
    lines.append(f"- Full domain reference: {ref.get('full_reference_doc')}")

    rules = data.get("validation_rules", {})
    sentinels = rules.get("sentinel_values", {}).get("values", [])
    lines.append(f"\n## Sentinel values (treat as artifacts): {sentinels}")

    stable_patterns = rules.get("artifact_patterns", {}).get("patterns", [])
    lines.append(f"\n## Stable artifact-detection rules ({len(stable_patterns)})")
    for p in stable_patterns:
        lines.append(f"- [{p['id']}] {p['pattern']}: {p['description']} -> {p['rule']}")

    active = [p for p in data.get("learned_patterns", []) if p.get("status") == "active"]
    active_sorted = sorted(active, key=lambda p: p["discovered"], reverse=True)[:max_learned_patterns]
    lines.append(f"\n## Recent active learned patterns ({len(active_sorted)} of {len(active)} total)")
    for p in active_sorted:
        loc = f"machine {p.get('machine_id')}" if p.get("machine_id") else "fleet-wide"
        lines.append(f"- [{p['id']}] ({loc}, {p.get('parameter') or 'n/a'}) {p['pattern']}: {p['description']} -> {p['rule']}")

    inventory = data.get("chiller_inventory", {}).get("entries", {})
    n_clusters = len(data.get("cluster_registry", {}).get("entries", {}))
    lines.append(f"\n## Chiller inventory: {len(inventory)} chillers known, {n_clusters} clusters formed")
    for mid, rec in list(inventory.items())[:10]:
        n_params = len(rec.get("parameters_tracked", {}))
        lines.append(f"- Chiller {mid}: {rec.get('name') or 'unnamed'} | {rec.get('chiller_type') or 'type unknown'} "
                      f"| {rec.get('manufacturer') or 'mfr unknown'} | {n_params} parameter(s) tracked")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test / demo — proves the round trip actually works end to end.
# Run directly: python knowledge_store.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil

    demo_path = os.path.join(tempfile.gettempdir(), "chiller_agent_knowledge_demo.json")
    shutil.copy(DEFAULT_STORE_PATH, demo_path)

    print("1) Loading store...")
    data = load(demo_path)
    print(f"   schema_version={data['schema_version']}, learned_patterns={len(data['learned_patterns'])}")

    print("\n2) Adding a learned pattern (simulating a Data Validation Agent finding)...")
    entry = add_learned_pattern(
        pattern="monotonic_ramp_then_reset",
        description="inlet_temperature climbed linearly from ~50 to ~390000 over 5.5 months, then reset to a normal value in one step.",
        rule="Flag the whole ramp window as one fault event; treat the single-step drop as recovery, not a new anomaly.",
        machine_id="2827",
        parameter="inlet_temperature",
        evidence="2026-01-01: 51.7 -> 2026-06-20 10:00: 392373.2 -> 2026-06-20 10:15: 22.6",
        path=demo_path,
    )
    print(f"   Added: {entry['id']}")

    print("\n3) Reloading from disk to confirm the write actually persisted...")
    reloaded = load(demo_path)
    assert len(reloaded["learned_patterns"]) == 1, "write did not persist!"
    print(f"   Confirmed: {len(reloaded['learned_patterns'])} pattern(s) on disk")

    print("\n4) Registering a sentinel value discovery...")
    values = add_sentinel_value(9999, path=demo_path)
    print(f"   Sentinel values now: {values}")

    print("\n5) Registering a chiller in the inventory...")
    register_chiller("2827", name="Chiller-2827", model_number=None, manufacturer=None,
                      location="DataCenterView_MumbaiDC1_Terrace",
                      chiller_type="Screw", industry="Data Centers",
                      confidence=0.82, evidence="param set matches screw-chiller fingerprint",
                      path=demo_path)
    record_parameter_observation("2827", "inlet_temperature", meaning="Return water temp entering the chiller",
                                  unit="degC", value=22.6, sanity_status="ok", path=demo_path)
    record_parameter_observation("2827", "inlet_temperature", value=23.1, path=demo_path)
    print("   Registered chiller 2827 with 1 tracked parameter (2 observations)")

    print("\n6) Promoting the pattern into stable rules...")
    promoted = promote_pattern(entry["id"], path=demo_path)
    print(f"   Promoted: {promoted['id']} -> status={promoted['status']}")

    print("\n7) Fetching agent-facing context bundle...")
    ctx = get_agent_context(path=demo_path)
    print("-" * 70)
    print(ctx)
    print("-" * 70)

    os.remove(demo_path)
    print("\nAll checks passed. Demo file cleaned up (real store was never touched).")
