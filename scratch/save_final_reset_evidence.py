import json
import pandas as pd

evidence = [
    {
        "machine_id": 1657,
        "sensor_id": 8409,
        "column": "CKT_1_RUN_Hours ValueY",
        "ramp_or_peak_start": "2023-08-12 11:59:09",
        "peak_val": 615.0,
        "reset_time": "2023-08-12 12:14:09",
        "reset_val": 0.0,
        "single_step_drop": 615.0,
        "artifact_type": "ramp_reset"
    },
    {
        "machine_id": 1657,
        "sensor_id": 8415,
        "column": "KW ValueY",
        "ramp_or_peak_start": "2024-09-22 13:10:54",
        "peak_val": 1023.0,
        "reset_time": "2024-09-22 13:25:54",
        "reset_val": 0.0,
        "single_step_drop": 1023.0,
        "artifact_type": "ramp_reset"
    },
    {
        "machine_id": 2761,
        "sensor_id": 21301,
        "column": "inlet_temperature ValueY",
        "ramp_or_peak_start": "2026-01-01 06:00:00",
        "peak_val": 89.478,
        "reset_time": "2026-01-01 06:45:00",
        "reset_val": 21.236,
        "single_step_drop": 68.242,
        "artifact_type": "ramp_reset"
    },
    {
        "machine_id": 2761,
        "sensor_id": 21300,
        "column": "Outlet_temperature ValueY",
        "ramp_or_peak_start": "2026-01-01 06:00:00",
        "peak_val": 93.635,
        "reset_time": "2026-01-01 06:45:00",
        "reset_val": 20.453,
        "single_step_drop": 73.182,
        "artifact_type": "ramp_reset"
    }
]

df = pd.DataFrame(evidence)
df.to_csv("data/reset_evidence.csv", index=False)
with open("data/reset_evidence.json", "w") as f:
    json.dump(evidence, f, indent=2)

print("Saved data/reset_evidence.csv and data/reset_evidence.json cleanly.")
