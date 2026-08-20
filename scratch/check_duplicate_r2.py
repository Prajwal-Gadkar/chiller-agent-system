import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
summary_path = os.path.join(PROJECT_ROOT, "data", "anomaly_agent_fit_summary.csv")

df = pd.read_csv(summary_path)
print(f"Total model summary rows: {len(df)}")
print(f"Distinct R2_CV values: {df['R2_CV'].nunique()}")

# Find any duplicate R2_CV values
dups = df[df.duplicated(subset=["R2_CV"], keep=False)]
print("\n=== DUPLICATE R2_CV ROWS IN FLEET SUMMARY ===")
print(dups[["machineId", "R2_CV", "RMSE_CV", "n_samples"]].to_string())
