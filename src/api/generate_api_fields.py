import pandas as pd
from pathlib import Path

TRAIN_CSV = Path("data/processed/train_features_final.csv")
df = pd.read_csv(TRAIN_CSV)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if "TARGET" in numeric_cols:
    numeric_cols.remove("TARGET")

print("class ApplicantData(BaseModel):")
for col in numeric_cols:
    # ensure valid Python identifier
    field = col.strip().replace(" ", "_").replace("-", "_")
    print(f"    {field}: float")
