import os
import glob
import pandas as pd
import logging

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="🧭 %(levelname)s: %(message)s"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "intrvw23")
OUTPUT_FILE = os.path.join(BASE_DIR, "column_index_report.csv")

# --- Collect CSVs ---
csv_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
logging.info(f"Found {len(csv_paths)} CSV files under /data/intrvw23")

# --- Index Collector ---
summary = []

for path in csv_paths:
    try:
        sample = pd.read_csv(path, nrows=5)
        num_rows = sum(1 for _ in open(path)) - 1  # subtract header
        coltypes = {col: str(sample[col].dtype) for col in sample.columns}
        summary.append({
            "file": os.path.basename(path),
            "columns": ", ".join(sample.columns),
            "row_count_est": num_rows,
            "column_types": "; ".join([f"{k}:{v}" for k, v in coltypes.items()])
        })
    except Exception as e:
        logging.warning(f"Could not read {os.path.basename(path)}: {e}")
        continue

# --- Save Output ---
df = pd.DataFrame(summary)
df.sort_values(by="file").to_csv(OUTPUT_FILE, index=False)
logging.info(f"📋 Column summary written to: {OUTPUT_FILE}")