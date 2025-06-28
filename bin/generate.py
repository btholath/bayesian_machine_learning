import os
import zipfile
import requests
import pandas as pd
import argparse
import glob
import logging

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="🪵 %(levelname)s: %(message)s"
)

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Generate consumer spending dataset from CE PUMD CSV ZIP.")
parser.add_argument("--skip-download", action="store_true", help="Skip downloading ZIP if it's already in data_source/")
args = parser.parse_args()

# --- Paths and Config ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_SOURCE_DIR = os.path.join(BASE_DIR, "data_source")
ZIP_NAME = "intrvw23.zip"
ZIP_PATH = os.path.join(DATA_SOURCE_DIR, ZIP_NAME)
EXTRACT_DIR = os.path.join(DATA_DIR, "intrvw23")
OUTPUT_CSV = os.path.join(DATA_DIR, "consumer_spending.csv")
CSV_DOWNLOAD_URL = "https://www.bls.gov/cex/pumd/data/csv/intrvw23.zip"

TARGET_EXPN_FILES = ["rnt23.csv", "utp23.csv", "xpb23.csv"]
SPENDING_COLS = {"COST", "VALUE", "AMOUNT", "EXPNAMT", "DOLAMT", "VAL", "EXPNS"}

# --- Download ZIP ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_SOURCE_DIR, exist_ok=True)

if not args.skip_download:
    if not os.path.exists(ZIP_PATH):
        logging.info("Downloading CE Interview ZIP...")
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(CSV_DOWNLOAD_URL, headers=headers)
        r.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            f.write(r.content)
    else:
        logging.info("ZIP already exists in data_source/")
else:
    logging.info("Skipping download step as requested.")

# --- Extract ZIP ---
logging.info("Extracting ZIP and flattening nested folders...")
if os.path.exists(EXTRACT_DIR):
    for root, dirs, files in os.walk(EXTRACT_DIR, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for d in dirs:
            os.rmdir(os.path.join(root, d))

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    for member in zip_ref.namelist():
        if member.endswith("/"):
            continue
        flattened_name = os.path.relpath(member, start=os.path.commonpath([member, "intrvw23"]))
        target_path = os.path.join(EXTRACT_DIR, flattened_name)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with zip_ref.open(member) as source, open(target_path, "wb") as target:
            target.write(source.read())

# --- Scan and Load FMLI ---
all_csvs = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)
logging.info(f"Found {len(all_csvs)} CSV files in extracted data.")

fmli_path = next((f for f in all_csvs if "fmli" in os.path.basename(f).lower()), None)
if not fmli_path:
    raise FileNotFoundError("❌ Could not locate FMLI CSV file.")
logging.info(f"Using FMLI file: {os.path.basename(fmli_path)}")
fmli = pd.read_csv(fmli_path)

# --- Load and Aggregate Spending ---
all_expns = []
for fname in TARGET_EXPN_FILES:
    fpath = next((f for f in all_csvs if os.path.basename(f).lower() == fname.lower()), None)
    if not fpath:
        logging.warning(f"EXPN file not found: {fname}")
        continue

    df = pd.read_csv(fpath)
    logging.info(f"{fname} columns: {df.columns.tolist()}")
    col = next((c for c in df.columns if c.upper() in SPENDING_COLS), None)

    if not col or "NEWID" not in df.columns:
        logging.warning(f"Skipping {fname}: missing NEWID or valid amount column")
        continue

    df[col] = pd.to_numeric(df[col], errors="coerce")
    agg = df.groupby("NEWID")[col].sum().reset_index()
    agg.columns = ["NEWID", f"spending_{fname.split('.')[0]}"]
    all_expns.append(agg)
    logging.info(f"✅ Aggregated from {fname} using column '{col}'")

if not all_expns:
    raise ValueError("❌ No valid EXPN files with usable spending data.")

# --- Merge EXPNS ---
spending_merged = all_expns[0]
for df in all_expns[1:]:
    spending_merged = pd.merge(spending_merged, df, on="NEWID", how="outer")
spending_merged.fillna(0, inplace=True)
spending_merged["TOTAL_SPENDING"] = spending_merged.drop(columns=["NEWID"]).sum(axis=1)

# --- Merge with FMLI ---
merged = pd.merge(fmli, spending_merged, on="NEWID")
merged = merged.dropna(subset=["TOTAL_SPENDING"])

# --- Categorize Spending ---
try:
    merged["spending_category"] = pd.qcut(
        merged["TOTAL_SPENDING"], q=3, labels=["Low", "Medium", "High"], duplicates="drop"
    )
except ValueError:
    merged["spending_category"] = "Uncategorized"
    logging.warning("Could not compute quantiles for TOTAL_SPENDING.")

# --- Rename & Clean ---
final_df = merged[[
    "NEWID", "AGE_REF", "EDUCA2", "REGION", "INCOMEY2", "TOTAL_SPENDING", "spending_category"
]].dropna()

final_df.rename(columns={
    "NEWID": "consumer_unit_id",
    "AGE_REF": "age_of_reference_person",
    "EDUCA2": "education_level",
    "REGION": "region_code",
    "INCOMEY2": "income_range_code",
    "TOTAL_SPENDING": "total_annual_spending",
    "spending_category": "spending_class"
}, inplace=True)

final_df["education_level"] = final_df["education_level"].map({
    31: "Less than HS", 32: "HS Graduate", 33: "Some College",
    34: "Associate’s Degree", 35: "Bachelor’s Degree", 36: "Advanced Degree"
})

final_df["region_code"] = final_df["region_code"].map({
    1: "Northeast", 2: "Midwest", 3: "South", 4: "West"
})

# --- Save & Report ---
final_df.to_csv(OUTPUT_CSV, index=False)
logging.info(f"🎉 consumer_spending.csv saved to: {OUTPUT_CSV}")
logging.info("\n📊 Sample rows:\n%s", final_df.head().to_string(index=False))
logging.info("\n📈 Spending class distribution:\n%s", final_df['spending_class'].value_counts())