import os
import zipfile
import requests
import pandas as pd
import argparse
import glob

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

# --- Step 1: Ensure Directories Exist ---
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_SOURCE_DIR, exist_ok=True)

# --- Step 2: Download ZIP ---
if not args.skip_download:
    if not os.path.exists(ZIP_PATH):
        print("⬇️ Downloading CE Interview ZIP...")
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(CSV_DOWNLOAD_URL, headers=headers)
        r.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            f.write(r.content)
    else:
        print("📦 ZIP already exists in data_source/")
else:
    print("⏭️ Skipping download step as requested.")

# --- Step 3: Extract and Flatten ZIP Contents ---
print("📂 Extracting ZIP and flattening nested folders...")

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

# --- Step 4: Locate Data Files ---
print("🔍 Scanning for FMLI and EXPN files...")
all_csvs = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)
print(f"📄 Found {len(all_csvs)} CSV files")

# Improved file match logic
fmli_path = next((f for f in all_csvs if "fmli" in os.path.basename(f).lower()), None)
expn_path = next(
    (f for f in all_csvs if any(key in os.path.basename(f).lower() for key in ["expn", "exp23"])),
    None
)

if not fmli_path or not expn_path:
    raise FileNotFoundError("❌ Could not locate required FMLI or EXPN CSV files.")

print(f"✅ Found FMLI file: {os.path.basename(fmli_path)}")
print(f"✅ Found EXPN file: {os.path.basename(expn_path)}")

# --- Step 5: Load and Inspect ---
fmli = pd.read_csv(fmli_path)
expn = pd.read_csv(expn_path)

# Detect expenditure column
print("🔎 Looking for spending column...")
spending_col = next((col for col in expn.columns if col.upper() in {"COST", "VALUE", "AMOUNT", "EXPNAMT"}), None)
if not spending_col:
    raise ValueError("❌ No spending column found (COST, VALUE, AMOUNT, EXPNAMT).")
print(f"✅ Using spending column: {spending_col}")

# --- Step 6: Aggregate and Merge ---
expn_total = expn.groupby("NEWID")[spending_col].sum().reset_index()
expn_total.columns = ["NEWID", "TOTAL_SPENDING"]
merged = pd.merge(fmli, expn_total, on="NEWID")

# --- Step 7: Categorize and Clean ---
merged["spending_category"] = pd.qcut(
    merged["TOTAL_SPENDING"], q=3, labels=["Low", "Medium", "High"]
)

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

# --- Step 8: Human-readable Mappings ---
final_df["education_level"] = final_df["education_level"].map({
    31: "Less than HS", 32: "HS Graduate", 33: "Some College",
    34: "Associate’s Degree", 35: "Bachelor’s Degree", 36: "Advanced Degree"
})

final_df["region_code"] = final_df["region_code"].map({
    1: "Northeast", 2: "Midwest", 3: "South", 4: "West"
})

# --- Step 9: Save Output ---
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"🎉 consumer_spending.csv saved to: {OUTPUT_CSV}")