import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_split(path="data/consumer_spending.csv", target_col="spending_class"):
    df = pd.read_csv(path)
    df["education_level"] = df["education_level"].fillna("Missing")
    df = df.dropna(subset=["region_code", "income_range_code", "age_of_reference_person", target_col])

    X = df[["age_of_reference_person", "education_level", "region_code", "income_range_code"]]
    y = df[target_col]

    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

def make_preprocessor():
    categorical = ["education_level", "region_code"]
    numeric = ["age_of_reference_person", "income_range_code"]

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("cat", cat_pipeline, categorical),
        ("num", num_pipeline, numeric)
    ])