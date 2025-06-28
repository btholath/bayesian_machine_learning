import pandas as pd
import bambi as bmb
import matplotlib.pyplot as plt

# Load and selectively clean
df = pd.read_csv("data/consumer_spending.csv")

# Drop only rows missing modeling inputs
df = df.dropna(subset=["age_of_reference_person", "region_code", "income_range_code", "spending_class"])
df["education_level"] = df["education_level"].fillna("Missing")

# Optional: view class distribution
print(df["spending_class"].value_counts())

# Fit Bayesian categorical model
model = bmb.Model(
    "spending_class ~ age_of_reference_person + education_level + region_code + income_range_code",
    df,
    family="categorical"
)

fitted = model.fit(draws=1000, tune=1000, chains=2)
print("✅ Model fit complete.")

# Visualize posterior means
model.plot(fitted)
plt.tight_layout()
plt.show()