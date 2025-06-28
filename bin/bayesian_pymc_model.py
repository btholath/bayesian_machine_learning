import pandas as pd
import bambi as bmb

# Load and clean
df = pd.read_csv("data/consumer_spending.csv").dropna()
df["education_level"] = df["education_level"].fillna("Missing")

# Create categorical model
model = bmb.Model(
    "spending_class ~ age_of_reference_person + education_level + region_code + income_range_code",
    df,
    family="categorical"
)

fitted = model.fit(tune=1000, draws=1000)
print("✅ Bayesian model fit complete.")

# Plot posterior means
model.plot(fitted)