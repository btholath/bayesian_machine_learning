import pandas as pd
import numpy as np
from scipy.stats import norm

# Load and clean data
df = pd.read_csv("data/consumer_spending.csv")
df = df.dropna(subset=["total_annual_spending", "income_range_code"])

# Determine Group B (High Income)
cutoff = df["income_range_code"].median()
group_b = df[df["income_range_code"] > cutoff]["total_annual_spending"].values

# Z-test against known population mean
sample_mean = group_b.mean()
sample_size = len(group_b)
population_mean = 12000  # Replace with domain-informed guess
population_std = 3500    # Replace with known or estimated std dev

# Compute Z-statistic
z = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
p_val = 2 * (1 - norm.cdf(abs(z)))

print(f"Z = {z:.2f}")
print(f"p-value = {p_val:.4f}")

if p_val < 0.05:
    print("✅ Statistically significant: High-income group differs from population mean")
else:
    print("⚠️ No significant difference detected")