import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["PYTENSOR_FLAGS"] = "mode=FAST_RUN,linker=py"
import arviz as az
import pymc as pm

# --- Load dataset ---
df = pd.read_csv("data/consumer_spending.csv")
df = df.dropna(subset=["total_annual_spending", "spending_class"])
df = df.sample(frac=1, random_state=42)  # Shuffle

# --- Simulate A/B groups ---
# For example, split into Low-income (Group A) vs High-income (Group B)
cutoff = df["income_range_code"].median()
group_a = df[df["income_range_code"] <= cutoff]["total_annual_spending"].values
group_b = df[df["income_range_code"] > cutoff]["total_annual_spending"].values

# Optional: apply log transform to reduce skewness
group_a = np.log1p(group_a)
group_b = np.log1p(group_b)

# --- Bayesian Model ---
with pm.Model() as model:
    mu_a = pm.Normal("mu_a", mu=group_a.mean(), sigma=10)
    mu_b = pm.Normal("mu_b", mu=group_b.mean(), sigma=10)
    sigma_a = pm.HalfNormal("sigma_a", sigma=10)
    sigma_b = pm.HalfNormal("sigma_b", sigma=10)

    obs_a = pm.Normal("obs_a", mu=mu_a, sigma=sigma_a, observed=group_a)
    obs_b = pm.Normal("obs_b", mu=mu_b, sigma=sigma_b, observed=group_b)

    diff = pm.Deterministic("diff", mu_b - mu_a)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

# --- Summary ---
az.summary(trace, var_names=["mu_a", "mu_b", "diff"])

# --- Visualization ---
az.plot_posterior(trace, var_names=["diff"], ref_val=0, hdi_prob=0.95)
plt.title("Posterior of μ_B - μ_A (Is Group B spending more?)")
plt.tight_layout()

# 📸 Save to file before showing
plt.savefig("posterior_difference.png", dpi=300, bbox_inches="tight")
plt.show()

