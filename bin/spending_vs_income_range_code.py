"""
Use Case: Spending vs Income Range Code
Suppose you want to model how annual spending depends on income bracket.

"""
import pymc as pm
import pandas as pd
import numpy as np

# Load & clean
df = pd.read_csv("data/consumer_spending.csv").dropna(subset=["total_annual_spending", "income_range_code"])
df["income_std"] = (df["income_range_code"] - df["income_range_code"].mean()) / df["income_range_code"].std()
y = np.log1p(df["total_annual_spending"].values)
x = df["income_std"].values

with pm.Model() as model:
    # --- Joint: define prior for both slope & intercept
    alpha = pm.Normal("alpha", mu=0, sigma=10)   # intercept
    beta = pm.Normal("beta", mu=0, sigma=1)      # slope
    sigma = pm.HalfNormal("sigma", sigma=1)

    # --- Conditional: mean depends on income
    mu = alpha + beta * x

    # --- Observed spending
    spending = pm.Normal("spending", mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True)