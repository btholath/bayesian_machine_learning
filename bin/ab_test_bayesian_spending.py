import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm

# Force PyTensor to run in Python-only mode (safer inside Codespaces)
os.environ["PYTENSOR_FLAGS"] = "mode=FAST_RUN,linker=py"

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def clean_and_split(df: pd.DataFrame):
    logger.info("Cleaning and splitting dataset into groups...")

    df = df.dropna(subset=["total_annual_spending", "spending_class", "income_range_code"])
    cutoff = df["income_range_code"].median()

    group_a = df[df["income_range_code"] <= cutoff]["total_annual_spending"].values
    group_b = df[df["income_range_code"] > cutoff]["total_annual_spending"].values

    # Remove invalid values before log1p
    group_a = group_a[~np.isnan(group_a) & (group_a > -1)]
    group_b = group_b[~np.isnan(group_b) & (group_b > -1)]

    # Optional: reduce skewness
    group_a = np.log1p(group_a)
    group_b = np.log1p(group_b)

    logger.info(f"Group A samples: {len(group_a)} | Group B samples: {len(group_b)}")
    return group_a, group_b

def run_model(group_a, group_b):
    logger.info("Building and sampling from Bayesian model...")

    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=group_a.mean(), sigma=10)
        mu_b = pm.Normal("mu_b", mu=group_b.mean(), sigma=10)
        sigma_a = pm.HalfNormal("sigma_a", sigma=10)
        sigma_b = pm.HalfNormal("sigma_b", sigma=10)

        obs_a = pm.Normal("obs_a", mu=mu_a, sigma=sigma_a, observed=group_a)
        obs_b = pm.Normal("obs_b", mu=mu_b, sigma=sigma_b, observed=group_b)

        diff = pm.Deterministic("diff", mu_b - mu_a)

        trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

    logger.info("Sampling complete.")
    return trace

def summarize_and_plot(trace):
    logger.info("Summarizing posterior...")
    summary = az.summary(trace, var_names=["mu_a", "mu_b", "diff"], round_to=2)
    print("\n🧾 Posterior Summary:\n", summary)

    logger.info("Plotting and saving posterior difference...")
    az.plot_posterior(trace, var_names=["diff"], ref_val=0, hdi_prob=0.95)
    plt.title("Posterior of μ_B - μ_A (Is Group B spending more?)")
    plt.tight_layout()
    plt.savefig("posterior_difference.png", dpi=300, bbox_inches="tight")
    plt.show()
    logger.info("Plot saved as 'posterior_difference.png'.")

if __name__ == "__main__":
    logger.info("🚀 Starting Bayesian A/B test for consumer spending...")
    df = pd.read_csv("data/consumer_spending.csv").sample(frac=1, random_state=42)
    group_a, group_b = clean_and_split(df)
    trace = run_model(group_a, group_b)
    summarize_and_plot(trace)