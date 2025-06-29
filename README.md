# bayesian_machine_learning
Practice traditional and bayesian A/B Testing

(.venv) @btholath ➜ /workspaces/bayesian_machine_learning (main) $
python bin/generate_consumer_spending_dataset.py --skip-download     # uses existing ZIP from /data_source/


@btholath ➜ /workspaces/bayesian_machine_learning (main) $ cat /etc/os-release 
PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
NAME="Debian GNU/Linux"
VERSION_ID="12"
VERSION="12 (bookworm)"
VERSION_CODENAME=bookworm
ID=debian
HOME_URL="https://www.debian.org/"
SUPPORT_URL="https://www.debian.org/support"
BUG_REPORT_URL="https://bugs.debian.org/"
@btholath ➜ /workspaces/bayesian_machine_learning (main) $ python --version
Python 3.11.13
@btholath ➜ /workspaces/bayesian_machine_learning (main) $ python -m venv .venv
@btholath ➜ /workspaces/bayesian_machine_learning (main) $ 
@btholath ➜ /workspaces/bayesian_machine_learning (main) $ source .venv/bin/activate
(.venv) @btholath ➜ /workspaces/bayesian_machine_learning (main) $ 
(.venv) @btholath ➜ /workspaces/bayesian_machine_learning (main) $ deactivate

# 📁 1. Navigate to your home directory
cd ~

# ⬇️ 2. Download the Miniconda installer for Linux (Python 3.11 version)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 🛠️ 3. Run the installer
bash Miniconda3-latest-Linux-x86_64.sh

# Accept the license, press ENTER to confirm defaults (recommend ~/miniconda3 as install path)

# ♻️ 4. Restart your shell or run:
source ~/miniconda3/etc/profile.d/conda.sh

# ✅ 5. (Optional) Automatically initialize Conda on shell startup
conda init bash

 6. Create Your Conda Environment
Let's spin up a fresh environment with all your favorite Bayesian ML tools:
conda create -n bayesian_env -c conda-forge python=3.11 bambi pymc arviz matplotlib seaborn scikit-learn pandas numpy


 7. Activate the Environment
conda activate bayesian_env


This puts you in your new Python playground.

🧼 8. (Optional but helpful) Check Python + Package Versions
which python
python --version
conda list


This confirms that you're using Conda's Python 3.11 and everything is in place.

Now you're ready to re-run your bayesian_pymc_model.py, ab_test_bayesian_spending.py, or any other script in your project — no linker errors, just modeling bliss.
If you’d like to auto-activate this environment when the Codespace launches or connect it to a Jupyter kernel, I can walk you through that too. Just say the word!

You can still install it in your existing environment:
conda activate bayesian_env
conda install -c conda-forge bambi


(base) @btholath ➜ /workspaces/bayesian_machine_learning (main) $ conda activate bayesian_env
(bayesian_env) @btholath ➜ /workspaces/bayesian_machine_learning (main) $ python bin/ab_test_bayesian_spending.py 
22:13:37 | INFO | 🚀 Starting Bayesian A/B test for consumer spending...
22:13:37 | INFO | Cleaning and splitting dataset into groups...
22:13:37 | INFO | Group A samples: 1103 | Group B samples: 384
22:13:37 | INFO | Building and sampling from Bayesian model...
Initializing NUTS using jitter+adapt_diag...
22:13:37 | INFO | Initializing NUTS using jitter+adapt_diag...
Sequential sampling (2 chains in 1 job)
22:13:39 | INFO | Sequential sampling (2 chains in 1 job)
NUTS: [mu_a, mu_b, sigma_a, sigma_b]
22:13:39 | INFO | NUTS: [mu_a, mu_b, sigma_a, sigma_b]
                                                                                                                                   
  Progress                                   Draws   Divergences   Step size   Grad evals   Sampling Speed    Elapsed   Remaining  
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   3000    0             0.87        3            2347.03 draws/s   0:00:01   0:00:00    
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   3000    0             0.79        3            1198.00 draws/s   0:00:02   0:00:00    
                                                                                                                                   
Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 3 seconds.
22:13:42 | INFO | Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 3 seconds.
We recommend running at least 4 chains for robust computation of convergence diagnostics
22:13:42 | INFO | We recommend running at least 4 chains for robust computation of convergence diagnostics
22:13:42 | INFO | Sampling complete.
22:13:42 | INFO | Summarizing posterior...

🧾 Posterior Summary:
        mean    sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
mu_a  11.42  0.03   11.37    11.48        0.0      0.0   6158.80   3115.27    1.0
mu_b  11.70  0.05   11.61    11.79        0.0      0.0   6260.73   3408.33    1.0
diff   0.28  0.06    0.17     0.39        0.0      0.0   6402.30   3561.83    1.0
22:13:42 | INFO | Plotting and saving posterior difference...
22:13:42 | INFO | Plot saved as 'posterior_difference.png'.
(bayesian_env) @btholath ➜ /workspaces/bayesian_machine_learning (main) $ 

You ran an A/B test that asked:
“Do people in the higher income group spend more annually than people in the lower income group?”

But instead of just comparing averages the usual way, you used a Bayesian approach, which means:
- You guessed what might be true (your prior)
- You collected data (actual spending)
- Then you updated your beliefs using that data, ending up with a posterior distribution — a fancy way of saying:
“Here’s what I now believe, and how sure I am.”

# 📊 The Output: What the Table Says
- | Term | What It Means (High School Edition) | 
- | mu_a | Average log spending in Group A (low income) = 11.42 | 
- | mu_b | Average log spending in Group B (high income) = 11.70 | 
- | diff | Difference between B and A = 0.28 (Group B spends more!) | 
- | hdi_3% to hdi_97% | “High Density Interval” – 95% of the likely values are between 0.17 and 0.39 for the difference | 
- | r_hat | Closeness to 1 means: “Yeah, I’m confident this result is stable.” | 


📌 So when you look at diff:
- It’s positive → Group B spends more
- The whole 95% HDI is above 0 → It's very unlikely this difference is just random chance
- Your model’s like:
"Yup. I'm about 95% sure high-income folks spend more... and here’s the range I’m confident about."

# 🖼 And the Plot?
You saved a file called posterior_difference.png, which is like a probability hill showing:
- Where you believe the difference lies
- How sure you are about that belief
The fatter and higher the hill? The more confident your model.

# Posterior_difference.png
# 🎯 What the Chart Shows:
- The curve is your belief about how much more Group B spends than Group A — after seeing the data.
- The center of the hill is around 0.28 → That’s the average difference in log spending between groups.
- The shaded range, from 0.17 to 0.4, is the 95% HDI → "There's a 95% chance the true difference lives here."

#💥 The Mic Drop Moment:
 See that text at the bottom?
 0.0% < 0 < 100.0%

# That means:
- There's virtually no chance Group B spends less than Group A.
- There’s a 100% probability the difference is greater than zero (in Bayesian speak, P(μ_B > μ_A) = 1.0)

# In plain English:
“All the evidence says Group B consistently spends more — not just by a little, but with high confidence.”

