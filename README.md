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