import pandas as pd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import numpy as np
from scipy import stats

# Setup paths
base_dir = Path("/mnt/z/Ultrasound-BMd/data/oisin/knot_density_results/main_experiments/estimation_error/S/4_24_32_L2")

# Collect data
weights = []
train_losses_mean = []
test_losses_mean = []
train_losses_sem = []
test_losses_sem = []

# Process each run directory
for run_dir in base_dir.glob("*"):
    if not run_dir.is_dir():
        continue
        
    # Read L2 weight from config
    with open(run_dir / "config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        weight = config['LISTA']['regularization']['weight']
    
    # Read all results from parameters.csv
    params_df = pd.read_csv(run_dir / "parameters.csv")
    
    # Calculate means and standard errors
    train_mean = params_df['train_loss_end'].mean()
    test_mean = params_df['test_loss_end'].mean()
    train_sem = stats.sem(params_df['train_loss_end'])
    test_sem = stats.sem(params_df['test_loss_end'])
    
    weights.append(weight)
    train_losses_mean.append(train_mean)
    test_losses_mean.append(test_mean)
    train_losses_sem.append(train_sem)
    test_losses_sem.append(test_sem)

# Sort by weights
sort_idx = np.argsort(weights)
weights = np.array(weights)[sort_idx]
train_losses_mean = np.array(train_losses_mean)[sort_idx]
test_losses_mean = np.array(test_losses_mean)[sort_idx]
train_losses_sem = np.array(train_losses_sem)[sort_idx]
test_losses_sem = np.array(test_losses_sem)[sort_idx]

# Create plot
plt.figure(figsize=(10, 6))
plt.errorbar(weights, train_losses_mean, yerr=train_losses_sem, fmt='b-', capsize=3, label='Train Loss')
plt.errorbar(weights, test_losses_mean, yerr=test_losses_sem, fmt='r-', capsize=3, label='Test Loss')
plt.xscale('log')
# plt.yscale('log')
plt.xlabel('L2 Penalty Weight')
plt.ylabel('Loss')
plt.title('Train and Test Loss vs L2 Penalty Weight (Mean ± SEM)')
plt.legend()
plt.grid(True)

# Save plot
plt.savefig('reg_weights_vs_losses.png')
plt.close()
print(f"✅ Saved to reg_weights_vs_losses.png")