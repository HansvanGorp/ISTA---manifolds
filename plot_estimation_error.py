from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 14
})

def collect_data(base_dir):
    """Collect metrics from experiment directory"""
    weights = []
    knot_densities_mean = []
    knot_densities_sem = []
    train_losses_mean = []
    test_losses_mean = []
    train_losses_sem = []
    test_losses_sem = []

    for run_dir in base_dir.glob("*"):
        if not run_dir.is_dir():
            continue
            
        with open(run_dir / "config.yaml", 'r') as f:
            weight = yaml.safe_load(f)['LISTA']['regularization']['weight']
        
        params_df = pd.read_csv(run_dir / "parameters.csv")
        
        weights.append(weight)
        knot_densities_mean.append(params_df['knot_density_end'].mean())
        knot_densities_sem.append(stats.sem(params_df['knot_density_end']))
        train_losses_mean.append(params_df['train_loss_end'].mean())
        test_losses_mean.append(params_df['test_loss_end'].mean())
        train_losses_sem.append(stats.sem(params_df['train_loss_end']))
        test_losses_sem.append(stats.sem(params_df['test_loss_end']))

    sort_idx = np.argsort(weights)
    return [arr[sort_idx] for arr in [
        np.array(weights),
        np.array(knot_densities_mean),
        np.array(knot_densities_sem),
        np.array(train_losses_mean),
        np.array(test_losses_mean),
        np.array(train_losses_sem),
        np.array(test_losses_sem)
    ]]

if __name__ == "__main__":
    # Setup paths
    S_dir = Path("/mnt/z/Ultrasound-BMd/data/oisin/knot_density_results/main_experiments/estimation_error/S/4_24_32_L2")
    D_dir = Path("/mnt/z/Ultrasound-BMd/data/oisin/knot_density_results/main_experiments/estimation_error/D/4_24_32_L2")
    
    # Collect data for both experiments
    S_data = collect_data(S_dir)
    D_data = collect_data(D_dir)
    
    # Unpack data
    S_weights, S_knot_mean, S_knot_sem, S_train_mean, S_test_mean, S_train_sem, S_test_sem = S_data
    D_weights, D_knot_mean, D_knot_sem, D_train_mean, D_test_mean, D_train_sem, D_test_sem = D_data
    
    # Find common weights for estimation error
    common_weights = np.intersect1d(S_weights, D_weights)
    S_idx = np.isin(S_weights, common_weights)
    D_idx = np.isin(D_weights, common_weights)
    
    # Calculate estimation error
    est_error = S_test_mean[S_idx] - D_test_mean[D_idx]
    est_error_sem = np.sqrt(S_test_sem[S_idx]**2 + D_test_sem[D_idx]**2)
    
    # Create plot with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(hspace=0.1)
    
    # Plot knot density
    ax1.errorbar(S_weights, S_knot_mean, yerr=S_knot_sem, fmt='-', color='blue', 
                 capsize=3, label='Small Dataset', marker='o')
    ax1.errorbar(D_weights, D_knot_mean, yerr=D_knot_sem, fmt='--', color='red',
                 capsize=3, label='Large Dataset', marker='s')
    ax1.set_ylabel('Knot Density')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False)
    
    # Plot losses
    ax2.errorbar(S_weights, S_train_mean, yerr=S_train_sem, fmt='-', color='blue',
                 capsize=3, label='Small Dataset (Train)', marker='o')
    ax2.errorbar(S_weights, S_test_mean, yerr=S_test_sem, fmt=':', color='blue',
                 capsize=3, label='Small Dataset (Test)', marker='s')
    ax2.errorbar(D_weights, D_train_mean, yerr=D_train_sem, fmt='-', color='red',
                 capsize=3, label='Large Dataset (Train)', marker='^')
    ax2.errorbar(D_weights, D_test_mean, yerr=D_test_sem, fmt=':', color='red',
                 capsize=3, label='Large Dataset (Test)', marker='v')
    ax2.set_ylabel('L1 Reconstruction Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False)
    
    # Plot estimation error
    ax3.errorbar(common_weights, est_error, yerr=est_error_sem, fmt='-', color='purple',
                 capsize=3, label='Estimation Error', marker='o')
    ax3.set_xlabel('L2 Penalty Weight')
    ax3.set_ylabel('Estimation Error')
    ax3.grid(True, alpha=0.3)
    ax3.legend(frameon=False)
    
    outpath = "combined_analysis.png"
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved to {outpath}")