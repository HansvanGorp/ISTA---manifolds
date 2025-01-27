from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# plt.style.use('/ISTA---manifolds/ieee.mplstyle')
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 16
})

color_map = {
    'K=4, M=24, N=32': 'blue',
    'K=8, M=64, N=64': 'red',

    'Post Training': 'blue',
    'Regular L2': 'red',
    'Regular Jacobian': 'red',
}

linestyle_map = {
    'K=4, M=24, N=32': 'dashed',
    'K=8, M=64, N=64': 'dotted',

    'Post Training': 'dashed',
    'Regular L2': 'dotted',
    'Regular Jacobian': 'dotted',
}

marker_map = {
    'K=4, M=24, N=32': '.',
    'K=8, M=64, N=64': 'x',
 
    'Post Training': '.',
    'Regular L2': 'x',
    'Regular Jacobian': 'x',
}

def collect_data(base_dir):
    weights = []
    knot_densities_mean = []
    knot_densities_sem = []
    train_losses_mean = []
    test_losses_mean = []
    train_losses_sem = []
    test_losses_sem = []
    gen_gaps_mean = []
    gen_gaps_sem = []

    for run_dir in base_dir.glob("*"):
        if not run_dir.is_dir():
            continue
            
        with open(run_dir / "config.yaml", 'r') as f:
            weight = yaml.safe_load(f)['LISTA']['regularization']['weight']
        
        params_df = pd.read_csv(run_dir / "parameters.csv")
        gen_gaps = params_df['test_loss_end'] - params_df['train_loss_end']
        
        weights.append(weight)
        knot_densities_mean.append(params_df['knot_density_end'].mean())
        knot_densities_sem.append(stats.sem(params_df['knot_density_end']))
        train_losses_mean.append(params_df['train_loss_end'].mean())
        test_losses_mean.append(params_df['test_loss_end'].mean())
        train_losses_sem.append(stats.sem(params_df['train_loss_end']))
        test_losses_sem.append(stats.sem(params_df['test_loss_end']))
        gen_gaps_mean.append(gen_gaps.mean())
        gen_gaps_sem.append(stats.sem(gen_gaps))

    sort_idx = np.argsort(weights)
    return [arr[sort_idx] for arr in [
        np.array(weights),
        np.array(knot_densities_mean),
        np.array(knot_densities_sem),
        np.array(train_losses_mean),
        np.array(test_losses_mean),
        np.array(train_losses_sem),
        np.array(test_losses_sem),
        np.array(gen_gaps_mean),
        np.array(gen_gaps_sem)
    ]]

if __name__ == "__main__":
    base_dir = Path("/mnt/z/Ultrasound-BMd/data/oisin/knot_density_results/main_experiments/estimation_error/D/4_24_32_L2")
    
    # Collect and sort data
    weights, knot_mean, knot_sem, train_mean, test_mean, train_sem, test_sem, gap_mean, gap_sem = collect_data(base_dir)
    
    # First plot: Knot density and losses vs L2 weight
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    ax1.errorbar(weights, knot_mean, yerr=knot_sem, fmt='b-', capsize=3, label='Knot Density')
    ax1.set_ylabel('Knot Density')
    ax1.set_xscale('log')
    ax1.grid(True)
    ax1.legend()
    
    ax2.errorbar(weights, train_mean, yerr=train_sem, fmt='g-', capsize=3, label='Train Loss')
    ax2.errorbar(weights, test_mean, yerr=test_sem, fmt='r-', capsize=3, label='Test Loss')
    ax2.set_xlabel('L2 Penalty Weight')
    ax2.set_ylabel('Loss')
    ax2.set_xscale('log')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('reg_weights_vs_metrics_D.png')
    plt.close()

    # Second plot: Correlation between knot density and generalization gap
    fig2, ax = plt.subplots(figsize=(6, 6))
    
    # Calculate correlation and fit line
    correlation = np.corrcoef(knot_mean, gap_mean)[0,1]
    z = np.polyfit(knot_mean, gap_mean, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(knot_mean), max(knot_mean), 100)
    
    # Plot scatter with errorbars and fitted line
    ax.errorbar(knot_mean, gap_mean, fmt='o', color='blue', capsize=3, label='Data')
    ax.plot(x_range, p(x_range), 'r--', label=f'Fit (r={correlation:.2f})')
    
    ax.set_xlabel('Knot Density')
    ax.set_ylabel('Generalization Gap')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('knot_density_vs_gen_gap_D.png')
    plt.close()
