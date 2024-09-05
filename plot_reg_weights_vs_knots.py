from pathlib import Path
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/ISTA---manifolds/ieee.mplstyle')

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

if __name__ == "__main__": 
    BASE_EXPERIMENT_ROOTS = {
        'K=4, M=24, N=32': Path("/ISTA---manifolds/knot_denisty_results/main_experiments/4_24_32_L2"),
        'K=8, M=64, N=64': Path("/ISTA---manifolds/knot_denisty_results/main_experiments/8_64_64_L2"), 
    }
    regularization_type = "L2"
    crop_after_zeros = 0
    
    for experiment_name, base_experiment_root in BASE_EXPERIMENT_ROOTS.items():
        child_dirs = [path for path in base_experiment_root.iterdir() if path.is_dir()]

        # get data from experiment result files
        knot_densities_mean = []
        knot_densities_stderr = []
        reg_weights = []
        for path in child_dirs:
            with open(path / "config.yaml", 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            reg_weights.append(config['LISTA']['regularization']['weight'])
            knot_density_df = pd.read_csv(path / "parameters.csv")['knot_density_end']
            mean_knot_density = knot_density_df.mean()
            knot_densities_mean.append(mean_knot_density)
            knot_densities_stderr.append(knot_density_df.std() / np.sqrt(len(knot_density_df)))
        
        reg_weights_sort_indices = np.argsort(reg_weights)
        reg_weights_sorted = np.array(reg_weights)[reg_weights_sort_indices]
        knot_density_means_sorted = np.array(knot_densities_mean)[reg_weights_sort_indices]
        knot_densities_stderr_sorted = np.array(knot_densities_stderr)[reg_weights_sort_indices]
        
        # plot
        # reg_weights_sorted = reg_weights_sorted[:-crop_after_zeros]
        # knot_density_means_sorted = knot_density_means_sorted[:-crop_after_zeros]
        # knot_densities_stderr_sorted = knot_densities_stderr_sorted[:-crop_after_zeros]
        ci = 1.96 * knot_densities_stderr_sorted
        plt.plot(reg_weights_sorted, knot_density_means_sorted, label=experiment_name, color=color_map[experiment_name], linestyle=linestyle_map[experiment_name], marker=marker_map[experiment_name])
        plt.fill_between(reg_weights_sorted, knot_density_means_sorted - ci, knot_density_means_sorted + ci, alpha=0.25, color=color_map[experiment_name], label=f'95% CI')
    plt.xlabel(f"{regularization_type} Weight α")
    plt.xscale('log')
    plt.ylabel("Knot Density")
    plt.legend(fontsize=7)
    plt.grid(True)
    filename = f'reg_weights_vs_knots_{regularization_type}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'reg_weights_vs_knots_{regularization_type}.pdf', bbox_inches='tight', pad_inches=0.1)
    print(f"✅ Saved fig to {filename}")
