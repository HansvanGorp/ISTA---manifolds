from pathlib import Path
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/ISTA---manifolds/ieee.mplstyle')

color_map = {
    'K=4, M=24, N=32': 'blue',
    'K=8, M=64, N=64': 'red',
}

linestyle_map = {
    'K=4, M=24, N=32': 'dashed',
    'K=8, M=64, N=64': 'dotted',
}

marker_map = {
    'K=4, M=24, N=32': '.',
    'K=8, M=64, N=64': 'x',
}

if __name__ == "__main__": 
    BASE_EXPERIMENT_ROOTS = {
        'K=4, M=24, N=32': Path("/ISTA---manifolds/knot_denisty_results/review_response/4_24_32_num_folds"),
        'K=8, M=64, N=64': Path("/ISTA---manifolds/knot_denisty_results/review_response/8_64_64_num_folds"),
    }   
    for experiment_name, base_experiment_root in BASE_EXPERIMENT_ROOTS.items():
        child_dirs = [path for path in base_experiment_root.iterdir() if path.is_dir()]

        # get data from experiment result files
        knot_densities_mean = []
        knot_densities_stderr = []
        num_folds = []
        for path in child_dirs:
            with open(path / "config.yaml", 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            num_folds.append(config['LISTA']['nr_folds'])
            knot_density_df = pd.read_csv(path / "parameters.csv")['knot_density_end']
            knot_densities_mean.append(knot_density_df.mean())
            knot_densities_stderr.append(knot_density_df.std() / np.sqrt(len(knot_density_df)))
        
        num_folds_sort_indices = np.argsort(num_folds)
        num_folds_sorted = np.array(num_folds)[num_folds_sort_indices]
        knot_density_means_sorted = np.array(knot_densities_mean)[num_folds_sort_indices]
        knot_densities_stderr_sorted = np.array(knot_densities_stderr)[num_folds_sort_indices]
        ci = 1.96 * knot_densities_stderr_sorted
        
        # plot
        plt.plot(num_folds_sorted, knot_density_means_sorted, label=experiment_name, color=color_map[experiment_name], linestyle=linestyle_map[experiment_name], marker=marker_map[experiment_name])
        plt.fill_between(num_folds_sorted, knot_density_means_sorted - ci, knot_density_means_sorted + ci, alpha=0.25, color=color_map[experiment_name], label=f'95% CI')
    plt.xlabel("Number of folds")
    plt.ylabel("Knot Density")
    plt.legend(fontsize=7)
    plt.grid(True)
    filename = 'num_folds_vs_knots.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.savefig('num_folds_vs_knots.pdf', bbox_inches='tight', pad_inches=0.1)
    print(f"✅ Saved fig to {filename}")
