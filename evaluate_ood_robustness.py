"""
Evaluate loss under test set with various perturbations to make it OOD
"""

# %% imports
# standard library imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('/ISTA---manifolds/ieee.mplstyle')
import os
import numpy as np
import yaml
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict

# local imports
import ista
from training import get_loss_on_dataset_over_folds, get_support_accuracy_on_dataset_over_folds
from data import ISTAData
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from matplotlib import colormaps
from matplotlib.colorbar import ColorbarBase

def parse_args():
    parser = argparse.ArgumentParser(description="Main experiment arguments")
    parser.add_argument(
        "-b",
        "--output_dir",
        type=str,
        default="/ISTA---manifolds/knot_denisty_results/review_response/plots",
        help="Path to dir containing original experiment results",
    )
    parser.add_argument(
        "-s",
        "--sweep_root",
        type=str,
        default="/ISTA---manifolds/knot_denisty_results/main_experiments/4_24_32_L2",
        help="Path to dir containing post-training results",
    )
    parser.add_argument(
        "-p",
        "--perturbation",
        type=str,
        default="noise",
        help="Which kind of perturbation to apply to the test data to make it OOD",
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="loss", # | "accuracy"
        help="Which metric to check",
    )
    return parser.parse_args()
args = parse_args()

colors = {
    "ISTA": "tab:blue", 
    "FISTA": "tab:red",
    "LISTA": "tab:orange", 
    "RLISTA": "tab:green",
    "ToeplitzLISTA": "tab:olive",
    "LISTA_L2": "tab:gray",
    "LISTA_Jacobian": "tab:pink",
}

model_type_to_reg_type = {
    "LISTA_L2": "L2",
    "LISTA_Jacobian": "jacobian",
}

def run_experiment(config, model_name, experiment_root, A_matrices, train_datasets, test_datasets): 
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    df = pd.DataFrame()

    results = []

    assert len(train_datasets) == config["max_nr_of_experiments"]
    assert len(test_datasets) == config["max_nr_of_experiments"]
    assert len(A_matrices) == config["max_nr_of_experiments"]

    for experiment_id in tqdm(range(config["max_nr_of_experiments"]), position=0, desc="running experiments", leave=True):
        experiment_run_path = experiment_root / str(experiment_id)
        state_dict_path = experiment_run_path / f"{model_name}/{model_name}_state_dict.tar"
        train_data = train_datasets[experiment_id]
        test_data = test_datasets[experiment_id]
        A = A_matrices[experiment_id]

        model_config = config["LISTA"]

        # create the model using the parameters in the config file
        model = ista.LISTA(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = model_config["nr_folds"], 
                        device = config["device"], initialize_randomly = False, share_weights=model_config["share_weights"], train_inputs = torch.stack([y for (y, _) in train_data]))
        
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        
        # apply OOD transform to test_data first
        if args.metric == "loss":
            result = get_loss_on_dataset_over_folds(model, test_data, l1_weight=1.0, l2_weight=0.0)
        elif args.metric == "accuracy":
            result = get_support_accuracy_on_dataset_over_folds(model, test_data)
        results.append(result.detach().cpu().numpy())
    
    # return mean and std err of losses
    return np.mean(results, axis=0)[-1], (np.std(results, axis=0)[-1] / np.sqrt(len(results)))

def perturb_dataset_with_measurement_noise(dataset: ISTAData, noise_std):
    dataset.y = dataset.y + (torch.randn_like(dataset.y) * noise_std)
    return dataset

def perturb_dataset_with_model_noise(dataset: ISTAData, noise_std):
    # y = (A+E)x + n
    # --> y - Ex = Ax + n
    model_noise_matrix = (torch.randn((dataset.y.shape[1], dataset.x.shape[1])) * noise_std)
    dataset.y = dataset.y - (dataset.x @ model_noise_matrix.T)
    return dataset

perturbation_fns = {
    'perturb_dataset_with_model_noise': perturb_dataset_with_model_noise,
    'perturb_dataset_with_measurement_noise': perturb_dataset_with_measurement_noise
}

if __name__ == "__main__": 
    OUTPUT_DIR = Path(args.output_dir)
    SWEEP_ROOT = Path(args.sweep_root)
    ZERO_COLORBAR = False
    
    perturbation_fn_name = 'perturb_dataset_with_measurement_noise'
    chosen_perturbation_fn = perturbation_fns[perturbation_fn_name]
        
    noise_levels = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125] 
    filename_append = "8_64_64"
    perturbations = {
        'noise=0.0': lambda data: chosen_perturbation_fn(data, 0.0),
        'noise=0.025': lambda data: chosen_perturbation_fn(data, 0.025),
        'noise=0.05': lambda data: chosen_perturbation_fn(data, 0.05),
        'noise=0.075': lambda data: chosen_perturbation_fn(data, 0.075),
        'noise=0.1': lambda data: chosen_perturbation_fn(data, 0.1),
        'noise=0.125': lambda data: chosen_perturbation_fn(data, 0.125),
    }
    if "4_24_32_L2" in args.sweep_root:
        noise_levels.append(0.15)
        perturbations['noise=0.15'] = lambda data: chosen_perturbation_fn(data, 0.15)
        filename_append = "4_24_32"
    
    # Plotting
    # plt.figure(figsize=(10, 6))  # Set figure size
    
    per_model_results = defaultdict(lambda: {'mean': [], 'stderr': [], 'knot_density': []})
    
    for name, perturbation_fn in perturbations.items():
        mean_knot_densities = []
        mean_results = []
        stderr_results = []
        for experiment_root in [Path(os.path.join(SWEEP_ROOT, name)) for name in os.listdir(SWEEP_ROOT) if name != "cache"]:
            # load data
            with open(experiment_root / "config.yaml", 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            train_datasets = []
            test_datasets = []
            A_matrices = []
            for experiment_id in tqdm(range(config["max_nr_of_experiments"]), position=0, desc="running experiments", leave=True):
                experiment_run_path = experiment_root / str(experiment_id)
                state_dict_path = experiment_run_path / f"LISTA/LISTA_state_dict.tar"
                train_datasets.append(torch.load(experiment_run_path / "data/train_data.tar"))
                test_dataset = torch.load(experiment_run_path / "data/test_data.tar")
                perturbed_test_dataset = perturbation_fn(test_dataset)            
                test_datasets.append(perturbed_test_dataset)
                A_matrices.append(torch.load(experiment_run_path / "A.tar"))

            # compute metrics
            mean_knot_density = pd.read_csv(experiment_root / "parameters.csv")['knot_density_end'].mean()
            mean_knot_densities.append(mean_knot_density)
            # loss_mean, loss_stderr = get_or_run_experiment(cache_root=SWEEP_ROOT / "cache", config=config, model_name="LISTA_L2", experiment_root=experiment_root, A_matrices=A_matrices, train_datasets=train_datasets, test_datasets=test_datasets)
            result_mean, result_stderr = run_experiment(config=config, model_name="LISTA", experiment_root=experiment_root, A_matrices=A_matrices, train_datasets=train_datasets, test_datasets=test_datasets)
            mean_results.append(result_mean)
            stderr_results.append(result_stderr)
            per_model_results[str(experiment_root)]['mean'].append(result_mean)
            per_model_results[str(experiment_root)]['stderr'].append(result_stderr)
            per_model_results[str(experiment_root)]['knot_density'].append(mean_knot_density)

        # Convert lists to numpy arrays for easier manipulation
        mean_knot_densities = np.array(mean_knot_densities)
        mean_results = np.array(mean_results)
        stderr_results = np.array(stderr_results)
        
        # sort
        sorted_indices = np.argsort(mean_knot_densities)
        mean_knot_densities = mean_knot_densities[sorted_indices]
        mean_results = mean_results[sorted_indices]
        stderr_results = stderr_results[sorted_indices]

        # Error bars: ± 1.96 * stderr for 95% confidence interval
        ci = 1.96 * stderr_results

        # plt.fill_between(mean_knot_densities, mean_results - ci, mean_results + ci, alpha=0.5)
        # # Plot the mean losses on top
        plt.plot(mean_knot_densities, mean_results, marker='o', linestyle='-', linewidth=2, markersize=8, label=name)
        # plt.errorbar(mean_knot_densities, mean_results, yerr=ci, fmt='o-', elinewidth=2, capsize=5, label=name)

    plt.xlabel('Knot Density', fontsize=20, fontfamily='serif')
    plt.ylabel('Loss', fontsize=20, fontfamily='serif')
    # plt.yscale('log')
    # plt.title('Knot Density vs Loss', fontsize=24, fontfamily='serif')
    plt.grid(True)
    plt.xticks(fontsize=16, fontfamily='serif')
    plt.yticks(fontsize=16, fontfamily='serif')
    plt.legend(fontsize=14)

    # Save the plot to PNG and PDF
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'knot_density_vs_{args.metric}_{perturbation_fn_name}.png', dpi=300)
    plt.savefig(OUTPUT_DIR / f'knot_density_vs_{args.metric}_{perturbation_fn_name}.pdf')
    print(f"✅ Saved fig to {OUTPUT_DIR / f'knot_density_vs_{args.metric}_{perturbation_fn_name}.png'}")
    
    plt.clf()
    # plot_barcharts(OUTPUT_DIR / f'knot_density_vs_loss_barchart_{perturbation_fn_name}.png', barchart_dict)
    # plt.figure(figsize=(3, 5))  # Set figure size
    fig, ax = plt.subplots(figsize=(3, 5))
    gray_cmap = ListedColormap(['gray'])

    # sort models by knot densities
    per_model_results = dict(sorted(per_model_results.items(), key=lambda item: item[1]['knot_density'][0], reverse=False))

    knot_densities = [result['knot_density'][0] for result in per_model_results.values()]
    
    fig.subplots_adjust(right=0.8)

    # Set up your colormap and ranges
    lower_cmap = colormaps.get_cmap('Blues').resampled(128)  # Colormap for the lower range
    upper_cmap = colormaps.get_cmap('viridis').resampled(128)  # Colormap for the upper range

    # Define your ranges
    nonzero_knot_densities = sorted([knot_density for knot_density in knot_densities if knot_density > 0])
    if ZERO_COLORBAR:
        # upper_boundaries = np.linspace(nonzero_knot_densities[1], nonzero_knot_densities[-1], 129)  # Upper range (e.g., 10 to 100)
        upper_boundaries = np.linspace(0, 70, 129)  # Upper range (e.g., 10 to 100)
    else:
        upper_boundaries = np.linspace(0, nonzero_knot_densities[-1], 129)  # Upper range (e.g., 10 to 100)

   # Set up normalization
    lower_norm = BoundaryNorm(boundaries=[-0.1, 0.1], ncolors=1)  # Only one color (gray) for the lower range
    upper_norm = Normalize(vmin=upper_boundaries.min(), vmax=upper_boundaries.max())

    if ZERO_COLORBAR:
        # Create a colorbar for the lower range (just gray)
        cax_lower = fig.add_axes([0.82, 0.125, 0.03, 0.1])  # Position the lower colorbar at the bottom
        cb_lower = ColorbarBase(cax_lower, cmap=gray_cmap, norm=lower_norm, orientation='vertical')
        cb_lower.set_ticks([0])  # Set a single tick
        cb_lower.ax.set_yticklabels(['0'])
        cb_lower.ax.tick_params(labelsize=12)


        # Create a colorbar for the upper range
        cax_upper = fig.add_axes([0.82, 0.25, 0.03, 0.6])  # Position the upper colorbar above the lower one
    else:
        cax_upper = fig.add_axes([0.82, 0.125, 0.03, 0.7])
    cb_upper = ColorbarBase(cax_upper, cmap=upper_cmap, norm=upper_norm, orientation='vertical')
    cb_upper.set_label('Knot Density', fontsize=18)
    cb_upper.ax.tick_params(labelsize=12)

    plotted_zero = False
    for key, result in per_model_results.items():
        assert len(np.unique(result['knot_density'])) == 1
        knot_density = result['knot_density'][0]
        
        if knot_density == 0 and ZERO_COLORBAR:
            if plotted_zero:
                continue
            else:
                color = 'gray'  # Use gray for the zero value
                plotted_zero = True
        else:
            color = upper_cmap(upper_norm(knot_density))

        ax.plot(noise_levels, result['mean'], '.-', color=color)
        ci = 1.96 * np.array(result['stderr'])
        ax.fill_between(noise_levels, result['mean'] - ci, result['mean'] + ci, alpha=0.2, color=color)


    # Final plot settings
    # legend.get_title().set_fontsize('10')
    ax.grid(True)
    if filename_append == "4_24_32":
        ax.set_ylabel("MAE", fontsize=18)
        cb_upper.remove()
    ax.set_xlabel("Added OOD Noise σ", fontsize=18)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    # plt.yscale("log")
    plt.savefig(OUTPUT_DIR / f'per_model_{args.metric}_{filename_append}.pdf', bbox_inches='tight', pad_inches=0.2)
    filename = OUTPUT_DIR / f'per_model_{args.metric}_{filename_append}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.2)
    print(f"✅ Saved figure to {filename}")