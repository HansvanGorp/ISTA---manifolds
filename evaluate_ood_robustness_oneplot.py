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
from matplotlib import colormaps, gridspec
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
        default="/ISTA---manifolds/knot_denisty_results/review_response/4_24_32_L2",
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
    
    
# def get_or_run_experiment(cache_root, config, model_name, experiment_root, A_matrices, train_datasets, test_datasets): 
#     if not cache_root.exists():
#         cache_root.mkdir(parents=True, exist_ok=True)
#     cache_path = cache_root / f"{str(experiment_root).replace('/', '_')}_{model_name}.npz"
#     if os.path.exists(cache_path):
#         cached_results = np.load(cache_path)
#         loss_mean, loss_stderr = cached_results['loss_mean'], cached_results['loss_stderr']
#     else:
#         loss_mean, loss_stderr = run_experiment(config=config, model_name=model_name, experiment_root=experiment_root, A_matrices=A_matrices, train_datasets=train_datasets, test_datasets=test_datasets)
#         np.savez(cache_path, loss_mean=loss_mean, loss_stderr=loss_stderr)
#     return loss_mean, loss_stderr

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
    SWEEP_ROOTS = [
        Path("/ISTA---manifolds/knot_denisty_results/review_response/4_24_32_L2_bias_reg/"),
        Path("/ISTA---manifolds/knot_denisty_results/review_response/8_64_64_L2_bias_reg/")
    ]
    ZERO_COLORBAR = False
    
    per_sweep_results = []
    for sweep_root in SWEEP_ROOTS:
    
        perturbation_fn_name = 'perturb_dataset_with_measurement_noise'
        chosen_perturbation_fn = perturbation_fns[perturbation_fn_name]
            
        # noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        noise_levels_8_64_64 = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125] # , 0.06, 0.07, 0.08, 0.09, 0.1]    
        # noise_levels = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06] # 0.07] # 0.08, 0.09, 0.1]
        filename_append = "8_64_64"
        perturbations = {
            'noise=0.0': lambda data: chosen_perturbation_fn(data, 0.0),
            'noise=0.025': lambda data: chosen_perturbation_fn(data, 0.025),
            'noise=0.05': lambda data: chosen_perturbation_fn(data, 0.05),
            'noise=0.075': lambda data: chosen_perturbation_fn(data, 0.075),
            'noise=0.1': lambda data: chosen_perturbation_fn(data, 0.1),
            'noise=0.125': lambda data: chosen_perturbation_fn(data, 0.125),
            # 'noise=0.05': lambda data: chosen_perturbation_fn(data, 0.05),
            # 'noise=0.06': lambda data: chosen_perturbation_fn(data, 0.06),
            # 'noise=0.07': lambda data: chosen_perturbation_fn(data, 0.07),
            # 'noise=0.08': lambda data: chosen_perturbation_fn(data, 0.08),
            # 'noise=0.09': lambda data: chosen_perturbation_fn(data, 0.09),
            # 'noise=0.1': lambda data: chosen_perturbation_fn(data, 0.1),
            # 'noise=0.15': lambda data: chosen_perturbation_fn(data, 0.15),
            # 'noise=0.2': lambda data: chosen_perturbation_fn(data, 0.2),
        }
        if "4_24_32_L2" in str(sweep_root):
            noise_levels_4_24_32 = noise_levels_8_64_64
            noise_levels_4_24_32.append(0.15)
            perturbations['noise=0.15'] = lambda data: chosen_perturbation_fn(data, 0.15)
            filename_append = "4_24_32"
        
        # Plotting
        # plt.figure(figsize=(10, 6))  # Set figure size
        
        per_model_results = defaultdict(lambda: {'mean': [], 'stderr': [], 'knot_density': []})
        
        for name, perturbation_fn in perturbations.items():
            mean_knot_densities = []
            mean_results = []
            stderr_results = []
            for experiment_root in [Path(os.path.join(sweep_root, name)) for name in os.listdir(sweep_root) if name != "cache"]:
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

        # sort models by knot densities
        per_model_results = dict(sorted(per_model_results.items(), key=lambda item: item[1]['knot_density'][0], reverse=False))
        per_sweep_results.append(per_model_results)

    knot_densities = [result['knot_density'][0] for result in per_model_results.values()]

    # Set up your colormap and ranges 
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

    fig = plt.figure(figsize=(5, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Create subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Plot the data for the first sweep on ax1
    for key, result in per_sweep_results[0].items():
        knot_density = result['knot_density'][0]
        color = upper_cmap(upper_norm(knot_density))
        ax1.plot(noise_levels_4_24_32, result['mean'], '.-', color=color)
        ci = 1.96 * np.array(result['stderr'])
        ax1.fill_between(noise_levels_4_24_32, result['mean'] - ci, result['mean'] + ci, alpha=0.2, color=color)

    # Plot the data for the second sweep on ax2
    for key, result in per_sweep_results[1].items():
        knot_density = result['knot_density'][0]
        color = upper_cmap(upper_norm(knot_density))
        ax2.plot(noise_levels_8_64_64, result['mean'], '.-', color=color)
        ci = 1.96 * np.array(result['stderr'])
        ax2.fill_between(noise_levels_8_64_64, result['mean'] - ci, result['mean'] + ci, alpha=0.2, color=color)

    # Adjust labels, titles, and other plot settings for ax1 and ax2
    ax1.set_xlabel("Added OOD Noise σ", fontsize=18)
    ax1.set_ylabel("Loss", fontsize=18)
    ax1.grid(True)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    ax2.set_xlabel("Added OOD Noise σ", fontsize=18)
    ax2.grid(True)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    # Add shared colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=upper_norm, cmap=upper_cmap), cax=cax)
    cb.set_label('Knot Density', fontsize=18)
    cb.ax.tick_params(labelsize=12)

    # Save the figure
    plt.savefig(OUTPUT_DIR / f'per_model_comparison_{args.metric}.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.savefig(OUTPUT_DIR / f'per_model_comparison_{args.metric}.png', bbox_inches='tight', pad_inches=0.2)
    outfile = OUTPUT_DIR / f'per_model_comparison_{args.metric}.png'
    print(f"✅ Saved figure to {outfile}")