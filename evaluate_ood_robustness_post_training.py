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
from training import get_loss_on_dataset_over_folds
from data import ISTAData

def parse_args():
    parser = argparse.ArgumentParser(description="Main experiment arguments")
    parser.add_argument(
        "-b",
        "--base_experiment_root",
        type=str,
        default="/ISTA---manifolds/knot_denisty_results/sweep_noise_levels/8_64_64/8_64_64_n=0.01_0d6d",
        help="Path to dir containing original experiment results",
    )
    parser.add_argument(
        "-s",
        "--sweep_root",
        type=str,
        default="/ISTA---manifolds/knot_denisty_results/sweep_noise_levels/8_64_64/8_64_64_n=0.01_0d6d/double_filtered_post_training_L2",
        help="Path to dir containing post-training results",
    )
    parser.add_argument(
        "-p",
        "--perturbation",
        type=str,
        default="noise",
        help="Which kind of perturbation to apply to the test data to make it OOD",
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

    losses = []

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
        test_loss = get_loss_on_dataset_over_folds(model, test_data, l1_weight=1.0, l2_weight=0.0)
        losses.append(test_loss.detach().cpu().numpy())
    
    # return mean and std err of losses
    return np.mean(losses, axis=0)[-1], (np.std(losses, axis=0)[-1] / np.sqrt(len(losses)))
    
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
    BASE_EXPERIMENT_ROOT = Path(args.base_experiment_root)
    SWEEP_ROOT = Path(args.sweep_root)

    with open(BASE_EXPERIMENT_ROOT / "config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
        perturbation_fn_name = 'perturb_dataset_with_measurement_noise'
    chosen_perturbation_fn = perturbation_fns[perturbation_fn_name]
    noise_levels = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125] # , 0.15]
    perturbations = {
        'noise=0.0': lambda data: chosen_perturbation_fn(data, 0.0),
        'noise=0.025': lambda data: chosen_perturbation_fn(data, 0.025),
        'noise=0.05': lambda data: chosen_perturbation_fn(data, 0.05),
        'noise=0.075': lambda data: chosen_perturbation_fn(data, 0.075),
        'noise=0.1': lambda data: chosen_perturbation_fn(data, 0.1),
        'noise=0.125': lambda data: chosen_perturbation_fn(data, 0.125),
        # 'noise=0.15': lambda data: chosen_perturbation_fn(data, 0.15),
    }
    
    # Plotting
    # plt.figure(figsize=(10, 6))  # Set figure size
    
    per_model_results = defaultdict(lambda: {'mean': [], 'stderr': [], 'knot_density': []})
    
    for name, perturbation_fn in perturbations.items():
        train_datasets = []
        test_datasets = []
        A_matrices = []
        for experiment_id in tqdm(range(config["max_nr_of_experiments"]), position=0, desc="running experiments", leave=True):
            experiment_run_path = BASE_EXPERIMENT_ROOT / str(experiment_id)
            state_dict_path = experiment_run_path / f"LISTA/LISTA_state_dict.tar"
            train_datasets.append(torch.load(experiment_run_path / "data/train_data.tar"))
            test_dataset = torch.load(experiment_run_path / "data/test_data.tar")
            perturbed_test_dataset = perturbation_fn(test_dataset)
            test_datasets.append(perturbed_test_dataset)
            A_matrices.append(torch.load(experiment_run_path / "A.tar"))

        mean_knot_densities = []
        mean_losses = []
        stderr_losses = []
        for experiment_root in [Path(os.path.join(SWEEP_ROOT, name)) for name in os.listdir(SWEEP_ROOT) if name != "cache"]:
            mean_knot_density = pd.read_csv(experiment_root / "parameters.csv")['knot_density_end'].mean()
            mean_knot_densities.append(mean_knot_density)
            # loss_mean, loss_stderr = get_or_run_experiment(cache_root=SWEEP_ROOT / "cache", config=config, model_name="LISTA_L2", experiment_root=experiment_root, A_matrices=A_matrices, train_datasets=train_datasets, test_datasets=test_datasets)
            loss_mean, loss_stderr = run_experiment(config=config, model_name="LISTA_L2", experiment_root=experiment_root, A_matrices=A_matrices, train_datasets=train_datasets, test_datasets=test_datasets)
            
            mean_losses.append(loss_mean)
            stderr_losses.append(loss_stderr)
            per_model_results[str(experiment_root)]['mean'].append(loss_mean)
            per_model_results[str(experiment_root)]['stderr'].append(loss_stderr)
            per_model_results[str(experiment_root)]['knot_density'].append(mean_knot_density)

        # Convert lists to numpy arrays for easier manipulation
        mean_knot_densities = np.array(mean_knot_densities)
        mean_losses = np.array(mean_losses)
        stderr_losses = np.array(stderr_losses)
        
        # sort
        sorted_indices = np.argsort(mean_knot_densities)
        mean_knot_densities = mean_knot_densities[sorted_indices]
        mean_losses = mean_losses[sorted_indices]
        stderr_losses = stderr_losses[sorted_indices]

        # Error bars: ± 1.96 * stderr for 95% confidence interval
        ci = 1.96 * stderr_losses

        # plt.fill_between(mean_knot_densities, mean_losses - ci, mean_losses + ci, alpha=0.5)
        # # Plot the mean losses on top
        plt.plot(mean_knot_densities, mean_losses, marker='o', linestyle='-', linewidth=2, markersize=8, label=name)
        # plt.errorbar(mean_knot_densities, mean_losses, yerr=ci, fmt='o-', elinewidth=2, capsize=5, label=name)

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
    plt.savefig(BASE_EXPERIMENT_ROOT / f'knot_density_vs_loss_{perturbation_fn_name}.png', dpi=300)
    plt.savefig(BASE_EXPERIMENT_ROOT / f'knot_density_vs_loss_{perturbation_fn_name}.pdf')
    print(f"✅ Saved fig to {BASE_EXPERIMENT_ROOT / f'knot_density_vs_loss_{perturbation_fn_name}.png'}")
    
    plt.clf()
    # plot_barcharts(BASE_EXPERIMENT_ROOT / f'knot_density_vs_loss_barchart_{perturbation_fn_name}.png', barchart_dict)
    plt.figure(figsize=(3, 5))  # Set figure size

    # sort models by knot densities
    per_model_results = dict(sorted(per_model_results.items(), key=lambda item: item[1]['knot_density'][0], reverse=False))

    for key, result in per_model_results.items():
        assert len(np.unique(result['knot_density'])) == 1
        plt.plot(noise_levels, result['mean'], '.-', label=str(round(result['knot_density'][0], 2)))
        ci = 1.96 * np.array(result['stderr'])
        plt.fill_between(noise_levels, result['mean'] - ci, result['mean'] + ci, alpha=0.2)
    legend = plt.legend(title="Model Knot Density")
    # legend.get_title().set_fontsize('10')
    plt.grid(True)
    plt.xlabel("Added Noise σ", fontsize=12)
    plt.ylabel("MAE", fontsize=12)
    # plt.yscale("log")
    plt.savefig(BASE_EXPERIMENT_ROOT / 'per_model_result.pdf', bbox_inches='tight', pad_inches=0.1)
    filename = BASE_EXPERIMENT_ROOT / 'per_model_result.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ Saved figure to {filename}")