from pathlib import Path
import yaml
from tqdm import tqdm
import math
from collections import defaultdict
import uuid
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors

from ista import LISTA
from training import get_loss_on_dataset_over_folds
from knot_density_analysis import knot_density_analysis
from hyper_plane_analysis import visual_analysis_of_ista

###

# WHAT MODELS TO ANALYSE
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_more_ill_posed_4/3_8_24_w=1e-4_aa78")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_large/32_64_64_w=1e-4_3379")
EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_large_sweep/32_64_64_w=1e-1_9d0e")
# MODEL_NAMES = ["LISTA", "RLISTA"]
MODEL_NAMES = ["RLISTA"]
# RUN_IDS = ["0", "1", "2"]
RUN_IDS = ["0"]
std_err = math.sqrt(len(RUN_IDS))
OUTDIR = Path("model_analysis")

# WHAT ANALYSES TO COMPUTE
COMPUTE_GENERALIZATION_GAP = True
COMPUTE_KNOT_DENSITY_ANALYSIS = True
COMPUTE_HYPERPLANE_ANALYSIS = True
ANCHOR_STDS = [1, 2, 5, 10]
FOLDS_TO_VISUALIZE = [0, 1, 15]

###
colors = {
    "ISTA": "tab:blue", 
    "FISTA": "tab:red",
    "LISTA": "tab:orange", 
    "RLISTA": "tab:green"
}

def plot_x_and_x_hat(x, x_hat):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image in the first subplot
    axes[0].imshow(x.cpu().detach().numpy()[...,0])
    axes[0].axis('off')  # Hide the axis
    axes[0].set_title('Target x')

    # Display the second image in the second subplot
    axes[1].imshow(x_hat.cpu().detach().numpy()[0].T)
    axes[1].set_title('x_hat')
    axes[1].set_ylabel('fold idx')
    axes[1].get_xaxis().set_visible(False)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure containing both images
    fig.savefig('x_hat.png')
    plt.clf()
    
def get_color_shade(value, cmap_name='viridis', vmin=0.0, vmax=1.0):
    # Normalize the value to be between vmin and vmax
    norm = mcolors.Normalize(vmin=-vmax, vmax=-vmin)
    
    # Get the colormap
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Map the normalized value to a color
    color = cmap(norm(-value))
    
    return color
    
def load_model(model_name, state_dict_path, A_path):
    model_config = config[model_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.load(A_path).to(device)
    model = LISTA(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = model_config["nr_folds"], device = config["device"], initialize_randomly = False, share_weights=model_config["share_weights"])
    model.to(device)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax_inset = inset_axes(ax1, width="40%", height="40%", loc='center right')
ax_inset.set_xlim(100, 130)
ax_inset.set_ylim(0.05, 0.12)

for model_name in MODEL_NAMES:
    if COMPUTE_GENERALIZATION_GAP:
        dataset_loss_means = {}
        for dataset_name in ["train", "test"]:
            losses_over_runs = []
            knots_per_distance = defaultdict(list)
            for run_id in RUN_IDS:
                experiment_run_path = EXPERIMENT_ROOT / run_id        
                with open(EXPERIMENT_ROOT / "config.yaml", 'r') as file:
                    config = yaml.load(file, Loader=yaml.FullLoader)

                datasets = {
                    'test': torch.load(experiment_run_path / "data/test_data.tar"),
                    'train': torch.load(experiment_run_path / "data/train_data.tar"),
                }
            
                print(f"Running for {model_name}")
                model = load_model(model_name, experiment_run_path / f"{model_name}/{model_name}_state_dict.tar", experiment_run_path / "A.tar")
                losses = get_loss_on_dataset_over_folds(model, datasets[dataset_name])
                losses_over_runs.append(losses)
                
                # dataset name doesn't matter, we just don't want to compute this twice, 
                # as it'll be the same for train or test
                if COMPUTE_KNOT_DENSITY_ANALYSIS and dataset_name == "test": 
                    for anchor_point_std in ANCHOR_STDS:
                        knot_density = knot_density_analysis(model, config[model_name]["nr_folds"], model.A, nr_paths = config["Path"]["nr_paths"], anchor_point_std = anchor_point_std,
                                                                    nr_points_along_path=config["Path"]["nr_points_along_path"], path_delta=config["Path"]["path_delta"], save_folder = ".",
                                                                    save_name = f"knot_density_{model_name}", verbose = True, tqdm_position=1)
                        knots_per_distance[anchor_point_std].append(knot_density)
                
            losses_over_runs_tensor = torch.stack(losses_over_runs)
            loss_means = torch.mean(losses_over_runs_tensor, axis=0)
            dataset_loss_means[dataset_name] = loss_means
            loss_std_err = torch.std(losses_over_runs_tensor, axis=0) / std_err
            num_folds = range(len(losses))
            ax1.plot(num_folds, loss_means, label=f'{model_name}_{dataset_name}', color=colors[model_name], linestyle = ("dashed" if dataset_name == "train" else "solid"))
            ax1.fill_between(num_folds, loss_means - loss_std_err, loss_means + loss_std_err, alpha=0.3, color=colors[model_name])
            ax_inset.plot(num_folds, loss_means, color=colors[model_name], linestyle = ("dashed" if dataset_name == "train" else "solid"))
            
            if COMPUTE_KNOT_DENSITY_ANALYSIS and dataset_name == "test":
                for anchor_point_std in ANCHOR_STDS:
                    knots_at_distance = torch.stack(knots_per_distance[anchor_point_std])
                    knot_mean = torch.mean(knots_at_distance, axis=0)
                    knot_std_err = torch.std(knots_at_distance, axis=0) / std_err
                    num_folds = range(len(knot_density))
                    cmap = "Oranges" if colors[model_name] == "tab:orange" else "Greens"
                    color_with_shade = get_color_shade(anchor_point_std, cmap_name=cmap, vmin=ANCHOR_STDS[0] - 1, vmax=ANCHOR_STDS[-1] + 3)
                    ax2.plot(num_folds, knot_mean, label=f'{model_name} std={anchor_point_std}', color=color_with_shade)
                    ax2.fill_between(num_folds, knot_mean - knot_std_err, knot_mean + knot_std_err, alpha=0.3, color=color_with_shade)
                
                if COMPUTE_HYPERPLANE_ANALYSIS:
                    hyperplane_outddir = OUTDIR / f"hyperplane_{model_name}_{str(uuid.uuid4())[:4]}"
                    os.mkdir(hyperplane_outddir)
                    visual_analysis_of_ista(model, config[model_name], config["Hyperplane"], model.A.cpu(), save_folder = hyperplane_outddir, tqdm_position=1, verbose = True, color_by="jacobian_label", folds_to_visualize=FOLDS_TO_VISUALIZE)
            
        print(f"Mean generalization gap for {model_name}={dataset_loss_means['test'][-1] - dataset_loss_means['train'][-1]}")
    

        
ax1.legend(loc="upper left")
ax1.grid(True)
ax1.set_title("L1 Loss per Fold")
ax1.set_xlabel("Fold Number")
ax1.set_ylabel("L1 Loss")

# Add labels and a title to the inset plot if necessary
ax_inset.grid(True)
ax_inset.set_title('Final Layers', fontsize=10)
ax_inset.tick_params(axis='both', which='major', labelsize=8)

ax2.legend()
ax2.grid(True)
ax2.set_title("Knot Density per Fold")
ax2.set_xlabel("Fold number")
ax2.set_ylabel("Knot Density")
outpath = OUTDIR / "loss_and_knots.png"
plt.savefig(outpath)
plt.close()
print(f"Saved plot at {outpath}")



########################
## KNOT DENSITY ANALYSIS
########################

# plot_x_and_x_hat(x, x_hat)



########################
## INDIVIDUAL LOSS TERMS
########################

# losses = torch.zeros((len(dataset), 3))
# for i, (y, x) in enumerate(tqdm(dataset)):
#     # add batch dim = 1
#     x = x[None, ...]
#     y = y[None, ...]
#     #
#     x_hat, _ = model(y, verbose = False, return_intermediate = True, calculate_jacobian = False)
#     x = x.unsqueeze(2).expand_as(x_hat).to(x_hat.device)

#     # calculate the loss
#     total_loss, reconstruction_loss, regularization_loss = calculate_loss(x_hat, x, model, config[model_name], False)

#     losses[i,0] = total_loss.item()
#     losses[i,1] = reconstruction_loss.item()
#     losses[i,2] = regularization_loss.item()