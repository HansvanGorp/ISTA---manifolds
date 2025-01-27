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
plt.rcParams['font.family'] = 'serif'
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

from ista import ISTA, LISTA, ToeplitzLISTA
from training import get_loss_on_dataset_over_folds
from knot_density_analysis import knot_density_analysis
from hyper_plane_analysis import visual_analysis_of_ista

###

# WHAT MODELS TO ANALYSE
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_more_ill_posed_4/3_8_24_w=1e-4_aa78")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_l1_2/3_8_32_w=1e-4_5978")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/interesting_results/used_in_paper/4_8_32_w=1e-4_f267")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/interesting_results/used_in_paper/4_8_32_from_sweep_bfc3")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/interesting_results/used_in_paper/3_8_24_w=1e-4_aa78")
EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/toeplitz/4_28_32_eec0")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/interesting_results/1_2_3_2fbd")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_large/32_64_64_w=1e-4_3379")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_large_sweep/32_64_64_w=1e-1_9d0e")
MODEL_NAMES = ["LISTA", "ToeplitzLISTA"]
RUN_IDS = ["0"] #, "2", "3", "4"]
std_err = math.sqrt(len(RUN_IDS))
OUTDIR = Path("model_analysis")
OUT_ID = "4_28_32"

# WHAT ANALYSES TO COMPUTE
COMPUTE_GENERALIZATION_GAP = True
COMPUTE_KNOT_DENSITY_ANALYSIS = True
COMPUTE_HYPERPLANE_ANALYSIS = False
ANCHOR_STDS = [1, 5, 10]
FOLDS_TO_VISUALIZE = [0, 1, 15]
if OUT_ID == "3_8_24":
    ZOOM_Y = [0.025, 0.05] # (3, 8, 24)
else:
    ZOOM_Y = [0.075, 0.09] # (4, 8, 32)
# YLIM = [0.07, 0.2]
YLIM = None
ZOOM_POS = 'upper right'
LOSS = "L1"
LOSS_NAME = "MAE" if LOSS == "L1" else "MSE"
ANCHOR_ON_INPUTS = False
NR_PATHS = 10
MAX_FOLDS = 128

if ANCHOR_ON_INPUTS:
    # in this case, anchor_std isn't used, so we only need to compute the knot density once.
    ANCHOR_STDS = [1]

###
colors = {
    "ISTA": "tab:blue", 
    "FISTA": "tab:red",
    "LISTA": "tab:orange", 
    "RLISTA": "tab:green"
}

L1_WEIGHT = 1.0 if LOSS == "L1" else 0.0
L2_WEIGHT = 1.0 if LOSS == "L2" else 0.0

# def plot_x_and_x_hat(x, x_hat):
#     fig, axes = plt.subplots(1, 2, figsize=(10, 12))

#     # Display the first image in the first subplot
#     axes[0].imshow(x.cpu().detach().numpy()[...,0])
#     axes[0].axis('off')  # Hide the axis
#     axes[0].set_title('Target x')

#     # Display the second image in the second subplot
#     axes[1].imshow(x_hat.cpu().detach().numpy()[0].T)
#     axes[1].set_title('x_hat')
#     axes[1].set_ylabel('fold idx')
#     axes[1].get_xaxis().set_visible(False)

#     # Adjust spacing between subplots
#     plt.tight_layout()

#     # Save the figure containing both images
#     fig.savefig('x_hat.png')
#     plt.clf()
    
def get_cmap(model_name):
    if colors[model_name] == "tab:orange":
        return "Oranges"
    elif colors[model_name] == "tab:green":
        return "Greens"
    elif colors[model_name] == "tab:blue":
        return "Blues"
    
anchor_std_linestyle = {
    1: 'solid',
    5: 'dashed',
    10: 'dotted'
}
    
def get_color_shade(value, cmap_name='viridis', vmin=0.0, vmax=1.0):
    # Normalize the value to be between vmin and vmax
    norm = mcolors.Normalize(vmin=-vmax, vmax=-vmin)
    
    # Get the colormap
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Map the normalized value to a color
    color = cmap(norm(-value))
    
    return color
    
def load_model(config, model_name, state_dict_path, A_path, train_dataset, experiment_run_path):
    if model_name == "ToeplitzLISTA":
        model_config = config["LISTA"]
    else:
        model_config = config[model_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.load(A_path).to(device)
    if model_name == "ISTA":
        with open(experiment_run_path / "ISTA" / "best_mu_and_lambda.yaml", 'r') as file:
            hyperparams = yaml.safe_load(file)
        model = ISTA(A, mu = hyperparams['mu'], _lambda = hyperparams['lambda'], nr_folds = min(model_config["nr_folds"], MAX_FOLDS), device = config["device"])
        model.to(device)
    elif model_name == "ToeplitzLISTA":
        model = ToeplitzLISTA(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = min(model_config["nr_folds"], MAX_FOLDS), device = config["device"], initialize_randomly = False, share_weights=model_config["share_weights"], train_inputs = torch.stack([y for (y, _) in train_dataset]))    
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = LISTA(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = min(model_config["nr_folds"], MAX_FOLDS), device = config["device"], initialize_randomly = False, share_weights=model_config["share_weights"], train_inputs = torch.stack([y for (y, _) in train_dataset]))
        model.to(device)
        state_dict = torch.load(state_dict_path, map_location=device)
        model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    inset_width = "60%"  # 40% of the width of the main plot
    inset_height = "60%"  # 40% of the height of the main plot

    # Positioning the inset using percentages and adding margins
    inset_x_margin = 0.025  # Margin on the x-axis (relative to the figure width)
    inset_y_margin = 0.06  # Margin on the y-axis (relative to the figure height)

    # Calculate the bottom left corner position (x0, y0)
    x0 = 1 - float(inset_width.strip('%')) / 100 - inset_x_margin
    y0 = 1 - float(inset_height.strip('%')) / 100 - inset_y_margin

    # Create the inset axes
    ax_inset = inset_axes(ax1, width=inset_width, height=inset_height,
                        bbox_to_anchor=(x0, y0, float(inset_width.strip('%')) / 100, float(inset_height.strip('%')) / 100),
                        bbox_transform=ax1.transAxes, loc=ZOOM_POS)
    # ax_inset = inset_axes(ax1, width="40%", height="40%", loc=ZOOM_POS)
    ax_inset.set_xlim(100, 130)
    ax_inset.set_ylim(ZOOM_Y[0], ZOOM_Y[1])

    for model_name in MODEL_NAMES:
        if COMPUTE_GENERALIZATION_GAP:
            dataset_loss_means = {}
            dataset_loss_std_errs = {}
            std1_knot_density = {}
            std1_knot_density_std_errs = {}
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
                    model = load_model(model_name, experiment_run_path / f"{model_name}/{model_name}_state_dict.tar", experiment_run_path / "A.tar", train_dataset = datasets["train"], experiment_run_path=experiment_run_path)
                    losses = get_loss_on_dataset_over_folds(model, datasets[dataset_name], l1_weight=L1_WEIGHT, l2_weight=L2_WEIGHT)
                    losses_over_runs.append(losses)
                    
                    # dataset name doesn't matter, we just don't want to compute this twice, 
                    # as it'll be the same for train or test
                    if COMPUTE_KNOT_DENSITY_ANALYSIS and dataset_name == "test": 
                        for anchor_point_std in ANCHOR_STDS:
                            knot_density = knot_density_analysis(model, min(config[model_name]["nr_folds"], MAX_FOLDS), model.A, nr_paths = NR_PATHS, anchor_point_std = anchor_point_std,
                                                                        nr_points_along_path=config["Path"]["nr_points_along_path"], path_delta=config["Path"]["path_delta"], anchor_on_inputs=ANCHOR_ON_INPUTS, save_folder = ".",
                                                                        save_name = f"knot_density_{model_name}", verbose = True, tqdm_position=1, anchor_on_sphere=False)
                            knots_per_distance[anchor_point_std].append(knot_density)
                        std1_knot_density[model_name] = knots_per_distance[1]
                    
                losses_over_runs_tensor = torch.stack(losses_over_runs)
                loss_means = torch.mean(losses_over_runs_tensor, axis=0)
                dataset_loss_means[dataset_name] = loss_means
                loss_std_err = torch.std(losses_over_runs_tensor, axis=0) / std_err
                dataset_loss_std_errs[dataset_name] = loss_std_err
                num_folds = range(len(losses))
                ax1.plot(num_folds, loss_means, label=f'{model_name} {dataset_name}', color=colors[model_name], linestyle = ("dashed" if dataset_name == "train" else "solid"))
                ax1.fill_between(num_folds, loss_means - loss_std_err, loss_means + loss_std_err, alpha=0.3, color=colors[model_name])
                ax_inset.plot(num_folds, loss_means, color=colors[model_name], linestyle = ("dashed" if dataset_name == "train" else "solid"))
                
                if COMPUTE_KNOT_DENSITY_ANALYSIS and dataset_name == "test":
                    for anchor_point_std in ANCHOR_STDS:
                        knots_at_distance = torch.stack(knots_per_distance[anchor_point_std])
                        knot_mean = torch.mean(knots_at_distance, axis=0)
                        knot_std_err = torch.std(knots_at_distance, axis=0) / std_err
                        num_folds = range(len(knot_density))
                        color_with_shade = get_color_shade(anchor_point_std, cmap_name=get_cmap(model_name), vmin=ANCHOR_STDS[0] - 1, vmax=ANCHOR_STDS[-1] + 5)
                        label = f'{model_name}' if ANCHOR_ON_INPUTS else f'{model_name} std={anchor_point_std}'
                        ax2.plot(num_folds, knot_mean, label=label, color=color_with_shade, linestyle=anchor_std_linestyle[anchor_point_std])
                        ax2.fill_between(num_folds, knot_mean - knot_std_err, knot_mean + knot_std_err, alpha=0.3, color=color_with_shade)
                    
                    if COMPUTE_HYPERPLANE_ANALYSIS:
                        hyperplane_outddir = OUTDIR / f"hyperplane_{model_name}_{str(uuid.uuid4())[:4]}"
                        os.mkdir(hyperplane_outddir)
                        visual_analysis_of_ista(model, config[model_name], config["Hyperplane"], model.A.cpu(), save_folder = hyperplane_outddir, tqdm_position=1, verbose = True, color_by="jacobian_label", folds_to_visualize=FOLDS_TO_VISUALIZE)
            
            print(f"Test loss {model_name}={dataset_loss_means['test'][-1]}")
            print(f"Test stderr {model_name}={dataset_loss_std_errs['test'][-1]}")
            print(f"Train loss {model_name}={dataset_loss_means['train'][-1]}")
            print(f"Train stderr {model_name}={dataset_loss_std_errs['train'][-1]}")
            print(f"Final knot density std1 {model_name}={torch.mean(torch.stack(std1_knot_density[model_name]), axis=0)[-1]}")
            print(f"Final knot density std1 stderr {model_name}={(torch.std(torch.stack(std1_knot_density[model_name]), axis=0) / std_err)[-1]}")
            print(f"Mean generalization gap for {model_name}={dataset_loss_means['test'][-1] - dataset_loss_means['train'][-1]}")
        

            
    ax1.legend(loc="upper left", fontsize=14)  # Increase legend font size
    ax1.grid(True)
    # ax1.set_title(f"{LOSS} Loss per Fold", fontsize=22)  # Increase title font size
    ax1.set_ylim(YLIM)
    ax1.set_xlabel("Fold Number", fontsize=22)  # Increase x-axis label font size
    ax1.set_ylabel(LOSS_NAME, fontsize=22)  # Increase y-axis label font size
    ax1.tick_params(axis='both', which='major', labelsize=18)  # Increase tick label size

    # Add labels and a title to the inset plot if necessary
    ax_inset.grid(True)
    ax_inset.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax_inset.set_title('Final Layers', fontsize=14)  # Increase inset title font size
    ax_inset.tick_params(axis='both', which='major', labelsize=10)  # Increase inset tick label size

    # Increase font size for ax2
    ax2.legend(fontsize=12)  # Increase legend font size
    ax2.grid(True)
    # ax2.set_title("Knot Density per Fold", fontsize=22)  # Increase title font size
    ax2.set_xlabel("Fold Number", fontsize=22)  # Increase x-axis label font size
    ax2.set_ylabel("Knot Density", fontsize=22)  # Increase y-axis label font size
    ax2.tick_params(axis='both', which='major', labelsize=18)  # Increase tick label size

    # Save the plot
    outpath = OUTDIR / f"loss_and_knots_{OUT_ID}.pdf"
    plt.tight_layout()  # Make sure layout is adjusted for bigger fonts
    plt.savefig(outpath)
    plt.close()

    print(f"Saved plot at {outpath}")