from pathlib import Path
import yaml
from tqdm import tqdm
import math
from collections import defaultdict
import uuid
import os
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors

from ista import LISTA
from training import get_loss_on_dataset_over_folds
from knot_density_analysis import knot_density_analysis
from hyper_plane_analysis import visual_analysis_of_ista

###

# WHAT MODELS TO ANALYSE
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_more_ill_posed_4/3_8_24_w=1e-4_aa78")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_l1_2/3_8_32_w=1e-4_5978")
EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/sweep_reg_weight_3")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_large/32_64_64_w=1e-4_3379")
# EXPERIMENT_ROOT = Path("/ISTA---manifolds/knot_denisty_results/x_large_sweep/32_64_64_w=1e-1_9d0e")
MODEL_NAMES = ["LISTA", "RLISTA"]
RUN_IDS = ["0"]
std_err = math.sqrt(len(RUN_IDS))
OUTPATH = Path("model_analysis/knots_vs_loss.csv")
OUT_ID = "32_64_64"

# WHAT ANALYSES TO COMPUTE
COMPUTE_GENERALIZATION_GAP = True
COMPUTE_KNOT_DENSITY_ANALYSIS = True
COMPUTE_HYPERPLANE_ANALYSIS = False
ANCHOR_STDS = [1, 2, 5, 10]
FOLDS_TO_VISUALIZE = [0, 1, 15]
ZOOM_Y = [0.0, 0.05]

###
colors = {
    "ISTA": "tab:blue", 
    "FISTA": "tab:red",
    "LISTA": "tab:orange", 
    "RLISTA": "tab:green"
}

def load_model(model_name, state_dict_path, A_path):
    model_config = config[model_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.load(A_path).to(device)
    model = LISTA(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = model_config["nr_folds"], device = config["device"], initialize_randomly = False, share_weights=model_config["share_weights"])
    model.to(device)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

with open(OUTPATH, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["reg_weight", "test_loss", "train_loss", "gen_gap", "knots"])

for experiment_path in [Path(os.path.join(EXPERIMENT_ROOT, file)) for file in os.listdir(EXPERIMENT_ROOT)]:
    model_name = "RLISTA"
    dataset_loss_means = {"train": [], "test": []}
    for dataset_name in ["train", "test"]:
        losses_over_runs = []
        for run_id in RUN_IDS:
            experiment_run_path = experiment_path / run_id        
            with open(experiment_path / "config.yaml", 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)

            datasets = {
                'test': torch.load(experiment_run_path / "data/test_data.tar"),
                'train': torch.load(experiment_run_path / "data/train_data.tar"),
            }
        
            model = load_model(model_name, experiment_run_path / f"{model_name}/{model_name}_state_dict.tar", experiment_run_path / "A.tar")
            losses = get_loss_on_dataset_over_folds(model, datasets[dataset_name])
            losses_over_runs.append(losses)
    
        dataset_loss_means[dataset_name].append(torch.mean(torch.vstack(losses_over_runs), axis=0)[-1])
    knot_density = knot_density_analysis(model, config[model_name]["nr_folds"], model.A, nr_paths = config["Path"]["nr_paths"], anchor_point_std = 1,
                                                nr_points_along_path=config["Path"]["nr_points_along_path"], path_delta=config["Path"]["path_delta"], save_folder = ".",
                                                save_name = f"knot_density_{model_name}", verbose = True, tqdm_position=1)

    with open(OUTPATH, 'a', newline='') as file:
        writer = csv.writer(file)
        reg_weight = float(config['RLISTA']['regularization']['weight'])
        test_loss = float(dataset_loss_means['test'][-1])
        train_loss = float(dataset_loss_means['train'][-1])
        gen_gap = float(dataset_loss_means['test'][-1] - dataset_loss_means['train'][-1])
        knots = float(knot_density[-1])
        writer.writerow([reg_weight, test_loss, train_loss, gen_gap, knots])
        print("------------------------")
        print(f"Reg weight: {reg_weight}")
        print(f"Test loss: {test_loss}")
        print(f"Train loss: {train_loss}")
        print(f"Generalization gap ={gen_gap}")
        print(f"Knots: {knots}")
        print("------------------------")
        print("\n")
