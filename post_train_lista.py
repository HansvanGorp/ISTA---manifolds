
# input experiment dir

# for loop over experiments

# load model and training data

# add squared jacobian frobenius 

# train_lista(model, train_data, validation_data, model_config,show_loss_plot = False,
#                                                                    loss_folder = model_folder, save_name = model_type, regularize = regularize,
#                                                                    tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave, save=save)

# compute same losses etc from experiment

# save all to a new subfolder inside the experiment dir

"""
This file creates a large experiment to test the knot density of (R)(L)ISTA in different conditions.
"""

# %% imports
# standard library imports
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import yaml
import shutil
import time
import pandas as pd
import argparse
import uuid
from pathlib import Path
from distutils.util import strtobool

import wandb

# local imports
import ista
from experiment_design import sample_experiment
from knot_density_analysis import knot_density_analysis
from hyper_plane_analysis  import visual_analysis_of_ista
from parallel_coordinates import plot_df_as_parallel_coordinates
from data import create_train_validation_test_datasets
from training import grid_search_ista, train_lista, get_loss_on_dataset_over_folds, get_support_accuracy_on_dataset_over_folds
from make_gif_from_figures_in_folder import make_gif_from_figures_in_folder

def parse_args():
    parser = argparse.ArgumentParser(description="Main experiment arguments")
    parser.add_argument(
        "-r",
        "--reg_type",
        type=str,
        default="L2",
        help="Type of regularization to use in post-training",
    )
    parser.add_argument(
        "-e",
        "--experiment_root",
        type=str,
        default="/ISTA---manifolds/knot_denisty_results/review_response/4_24_32_num_folds/4_24_32_n=0.01_num_folds=10_c9a0",
        help="Root dir of experiment to post-train on",
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

def run_experiment(config, reg_types, reg_config, experiment_dir, plot=False, wandb_sweep=False, save=True, experiment_name = ""): 
    model_types = [f"LISTA_{reg_type}" for reg_type in reg_types]
    nr_of_model_types = len(model_types) # the number of models we are comparing, ISTA, LISTA and RLISTA
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    results_dir_with_parent = experiment_dir / f"post_training_{experiment_name}{str(uuid.uuid4())[:4]}"
    os.makedirs(results_dir_with_parent)

    # %% loop over the experiments to run
    print("\nStarting the experiments")

    # inialize lists to store the results, to show them together in the end
    knot_density_over_experiments  = [[] for _ in range(nr_of_model_types)]
    named_knot_density_over_experiments  = {model_type: [] for model_type in model_types}
    test_loss_over_experiments     = [[] for _ in range(nr_of_model_types)]
    named_test_loss_over_experiments     = {model_type: [] for model_type in model_types}
    named_train_loss_over_experiments     = {model_type: [] for model_type in model_types}
    test_accuracy_over_experiments = [[] for _ in range(nr_of_model_types)]
    train_accuracy_over_experiments = [[] for _ in range(nr_of_model_types)]
    train_loss_over_experiments     = [[] for _ in range(nr_of_model_types)]

    df = pd.DataFrame()

    for model_idx, model_type in enumerate(model_types):
        for experiment_id in tqdm(range(config["max_nr_of_experiments"]), position=0, desc="running experiments", leave=True):
            tqdm_leave = False if experiment_id < config["max_nr_of_experiments"]-1 else True # set tqdm_leave to False if this is not the last experiment
            experiment_run_path = EXPERIMENT_ROOT / str(experiment_id)
            # vars to populate
            results_dir_this_experiment = results_dir_with_parent / str(experiment_id)
            state_dict_path = experiment_run_path / f"LISTA/LISTA_state_dict.tar"
            train_data = torch.load(experiment_run_path / "data/train_data.tar")
            test_data = torch.load(experiment_run_path / "data/test_data.tar")
            validation_data = torch.load(experiment_run_path / "data/validation_data.tar")
            A = torch.load(experiment_run_path / "A.tar")
            with open(experiment_run_path / "parameters.yaml", 'r') as file:
                experiment_parameters = yaml.load(file, Loader=yaml.FullLoader)
            M = experiment_parameters["M"]
            N = experiment_parameters["N"]
            K = experiment_parameters["K"]

            model_config = config["LISTA"]

            # create a directory for this model type
            model_folder = os.path.join(results_dir_this_experiment, model_type)
            os.makedirs(model_folder, exist_ok=True)
            # create the model using the parameters in the config file
            model = ista.LISTA(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = model_config["nr_folds"], 
                            device = config["device"], initialize_randomly = False, share_weights=model_config["share_weights"], train_inputs = torch.stack([y for (y, _) in train_data]))
            
            # load state dict to pick up training where it left
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict)
             
            model_config["regularization"] = {"type": model_type_to_reg_type[model_type], "weight": reg_config["weight"]}
            model, train_losses, val_losses  =  train_lista(model, train_data, validation_data, model_config, show_loss_plot = False,
                                                            loss_folder = model_folder, save_name = model_type, regularize = True,
                                                            tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave, save=save)
                
                    
            # perform knot density analysis on the model
            knot_density = knot_density_analysis(model, model_config["nr_folds"], A, nr_paths = config["Path"]["nr_paths"], anchor_point_std = config["Path"]["anchor_point_std"],
                                                nr_points_along_path=config["Path"]["nr_points_along_path"], path_delta=config["Path"]["path_delta"], anchor_on_inputs=config["Path"].get("anchor_on_inputs", False), save_folder = model_folder,
                                                save_name = "knot_density_"+ model_type, verbose = True, color = colors[model_type], tqdm_position=1, tqdm_leave=tqdm_leave)
            
            if save:
                # save the knot densities in a .tar file and to the lists
                torch.save(knot_density, os.path.join(model_folder, "knot_density.tar"))
            knot_density_over_experiments[model_idx].append(knot_density)
            named_knot_density_over_experiments[model_type].append(knot_density)

            # evaluate the model on the test set
            test_loss = get_loss_on_dataset_over_folds(model, test_data, l1_weight=1.0, l2_weight=0.0)
            test_accuracy = get_support_accuracy_on_dataset_over_folds(model, test_data)
            
            train_accuracy = get_support_accuracy_on_dataset_over_folds(model, train_data)
            train_accuracy_over_experiments[model_idx].append(train_accuracy)

            # save the test loss in a .tar file and to the lists
            test_loss_over_experiments[model_idx].append(test_loss)
            named_test_loss_over_experiments[model_type].append(test_loss)
            test_accuracy_over_experiments[model_idx].append(test_accuracy)
            
            train_loss = get_loss_on_dataset_over_folds(model, train_data, l1_weight=1.0, l2_weight=0.0)
            named_train_loss_over_experiments[model_type].append(train_loss)
            train_loss_over_experiments[model_idx].append(train_loss)
            if save:
                torch.save(test_loss, os.path.join(model_folder, "test_loss.tar"))
                torch.save(test_accuracy, os.path.join(model_folder, "test_accuracy.tar"))

            # visualize the results in a 2D plane
            hyperplane_config = config["Hyperplane"]
            if hyperplane_config["enabled"]:
                hyperplane_folder_norm           = os.path.join(model_folder, "hyperplane","norm")  
                hyperplane_folder_jacobian_label = os.path.join(model_folder, "hyperplane","jacobian_label")
                hyperplane_folder_jacobian_pca = os.path.join(model_folder, "hyperplane","jacobian_pca")

                visual_analysis_of_ista(model, model_config, hyperplane_config, A, save_folder = hyperplane_folder_norm,           tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, color_by="norm", folds_to_visualize=[0, 1, 5, 9])
                visual_analysis_of_ista(model, model_config, hyperplane_config, A, save_folder = hyperplane_folder_jacobian_label, tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, color_by="jacobian_label", folds_to_visualize=[0, 1, 5, 9])
                visual_analysis_of_ista(model, model_config, hyperplane_config, A, save_folder = hyperplane_folder_jacobian_pca   , tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, color_by="jacobian_pca", folds_to_visualize=[0, 1, 5, 9])

                # make gifs of the results?
                if hyperplane_config["make_gif"]:
                    make_gif_from_figures_in_folder(hyperplane_folder_norm,   10)
                    make_gif_from_figures_in_folder(hyperplane_folder_jacobian_label, 10)
                    make_gif_from_figures_in_folder(hyperplane_folder_jacobian_pca,   10)
            if plot:            
                df = make_plots(results_dir_this_experiment, model_types, knot_density_over_experiments, test_accuracy_over_experiments, test_loss_over_experiments, train_accuracy_over_experiments, train_loss_over_experiments, results_dir_with_parent, config, M, N, K, df)

    # save the configuration file to the results directory
    with open(os.path.join(results_dir_with_parent, "config.yaml"), 'w') as file:
        yaml.dump(config, file)

    return test_accuracy_over_experiments, test_loss_over_experiments, knot_density_over_experiments

def make_plots(results_dir_this_experiment, model_types, knot_density_over_experiments, test_accuracy_over_experiments, test_loss_over_experiments, train_accuracy_over_experiments, train_loss_over_experiments, results_dir_with_parent, config, M, N, K, df):    
    # %% after looping over each model type, make combined plots of all model types together
    # create a directory for the combined results
    combined_folder = os.path.join(results_dir_this_experiment, "combined")
    os.makedirs(combined_folder, exist_ok=True)

    # make a joint plot of the knot densities
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        knot_density = knot_density_over_experiments[model_idx][-1]
        folds = np.arange(0, len(knot_density))
        plt.plot(folds, knot_density, label = model_type, c = colors[model_type])
        max_folds = max(max_folds, len(knot_density))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("knot density")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.tight_layout()
    plt.savefig(os.path.join(combined_folder, "knot_density.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(combined_folder, "knot_density.svg"), bbox_inches='tight')
    plt.close()
    
    # make a joint plot of the test losses
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        test_loss = test_loss_over_experiments[model_idx][-1]
        folds = np.arange(0, len(test_loss))
        plt.plot(folds, test_loss, label = model_type, c = colors[model_type])
        max_folds = max(max_folds, len(test_loss))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("test loss")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.tight_layout()
    plt.savefig(os.path.join(combined_folder, "test_loss.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(combined_folder, "test_loss.svg"), bbox_inches='tight')
    plt.close()

    # make a joint plot of the test accuracies
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        test_accuracy = test_accuracy_over_experiments[model_idx][-1]
        folds = np.arange(0, len(test_accuracy))
        plt.plot(folds, test_accuracy, label = f"{model_type}_test", c = colors[model_type])
        
        train_accuracy = train_accuracy_over_experiments[model_idx][-1]
        folds = np.arange(0, len(train_accuracy))
        plt.plot(folds, train_accuracy, label = f"{model_type}_train", c = colors[model_type], linestyle='--')
        max_folds = max(max_folds, len(test_accuracy))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("test accuracy")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.tight_layout()
    plt.savefig(os.path.join(combined_folder, "test_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(combined_folder, "test_accuracy.svg"), bbox_inches='tight')
    plt.close()


    # %% now plot the results across all experiments
    # mean and standard deviation of the knot densities
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        knot_density_over_experiments_this_model = knot_density_over_experiments[model_idx]
        knot_density_over_experiments_this_model = torch.stack(knot_density_over_experiments_this_model)
        knot_density_mean = knot_density_over_experiments_this_model.mean(dim=0)
        knot_density_std  = knot_density_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(knot_density_mean))
        plt.plot(folds, knot_density_mean, label = model_type, c = colors[model_type])
        plt.fill_between(folds, knot_density_mean - knot_density_std, knot_density_mean + knot_density_std, alpha=0.3, color=colors[model_type])
        max_folds = max(max_folds, len(knot_density_mean))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("knot density")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.title("mean and std of the knot density per fold over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "knot_density.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "knot_density.svg"), bbox_inches='tight')
    plt.close()

    # mean and standard deviation of the test losses
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        test_loss_over_experiments_this_model = test_loss_over_experiments[model_idx]
        test_loss_over_experiments_this_model = torch.stack(test_loss_over_experiments_this_model)
        test_loss_mean = test_loss_over_experiments_this_model.mean(dim=0)
        test_loss_std  = test_loss_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(test_loss_mean))
        plt.plot(folds, test_loss_mean, label = f"{model_type} test", c = colors[model_type])
        plt.fill_between(folds, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, alpha=0.3, color=colors[model_type])
        
        train_loss_over_experiments_this_model = train_loss_over_experiments[model_idx]
        train_loss_over_experiments_this_model = torch.stack(train_loss_over_experiments_this_model)
        train_loss_mean = train_loss_over_experiments_this_model.mean(dim=0)
        train_loss_std  = train_loss_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(train_loss_mean))
        plt.plot(folds, train_loss_mean, label = f"{model_type} train", c = colors[model_type], linestyle="dashed")
        plt.fill_between(folds, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.3, color=colors[model_type])
        max_folds = max(max_folds, len(test_loss_mean))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("L1 loss")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.title("mean and std of train and test loss per fold over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "train_test_loss.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "train_test_loss.svg"), bbox_inches='tight')
    plt.close()

    # mean and standard deviation of the test accuracies
    plt.figure()

    max_folds = 0
    for model_idx, model_type in enumerate(model_types):
        test_accuracy_over_experiments_this_model = test_accuracy_over_experiments[model_idx]
        test_accuracy_over_experiments_this_model = torch.stack(test_accuracy_over_experiments_this_model)
        test_accuracy_mean = test_accuracy_over_experiments_this_model.mean(dim=0)
        test_accuracy_std  = test_accuracy_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(test_accuracy_mean))
        plt.plot(folds, test_accuracy_mean, label = f"{model_type}_test", c = colors[model_type], linestyle="-")
        plt.fill_between(folds, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.3, color=colors[model_type])
        max_folds = max(max_folds, len(test_accuracy_mean))
        
        train_accuracy_over_experiments_this_model = train_accuracy_over_experiments[model_idx]
        train_accuracy_over_experiments_this_model = torch.stack(train_accuracy_over_experiments_this_model)
        train_accuracy_mean = train_accuracy_over_experiments_this_model.mean(dim=0)
        train_accuracy_std  = train_accuracy_over_experiments_this_model.std(dim=0)
        folds = np.arange(0, len(train_accuracy_mean))
        plt.plot(folds, train_accuracy_mean, label = f"{model_type}_train", c = colors[model_type], linestyle="--")
        plt.fill_between(folds, train_accuracy_mean - train_accuracy_std, train_accuracy_mean + train_accuracy_std, alpha=0.3, color=colors[model_type])

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("test accuracy")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.title("mean and std of the test accuracy per fold over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "test_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "test_accuracy.svg"), bbox_inches='tight')
    plt.close()


    # add results to the df for the parallel coordinates plot
    new_row = pd.DataFrame({"M": M, "N": N, "K": K}, index=[0])

    for model_idx, model_type in enumerate(model_types):
        knot_density_last_experiment_this_model = knot_density_over_experiments[model_idx][-1]

        new_row["model_type"] = model_type
        new_row["regularization_type"] = config['RLISTA']['regularization']['type'] if model_type == "RLISTA" else None
        new_row["knot_density_max"] = knot_density_last_experiment_this_model.max().item()
        new_row["knot_density_end"] = knot_density_last_experiment_this_model[-1].item()

        test_loss_last_experiment_this_model = test_loss_over_experiments[model_idx][-1]
        new_row["test_loss_end"] = test_loss_last_experiment_this_model[-1].item()
        
        train_loss_last_experiment_this_model = train_loss_over_experiments[model_idx][-1]
        new_row["train_loss_end"] = train_loss_last_experiment_this_model[-1].item()

        test_accuracy_last_experiment_this_model = test_accuracy_over_experiments[model_idx][-1]
        new_row["test_accuracy_end"] = test_accuracy_last_experiment_this_model[-1].item()
        new_row["noise_std"] = config["data_that_stays_constant"]["noise_std"]

        df = pd.concat([df, new_row], ignore_index=True)

    parameters_output_path = os.path.join(results_dir_with_parent, "parameters.csv")
    df.to_csv(parameters_output_path)
    print(f"Saved results to {parameters_output_path}")

    # make the parallel coordinates plot
    for model_idx, model_type in enumerate(model_types):
        fig, ax = plt.subplots(1,1, figsize=(16,8))
        plot_df_as_parallel_coordinates(df[df['model_type'] == model_type], 
                                        ["knot_density_max",  "knot_density_end", "test_loss_end", "test_accuracy_end"], 
                                        "test_accuracy_end",
                                        host = ax, title=model_type, accuracy_scale_collumns = ["test_accuracy_end"],
                                        same_y_scale_collumns = [["knot_density_max",  "knot_density_end"]])    
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir_with_parent, f"parallel_coordinates_{model_type}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir_with_parent, f"parallel_coordinates_{model_type}.svg"), bbox_inches='tight')
        plt.close()
    
    return df


if __name__ == "__main__": 
    
    EXPERIMENT_ROOT = Path(args.experiment_root)
    REG_TYPE = args.reg_type
    # REG_WEIGHTS = [0.0, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    REG_WEIGHTS = [0.1]
    with open(EXPERIMENT_ROOT / "config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    for reg_weight in REG_WEIGHTS:
        run_experiment(config=config, reg_types=[REG_TYPE], reg_config={"weight": reg_weight}, experiment_dir=EXPERIMENT_ROOT, plot=True, experiment_name=f"_{REG_TYPE}_w={reg_weight}_")