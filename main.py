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
        "-c",
        "--config",
        type=str,
        default="configs/config_knot_density_experiment_debug.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "-m",
        "--model_types",
        nargs='+',
        default=["LISTA"],
        help="Specify which set of algorithms to run.",
    )
    parser.add_argument(
        "--wandb_sweep",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run a wandb sweep",
    )
    parser.add_argument(
        "--sweep_L2",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run an L2 regularization weight sweep",
    )
    parser.add_argument(
        "--sweep_jacobian",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run a jacobian regularization weight sweep",
    )
    parser.add_argument(
        "--sweep_num_folds",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to run a num folds sweep",
    )
    return parser.parse_args()
args = parse_args()

colors = {
    "ISTA": "tab:blue", 
    "FISTA": "tab:red",
    "LISTA": "tab:orange", 
    "RLISTA": "tab:green",
    "ToeplitzLISTA": "tab:olive"
}

def run_experiment(config, model_types, plot=False, wandb_sweep=False, save=True, output_identifier=""):
    with wandb.init():
        if wandb_sweep:
            sweep_config = wandb.config
            updated_config = {key: sweep_config.get(key, value) for key, value in config['data_that_stays_constant'].items()}
            config['data_that_stays_constant'] = updated_config
        
        nr_of_model_types = len(model_types) # the number of models we are comparing, ISTA, LISTA and RLISTA
        config_file_name = os.path.splitext(os.path.basename(Path(args.config)))[0]
        # create the directory to save the results, check first if it already exists, if so stop, and query the user if it should be overwritten
        results_dir_with_parent = os.path.join("knot_denisty_results", config["results_dir"], config_file_name+"_"+output_identifier+"_"+str(uuid.uuid4())[:4])
        if os.path.exists(results_dir_with_parent):
            print(f"\nThis results directory already exists: {config['results_dir']}")
            print("Do you want to overwrite it? (y/n)")	
            answer = input()
            if answer == "y":
                shutil.rmtree(results_dir_with_parent, ignore_errors=True) # remove the directory and its contents
                time.sleep(1) # wait for the directory to be deleted
                os.makedirs(results_dir_with_parent, exist_ok=True)
            else:
                raise FileExistsError(f"The results directory {config['results_dir']} already exists.")
        else:
            os.makedirs(results_dir_with_parent)

        # save the configuration file to the results directory
        with open(os.path.join(results_dir_with_parent, "config.yaml"), 'w') as file:
            yaml.dump(config, file)

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

        # loop
        for experiment_id in tqdm(range(config["max_nr_of_experiments"]), position=0, desc="running experiments", leave=True):
            tqdm_leave = False if experiment_id < config["max_nr_of_experiments"]-1 else True # set tqdm_leave to False if this is not the last experiment
            
            # sample the parameters for the experiment that vary, untill a valid experiment is found
            M, N, K, A = sample_experiment(config)

            # create the directory for the experiment
            results_dir_this_experiment = os.path.join(results_dir_with_parent, str(experiment_id))
            os.makedirs(results_dir_this_experiment, exist_ok=True)

            # save the parameters of the experiment in a .yaml file in the experiment folder/str(experiment_id)
            with open(os.path.join(results_dir_this_experiment, "parameters.yaml"), 'w') as file:
                yaml.dump({"M": M, "N": N, "K": K}, file)

            if save:
                # save the A matrix in a .tar file
                torch.save(A, os.path.join(results_dir_this_experiment, "A.tar"))

            # create the data for the experiment   
            train_data, validation_data, test_data = create_train_validation_test_datasets(A, maximum_sparsity = K, x_magnitude=config["data_that_stays_constant"]["x_magnitude"], 
                                                                                        N=N, noise_std = config["data_that_stays_constant"]["noise_std"],
                                                                                        nr_of_examples_train = config["data_that_stays_constant"]["nr_training_samples"],
                                                                                        nr_of_examples_validation = config["data_that_stays_constant"]["nr_validation_samples"],
                                                                                        nr_of_examples_test = config["data_that_stays_constant"]["nr_test_samples"],
                                                                                        test_magnitude_shift_epsilon = config["data_that_stays_constant"]["test_distribution_shift_epsilon"])
            
            if save:
                # save the data in .tar files
                results_dir_this_experiment_data = os.path.join(results_dir_with_parent, str(experiment_id), "data")
                os.makedirs(results_dir_this_experiment_data, exist_ok=True)
                torch.save(train_data,      os.path.join(results_dir_this_experiment_data, "train_data.tar"))
                torch.save(validation_data, os.path.join(results_dir_this_experiment_data, "validation_data.tar"))
                torch.save(test_data,       os.path.join(results_dir_this_experiment_data, "test_data.tar"))

            # %% loop over each model type
            for model_idx, model_type in enumerate(model_types):
                config["ToeplitzLISTA"] = config["LISTA"]
                # get the model config for this model type
                model_config = config[model_type]
                
                if model_type == "RLISTA" and wandb_sweep:
                    updated_config = {key: sweep_config.get(key, value) for key, value in model_config['regularization'].items()}
                    model_config['regularization'] = updated_config
                # create a directory for this model type
                model_folder = os.path.join(results_dir_this_experiment, model_type)
                os.makedirs(model_folder, exist_ok=True)
                    
                # check if this is ISTA or FISTA, in which case the parameters are found by grid search
                if model_type == "ISTA" or model_type == "FISTA":
                    model_class = ista.ISTA if model_type == "ISTA" else ista.FISTA
                    # create the ISTA/FISTA model with mu=0 and lambda=0
                    model = model_class(A, mu = 0, _lambda = 0, nr_folds = model_config["nr_folds"], device = config["device"])

                    # perform grid search on ISTA for the best lambda and mu for these parameters
                    model, mu, _lambda, losses, tested_mus, tested_lambdas = grid_search_ista(model, train_data, validation_data, model_config, tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave)
                    
                    # save the results of the grid search in the results directroy in a .yaml file
                    with open(os.path.join(model_folder, "best_mu_and_lambda.yaml"), 'a') as file:
                        yaml.dump({"mu": mu.cpu().item(), "lambda": _lambda.cpu().item()}, file)   

                    # put the losses in a .csv file, with the tested mus and lambdas as the rows and columns
                    loss_df = pd.DataFrame(losses, index=tested_mus, columns=tested_lambdas)
                    loss_df.to_csv(os.path.join(model_folder, "losses.csv"))


                # otherwise, the model is LISTA or RLISTA, and needs to be trained
                else:
                    lista_class = ista.ToeplitzLISTA if model_type == "TeoplitzLista" else ista.LISTA
                    # create the model using the parameters in the config file
                    model = lista_class(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = model_config["nr_folds"], 
                                    device = config["device"], initialize_randomly = False, share_weights=model_config["share_weights"], train_inputs = torch.stack([y for (y, _) in train_data]))
                    
                    regularize = "regularization" in fixed_config[model_type].keys()
                    model, train_losses, val_losses  =  train_lista(model, train_data, validation_data, model_config,show_loss_plot = False,
                                                                    loss_folder = model_folder, save_name = model_type, regularize = regularize,
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

        lista_minus_rlista = None
        lista_test_loss = None
        rlista_test_loss = None
        lista_train_loss = None
        rlista_train_loss = None
        lista_knot_density = None
        rlista_knot_density = None
        rlista_gen_gap = None
        if 'RLISTA' in model_types:
            rlista_test_loss = torch.mean(torch.vstack(named_test_loss_over_experiments['RLISTA']), axis=0)[-1]
            rlista_train_loss = torch.mean(torch.vstack(named_train_loss_over_experiments['RLISTA']), axis=0)[-1]
            rlista_knot_density = torch.mean(torch.vstack(named_knot_density_over_experiments['RLISTA']), axis=0)[-1]
            rlista_gen_gap = rlista_test_loss - rlista_train_loss
            
        if 'LISTA' in model_types:
            lista_test_loss = torch.mean(torch.vstack(named_test_loss_over_experiments['LISTA']), axis=0)[-1]
            lista_train_loss = torch.mean(torch.vstack(named_train_loss_over_experiments['LISTA']), axis=0)[-1]
            lista_knot_density = torch.mean(torch.vstack(named_knot_density_over_experiments['LISTA']), axis=0)[-1]
            
        if 'RLISTA' in model_types and 'LISTA' in model_types:
            lista_minus_rlista = lista_test_loss - rlista_test_loss

        wandb.log({
            "LISTA_test_loss": lista_test_loss,
            "LISTA_train_loss": lista_train_loss,
            "RLISTA_test_loss": rlista_test_loss,
            "RLISTA_train_loss": rlista_train_loss,
            "RLISTA_generalization_gap": rlista_gen_gap,
            "LISTA_minus_RLISTA": lista_minus_rlista,
            "LISTA_knot_density": lista_knot_density,
            "RLISTA_knot_density": rlista_knot_density,
        })
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
    with open(args.config, 'r') as file:
        fixed_config = yaml.load(file, Loader=yaml.FullLoader)
      
    sweep_config = {
        'name': '16_64_64_newRLISTA',
        'method': 'random',  # Intelligent sampling method
        'metric': {
            # 'name': 'LISTA_minus_RLISTA',
            # 'goal': 'maximize'
            'name': 'RLISTA_test_loss',
            'goal': 'minimize'
        },
        'parameters': {
            # 'nr_training_samples': {
            #     'min': 10,
            #     'max': 1024,
            # },
            # 'noise_std': {
            #     'min': 0.001,
            #     'max': 0.1
            # },
            'N_points': {
                'min': 10,
                'max': 100
            },
            'cloud_scale': {
                'min': 5,
                'max': 50
            },
            'weight': {
                'min': 0.00001,
                'max': 10,
                'distribution': 'log_uniform_values'
            },
            'num_clouds': {
                'min': 1,
                'max': 100
            }
        }
    }
    

    torch.manual_seed(fixed_config["seed"])
    np.random.seed(fixed_config["seed"])

    if args.wandb_sweep == True:
        sweep_id = wandb.sweep(sweep_config, project="ISTA geom")
        
        def sweep_run():
            run_experiment(config=fixed_config, model_types=args.model_types, wandb_sweep=True, save=False)
        
        # Run the sweep
        agent = wandb.agent(sweep_id, function=sweep_run, count=200)
    elif args.sweep_L2 == True:
        weights = [0.0, 0.000005, 0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025]
        
        for weight in weights:
            fixed_config['LISTA']['regularization'] = {"type": "L2", "weight": weight}
            run_experiment(config=fixed_config, model_types=args.model_types, save=True, plot=True, output_identifier=f"L2={weight}")
    elif args.sweep_jacobian == True:
        weights = [0.0, 0.000005, 0.00001, 0.000025, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025]
        
        for weight in weights:
            fixed_config['LISTA']['regularization'] = {"type": "jacobian", "weight": weight}
            run_experiment(config=fixed_config, model_types=args.model_types, save=True, plot=True, output_identifier=f"jacob={weight}")
    elif args.sweep_num_folds == True:
        num_foldss = [5, 6, 7, 8, 9, 10]
        
        for num_folds in num_foldss:
            fixed_config['LISTA']['nr_folds'] = num_folds
            run_experiment(config=fixed_config, model_types=args.model_types, save=True, plot=True, output_identifier=f"num_folds={num_folds}")
    else:
        run_experiment(config=fixed_config, model_types=args.model_types, plot=True)
        
    # test_accuracy_over_experiments, test_loss_over_experiments, knot_density_over_experiments = run_experiment(fixed_config, sweep_config, args.model_types)
    
    # # this is probably what we want to optimize
    # mean_test_loss = torch.mean(torch.stack(test_loss_over_experiments[0]), axis=0)[-1]
    # # just sum all densities over all folds
    # total_knot_density = torch.sum(torch.stack(test_loss_over_experiments[0]))
    # print("✅")