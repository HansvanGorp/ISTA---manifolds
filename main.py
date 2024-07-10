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
        default="configs/config_knot_density_experiment.yaml",
        help="Path to the experiment config file.",
    )
    return parser.parse_args()
args = parse_args()

# %% constants
model_types = ["ISTA", "LISTA", "RLISTA"] # the model types we are comparing
nr_of_model_types = len(model_types) # the number of models we are comparing, ISTA, LISTA and RLISTA
colors = ["tab:blue", "tab:orange", "tab:green"] # the colors for the models

# %% load the configuration file
with open(args.config, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# %% preambule
# set the seed
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

# create the directory to save the results, check first if it already exists, if so stop, and query the user if it should be overwritten
results_dir_with_parent = os.path.join("knot_denisty_results", config["results_dir"])
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
test_loss_over_experiments     = [[] for _ in range(nr_of_model_types)]
test_accuracy_over_experiments = [[] for _ in range(nr_of_model_types)]

# initiale the df for the parallel coordinates plot, each row will be an experiment
# the df has the following columns: M, N, K, noise_std, mu, lambda, knot_density_ista_max,  knot_density_ista_end, knot_density_lista_max, knot_density_lista_end
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

    # save the A matrix in a .tar file
    torch.save(A, os.path.join(results_dir_this_experiment, "A.tar"))

    # create the data for the experiment   
    train_data, validation_data, test_data = create_train_validation_test_datasets(A, maximum_sparsity = K, x_magnitude=config["data_that_stays_constant"]["x_magnitude"], 
                                                                                   N=N, noise_std = config["data_that_stays_constant"]["noise_std"],
                                                                                   nr_of_examples_train = config["data_that_stays_constant"]["nr_training_samples"],
                                                                                   nr_of_examples_validation = config["data_that_stays_constant"]["nr_validation_samples"],
                                                                                   nr_of_examples_test = config["data_that_stays_constant"]["nr_test_samples"])
    
    # save the data in .tar files
    results_dir_this_experiment_data = os.path.join(results_dir_with_parent, str(experiment_id), "data")
    os.makedirs(results_dir_this_experiment_data, exist_ok=True)
    torch.save(train_data,      os.path.join(results_dir_this_experiment_data, "train_data.tar"))
    torch.save(validation_data, os.path.join(results_dir_this_experiment_data, "validation_data.tar"))
    torch.save(test_data,       os.path.join(results_dir_this_experiment_data, "test_data.tar"))

    # %% loop over each model type
    for model_idx, model_type in enumerate(model_types):
        # get the model config for this model type
        model_config = config[model_type]

        # create a directory for this model type
        model_folder = os.path.join(results_dir_this_experiment, model_type)
        os.makedirs(model_folder, exist_ok=True)

        # check if this is ISTA, in which case the parameters are found by grid search
        if model_type == "ISTA":
            # create the ISTA model with mu=0 and lambda=0
            model = ista.ISTA(A, mu = 0, _lambda = 0, nr_folds = model_config["nr_folds"], device = config["device"])

            # perform grid search on ISTA for the best lambda and mu for these parameters
            model, mu, _lambda, losses, tested_mus, tested_lambdas = grid_search_ista(model, train_data, validation_data, model_config, tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave)
            
            # save the results of the grid search in the results directroy in a .yaml file
            with open(os.path.join(model_folder, "best_mu_and_lambda.yaml"), 'a') as file:
                yaml.dump({"mu": mu.cpu().item(), "lambda": _lambda.cpu().item()}, file)   

            # put the losses in a .csv file, with the tested mus and lambdas as the rows and columns
            df = pd.DataFrame(losses, index=tested_mus, columns=tested_lambdas)
            df.to_csv(os.path.join(model_folder, "losses.csv"))


        # otherwise, the model is LISTA or RLISTA, and needs to be trained
        else:
            # create the model using the parameters in the config file
            model = ista.LISTA(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = model_config["nr_folds"], 
                               device = config["device"], initialize_randomly = False)
            
            regularize = (model_type == "RLISTA")
            model, train_losses, val_losses  =  train_lista(model, train_data, validation_data, model_config,show_loss_plot = False,
                                                            loss_folder = model_folder, save_name = model_type, regularize = regularize,
                                                            tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave)
            
                
        # perform knot density analysis on the model
        knot_density = knot_density_analysis(model, model_config["nr_folds"], A, nr_paths = config["Path"]["nr_paths"], anchor_point_std = config["Path"]["anchor_point_std"],
                                             nr_points_along_path=config["Path"]["nr_points_along_path"], path_delta=config["Path"]["path_delta"], save_folder = model_folder,
                                             save_name = "knot_density_"+ model_type, verbose = True, color = colors[model_idx], tqdm_position=1, tqdm_leave=tqdm_leave)
        
        # save the knot densities in a .tar file and to the lists
        torch.save(knot_density, os.path.join(model_folder, "knot_density.tar"))
        knot_density_over_experiments[model_idx].append(knot_density)

        # evaluate the model on the test set
        test_loss = get_loss_on_dataset_over_folds(model, test_data)
        test_accuracy = get_support_accuracy_on_dataset_over_folds(model, test_data)

        # save the test loss in a .tar file and to the lists
        torch.save(test_loss, os.path.join(model_folder, "test_loss.tar"))
        test_loss_over_experiments[model_idx].append(test_loss)
        torch.save(test_accuracy, os.path.join(model_folder, "test_accuracy.tar"))
        test_accuracy_over_experiments[model_idx].append(test_accuracy)

        # visualize the results in a 2D plane
        hyperplane_config = config["Hyperplane"]
        if hyperplane_config["enabled"]:
            hyperplane_folder_norm           = os.path.join(model_folder, "hyperplane","norm")  
            hyperplane_folder_jacobian_label = os.path.join(model_folder, "hyperplane","jacobian_label")
            hyperplane_folder_jacobian_pca = os.path.join(model_folder, "hyperplane","jacobian_pca")

            visual_analysis_of_ista(model, model_config, hyperplane_config, A, save_folder = hyperplane_folder_norm,           tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, color_by="norm")
            visual_analysis_of_ista(model, model_config, hyperplane_config, A, save_folder = hyperplane_folder_jacobian_label, tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, color_by="jacobian_label")
            visual_analysis_of_ista(model, model_config, hyperplane_config, A, save_folder = hyperplane_folder_jacobian_pca   , tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, color_by="jacobian_pca")

            # make gifs of the results?
            if hyperplane_config["make_gif"]:
                make_gif_from_figures_in_folder(hyperplane_folder_norm,   10)
                make_gif_from_figures_in_folder(hyperplane_folder_jacobian_label, 10)
                make_gif_from_figures_in_folder(hyperplane_folder_jacobian_pca,   10)

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
        plt.plot(folds, knot_density, label = model_type, c = colors[model_idx])
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
        plt.plot(folds, test_loss, label = model_type, c = colors[model_idx])
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
        plt.plot(folds, test_accuracy, label = model_type, c = colors[model_idx])
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
        plt.plot(folds, knot_density_mean, label = model_type, c = colors[model_idx])
        plt.fill_between(folds, knot_density_mean - knot_density_std, knot_density_mean + knot_density_std, alpha=0.3, color=colors[model_idx])
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
        plt.plot(folds, test_loss_mean, label = model_type, c = colors[model_idx])
        plt.fill_between(folds, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, alpha=0.3, color=colors[model_idx])
        max_folds = max(max_folds, len(test_loss_mean))

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("test loss")
    plt.legend(loc='best')
    plt.xlim([0,max_folds-1])
    plt.title("mean and std of the test loss per fold over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "test_loss.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "test_loss.svg"), bbox_inches='tight')
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
        plt.plot(folds, test_accuracy_mean, label = model_type, c = colors[model_idx])
        plt.fill_between(folds, test_accuracy_mean - test_accuracy_std, test_accuracy_mean + test_accuracy_std, alpha=0.3, color=colors[model_idx])
        max_folds = max(max_folds, len(test_accuracy_mean))

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

        new_row[model_type+"_knot_density_max"] = knot_density_last_experiment_this_model.max().item()
        new_row[model_type+"_knot_density_end"] = knot_density_last_experiment_this_model[-1].item()

        test_loss_last_experiment_this_model = test_loss_over_experiments[model_idx][-1]
        new_row[model_type+"_test_loss_end"] = test_loss_last_experiment_this_model[-1].item()

        test_accuracy_last_experiment_this_model = test_accuracy_over_experiments[model_idx][-1]
        new_row[model_type+"_test_accuracy_end"] = test_accuracy_last_experiment_this_model[-1].item()


    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(os.path.join(results_dir_with_parent, "parameters.csv"))

    # make the parallel coordinates plot
    for model_idx, model_type in enumerate(model_types):
        fig, ax = plt.subplots(1,1, figsize=(16,8))
        plot_df_as_parallel_coordinates(df, 
                                        [model_type+"_knot_density_max",  model_type+"_knot_density_end", model_type+"_test_loss_end", model_type+"_test_accuracy_end"], 
                                        model_type+"_test_accuracy_end",
                                        host = ax, title=model_type, accuracy_scale_collumns = [model_type+"_test_accuracy_end"],
                                        same_y_scale_collumns = [[model_type+"_knot_density_max",  model_type+"_knot_density_end"]])    
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir_with_parent, f"parallel_coordinates_{model_type}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir_with_parent, f"parallel_coordinates_{model_type}.svg"), bbox_inches='tight')
        plt.close()

