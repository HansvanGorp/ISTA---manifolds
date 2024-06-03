"""
This file creates a large experiment to test the knot density of ISTA and LISTA in different conditions.
For this, we use the functions layed out in ista.py
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

# local imports
import ista
from parallel_coordinates import plot_df_as_parallel_coordinates

# %% constants
model_types = ["ISTA", "LISTA", "RLISTA"] # the model types we are comparing
nr_of_model_types = len(model_types) # the number of models we are comparing, ISTA, LISTA and RLISTA
colors = ["tab:blue", "tab:orange", "tab:green"] # the colors for the models

# %% functions
def sample_experiment(config):
    """
    This function will sample parameters that vary to create an experiment.
    """
    valid_experiment = False

    while valid_experiment is False:
        # sample the parameters that vary
        M = torch.randint(config["data_that_varies"]["M"]["min"], config["data_that_varies"]["M"]["max"] + 1, (1,)).item()
        N = torch.randint(config["data_that_varies"]["N"]["min"], config["data_that_varies"]["N"]["max"] + 1, (1,)).item()
        K = torch.randint(config["data_that_varies"]["K"]["min"], config["data_that_varies"]["K"]["max"] + 1, (1,)).item()

        # check if the parameters are valid
        if M <= N and K <= M:
            valid_experiment = True

    # create the A matrix that belongs to these parameters
    A = ista.create_random_matrix_with_good_singular_values(M, N)
    
    return M, N, K, A


# %% load the configuration file
config_file = "config_knot_density_experiment_refactored.yaml"
with open(config_file, 'r') as file:
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

# initiale the df for the parallel coordinates plot, each row will be an experiment
# the df has the following columns: M, N, K, noise_std, mu, lambda, knot_density_ista_max,  knot_density_ista_end, knot_density_lista_max, knot_density_lista_end
df = pd.DataFrame()

# TODO: delete this later
regularization_weights = (np.arange(100) + 1)/100.0 # the regularization weights to test, from 0.01 to 1

# loop
for experiment_id in tqdm(range(config["max_nr_of_experiments"]), position=0, desc="running experiments", leave=True):
    tqdm_leave = False if experiment_id < config["max_nr_of_experiments"]-1 else True # set tqdm_leave to False if this is not the last experiment

    # TODO: delete this later
    config["RLISTA"]["regularization"]["weight"] = float(regularization_weights[experiment_id])
    
    # sample the parameters for the experiment that vary, untill a valid experiment is found
    M, N, K, A = sample_experiment(config)

    # create the directory for the experiment
    results_dir_this_experiment = os.path.join(results_dir_with_parent, str(experiment_id))
    os.makedirs(results_dir_this_experiment, exist_ok=True)

     # save the parameters of the experiment in a .yaml file in the experiment folder/str(experiment_id)
    with open(os.path.join(results_dir_this_experiment, "parameters.yaml"), 'w') as file:
        yaml.dump({"M": M, "N": N, "K": K, "regularization_weight": config["RLISTA"]["regularization"]["weight"]}, file)

    # save the A matrix in a .tar file
    torch.save(A, os.path.join(results_dir_this_experiment, "A.tar"))


    # %% loop over each model type
    for model_idx, model_type in enumerate(model_types):
        # get the model config for this model type
        model_config = config[model_type]

        # create a directory for this model type
        model_folder = os.path.join(results_dir_this_experiment, model_type)
        os.makedirs(model_folder, exist_ok=True)

        # check if this is ISTA, in which case the parameters are found by grid search
        if model_type == "ISTA":
            # perform grid search on ISTA for the best lambda and mu for these parameters
            data_generator_initialized = lambda: ista.data_generator(A, model_config["nr_points_to_use"], K, config["data_that_stays_constant"]["x_magnitude"], 
                                                                     N, config["device"], noise_std = config["data_that_stays_constant"]["noise_std"])
            
            mus      = torch.linspace(model_config["mu"]["min"], model_config["mu"]["max"], model_config["mu"]["nr_points"])
            _lambdas = torch.linspace(model_config["lambda"]["min"], model_config["lambda"]["max"], model_config["lambda"]["nr_points"])

            mu, _lambda = ista.grid_search_ista(A, data_generator_initialized, mus, _lambdas, model_config["nr_folds"], forgetting_factor = model_config["weighting_for_first_fold"], 
                                                device=config["device"], tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave)
            
            # save the results of the grid search in the results directroy in a .yaml file
            with open(os.path.join(model_folder, "grid_search_results.yaml"), 'a') as file:
                yaml.dump({"mu": mu.cpu().item(), "lambda": _lambda.cpu().item()}, file)

            # create the ISTA model with the found parameters
            model = ista.ISTA(A, mu = mu, _lambda = _lambda, K = model_config["nr_folds"], device = config["device"])

        # otherwise, the model is LISTA or RLISTA, and needs to be trained
        else:
            # create the model using the parameters in the config file
            model = ista.LISTA(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], K = model_config["nr_folds"], 
                               device = config["device"], initialize_randomly = False)
            
            # train the model
            data_generator_initialized = lambda: ista.data_generator(A, model_config["batch_size"], K, config["data_that_stays_constant"]["x_magnitude"], N,
                                                                     config["device"], noise_std = config["data_that_stays_constant"]["noise_std"])
    
            forgetting_factor = model_config["weighting_for_first_fold"]**(1/config["LISTA"]["nr_folds"])

            if model_type == "RLISTA":
                 model, train_loss  =  ista.train_lista(model, data_generator_initialized, model_config["nr_of_batches"], forgetting_factor,
                                                        model_config["learning_rate"], patience= model_config["patience"], show_loss_plot = False, tqdm_position=1,
                                                        verbose=True, tqdm_leave=tqdm_leave, loss_folder =model_folder, save_name = model_type,
                                                        regularize = True, regularize_config = model_config["regularization"])
            else:    
                model, train_loss   =  ista.train_lista(model, data_generator_initialized, model_config["nr_of_batches"], forgetting_factor,
                                                        model_config["learning_rate"], patience= model_config["patience"], show_loss_plot = False, tqdm_position=1,
                                                        verbose=True, tqdm_leave=tqdm_leave, loss_folder =model_folder, save_name = model_type,
                                                        regularize = False)
                
            # save the losses in a .tar files
            torch.save(train_loss, os.path.join(model_folder, "train_loss.tar"))
            

        # perform knot density analysis on the model
        knot_density = ista.knot_density_analysis(model, model_config["nr_folds"], A, nr_paths = config["Path"]["nr_paths"], anchor_point_std = config["Path"]["anchor_point_std"],
                                                  nr_points_along_path=config["Path"]["nr_points_along_path"], path_delta=config["Path"]["path_delta"], save_folder = model_folder,
                                                  save_name = "knot_density_"+ model_type, verbose = True, color = colors[model_idx], tqdm_position=1, tqdm_leave=tqdm_leave)
        
        # save the knot densities in a .tar file and to the lists
        torch.save(knot_density, os.path.join(model_folder, "knot_density.tar"))
        knot_density_over_experiments[model_idx].append(knot_density)

        # evaluate the model on the test set
        y, x         = ista.data_generator(A, config["support_accuracy_nr_points_to_use"],  K, config["data_that_stays_constant"]["x_magnitude"],     
                                           N, config["device"], noise_std = config["data_that_stays_constant"]["noise_std"])
        
        test_accuracy, test_loss = ista.support_accuracy_analysis( model, model_config["nr_folds"],  A, y, x, save_folder = model_folder, save_name = model_type, 
                                                                  verbose = True, color = 'tab:blue', tqdm_position=1, tqdm_leave=tqdm_leave)
        
        # save the test loss in a .tar file and to the lists
        torch.save(test_loss, os.path.join(model_folder, "test_loss.tar"))
        test_loss_over_experiments[model_idx].append(test_loss)


        # visualize the results in a 2D plane
        hyperplane_config = config["Hyperplane"]
        if hyperplane_config["enabled"]:
            hyperplane_folder = os.path.join(model_folder, "hyperplane")

            ista.visual_analysis_of_ista(model, model_config["nr_folds"], hyperplane_config["nr_points_along_axis"], hyperplane_config["margin"], hyperplane_config["indices_of_projection"],
                                         A, save_folder = hyperplane_folder, tqdm_position=1, tqdm_leave= tqdm_leave, verbose = True, magntiude=hyperplane_config["magnitude"])
            
            # make gifs of the results?
            if hyperplane_config["make_gif"]:
                ista.make_gif_from_figures_in_folder(hyperplane_folder,   10)

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


    # add results to the df for the parallel coordinates plot
    new_row = pd.DataFrame({"M": M, "N": N, "K": K}, index=[0])

    for model_idx, model_type in enumerate(model_types):
        knot_density_over_experiments_this_model = knot_density_over_experiments[model_idx]
        knot_density_over_experiments_this_model = torch.stack(knot_density_over_experiments_this_model)
        knot_density_max  = knot_density_over_experiments_this_model.max(dim=0)
        knot_density_end  = knot_density_over_experiments_this_model[:,-1]

        new_row[model_type+"_knot_density_max"] = knot_density_mean.max().item()
        new_row[model_type+"_knot_density_end"] = knot_density_end.mean().item()

        test_loss_over_experiments_this_model = test_loss_over_experiments[model_idx]
        test_loss_over_experiments_this_model = torch.stack(test_loss_over_experiments_this_model)
        test_loss_end = test_loss_over_experiments_this_model[:,-1]

        new_row[model_type+"_test_loss_end"] = test_loss_end.mean().item()


    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(os.path.join(results_dir_with_parent, "parameters.csv"))

    # make the parallel coordinates plot
    for model_idx, model_type in enumerate(model_types):
        fig, ax = plt.subplots(1,1, figsize=(16,8))
        plot_df_as_parallel_coordinates(df, 
                                        [model_type+"_knot_density_max",  model_type+"_knot_density_end", model_type+"_test_loss_end"], model_type+"_test_loss_end",
                                        host = ax, title=model_type,
                                        same_y_scale_collumns = [[model_type+"_knot_density_max",  model_type+"_knot_density_end"]])    
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir_with_parent, f"parallel_coordinates_{model_type}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(results_dir_with_parent, f"parallel_coordinates_{model_type}.svg"), bbox_inches='tight')
        plt.close()

