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
config_file = "config_knot_density_experiment.yaml"
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
        while(os.path.exists(results_dir_with_parent)):
            time.sleep(1) # wait for the directory to be deleted
        os.makedirs(results_dir_with_parent)
    else:
        raise FileExistsError(f"The results directory {config['results_dir']} already exists.")
else:
    os.makedirs(results_dir_with_parent)

# save the configuration file to the results directory
with open(os.path.join(results_dir_with_parent, "config.yaml"), 'w') as file:
    yaml.dump(config, file)

# %% loop over the experiments to run
print("\nStarting the experiments")

# inialize two lists to store the knot densities of ISTA and LISTA
ista_knot_density_over_experiments  = []
lista_knot_density_over_experiments = []
ista_support_accuracy_over_experiments = []
lista_support_accuracy_over_experiments = []
ista_support_accuracy_over_experiments_ood = []
lista_support_accuracy_over_experiments_ood = []
losses_over_experiments = []

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
    os.makedirs(results_dir_this_experiment)

    # save the parameters of the experiment in a .yaml file in the experiment folder/str(experiment_id)
    with open(os.path.join(results_dir_this_experiment, "parameters.yaml"), 'w') as file:
        yaml.dump({"M": M, "N": N, "K": K}, file)

    # save the A matrix in a .tar file
    torch.save(A, os.path.join(results_dir_this_experiment, "A.tar"))

    # perform grid search on ISTA for the best lambda and mu for these parameters
    data_generator_initialized = lambda: ista.data_generator(A, config["ISTA"]["nr_points_to_use"], K, config["data_that_stays_constant"]["x_magnitude"], N, config["device"], noise_std = config["data_that_stays_constant"]["noise_std"])
    mus = torch.linspace(config["ISTA"]["mu"]["min"], config["ISTA"]["mu"]["max"], config["ISTA"]["mu"]["nr_points"])
    _lambdas = torch.linspace(config["ISTA"]["lambda"]["min"], config["ISTA"]["lambda"]["max"], config["ISTA"]["lambda"]["nr_points"])
    mu, _lambda = ista.grid_search_ista(A, data_generator_initialized, mus, _lambdas, config["ISTA"]["nr_folds"], forgetting_factor = config["ISTA"]["weighting_for_first_fold"], device=config["device"],
                                        tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave)

    # save the results of the grid search in the results directroy in a .yaml file
    with open(os.path.join(results_dir_this_experiment, "grid_search_results.yaml"), 'a') as file:
        yaml.dump({"mu": mu.cpu().item(), "lambda": _lambda.cpu().item()}, file)

    # Using the opimized mu and lambda, initialize the ISTA and LISTA models
    model_ista  = ista.ISTA( A, mu = mu, _lambda= _lambda, K = config["ISTA"]["nr_folds"],  device = config["device"])
    model_lista = ista.LISTA(A, mu = mu, _lambda= _lambda, K = config["LISTA"]["nr_folds"], device = config["device"], initialize_randomly = False)

    # train LISTA
    data_generator_initialized = lambda: ista.data_generator(A, config["LISTA"]["batch_size"], K, config["data_that_stays_constant"]["x_magnitude"], N, config["device"], noise_std = config["data_that_stays_constant"]["noise_std"])
    model_lista,losses = ista.train_lista(model_lista, data_generator_initialized, config["LISTA"]["nr_of_batches"], config["LISTA"]["weighting_for_first_fold"],  config["LISTA"]["learning_rate"],
                                          patience= config["LISTA"]["patience"], show_loss_plot = False, tqdm_position=1, verbose=True, tqdm_leave=tqdm_leave, loss_folder =results_dir_this_experiment)
    
    # save the losses in a .tar files
    torch.save(losses, os.path.join(results_dir_this_experiment, "losses.tar"))


    # store the losses in the list
    losses_over_experiments.append(losses)

    # perform knot density analysis on the mdels
    knot_density_ista = ista.knot_density_analysis(model_ista, config["ISTA"]["nr_folds"], A, 
                                                   nr_paths = config["Path"]["nr_paths"],
                                                   anchor_point_std = config["Path"]["anchor_point_std"],
                                                   nr_points_along_path=config["Path"]["nr_points_along_path"], 
                                                   path_delta=config["Path"]["path_delta"],
                                                   save_folder = results_dir_this_experiment,
                                                   save_name = "knot_density_ISTA", 
                                                   verbose = True, color = 'tab:blue',
                                                   tqdm_position=1, tqdm_leave=tqdm_leave)
    
    knot_density_lista = ista.knot_density_analysis(model_lista, config["LISTA"]["nr_folds"], A, 
                                                   nr_paths = config["Path"]["nr_paths"],
                                                   anchor_point_std = config["Path"]["anchor_point_std"],
                                                   nr_points_along_path=config["Path"]["nr_points_along_path"], 
                                                   path_delta=config["Path"]["path_delta"],
                                                   save_folder = results_dir_this_experiment,
                                                   save_name = "knot_density_LISTA", 
                                                   verbose = True, color = 'tab:orange',
                                                   tqdm_position=1, tqdm_leave=tqdm_leave)
    
    
    # save the knot densities in a .tar file
    torch.save(knot_density_ista,  os.path.join(results_dir_this_experiment, "knot_density_ISTA.tar"))
    torch.save(knot_density_lista, os.path.join(results_dir_this_experiment, "knot_density_LISTA.tar"))

    # make a joint plot of the knot densities
    max_folds = max(config["ISTA"]["nr_folds"], config["LISTA"]["nr_folds"])
    folds_ista = np.arange(0,config["ISTA"]["nr_folds"]+1)
    folds_lista = np.arange(0,config["LISTA"]["nr_folds"]+1)

    plt.figure()
    plt.plot(folds_ista,knot_density_ista,  '-', label = "ISTA",  c = 'tab:blue')
    plt.plot(folds_lista,knot_density_lista,'-', label = "LISTA", c = 'tab:orange')
    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("knot density")
    plt.legend(loc='best')
    plt.xlim([0,max_folds])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_this_experiment, "knot_density_ISTA_and_LISTA.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_this_experiment, "knot_density_ISTA_and_LISTA.svg"), bbox_inches='tight')
    plt.close()

    # store the knot densities in the lists
    ista_knot_density_over_experiments.append(knot_density_ista.cpu())
    lista_knot_density_over_experiments.append(knot_density_lista.cpu())

    # %% support reconstruction accuracy over the folds
    y, x         = ista.data_generator(A, config["support_accuracy_nr_points_to_use"],  K, config["data_that_stays_constant"]["x_magnitude"],     N, config["device"], noise_std = config["data_that_stays_constant"]["noise_std"])
    y_ood, x_ood = ista.data_generator(A, config["support_accuracy_nr_points_to_use"],  K, config["data_that_stays_constant"]["x_magnitude_ood"], N, config["device"], noise_std = config["data_that_stays_constant"]["noise_std_ood"])

    support_accuracy_ista = ista.support_accuracy_analysis( model_ista, config["ISTA"]["nr_folds"],  A, y, x,
                                                            save_folder = results_dir_this_experiment,
                                                            save_name = "support_accuracy_ISTA", 
                                                            verbose = True, color = 'tab:blue',
                                                            tqdm_position=1, tqdm_leave=tqdm_leave)
    
    support_accuracy_ista_ood = ista.support_accuracy_analysis( model_ista, config["ISTA"]["nr_folds"],  A, y_ood, x_ood,
                                                            save_folder = results_dir_this_experiment,
                                                            save_name = "support_accuracy_ISTA_ood", 
                                                            verbose = True, color = 'tab:blue',
                                                            tqdm_position=1, tqdm_leave=tqdm_leave)
    
    support_accuracy_lista = ista.support_accuracy_analysis(model_lista, config["LISTA"]["nr_folds"], A, y, x,
                                                            save_folder = results_dir_this_experiment,
                                                            save_name = "support_accuracy_LISTA", 
                                                            verbose = True, color = 'tab:orange',
                                                            tqdm_position=1, tqdm_leave=tqdm_leave)
    
    support_accuracy_lista_ood = ista.support_accuracy_analysis(model_lista, config["LISTA"]["nr_folds"], A, y_ood, x_ood,
                                                            save_folder = results_dir_this_experiment,
                                                            save_name = "support_accuracy_LISTA_ood", 
                                                            verbose = True, color = 'tab:orange',
                                                            tqdm_position=1, tqdm_leave=tqdm_leave)
    
    
    # make a joint plot of the support accuracies
    max_folds = max(config["ISTA"]["nr_folds"], config["LISTA"]["nr_folds"])
    folds_ista = np.arange(0,config["ISTA"]["nr_folds"]+1)
    folds_lista = np.arange(0,config["LISTA"]["nr_folds"]+1)

    plt.figure()
    plt.plot(folds_ista,support_accuracy_ista,      '-',  label = "ISTA",     c = 'tab:blue')
    plt.plot(folds_lista,support_accuracy_ista_ood, '--', label = "ISTA OOD", c = 'tab:blue')
    plt.plot(folds_lista,support_accuracy_lista,    '-',  label = "LISTA",    c = 'tab:orange')
    plt.plot(folds_lista,support_accuracy_lista_ood,'--', label = "LISTA OOD",c = 'tab:orange')
    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("support accuracy")
    plt.legend(loc='best')
    plt.xlim([0,max_folds])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_this_experiment, "support_accuracy_ISTA_and_LISTA.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_this_experiment, "support_accuracy_ISTA_and_LISTA.svg"), bbox_inches='tight')
    plt.close()

    # save the support accuracies in a .tar file
    torch.save(support_accuracy_ista,  os.path.join(results_dir_this_experiment, "support_accuracy_ISTA.tar"))
    torch.save(support_accuracy_lista, os.path.join(results_dir_this_experiment, "support_accuracy_LISTA.tar"))
    torch.save(support_accuracy_ista_ood,  os.path.join(results_dir_this_experiment, "support_accuracy_ISTA_ood.tar"))
    torch.save(support_accuracy_lista_ood, os.path.join(results_dir_this_experiment, "support_accuracy_LISTA_ood.tar"))

    # store the support accuracies in the lists
    ista_support_accuracy_over_experiments.append(support_accuracy_ista.cpu())
    lista_support_accuracy_over_experiments.append(support_accuracy_lista.cpu())
    ista_support_accuracy_over_experiments_ood.append(support_accuracy_ista_ood.cpu())
    lista_support_accuracy_over_experiments_ood.append(support_accuracy_lista_ood.cpu())

    # %%
    """
    ------------------------------------------------------------------------------------------------------------------------------------------------
    After this follow plots and analysis that are done over all the experiments, and not per experiment
    ------------------------------------------------------------------------------------------------------------------------------------------------
    """

    # %% plot the knot densities over the experiments
    plt.figure()

    ista_mean_over_experiments  = torch.stack(ista_knot_density_over_experiments).mean(dim=0)
    ista_std_over_experiments   = torch.stack(ista_knot_density_over_experiments).std(dim=0)
    lista_mean_over_experiments = torch.stack(lista_knot_density_over_experiments).mean(dim=0)
    lista_std_over_experiments  = torch.stack(lista_knot_density_over_experiments).std(dim=0)

    plt.plot(folds_ista, ista_mean_over_experiments, '-', c = 'tab:blue', label = "ISTA")
    plt.fill_between(folds_ista, ista_mean_over_experiments - ista_std_over_experiments, ista_mean_over_experiments + ista_std_over_experiments, alpha=0.3, color='tab:blue')
    plt.plot(folds_lista, lista_mean_over_experiments, '-', c = 'tab:orange', label = "LISTA")
    plt.fill_between(folds_lista, lista_mean_over_experiments - lista_std_over_experiments, lista_mean_over_experiments + lista_std_over_experiments, alpha=0.3, color='tab:orange')
    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("knot density")
    plt.legend(loc='best')
    plt.xlim([0,max_folds])
    plt.title("mean and std of the knot density per fold over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "knot_density_ISTA_and_LISTA_over_experiments.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "knot_density_ISTA_and_LISTA_over_experiments.svg"), bbox_inches='tight')
    plt.close()

    # %% plot the losses over the experiments
    plt.figure()
    batches = np.arange(1, len(losses_over_experiments[0])+1)
    losses_mean_over_experiments = torch.stack(losses_over_experiments).mean(dim=0)
    losses_std_over_experiments = torch.stack(losses_over_experiments).std(dim=0)

    plt.plot(batches,losses_mean_over_experiments, '-', c = 'tab:blue', label = "loss")
    plt.fill_between(batches, losses_mean_over_experiments - losses_std_over_experiments, losses_mean_over_experiments + losses_std_over_experiments, alpha=0.3, color='tab:blue')
    plt.grid()
    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.ylim(0,0.15)
    plt.xlim(1,len(losses_over_experiments[0]))
    plt.title("mean and std of the loss over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "loss_over_experiments.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "loss_over_experiments.svg"), bbox_inches='tight')
    plt.close()

    # %% plot the support accuracy over the experiments
    plt.figure()
    ista_mean_over_experiments  = torch.stack(ista_support_accuracy_over_experiments).mean(dim=0)
    ista_std_over_experiments   = torch.stack(ista_support_accuracy_over_experiments).std(dim=0)
    lista_mean_over_experiments = torch.stack(lista_support_accuracy_over_experiments).mean(dim=0)
    lista_std_over_experiments  = torch.stack(lista_support_accuracy_over_experiments).std(dim=0)

    ista_mean_over_experiments_ood  = torch.stack(ista_support_accuracy_over_experiments_ood).mean(dim=0)
    ista_std_over_experiments_ood   = torch.stack(ista_support_accuracy_over_experiments_ood).std(dim=0)
    lista_mean_over_experiments_ood = torch.stack(lista_support_accuracy_over_experiments_ood).mean(dim=0)
    lista_std_over_experiments_ood  = torch.stack(lista_support_accuracy_over_experiments_ood).std(dim=0)

    plt.plot(folds_ista, ista_mean_over_experiments, '-', c = 'tab:blue', label = "ISTA")
    plt.fill_between(folds_ista, ista_mean_over_experiments - ista_std_over_experiments, ista_mean_over_experiments + ista_std_over_experiments, alpha=0.3, color='tab:blue')
    plt.plot(folds_lista, lista_mean_over_experiments, '-', c = 'tab:orange', label = "LISTA")
    plt.fill_between(folds_lista, lista_mean_over_experiments - lista_std_over_experiments, lista_mean_over_experiments + lista_std_over_experiments, alpha=0.3, color='tab:orange')

    plt.plot(folds_ista, ista_mean_over_experiments_ood, '--', c = 'tab:blue', label = "ISTA OOD")
    plt.fill_between(folds_ista, ista_mean_over_experiments_ood - ista_std_over_experiments_ood, ista_mean_over_experiments_ood + ista_std_over_experiments_ood, alpha=0.3, color='tab:blue')
    plt.plot(folds_lista, lista_mean_over_experiments_ood, '--', c = 'tab:orange', label = "LISTA OOD")
    plt.fill_between(folds_lista, lista_mean_over_experiments_ood - lista_std_over_experiments_ood, lista_mean_over_experiments_ood + lista_std_over_experiments_ood, alpha=0.3, color='tab:orange')

    plt.grid()
    plt.xlabel("fold")
    plt.ylabel("support accuracy")
    plt.legend(loc='best')
    plt.xlim([0,max_folds])
    plt.title("mean and std of the support accuracy per fold over the random experiments")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "support_accuracy_ISTA_and_LISTA_over_experiments.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "support_accuracy_ISTA_and_LISTA_over_experiments.svg"), bbox_inches='tight')
    plt.close()


    # %% make a parralel coordinate plot of the parameters
    # we want to plot the following parameters: M, N, K, noise_std, mu, lambda, knot_density_ista_max,  knot_density_ista_end, knot_density_lista_max, knot_density_lista_end
    #                                           support_accuracy_ista_end, support_accuracy_lista_end, support_accuracy_ista_end_ood, support_accuracy_lista_end_ood

    # step_1, add a new row to the df with the parameters of this experiment
    new_row = pd.DataFrame({"M": M, "N": N, "K": K, "noise_std": noise_std, "mu": mu.cpu().item(), "lambda": _lambda.cpu().item(), 
                            "knot_density_ista_max": knot_density_ista.max().cpu().item(), "knot_density_ista_end": knot_density_ista[-1].cpu().item(),
                            "knot_density_lista_max": knot_density_lista.max().cpu().item(), "knot_density_lista_end": knot_density_lista[-1].cpu().item(),
                            "support_accuracy_ista_end": support_accuracy_ista[-1].cpu().item(), "support_accuracy_lista_end": support_accuracy_lista[-1].cpu().item(),
                            "support_accuracy_ista_end_ood": support_accuracy_ista_ood[-1].cpu().item(), "support_accuracy_lista_end_ood": support_accuracy_lista_ood[-1].cpu().item()
                            }, index=[0])
    
    df = pd.concat([df, new_row], ignore_index=True)

    # step_2, plot the df for ISTa and LISTA
    # ISTA
    fig, ax = plt.subplots(1,1, figsize=(16,8))
    plot_df_as_parallel_coordinates(df, ["K", "M", "N", "knot_density_ista_max",  "knot_density_ista_end", "support_accuracy_ista_end","support_accuracy_ista_end_ood"], 
                                    "support_accuracy_ista_end_ood",
                                    host = ax, title="ISTA",
                                    perturbation_collumns = ["K", "M", "N"], 
                                    same_y_scale_collumns = [["K", "M", "N"],["knot_density_ista_max",  "knot_density_ista_end"]],
                                    accuracy_scale_collumns = ["support_accuracy_ista_end", "support_accuracy_ista_end_ood"])    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "parallel_coordinates_ISTA.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "parallel_coordinates_ISTA.svg"), bbox_inches='tight')
    plt.close()

    # LISTA
    fig, ax = plt.subplots(1,1, figsize=(16,8))
    plot_df_as_parallel_coordinates(df, ["K", "M", "N", "knot_density_lista_max", "knot_density_lista_end", "support_accuracy_lista_end","support_accuracy_lista_end_ood"], 
                                    "support_accuracy_lista_end_ood",
                                    host = ax, title="LISTA",
                                    perturbation_collumns = ["K", "M", "N"], 
                                    same_y_scale_collumns = [["K", "M", "N"],["knot_density_lista_max",  "knot_density_lista_end"]],
                                    accuracy_scale_collumns = ["support_accuracy_lista_end", "support_accuracy_lista_end_ood"])      
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "parallel_coordinates_LISTA.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "parallel_coordinates_LISTA.svg"), bbox_inches='tight')
    plt.close()

    # save the df to a .csv file
    df.to_csv(os.path.join(results_dir_with_parent, "parameters.csv"))


    # %% make a scatter plot ith on the x-axis the max knot_density and on the y-axis the end knot density, do this for ISTA and LISTA
    # get the data from the df for the scatter plot
    knot_density_ista_max = df["knot_density_ista_max"].to_numpy()
    knot_density_ista_end = df["knot_density_ista_end"].to_numpy()

    knot_density_lista_max = df["knot_density_lista_max"].to_numpy()
    knot_density_lista_end = df["knot_density_lista_end"].to_numpy()

    # get the min and max values of the data
    minimum = min(knot_density_ista_max.min(), knot_density_ista_end.min(), knot_density_lista_max.min(), knot_density_lista_end.min())
    maximum = max(knot_density_ista_max.max(), knot_density_ista_end.max(), knot_density_lista_max.max(), knot_density_lista_end.max())
    delta   = (maximum-minimum)

    if delta == 0:
        delta = 1 # if the min and max values are the same, set the delta to 1

    # add 5% to the min and max values
    minimum -= 0.05*delta
    maximum += 0.05*delta

    
    # the plot
    plt.figure(figsize=(6,6))
    plt.scatter(knot_density_ista_max, knot_density_ista_end, c='tab:blue', label="ISTA")
    plt.scatter(knot_density_lista_max, knot_density_lista_end, c='tab:orange', label="LISTA")
    plt.plot([minimum, maximum], [minimum, maximum], '--', c='black')
    plt.xlim(minimum, maximum)
    plt.ylim(minimum, maximum)
    plt.xlabel("max knot density")
    plt.ylabel("end knot density")
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_with_parent, "scatter_plot_knot_density_max_vs_end.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir_with_parent, "scatter_plot_knot_density_max_vs_end.svg"), bbox_inches='tight')
    plt.close()
    