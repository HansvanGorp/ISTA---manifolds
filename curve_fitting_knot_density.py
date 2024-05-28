""" 
In this script we test out fitting a curve through an estimation of the knot density of ISTA and LISTA
"""

# %% imports
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import os

# %% parameters
parent_experiment_folder= "knot_denisty_results"
experiment_folder = "knot_density_experiment_only_changing_A_M8_N16_K4"
experiment_id = 0 

results_dir_this_experiment = os.path.join(parent_experiment_folder, experiment_folder, str(experiment_id))

# %% load the data
knot_density_ista  = torch.load(os.path.join(results_dir_this_experiment, "knot_density_ISTA.tar"))
knot_density_lista = torch.load(os.path.join(results_dir_this_experiment, "knot_density_LISTA.tar"))


# %% now plot this
fig, axs = plt.subplots(2,1, figsize=(10,10))

axs[0].plot(knot_density_ista, label="ISTA")
axs[0].plot(knot_density_lista, label="LISTA")
axs[0].set_title("Knot density")


plt.show()