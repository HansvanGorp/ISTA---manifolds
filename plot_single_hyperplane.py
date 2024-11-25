"""
This script will load a single experiment as specified in the config file and plot the hyperplane of the model.
"""

# %% imports
from pathlib import Path
# standard library imports
import torch
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("/ISTA---manifolds/ieee.mplstyle")
plt.rcParams['text.usetex'] = True
import os
import numpy as np
import warnings
import yaml
import shutil
import time
import pandas as pd

# local imports
import ista
import hyper_plane_analysis  as ha

# set the seed for reproducability
np.random.seed(0)
torch.manual_seed(0)

# %% load the configuration file
plot_config_file = Path("configs/config_plot_single_hyperplane.yaml")
with open(plot_config_file, 'r') as file:
    plot_config = yaml.load(file, Loader=yaml.FullLoader)

model_type = plot_config["model"]

# %% using the config file, load the configuration of the experiment
results_dir    = os.path.join("knot_denisty_results", plot_config["results_dir"])
experiment_dir = os.path.join(results_dir, plot_config["experiment_id"])
model_dir      = os.path.join(experiment_dir, model_type)

# load the results configuration
results_config_file = os.path.join(results_dir, "config.yaml")
with open(results_config_file, 'r') as file:
    results_config = yaml.load(file, Loader=yaml.FullLoader)

# load the A matrix
A_file = os.path.join(experiment_dir, "A.tar")
A = torch.load(A_file)

# %% load the model
if model_type == "ISTA":
    # get the model config
    model_config = results_config[model_type]

    # if this is ista, we can load grid search results
    best_mu_and_lambda_file = os.path.join(model_dir, "best_mu_and_lambda.yaml")
    with open(best_mu_and_lambda_file, 'r') as file:
        best_mu_and_lambda = yaml.load(file, Loader=yaml.FullLoader)

    mu      = best_mu_and_lambda["mu"]
    _lambda = best_mu_and_lambda["lambda"]

    # create the ISTA model with the loaded parameters
    model = ista.ISTA(A, mu = mu, _lambda = _lambda, nr_folds = model_config["nr_folds"], device = results_config["device"])

else:
    # get the model config
    model_config = results_config[model_type]

    # load the LISTA/RLISTA model
    state_dict_file = os.path.join(model_dir, f"{model_type}_state_dict.tar")
    state_dict = torch.load(state_dict_file)

    model = ista.LISTA(A, mu = model_config["initial_mu"], _lambda = model_config["initial_lambda"], nr_folds = model_config["nr_folds"], 
                               device = results_config["device"], initialize_randomly = False)

    model.load_state_dict(state_dict)

# %% get the resulting hyperplane at the desired iteration
# get the hyperplane config, with hasattr to allow for defaults
hyperplane_config       = plot_config["Hyperplane"]
nr_points_in_batch      = hyperplane_config.get("nr_points_in_batch", 1024)
nr_points_along_axis    = hyperplane_config.get("nr_points_along_axis", 1024)
margin                  = hyperplane_config.get("margin", 0.5)
indices_of_projection   = hyperplane_config.get("indices_of_projection", [None,0,1])
anchor_on_y_instead     = hyperplane_config.get("anchor_on_y_instead", False)
magntiude               = hyperplane_config.get("magnitude", 1.0)
tolerance               = hyperplane_config.get("tolerance", None)
draw_decision_boundary  = hyperplane_config.get("draw_decision_boundary", False)
plot_data_regions       = hyperplane_config.get("plot_data_regions", False)
data_region_extend      = hyperplane_config.get("data_region_extend", [0.5, 1.5])
K                       = hyperplane_config.get("K", 4)
symmetric               = hyperplane_config.get("symmetric", False)

color_by = plot_config["color_by"]
max_magnitude = magntiude

# anchor point 0 is where the x-vector is [0,0,0,...,0]
y_anchors, _ = ha.create_anchors_from_x_indices(indices_of_projection, A, anchor_on_y_instead= anchor_on_y_instead)

# create the projection matrix
jacobian_projection = ha.create_jacobian_projection_from_anchors(y_anchors)

# create y data from the projection
y,Z1,Z2 = ha.create_y_from_projection(y_anchors, nr_points_along_axis, margin = margin, max_magnitude = max_magnitude, symmetric = symmetric)

# %% run the model to the desired iteration
# run the initials function to get the initial x and jacobian
x, jacobian = model.get_initial_x_and_jacobian(y.shape[0], calculate_jacobian = True, jacobian_projection = jacobian_projection, overwite_device = "cpu")

# figure out how to batch the input
total_nr_points = nr_points_along_axis * nr_points_along_axis
assert total_nr_points % nr_points_in_batch == 0, "nr_points_in_batch should be a divisor of total_nr_points"
nr_batches = total_nr_points // nr_points_in_batch

# loop over the data iterations untill we reach the desired iteration
for batch_idx in tqdm(range(nr_batches), desc="running over batches", position=0, leave=True):
    # get the batched data and move it to the device
    x_batch = x[batch_idx * nr_points_in_batch : (batch_idx + 1) * nr_points_in_batch].to(results_config["device"])
    y_batch = y[batch_idx * nr_points_in_batch : (batch_idx + 1) * nr_points_in_batch].to(results_config["device"])
    jacobian_batch = jacobian[batch_idx * nr_points_in_batch : (batch_idx + 1) * nr_points_in_batch].to(results_config["device"])

    for fold_idx in tqdm(range(plot_config["iteration"]), desc="running to correct iteration", position=1, leave=False):
        with torch.no_grad():
            x_batch, jacobian_batch = model.forward_at_iteration(x_batch, y_batch, fold_idx, jacobian_batch, jacobian_projection)

    # move the data back to the cpu and save to the correct location
    x[batch_idx * nr_points_in_batch : (batch_idx + 1) * nr_points_in_batch] = x_batch.cpu()
    jacobian[batch_idx * nr_points_in_batch : (batch_idx + 1) * nr_points_in_batch] = jacobian_batch.cpu()
    
# %% we have now reached the desired iteration, we can now plot the hyperplane
# create the xmin, xmax, ymin, ymax
if symmetric:
    xmin = -max_magnitude - margin
    ymin = -max_magnitude - margin
else:
    xmin = -margin
    ymin = -margin

xmax = max_magnitude + margin
ymax = max_magnitude + margin

# if we are using jacboian labels, we need to create a map to colors object
if color_by == "jacobian_label":
    map_to_colors = ha.MapToColors(10)

# extract the linear regions from the jacobian
nr_of_regions, norms, _, jacobian_labels = ha.extract_linear_regions_from_jacobian(jacobian, tolerance = tolerance)    

# extract the sparsity label from x
sparsity_label, unique_labels = ha.extract_sparsity_label_from_x(x)
sparsity_label_reshaped = sparsity_label.reshape(nr_points_along_axis, nr_points_along_axis)

# compress the sparsity label to the unique labels
sparsity_label_reshaped = sparsity_label_reshaped.unique(return_inverse=True)[1].reshape(nr_points_along_axis, nr_points_along_axis)
unique_labels = sparsity_label_reshaped.unique()

# figure out what to color by
if color_by == "norm":
    norms_reshaped = norms.reshape(nr_points_along_axis, nr_points_along_axis)
    norms_reshaped = torch.log(norms_reshaped + 1) 
    color_data = norms_reshaped.cpu()
    cmap = 'cividis'
elif color_by == "jacobian_label":
    jacobian_labels_reshaped = jacobian_labels.reshape(nr_points_along_axis, nr_points_along_axis)
    # color_data = map_to_colors(jacobian_labels_reshaped).cpu()
    color_data = jacobian_labels_reshaped
    cmap = 'tab20'
else:
    raise ValueError("color_by should be either 'norm' or 'jacobian_label'")

# plot the results with exact dpi
# dpi = matplotlib.rcParams['figure.dpi']
# nr_pixels_along_axis = nr_points_along_axis
# fig_size_along_axis = nr_pixels_along_axis / dpi

# fig = plt.figure(figsize=(fig_size_along_axis, fig_size_along_axis))
fig = plt.figure(figsize=(3, 3))


if plot_config["axis_off"]:
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
elif anchor_on_y_instead:
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ax.set_xlabel(r'$\mathbf{y}_1$', fontsize=11)
    ax.set_ylabel(r'$\mathbf{y}_2$', fontsize=11)
else:
    ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

ax.imshow(color_data, cmap=cmap, origin="lower", extent=[xmin, xmax, ymin, ymax])

if plot_data_regions:
    data_on_plane = ha.DataOnPlane(A, data_region_extend, y_anchors, K=K, consider_partials=False, only_positive = True)
    data_on_plane.plot_data_regions(show_legend=False, colors = ["white","white","white"], ax = ax)

if draw_decision_boundary:
    ax.contour(Z2, Z1, sparsity_label_reshaped.cpu(), levels=unique_labels.cpu(), colors='k', linewidths=0.5, linestyles='solid', extent=[xmin, xmax, ymin, ymax], zorder = 1, origin="lower")

# set limits
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])


if plot_config["draw_path"]:
    # draw a random path with knots
    point_0 = np.array([-1.7,-2])
    point_1 = np.array([0,2])
    point_2 = np.array([2,0])
    point_3 = np.array([0,0])
    point_4 = np.array([1,-1.5])
    path = np.array([point_0, point_1, point_2, point_3, point_4])

    # draw the path
    ax.plot(path[:,0], path[:,1], color = "white", zorder = 2, linewidth = 1)

    # figure out where the path crosses from one jacobian to another, for this we leverage jacobian_labels_reshaped
    for i in range(1, path.shape[0]):
        # find the start and end point
        point_start = path[i-1]
        point_end = path[i]

        # create a linspace of many points between the start and end point
        linspace = np.linspace(point_start, point_end, 1000)

        # now loop over the linspace and find the jacobian label at each point
        for point_0, point_1 in zip(linspace[:-1], linspace[1:]):
            # get the jacobian label at the start and end point
            jacobian_label_start = jacobian_labels_reshaped[int((point_0[1] - ymin) / (ymax - ymin) * nr_points_along_axis), int((point_0[0] - xmin) / (xmax - xmin) * nr_points_along_axis)]
            jacobian_label_end   = jacobian_labels_reshaped[int((point_1[1] - ymin) / (ymax - ymin) * nr_points_along_axis), int((point_1[0] - xmin) / (xmax - xmin) * nr_points_along_axis)]

            # if they are different, we need to draw a knot
            if jacobian_label_start != jacobian_label_end:
                ax.plot(point_0[0], point_0[1], 'o', color = "white", markersize = 2, zorder = 2)
    


# create a figure subfolder
figure_dir = "hyperplane_analysis_figures"
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# save the figure to the correct location in .svg format
figure_name = os.path.join(figure_dir, plot_config["figure_name"]+".png")
plt.savefig(figure_name, bbox_inches='tight', pad_inches=0.1)
plt.savefig(os.path.join(figure_dir, plot_config["figure_name"]+".pdf"), bbox_inches='tight', pad_inches=0.1)
print(f"✅ Saved fig to {figure_name}")