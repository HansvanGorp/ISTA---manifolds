"""
This the effects of the sampling parameters on the performance of the model.
"""

# %% imports
# standard libraries
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.path import Path as pltPath
import pandas as pd
import matplotlib.patheffects as PathEffects

# add parent directory to path
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# local import


# %% seed
torch.random.manual_seed(0)

# %% create a plotting function
def plot_df_as_parallel_coordinates(df, collumns_to_plot, color_collumn, 
                                    perturbation_collumns = [], same_y_scale_collumns = [[]], accuracy_scale_collumns =[], logaritmic_collumns = [], 
                                    title = None, host = None): #NOSONAR
    """
    This function plots a dataframe as a parallel coordinates plot, with bezier curves instead of straight lines.
    The dataframe should have the collumns to plot in the right order.

    inputs:
        df: dataframe to plot
        collumns_to_plot: list of collumns to plot in the right order
        color_collumn: name of the collumn to use for the color of the lines
        perturbation_collumns: list of collumns to always add a small bit of noise to
        same_y_scale_collumns: list of list with the second list containing the collumns that should have the same y scale
        accuracy_scale_collumns: list of collumns that should be scaled between 0 and 100
        logaritmic_collumns: list of collumns to plot on a logaritmic scale
        title: title of the plot
        host: host of the plot, if None, a new figure is created

    outputs:
        fig: figure of the plot
        host: host of the plot
    """


    # transform the dataframe to a numpy array in the right order
    df = df[collumns_to_plot]
    df = df.sort_values(by=[color_collumn])
    df = df.reset_index(drop=True)
    df = df.dropna()
    ys = df.to_numpy()

    # transform the collumns that are to be plotted on a logaritmic scale
    for collumn in logaritmic_collumns:
        collumn_index = collumns_to_plot.index(collumn)
        ys[:, collumn_index] = np.log(ys[:, collumn_index]) / np.log(2)

    # figure out the min and max values of each column
    ymins = ys.min(axis=0) 
    ymaxs = ys.max(axis=0)
    dys   = ymaxs - ymins

    # find the collumns that should have the same y scale, make them have the largest y scale of the group (most min and most max)
    for same_y_scale_group in same_y_scale_collumns:
        # get the idxs
        same_y_scale_group_indices = [collumns_to_plot.index(x) for x in same_y_scale_group]
        # get the values
        ymins_group = ymins[same_y_scale_group_indices]
        ymaxs_group = ymaxs[same_y_scale_group_indices]
        # set the values to the largest values
        ymins[same_y_scale_group_indices] = ymins_group.min()
        ymaxs[same_y_scale_group_indices] = ymaxs_group.max()
        dys[same_y_scale_group_indices] = ymaxs[same_y_scale_group_indices] - ymins[same_y_scale_group_indices]


    #  all collumns that should be scaled between 0 and 100
    for accuracy in accuracy_scale_collumns:
        accuracy_idx = collumns_to_plot.index(accuracy)
        
        # set the ymins and ymaxs
        ymins[accuracy_idx] = 0
        ymaxs[accuracy_idx] = 100
        # limit the values of ys to 0 and 100
        ys[:, accuracy_idx] = np.clip(ys[:, accuracy_idx], 0, 100)

    if "kappa" in collumns_to_plot:
        kappa_idx = collumns_to_plot.index("kappa")
        # set the ymins and ymaxs
        ymins[kappa_idx] = 0
        ymaxs[kappa_idx] = 1
        # limit the values of ys to 0 and 1
        ys[:, kappa_idx] = np.clip(ys[:, kappa_idx], 0, 1)

    # if y min is close to 0, i.e. 1 or less, set it to 0
    ymins[ymins <= 1] = 0

    # if dys is 0, set it to 1, so as to not get a division by 0 error
    dys[dys == 0] = 1

    # perturbation, loop over collums, if collum is in perturbation collumns, always add a small bit of noise
    noise_generator = np.random.default_rng(0)
    for i in range(ys.shape[1]):
        if collumns_to_plot[i] in perturbation_collumns:
            # add a small bit of noise to the collumn
            ys[:, i] += noise_generator.normal(0, dys[i] * 0.005, ys.shape[0])

    # add 5% padding
    ymins = ymins - dys * 0.05 # add 5% padding
    ymaxs = ymaxs + dys * 0.05 # add 5% padding
    dys   = ymaxs - ymins

    # transform 
    zs = np.zeros_like(ys)
    zs[:, 0]  = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    # get the coloring values and normalize them between 0 and 1
    color_collumn_index = collumns_to_plot.index(color_collumn)
    color_values = ys[:, color_collumn_index]
    if color_values.max() == color_values.min():
        color_values = np.zeros_like(color_values)
    else:
        color_values = (color_values - color_values.min()) / (color_values.max() - color_values.min())
    
    # create the figure
    if host is None:
        fig, host = plt.subplots(figsize=(10, 6))
    else:
        fig = host.get_figure()

    # create the context for the plot
    collum_names = collumns_to_plot
    for collumn in logaritmic_collumns:
        collumn_index = collumns_to_plot.index(collumn)
        collum_names[collumn_index] = "log2(" + collumns_to_plot[collumn_index] + ")" # change the name of the collumn to log2(collumn_name)

    axes = [host] + [host.twinx() for _ in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        # set the limits of the axis, and make the top and bottom spine invisible
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # if not the first axis, remove the left spine and move the right spine to the left
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks_position('right')
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

        # using path effects, set the tick labels to have a black outline and the text itself to be white
        for label in ax.get_yticklabels():
            label.set_fontsize(12)
            label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])
            label.set_color("w")

    # change some collumn_names to be easier to read:
    collum_names = [x.replace("knot_density_ista_max",  "Max Knot Density") for x in collum_names]
    collum_names = [x.replace("knot_density_ista_end",  "End Knot Density") for x in collum_names]
    collum_names = [x.replace("knot_density_lista_max", "Max Knot Density") for x in collum_names]
    collum_names = [x.replace("knot_density_lista_end", "End Knot Density") for x in collum_names]
    collum_names = [x.replace("support_accuracy_ista_end_ood",  "Support Accuracy OOD") for x in collum_names]
    collum_names = [x.replace("support_accuracy_lista_end_ood", "Support Accuracy OOD") for x in collum_names]
    collum_names = [x.replace("support_accuracy_ista_end",  "Support Accuracy") for x in collum_names]
    collum_names = [x.replace("support_accuracy_lista_end", "Support Accuracy") for x in collum_names]

    # set the x axis
    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(collum_names, fontsize=14)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()

    # set the title
    if title is not None:
        host.set_title(title, fontsize=18)

    # plot the lines
    colormap = cm.cividis
    for j in range(ys.shape[0]):
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, ys.shape[1] - 1, ys.shape[1] * 3 - 2, endpoint=True)],
                        np.repeat(zs[j, :], 3)[1:-1]))
        codes = [pltPath.MOVETO] + [pltPath.CURVE4 for _ in range(len(verts) - 1)]
        path = pltPath(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', lw=3, alpha=0.5, edgecolor=colormap(color_values[j]))
        host.add_patch(patch)

    # end of function
    plt.tight_layout()
    return fig, host