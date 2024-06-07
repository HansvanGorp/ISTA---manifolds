"""
This script creates the functions used to analyze the linear regions of (RL)ISTA along a hyperplane.
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

from ista import ISTA

# %% helper functions
def extract_linear_regions_from_jacobian(jacobian: torch.tensor):
        """
        Given a Jacobian matrix, extract the linear regions from it.

        inputs:
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M) or (batch_size, N, 2) if jacobian_projection was used, we will call the last dimension Z

        outputs:
        - nr_of_regions (int): the number of linear regions 
        - norms (torch.tensor of floats): the norm of the jacobian matrices in the same order as the linear regions of shape (batch_size)
        - unique_entries (torch.tensor): the unique entries of the jacobian, of shape (nr_of_regions, Z)
        """

        # get the shape of the jacobian
        batch_size, N, Z  = jacobian.shape # let's just call the last dimension Z for now

        # calculate the norm of the jacobian
        norms = torch.linalg.norm(jacobian.view(batch_size, N*Z), ord=2, dim=1)
        
        # reshape jacobian to (batch, N*Z)
        jacobian = jacobian.view(batch_size, N*Z)
    
        # find the unique rows using consecutive on the sorted jacobian
        unique_entries, _ = torch.unique(jacobian, dim=0, return_inverse=True)
        nr_of_regions = len(unique_entries)

        return nr_of_regions, norms, unique_entries

def create_y_from_projection(anchors: torch.tensor, nr_points_along_axis: int, margin: float = 0.5, max_magnitude: float = 1.0):
    """
    Given three anchor points, create a meshgrid of y values that forms the plane of the three anchor points.
    The meshgrid is of size (nr_points_along_axis, nr_points_along_axis)

    inputs:
    - anchors (torch.tensor): the anchor points, of shape (3, M)
    - nr_points_along_axis (int): the number of points along the axis
    - margin, by how much to extend both positive and negative along the axis
    - max_magnitude: the maximum magnitude of the anchor points

    outputs:
    - y (torch tensor): the points in a batch of shape (nr_points_along_axis*nr_points_along_axis, M)
    - Z1 (torch tensor): the first axis of the meshgrid, of shape (nr_points_along_axis, nr_points_along_axis)
    - Z2 (torch tensor): the second axis of the meshgrid, of shape (nr_points_along_axis, nr_points_along_axis)
    """
    # create the meshgrid in Z-space, which is the 2D space given by the anchor points
    line = torch.linspace(- margin, max_magnitude + margin, nr_points_along_axis)
    Z1, Z2 = torch.meshgrid(line, line, indexing = 'ij')
    
    # reshape the mesh into a batch and create the three z values for the three anchor points
    z1 = Z1.reshape(-1)
    z2 = Z2.reshape(-1)

    # create the y values from the anchor points
    y = anchors[0,:] + z1.unsqueeze(1)*(anchors[1,:] - anchors[0,:]) + z2.unsqueeze(1)*(anchors[2,:] - anchors[0,:])

    # return the results
    return y, Z1, Z2

def create_anchors_from_x_indices(indices: tuple[int,int,int], A:torch.tensor):
    """
    creates anchor points given two indices that should be non-zero in the x-vector.

    inputs:
    - indices: which indices should be non-zero for the x-vectors used to create the three anchor points, if an index is None, that x-vector will be all zeros
    - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension

    outputs:
    - y_anchors (torch.tensor): the y-anchors, of shape (3, M)
    - x_anchors (torch.tensor): the x-anchors, of shape (3, N)
    """
    # extract the N size
    M, N = A.shape

    # create the anchor points
    x_anchors = torch.zeros(3, N)
    y_anchors = torch.zeros(3, M)
    for i in range(3):
        if indices[i] is not None:
            x_anchors[i, indices[i]] = 1

        y_anchors[i] = A @ x_anchors[i]

    return y_anchors, x_anchors

def create_jacobian_projection_from_anchors(anchors: torch.tensor):
    """
    Given three points that a plane needs to pass through, create a projection matrix that projects the Jacobian to a 2D space.

    inputs:
    - anchors (torch.tensor): the anchor points, of shape (3, M)

    outputs:
    - jacobian_projection (torch.tensor): the projection matrix, of shape (M, 2)
    """
    # create the projection matrix
    jacobian_projection = torch.zeros(anchors.shape[1], 2)

    # create the first vector
    jacobian_projection[:,0] = anchors[1,:] - anchors[0,:]

    # create the second vector
    jacobian_projection[:,1] = anchors[2,:] - anchors[0,:]


    # ensure the projection matrix is normalized
    jacobian_projection = torch.nn.functional.normalize(jacobian_projection, dim=0)

    return jacobian_projection

def extract_sparsity_label_from_x(x):
        """
        Given a bunch of x's, extract the sparsity label. That is to say, is the value zero, or non-zero?
        If x is 8 long, each element is either 0 or 1, then we can assign to each x an index from 0 to 2^8-1, i.e. 0 to 255
        This gives me regions of support of x, and I can assign a color to each region of support.

        input:
            x: the x's of shape (batch, N, 1)

        output:
            sparsity_label: the sparsity index of the x's of shape (batch) as in integer
            labels: the unique labels found in the x's
        """
        # step one, binarize the x's into 0 and non-zero
        x = (x!=0) # make sure to use a small value to avoid numerical issues

        # now use binary to decimal conversion
        sparsity_label = torch.sum(x[:,:]*2**(torch.arange(x.shape[1]).to(x.device)), dim=1).long()

        # get the unqiue labels
        labels = torch.unique(sparsity_label)

        return sparsity_label, labels
# %% visual analysis of ISTA along a hyperplane
def visual_analysis_of_ista(ista: ISTA, nr_folds: int, nr_points_along_axis: int, margin: float, indices_of_projection: tuple[int,int,int], A: torch.tensor, 
                            save_folder: str = "test_figures", tqdm_position: int = 0, tqdm_leave: bool = True, verbose: bool = False,
                            magntiude: float = 1.0, magnitude_ood: float = None):
    """
    Creates a visual analysis of the ISTA module. This is done by visualizing the linear regions of the Jacobian, and the sparsity of the x-vector.
    We only visualize in part of the space, namely a hyperplane that passes through three anchor points. This hyperplane is embedded in y-space,
    which is the actual input space of the ISTA module. We project the Jacobian to a 2D space, and visualize the linear regions in this 2D space.

    inputs:
    - ista: the ISTA module
    - nr_folds: the number of iterations
    - nr_points_along_axis: the number of points along the axis of the hyperplane
    - margin: the margin around the hyperplane to visualize
    - indices_of_projection: the indices of the anchor points, A none means the origin, a 0 means x=[1,0,0,0,..] and a 1 means x=[0,1,0,0,..], and so on.
    - A: the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
    - save_folder: the folder to save the figures in

    - tqdm_position: the position of the tqdm bar
    - tqdm_leave: if True, leave the tqdm bar
    - verbose: if True, print a progress bar

    - magntiude: the magnitude of the anchor points, default is 1.0
    - magnitude_ood: the magnitude of the out of distribution anchor points, if None, we do not use out of distribution anchor points. 
                     Note if we do this, the first anchor point should be the origin, i.e. indices_of_projection[0] should be None

    """
    # create the save folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        # empty the folder of its contents
        for file in os.listdir(save_folder):
            os.remove(f"{save_folder}/{file}")

    # figure out if we need to up to magnitude
    if magnitude_ood is not None:
        max_magnitude = magnitude_ood
        assert indices_of_projection[0] is None # if we are using magnitude_ood, the first anchor should be the origin
    else:
        max_magnitude = magntiude

    # we create a projection matrix that projects the jacobian to a 2d space, for visualization, this is done by specifying three anchor points
    # anchor point 0 is where the x-vector is [0,0,0,...,0]
    y_anchors, _ = create_anchors_from_x_indices(indices_of_projection, A)
    
    # create the projection matrix
    jacobian_projection = create_jacobian_projection_from_anchors(y_anchors)

    # create y data from the projection
    y,Z1,Z2 = create_y_from_projection(y_anchors, nr_points_along_axis, margin = margin, max_magnitude = max_magnitude)

    # run the initials function to get the initial x and jacobian
    x, jacobian = ista.get_initial_x_and_jacobian(y.shape[0], calculate_jacobian = True, jacobian_projection = jacobian_projection)

    # create an array of nr regions over the iterations
    nr_regions_arrray = torch.zeros(nr_folds)

    # loop over the iterations
    for fold_idx in tqdm(range(nr_folds), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="visual analysis of ISTA, runnning over folds"):
        with torch.no_grad():
            x, jacobian = ista.forward_at_iteration(x, y, fold_idx, jacobian, jacobian_projection)

        # extract the linear regions from the jacobian
        nr_of_regions, norms, _ = extract_linear_regions_from_jacobian(jacobian)
        norms_reshaped = norms.reshape(nr_points_along_axis, nr_points_along_axis)

        # extract the sparsity label from x
        sparsity_label, unique_labels = extract_sparsity_label_from_x(x)
        sparsity_label_reshaped = sparsity_label.reshape(nr_points_along_axis, nr_points_along_axis)

        # compress the sparsity label to the unique labels
        sparsity_label_reshaped = sparsity_label_reshaped.unique(return_inverse=True)[1].reshape(nr_points_along_axis, nr_points_along_axis)
        unique_labels = sparsity_label_reshaped.unique()

        # create the three names of the anchor points
        anchor_names = ["anchor x-index: "] * 3
        for i in range(3):
            anchor_names[i] += str(indices_of_projection[i]) if indices_of_projection[i] is not None else "origin"

        # plot the results
        plt.figure(figsize=(10, 10))
        plt.title(f"number of linear regions: {nr_of_regions}")
        plt.imshow(norms_reshaped.cpu(), extent=[-margin, max_magnitude + margin, -margin, max_magnitude + margin], cmap = 'cividis', vmin = 0, vmax = torch.quantile(norms_reshaped, 0.95), origin="lower", zorder = -10)
        # scatter three points, at 0,0 and 1,0 and 0,1 and put a legen with the anchor points
        plt.scatter(0,         0,         c= 'white', label = anchor_names[0], zorder = 10, marker='x', s = 50)
        plt.scatter(magntiude, 0,         c= 'white', label = anchor_names[1], zorder = 10, marker='o', s = 50)
        plt.scatter(0,         magntiude, c= 'white', label = anchor_names[2], zorder = 10, marker='s', s = 50)

        # add the ood anchor if it is there
        if magnitude_ood is not None:
            plt.scatter(magnitude_ood, 0,             c= 'white', label = anchor_names[1] + " ood", zorder = 10, marker='o', s = 50)
            plt.scatter(0,             magnitude_ood, c= 'white', label = anchor_names[2] + " ood", zorder = 10, marker='s', s = 50)

        plt.legend()

        # put a contour plot around the sparsity labels
        plt.contour(Z2, Z1, sparsity_label_reshaped.cpu(), levels=unique_labels.cpu(), colors='k', linewidths=1, linestyles='solid', extent=[-margin, max_magnitude + margin, -margin, max_magnitude + margin], zorder = 1, origin="lower")
        
        # make the plot look nice
        plt.tight_layout()

        # save the figure
        plt.savefig(f"{save_folder}/iteration_{fold_idx}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # save the number of regions
        nr_regions_arrray[fold_idx] = nr_of_regions
    
    # plot the number of regions over the iterations
    plt.figure()
    plt.plot(nr_regions_arrray,'-')
    plt.xlabel("iteration")
    plt.ylabel("number of linear regions")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_folder}/nr_regions_over_iterations.jpg", dpi=300, bbox_inches='tight') # save the figure as a jpg
    plt.savefig(f"{save_folder}/nr_regions_over_iterations.svg", bbox_inches='tight')          # save the figure as a svg
    plt.close()

    return nr_regions_arrray