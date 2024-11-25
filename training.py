"""
This file specifies different functions to train (R)(L)ISTA modules
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
from typing import Union
from scipy.spatial import distance

from ista import ISTA, FISTA, LISTA
from data import data_generator, ISTAData
from knot_density_analysis import generate_path

# %% ISTA
def get_support_accuracy(x_hat: torch.tensor, x: torch.tensor, tolerance=0):
    """
    get the support accuracy of a batch of x_hat compared to x
    """
    # make x and x_hat on the same device
    x = x.to(x_hat.device)

    # make the x and x_hat the same shape
    if len(x_hat.shape) != len(x.shape):
        x = x.unsqueeze(2).expand_as(x_hat)

    # get the support accuracy
    support_of_x = (torch.abs(x) <= tolerance)
    support_of_x_hat = (torch.abs(x_hat) <= tolerance)

    equal_support = (support_of_x == support_of_x_hat)

    all_N_equal = equal_support.all(dim=1)

    support_accuracy = 100.0 * all_N_equal.float().mean()

    return support_accuracy

def grid_search_ista(model: Union[ISTA, FISTA], train_data: ISTAData, validation_data: ISTAData, model_config: dict, tqdm_position: int=0, verbose: bool=True, tqdm_leave: bool=True):
    """
    perfrom a grid search for the best mu and lambda for the ISTA module. using the data generator.
    """
    model_name = "ISTA" if isinstance(model, ISTA) else "FISTA"

    # step 1, extract the data from both the training and validation data and combine them, we will use this data to calculate the loss for the grid search
    y_train, x_train = train_data.y, train_data.x
    y_val, x_val     = validation_data.y, validation_data.x

    y = torch.cat((y_train, y_val), dim=0)
    x = torch.cat((x_train, x_val), dim=0)

    # step 2, create the grid
    mus      = torch.linspace(model_config["mu"]["min"], model_config["mu"]["max"], model_config["mu"]["nr_points"])
    _lambdas = torch.linspace(model_config["lambda"]["min"], model_config["lambda"]["max"], model_config["lambda"]["nr_points"])
    losses = torch.zeros(len(mus), len(_lambdas)) + 1e32
    #accuracies = torch.zeros(len(mus), len(_lambdas))

    # step 3, loop over the grid
    for i, mu in enumerate(tqdm(mus, position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc=f"grid search for {model_name}, runnning over mus")):
        for j, _lambda in enumerate(tqdm(_lambdas, position=tqdm_position+1, leave=(tqdm_leave and (i+1)==len(mus)), disable=not verbose, desc=f"grid search for {model_name}, runnning over lambdas")):           
            # change the mu and lambda of the model
            model.reset_params_using_mu_and_lambda(mu, _lambda)
            
            # run the ISTA module
            x_hat,_ = model(y, verbose = False, return_intermediate = True, calculate_jacobian = False)           

            # calculate the loss over the K folds
            losses[i,j] = get_reconstruction_loss(x,x_hat)

            # calculate the support accuracy
            #accuracies[i,j] = get_support_accuracy(x_hat, x)

    # step 4, find the best mu and lambda
    best_idx = torch.argmin(losses)
    #best_idx = torch.argmax(accuracies)
    best_mu_idx = best_idx // len(_lambdas)
    best_lambda_idx = best_idx % len(_lambdas)

    # step 5, get the best mu and lambda
    best_mu = mus[best_mu_idx]
    best_lambda = _lambdas[best_lambda_idx]

    # also reset the model with the best mu and lambda
    model.reset_params_using_mu_and_lambda(best_mu, best_lambda)

    return model, best_mu, best_lambda, losses, mus.numpy(), _lambdas.numpy()

# %% LISTA
def get_loss_on_dataset_over_folds(model: ISTA, datset: ISTAData, l1_weight=1.0, l2_weight=1.0):
    """
    get the loss of a model on an entire dataset over the folds
    """

    y, x = datset.y, datset.x

    with torch.no_grad():
        x_hat,_ = model(y, verbose = False, return_intermediate = True, calculate_jacobian = False)

    loss_per_fold = torch.zeros(model.nr_folds)
    for k in range(model.nr_folds):
        loss_per_fold[k] = get_reconstruction_loss(x_hat[:,:,k], x, l1_weight=l1_weight, l2_weight=l2_weight)

    return loss_per_fold

def get_support_accuracy_on_dataset_over_folds(model: ISTA, datset: ISTAData, tolerance=0):
    """
    get the support accuracy of a model on an entire dataset over the folds
    """

    y, x = datset.y, datset.x

    with torch.no_grad():
        x_hat,_ = model(y, verbose = False, return_intermediate = True, calculate_jacobian = False)

    support_accuracy_per_fold = torch.zeros(model.nr_folds)
    for k in range(model.nr_folds):
        support_accuracy_per_fold[k] = get_support_accuracy(x_hat[:,:,k], x, tolerance=tolerance)

    return support_accuracy_per_fold


def calculate_loss(x_hat: torch.tensor, x: torch.tensor, y: torch.tensor, model: LISTA, model_config: dict, regularize: bool=False):
    # calculate the l1 loss over the K folds
    reconstruction_loss = get_reconstruction_loss(x, x_hat, model_config["l1_weight"], model_config["l2_weight"])

    # now check if we need to regularize
    if regularize:
        regularization_loss = get_regularization_loss(model, model_config["regularization"], y)
    else:
        regularization_loss = torch.zeros(1, device=x.device)

    # get the total loss
    total_loss = reconstruction_loss + regularization_loss

    return total_loss, reconstruction_loss, regularization_loss

def get_reconstruction_loss(x: torch.tensor, x_hat: torch.tensor, l1_weight: float=0.0, l2_weight: float=1.0):
    """
    get the loss of a batch of x_hat compared to x
    """
    # make x and x_hat on the same device
    x = x.to(x_hat.device)

    # make the x and x_hat the same shape
    if len(x_hat.shape) != len(x.shape):
        x = x.unsqueeze(2).expand_as(x_hat)

    # get the L1 loss
    l1 = (torch.abs(x_hat - x)).mean()

    # get the L2 loss
    l2 = ((x_hat - x)**2).mean()
        
    # calculate loss
    loss = l1_weight * l1 + l2_weight * l2

    return loss

def go_over_validation_set(model: LISTA, dataloader_val: torch.utils.data.DataLoader, model_config: dict, regularize: bool=False, tqdm_position: int=0, verbose: bool=True, tqdm_leave: bool=True):
    model.eval()
    val_loss = torch.zeros(3)
    nr_batches = len(dataloader_val)

    for i, (y, x) in enumerate(tqdm(dataloader_val, position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="Going over validation batches")):

        with torch.no_grad():
            x_hat, _ = model(y, verbose = False, return_intermediate = True, calculate_jacobian = False)
            x = x.unsqueeze(2).expand_as(x_hat).to(x_hat.device)

        total_loss, reconstruction_loss, regularization_loss = calculate_loss(x_hat, x, y, model, model_config, regularize)


        # save the losses
        val_loss[0] += total_loss.item()/ nr_batches
        val_loss[1] += reconstruction_loss.item()/ nr_batches
        val_loss[2] += regularization_loss.item()/ nr_batches

    return val_loss

def plot_loss(fraction_idx: torch.tensor, epoch_idx: torch.tensor, fractions: torch.tensor, epochs: torch.tensor,
              train_losses: torch.tensor, val_losses: torch.tensor, 
              save_name: str, show_loss_plot: bool=False, loss_folder: str=None, regularize: bool=False):
    # plot the loss
    train_color = "tab:blue"
    val_color = "tab:orange"

    plt.figure()
    plt.plot(fractions[:fraction_idx+1],train_losses[:fraction_idx+1,0].cpu().numpy(), label="total training loss", linestyle="-", c = train_color)
    plt.plot(epochs[:epoch_idx+1],val_losses[:epoch_idx+1,0].cpu().numpy(),          label="total validation loss", linestyle="-", c = val_color)

    if regularize:
        plt.plot(fractions[:fraction_idx+1],train_losses[:fraction_idx+1,1].cpu().numpy(), label="reconstruction training loss", linestyle="--", c = train_color)
        plt.plot(epochs[:epoch_idx+1],val_losses[:epoch_idx+1,1].cpu().numpy(),          label="reconstruction validation loss", linestyle="--", c = val_color)

        plt.plot(fractions[:fraction_idx+1],train_losses[:fraction_idx+1,2].cpu().numpy(), label="regularization training loss", linestyle=":", c = train_color)
        plt.plot(epochs[:epoch_idx+1],val_losses[:epoch_idx+1,2].cpu().numpy(),          label="regularization validation loss", linestyle=":", c = val_color)

    plt.xlim(0, fractions[fraction_idx])
    plt.ylim(0, val_losses[:epoch_idx+1,0].max()*1.5)
    plt.grid()
    plt.title("loss over the batches")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    
    if loss_folder is None:
        try:
            plt.savefig(f"loss_{save_name}.jpg", dpi=300, bbox_inches='tight')
            plt.savefig(f"loss_{save_name}.svg", bbox_inches='tight')
        except: #NOSONAR
            pass #sometimes the plot is not saved, but that is not a problem, we will save it next time
    else:
        try:
            plt.savefig(f"{loss_folder}/loss_{save_name}.jpg", dpi=300, bbox_inches='tight')
            plt.savefig(f"{loss_folder}/loss_{save_name}.svg", bbox_inches='tight')
        except: #NOSONAR
            pass #sometimes the plot is not saved, but that is not a problem, we will save it next time

    if show_loss_plot:
        plt.show()
    else:
        plt.close()

def train_lista(model: LISTA, train_data: ISTAData, validation_data: ISTAData, model_config: dict, 
                show_loss_plot: bool=False, loss_folder: str=None, save_name: str=None, regularize: bool=False,
                tqdm_position: int=0, verbose: bool=True, tqdm_leave: bool=True, save: bool = True, compute_loss_on_final_output: bool = False):
    """
    perfrom training of (R)LISTA module. using the data.
    """
    # preambule
    dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=model_config["batch_size"],    shuffle=True, drop_last=False)
    dataloader_val = torch.utils.data.DataLoader(validation_data, batch_size=model_config["batch_size"], shuffle=False, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])

    if regularize and model_config['regularization']['type'] == 'pointcloud':
        model_config['regularization']['sigma_y'] = estimate_y_std(train_data)
        # model_config['regularization']['sigma_cloud'] = average_nearest_neighbor_distance(model.train_inputs)
                                 
    nr_of_epochs = model_config["nr_of_epochs"]
    nr_batches_per_epoch = len(dataloader_train)

    train_losses = torch.zeros(nr_of_epochs*nr_batches_per_epoch+1, 3) # the 3 dimensions are for, total, reconstruction and regularization loss
    val_losses   = torch.zeros(nr_of_epochs+1,3)

    epochs    = torch.arange(nr_of_epochs+1)
    fractions = torch.arange(nr_of_epochs*nr_batches_per_epoch + 1)/nr_batches_per_epoch
    patience_counter = 0

    # initial validation
    val_losses[0,:] = go_over_validation_set(model, dataloader_val, model_config, regularize, tqdm_position, verbose, tqdm_leave=False)
    best_loss = val_losses[0,0]
    train_losses[0,:] = val_losses[0,:]

    # loop over the epochs
    for epoch_idx in  tqdm(range(nr_of_epochs), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc=f"training {save_name}, runnning over epochs"):
        # check if we leave the nested bar (only if we are in the last epoch)
        leave_nested_bar = tqdm_leave and (epoch_idx == nr_of_epochs-1)

        # training
        model.train()
        for i, (y, x) in enumerate(tqdm(dataloader_train, position=tqdm_position+1, leave=leave_nested_bar, disable=not verbose, desc="Going over training batches")):
            x_hat, _ = model(y, verbose = False, return_intermediate = (not compute_loss_on_final_output), calculate_jacobian = False)
            if x_hat.shape != x.shape:
                x = x.unsqueeze(2).expand_as(x_hat)
            x = x.to(x_hat.device)

            # calculate the loss
            total_loss, reconstruction_loss, regularization_loss = calculate_loss(x_hat, x, y, model, model_config, regularize)

            # optimizer step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # save the losses
            fraction_idx = epoch_idx*nr_batches_per_epoch + i + 1
            train_losses[fraction_idx,0] = total_loss.item()
            train_losses[fraction_idx,1] = reconstruction_loss.item()
            train_losses[fraction_idx,2] = regularization_loss.item()

            # plot the loss, every so often
            if i % 100 == 0:
                plot_loss(fraction_idx, epoch_idx, fractions, epochs, train_losses, val_losses, save_name, show_loss_plot, loss_folder, regularize)

        # validation
        val_losses[epoch_idx+1,:] = go_over_validation_set(model, dataloader_val, model_config, regularize, tqdm_position = tqdm_position+1, verbose = verbose , tqdm_leave = leave_nested_bar)
        plot_loss(fraction_idx, epoch_idx+1, fractions, epochs, train_losses, val_losses, save_name, show_loss_plot, loss_folder, regularize)
        
        # check if this loss is the current best loss
        if val_losses[epoch_idx+1,0] < best_loss:
            best_loss = val_losses[epoch_idx,0]
            patience_counter = 0
        else:
            patience_counter += 1

        # check if patience is reached, if so, stop
        if patience_counter == model_config["patience"]:
            break

        # after each epoch, save the model of the current epoch
        state_dict = model.state_dict()
        if save:
            torch.save(state_dict,   os.path.join(loss_folder, f"{save_name}_state_dict_epoch_{epoch_idx}.tar"))
            torch.save(train_losses, os.path.join(loss_folder, "train_loss_epoch_{epoch_idx}.tar"))
            torch.save(val_losses,   os.path.join(loss_folder, "val_loss_epoch_{epoch_idx}.tar"))
            
    # save some stuff
    state_dict = model.state_dict()
    torch.save(state_dict,   os.path.join(loss_folder, f"{save_name}_state_dict.tar"))
    torch.save(train_losses, os.path.join(loss_folder, "train_loss.tar"))
    torch.save(val_losses,   os.path.join(loss_folder, "val_loss.tar"))

    return model, train_losses, val_losses
   

# %% regularization
def get_regularization_loss(model: LISTA, regularize_config: dict, y: torch.tensor):
    if regularize_config["type"] == "smooth_jacobian":
        regularization_loss = get_regularization_loss_smooth_jacobian(model, regularize_config)
    elif regularize_config["type"] == "tv_jacobian":
        regularization_loss = get_regularization_loss_tv_jacobian(model, regularize_config)
    elif regularize_config["type"] == "tie_weights":
        regularization_loss = get_regularization_loss_tie_weights(model)
    elif regularize_config["type"] == "pointcloud":
        regularization_loss = get_regularization_loss_pointcloud(model, regularize_config)
    elif regularize_config["type"] == "L2":
        regularization_loss = get_regularization_loss_L2(model)
    elif regularize_config["type"] == "jacobian":
        regularization_loss = get_regularization_loss_jacobian(model, y)
    else:
        raise ValueError("regularize_config['type'] is not valid")
    
    return regularization_loss * regularize_config["weight"]

def get_regularization_loss_jacobian(model: LISTA, y: torch.tensor):
    batch_size = len(y) 
    x, jacobian = model.get_initial_x_and_jacobian(batch_size, calculate_jacobian = True)
    for k in range(model.nr_folds):
        x, jacobian = model.forward_at_iteration(x, y, k, jacobian)
        
    return torch.norm(jacobian)

def get_regularization_loss_L2(model):
    l2_loss = 0
    for param in model.W1:
        l2_loss += (torch.norm(param)**2)
    for param in model.W2:
        l2_loss += (torch.norm(param)**2)
    for param in model.bias:
        l2_loss += (torch.norm(param)**2)
    return l2_loss


def min_nearest_neighbor_distance(vectors):
    """
    Computes the average distance to the nearest neighbor for a list of vectors.

    Parameters:
    vectors (numpy.ndarray): A 2D array where each row represents a vector.

    Returns:
    float: The average nearest neighbor distance.
    """
    n = len(vectors)
    if n < 2:
        raise ValueError("At least two vectors are required to compute nearest neighbor distances.")

    # Compute the pairwise distance matrix
    dist_matrix = distance.cdist(vectors, vectors, 'euclidean')

    # Set the diagonal to infinity to ignore self-distances
    np.fill_diagonal(dist_matrix, np.inf)

    # Find the nearest neighbor distance for each vector
    nearest_distances = np.min(dist_matrix, axis=1)
    
    # Compute the average nearest neighbor distance
    avg_distance = np.median(nearest_distances)
    
    return avg_distance

def get_regularization_loss_pointcloud(lista: LISTA, regularize_config: dict):
    M, N = lista.A.shape
    num_points = regularize_config['N_points']    
    sigma_y = regularize_config['sigma_y']
    cloud_scale = regularize_config['cloud_scale']
    num_clouds = regularize_config['num_clouds']
    if not regularize_config.get("input_sampling", False):
        sampling_center_points = sigma_y*torch.randn(num_clouds, M)
    else:
        sampled_indices = torch.randint(low=0, high=lista.train_inputs.shape[0], size=(num_clouds,))
        sampling_center_points = lista.train_inputs[sampled_indices]
    sigma_reg = sigma_y / cloud_scale
    random_y_points = torch.vstack([center_point + sigma_reg*torch.randn(num_points, M) for center_point in sampling_center_points])
    batch_size = len(random_y_points)
        
    regularization_loss = 0
    x, jacobian = lista.get_initial_x_and_jacobian(batch_size, calculate_jacobian = True)

    # step 3, initialze a jacobian over time tensor of shape (nr_fold, nr_points_along_path, N, M)
    nr_folds = lista.nr_folds
    # jacobian_over_time = torch.zeros(nr_folds, regularize_config["N_points"], N, M, device = lista.device)

    # step 4, loop over the iterations, saving the jacobian at each iteration into the jacobian_over_time tensor
    for k in range(nr_folds):
        x, jacobian = lista.forward_at_iteration(x, random_y_points, k, jacobian)
        jacobian_reshaped = jacobian.reshape((num_clouds, num_points, N*M))
        # compute the difference in jacobian between random pairs of points in the same cloud
        differences = torch.mean(
            torch.tensor([
                torch.mean(torch.abs(pointcloud_jacobians[1:] - pointcloud_jacobians[:-1])) 
                for pointcloud_jacobians in jacobian_reshaped
            ])
        )
        regularization_loss += differences
        # outer_product = jacobian_reshaped @ jacobian_reshaped.T
        # regularization_loss -= torch.mean(outer_product)
        
    return regularization_loss
    

def get_regularization_loss_smooth_jacobian(lista: LISTA, regularize_config: dict):
    """
    get the regularization loss for a LISTA module. This loss is defined as taking a 1D path along the input space, and then taking the jacobian. 
    From the jacobian, we extract the individual regions, based on the fact that the derivative is zero between the points inside a region.
    We then put an l1 loss on the difference between any of the points on a region compared to the neighbouring region that it is closest to.

    e.g we have three regions with 1D jacobians that are:   -5,-2, -2, -2, 1, 1, 1, 7, 7, 7, 7, 7, 17
    Then the first region will have the loss: |-2-1|, the second region will have the loss |-2-2|, and the third region will have the loss |1-7| 
    Note that we ignore the first and last region, as they do not have a left or right neighbour.
    """
    M, N = lista.A.shape

    # step 1, generate a path
    y = generate_path(M, regularize_config["nr_points_along_path"], regularize_config["path_delta"], regularize_config["anchor_point_std"], lista.device)

    # step 2, inialize x and the jacboian
    x, jacobian = lista.get_initial_x_and_jacobian(regularize_config["nr_points_along_path"], calculate_jacobian = True)

    # step 3, initialze a jacobian over time tensor of shape (nr_fold, nr_points_along_path, N, M)
    nr_folds = lista.nr_folds
    jacobian_over_time = torch.zeros(nr_folds, regularize_config["nr_points_along_path"], N, M, device = lista.device)

    # step 4, loop over the iterations, saving the jacobian at each iteration into the jacobian_over_time tensor
    for k in range(nr_folds):
        x, jacobian = lista.forward_at_iteration(x, y, k, jacobian)
        jacobian_over_time[k] = jacobian

    # step 5, reshape the jacobian_over_time tensor to (nr_folds, nr_points_along_path, N*M)
    jacobian_over_time = jacobian_over_time.view(nr_folds, regularize_config["nr_points_along_path"], N*M)

    # step 6, calculate the differences between consecutive points
    with torch.no_grad():
        differences = torch.mean(torch.abs(jacobian_over_time[:,1:,:] - jacobian_over_time[:,:-1,:]), dim = -1)

    # step 7, randomly select a fold index
    fold_idx = torch.randint(0, nr_folds, (1,)).item()

    # get the nr of knots, and the location of the knots
    knot_locations = torch.nonzero(differences[fold_idx,:])[:,0]
    nr_of_knots = len(knot_locations)

    # step 8, loop over each region in the jacobian, except the two edge regions (first and last)
    regularization_loss = 0
    for region_idx in range(1,nr_of_knots):
        # get the indices of the region
        start_idx = knot_locations[region_idx-1].item()
        end_idx   = knot_locations[region_idx].item()

        # get the value of the jacobian of this region, as well as its left and right neighbour
        jacobian_region = jacobian_over_time[fold_idx, start_idx+1:end_idx+1, :]

        # calculate the loss, as the l1 loss to the jacobian of the closest neighbour
        if differences[fold_idx, start_idx] < differences[fold_idx, end_idx]:
            # the left neighbour is the closest
            regularization_loss += torch.abs(jacobian_region - jacobian_over_time[fold_idx, start_idx, :]).mean()
        else:
            # the right neighbour is the closest   
            regularization_loss += torch.abs(jacobian_region - jacobian_over_time[fold_idx, end_idx+1, :]).mean()

    return regularization_loss

def get_regularization_loss_tv_jacobian(lista: LISTA, regularize_config: dict):
    """
    get the regularization loss for a LISTA module. This loss is defined as taking a 1D path along the input space, and then taking the jacobian. 
    We then calculate the total-variation loss on the jacobian, which is the sum of the absolute differences between consecutive points.
    This should smooth it out over time and recude the number of knots in the jacobian.
    """
    M, N = lista.A.shape

    # step 1, generate a path
    y = generate_path(M, regularize_config["nr_points_along_path"], regularize_config["path_delta"], regularize_config["anchor_point_std"], lista.device)

    # step 2, inialize x and the jacboian
    x, jacobian = lista.get_initial_x_and_jacobian(regularize_config["nr_points_along_path"], calculate_jacobian = True)

    # step 3, initialze a jacobian over time tensor of shape (nr_fold, nr_points_along_path, N, M)
    nr_folds = lista.nr_folds
    jacobian_over_time = torch.zeros(nr_folds, regularize_config["nr_points_along_path"], N, M, device = lista.device)

    # step 4, loop over the iterations, saving the jacobian at each iteration into the jacobian_over_time tensor
    for k in range(nr_folds):
        x, jacobian = lista.forward_at_iteration(x, y, k, jacobian)
        jacobian_over_time[k] = jacobian

    # step 5, reshape the jacobian_over_time tensor to (nr_folds, nr_points_along_path, N*M)
    jacobian_over_time = jacobian_over_time.view(nr_folds, regularize_config["nr_points_along_path"], N*M)

    # step 6, calculate the differences between consecutive points
    with torch.no_grad():
        differences = torch.mean(torch.abs(jacobian_over_time[:,1:,:] - jacobian_over_time[:,:-1,:]), dim = -1)

    # step 7, calculate the total variation loss as the mean of all the differences
    regularization_loss = torch.mean(differences)

    return regularization_loss

def get_regularization_loss_tie_weights(lista: LISTA):
    """
    This second regularization loss is imply an l1 norm between al W matrices of the lista module
    Note that lista.W1 is a parameter list, so we need to take the mean of all the W1 matrices
    same aplies to lista.W2, and lista.bias

    """

    # l1 for W1
    W1_stacked = torch.stack([W1 for W1 in lista.W1])
    average_W1 = torch.mean(W1_stacked, dim=0)
    l1_W1 = torch.mean(torch.abs(W1_stacked - average_W1))

    # l1 for W2
    W2_stacked = torch.stack([W2 for W2 in lista.W2])
    average_W2 = torch.mean(W2_stacked, dim=0)
    l1_W2 = torch.mean(torch.abs(W2_stacked - average_W2))

    # l1 for bias
    bias_stacked = torch.stack([bias for bias in lista.bias])
    average_bias = torch.mean(bias_stacked, dim=0)
    l1_bias = torch.mean(torch.abs(bias_stacked - average_bias))

    # calculate the total loss
    regularization_loss = l1_W1 + l1_W2 + l1_bias

    return regularization_loss

def estimate_y_std(train_data):
    ys = torch.zeros((train_data.nr_of_examples, train_data.A.shape[0]))
    for i, (y, _) in enumerate(torch.utils.data.DataLoader(train_data, batch_size=1)):
        ys[i] = y
    return torch.std(y)