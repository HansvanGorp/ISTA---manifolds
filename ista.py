"""
We here implement ISTA as a pytorch module
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

from make_gif_from_figures_in_folder import make_gif_from_figures_in_folder

# %% create an ISTA prototype module
class ISTAPrototype(torch.nn.Module):
    def __init__(self, A: torch.tensor, K: int = 16, device: str = "cpu"):
        super(ISTAPrototype, self).__init__()
        """
        Create the ISTA prototype module. Other modules will inherit from this module. Such as ISTA, LISTA, etc.
        We thus create all functionalities that are common to all these modules here.
        Note that we will require the inherited modules to implement some functions that are specific to them. Specifically:
        - the get_W1_at_iteration function. This function returns the W1 matrix at a specific iteration of the iteration.
        - the get_W2_at_iteration function. This function returns the W2 matrix at a specific iteration of the iteration.
        - the get_lambda_at_iteration function. This function returns the lambda value at a specific iteration of the iteration.

        input parameters:
        - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
        - K (int): the number of iterations for ISTA
        - device (str): the device to run the module on, default is "cpu"

        inferred:
        - N (int): the signal dimension of x, inferred from A
        - M (int): the measurement dimension of y, inferred from A
        """

        # it can be a numpy array, we convert it to a tensor
        if not torch.is_tensor(A):
            self.A = torch.tensor(A, dtype=torch.float32, device=device)
        else:
            self.A = A.to(device)

        # get the shape of A
        self.N = A.shape[1]
        self.M = A.shape[0]

        # set the number of iterations
        self.K = K
        self.device = device

    def forward(self, y: torch.tensor, verbose: bool = False, calculate_jacobian:bool = True, jacobian_projection: torch.tensor = None, return_intermediate: bool = False, tqdm_position: int = 0, tqdm_leave: bool = True):
        """
        Implements the forward function of the prototype. This function is common to all inherited modules.
        This function needs the two functions get_W1_at_iteration and get_W2_at_iteration to be implemented by the inherited modules.

        It can optionally compute the jacobian of the forward function. i.e. dx/dy at the end of the iterations. relating how each x element changes with each y element.

        inputs:
        - y (torch.tensor): the input y, of shape (batch_size, M)
        - verbose (bool): if True, print a progress bar
        - calculate_jacobian (bool): if True, calculate the Jacobian of the forward function
        - jacobian_projection (torch.tensor): a projection matrix to project the Jacobian to a 2D space, for visualization, we add that here because 
                                              it will save memory and computation if we calculate the Jacobian in a 2D space, of shape (N, 2)

        - return_intermediate (bool): if True, return the intermediate x's, instead of only the final x (not implemented for intermediate jacobian)

        outputs:
        - x (torch.tensor): the output x, of shape (batch_size, N) or (batch_size, N, K) if return_intermediate is True
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M) or (batch_size, N, 2) if jacobian_projection is not None
        """
        # push y to the correct device if it is not already there
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(self.device)

        # get the initial x and jacobian
        x, jacobian = self.get_initial_x_and_jacobian(y.shape[0], calculate_jacobian, jacobian_projection)

        # if we are returning the intermediate x's, we need to store them somewhere
        if return_intermediate:
            x_intermediate = torch.zeros(y.shape[0], self.N, self.K, dtype=torch.float32, device=self.device)

        # Now start going over the iterations
        for k in tqdm(range(self.K), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="running ISTA folds"):
            # perform the forward function at this iteration
            x, jacobian = self.forward_at_iteration(x, y, k, jacobian, jacobian_projection)

            # if we are returning the intermediate x's, store them
            if return_intermediate:
                x_intermediate[:,:,k] = x

        # if we are returning the intermediate x's, return them
        if return_intermediate:
            return x_intermediate, jacobian
        else:
            return x, jacobian
    
    def forward_at_iteration(self, x:torch.tensor, y: torch.tensor, k: int, jacobian:torch.tensor = None, jacobian_projection: torch.tensor = None):
        """
        Implements the forward function of the prototype at a specific iteration.

        inputs:
        - x (torch.tensor): the current x, of shape (batch_size, N)
        - y (torch.tensor): the input y, of shape (batch_size, M)
        - k (int): the current iteration (this is needed to get the correct W1 and W2 matrices)
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M), or (batch_size, N, 2) if jacobian_projection is not None
        - jacobian_projection (torch.tensor): a projection matrix to project the Jacobian to a 2D space, for visualization, of shape (2, M)
        """
        # make sure things are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)
        jacobian = jacobian.to(self.device) if jacobian is not None else None
        jacobian_projection = jacobian_projection.to(self.device) if jacobian_projection is not None else None

        # step 1, perform data-consistency
        x, jacobian = self.data_consistency( x, y, k, jacobian, jacobian_projection)

        # step 2, perform thresholding
        x, jacobian = self.soft_thresholding(x,    k, jacobian)

        return x, jacobian

    def get_initial_x_and_jacobian(self, batch_size: int, calculate_jacobian: bool, jacobian_projection: torch.tensor = None):
        return self.get_initial_x(batch_size), self.initalize_jacobian(batch_size, calculate_jacobian, jacobian_projection)
    
    def get_initial_x(self, batch_size: int):
        """
        Initializes the x vector.

        inputs:
        - batch_size (int): the batch size
        """
        return torch.zeros(batch_size, self.N, dtype=torch.float32, device=self.device)
    
    def initalize_jacobian(self, batch_size: int, calculate_jacobian: bool, jacobian_projection: torch.tensor):
        """
        Initializes the Jacobian matrix.

        inputs:
        - batch_size (int): the batch size
        - calculate_jacobian (bool): if True, calculate the Jacobian of the forward function
        - jacobian_projection (torch.tensor): a projection matrix to project the Jacobian to a 2D space, for visualization, of shape (2, M)
        """
        if calculate_jacobian and jacobian_projection is None:
            # initialize the Jacobian with all zeros
            jacobian = torch.zeros(batch_size, self.N, self.M, dtype=torch.float32, device=self.device)

        elif calculate_jacobian and jacobian_projection is not None:
            # initialize the Jacobian with all zeros in the 2D space
            jacobian = torch.zeros(batch_size, self.N,       2, dtype=torch.float32, device=self.device)

        else:
            # put it to None
            jacobian = None

        return jacobian
        
    def data_consistency(self, x: torch.tensor, y: torch.tensor, k: int, jacobian: torch.tensor = None, jacobian_projection: torch.tensor = None):
        """
        Implements the data consistency step of the ISTA algorithm.

        inputs:
        - x (torch.tensor): the current x, of shape (batch_size, N)
        - y (torch.tensor): the input y, of shape (batch_size, M)
        - k (int): the current iteration (this is needed to get the correct W1 and W2 matrices)
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M) (if None, it is not calculated)
        - jacobian_projection (torch.tensor): a projection matrix to project the Jacobian to a 2D space, for visualization, of shape (2, M)
        """
        # get W1 and W2 at the current iteration
        W1   = self.get_W1_at_iteration(k)
        W2   = self.get_W2_at_iteration(k)
        bias = self.get_bias_at_iteration(k)

        # perform data consistency
        W1_times_y = torch.nn.functional.linear(y, W1)
        W2_times_x = torch.nn.functional.linear(x, W2)
        x = W1_times_y + W2_times_x + bias.unsqueeze(0)

        # calculate the Jacobian if needed
        if jacobian is not None:
            # the jacobian gets multiplied by W2, which is a linear operation
            jacobian = torch.matmul(W2, jacobian)

            # then W1 gets added to the result, which is also a linear operation
            additive_jacobian = W1
            if jacobian_projection is not None:
                # if we are projecting the Jacobian to a 2D space, we need to project the additive Jacobian as well
                additive_jacobian = torch.matmul(additive_jacobian, jacobian_projection)
                
            jacobian = jacobian + additive_jacobian.unsqueeze(0)

        # return the result
        return x, jacobian
    
    def soft_thresholding(self, x: torch.tensor, k: int, jacobian: torch.tensor = None, max_clip: float = 10):
        """
        Implements the soft thresholding step of the ISTA algorithm.

        inputs:
        - x (torch.tensor): the current x, of shape (batch_size, N)
        - k (int): the current iteration (this is needed to get the correct lambda value)
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M) (if None, it is not calculated)
        - max_clip (float): the maximum magnitude value to clip x to (to prevent numerical instability)
        """
        # get lambda at the current iteration
        _lambda = self.get_lambda_at_iteration(k)

        # if we are calculating the Jacobian, do so now, before x gets modified
        if jacobian is not None:
            # more efficient implementation: we first create a mask to see where x is above the threshold
            mask = (torch.abs(x[:,:]) > _lambda).float() * (torch.abs(x[:,:]) < max_clip).float()

            # the mask will be of shape (batch_size, N), while jacbian is of shape (batch_size, N, M)

            # this is fine, the mask should do the following, where it is 1, the entire (M) collumn of the jacobian stays the same
            # where it is 0, the entire (M) collumn of the jacobian becomes 0
            jacobian = jacobian * mask.unsqueeze(2)

            # # the new jacobian with which we need to multiply the current jacobian is simple a diagonal matrix with 1s where x is above the threshold
            # new_jacobian = torch.diag_embed((torch.abs(x[:,:]) > _lambda).float())

            # # multiply the current jacobian with the new jacobian
            # jacobian = torch.matmul(new_jacobian, jacobian)

        # clip the x values to prevent numerical instability
        x = torch.clamp(x, -max_clip, max_clip)
        
        # perform soft thresholding
        x = torch.nn.functional.softshrink(x, _lambda)

        # return the result
        return x, jacobian

# %% create an ISTA module that inherits from ISTAPrototype
class ISTA(ISTAPrototype):
    def __init__(self, A: torch.tensor, mu: float = 0.5, _lambda: float = 0.5, K: int = 16, device: str = "cpu"):
        super(ISTA, self).__init__(A, K, device)
        """Create the ISTA module with the input parameters:
        - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
        - mu (float): the step size for ISTA
        - _lambda (float): the threshold for ISTA
        - K (int): the number of iterations for ISTA
        - device (str): the device to run the module on, default is "cpu"

        Since ISTA inherits from ISTAPrototype, it has all the functionalities of ISTAPrototype.
        We only need to specify the parameters that are specific to ISTA.
        """

        # save mu and _lambda
        self.mu = mu
        self._lambda = _lambda

        # create W1 and W2 of Ista
        self.W1 = self.mu*self.A.t()
        self.W2 = torch.eye(self.N).to(self.device) - self.mu*self.A.t()@self.A

    # The prototype requires three functions to be implemented by the inherited modules:
    def get_W1_at_iteration(self, k):
        # simply return W1, ISTA does not change W1 over iterations
        return self.W1
    
    def get_W2_at_iteration(self, k):
        # simply return W2, ISTA does not change W2 over iterations
        return self.W2
    
    def get_bias_at_iteration(self, k):
        # simply return 0, ISTA does not have a bias
        return torch.zeros(1, device = self.device)
    
    def get_lambda_at_iteration(self, k):
        # simply return _lambda, ISTA does not change _lambda over iterations
        return self._lambda

# %% create a LISTA module that inherits from ISTAPrototype
class LISTA(ISTAPrototype):
    def __init__(self, A: torch.tensor, mu: float = 0.5, _lambda: float = 0.5, K: int = 16, device: str = "cpu", initialize_randomly: bool = True):
        super(LISTA, self).__init__(A, K, device)
        """Create the LISTA module with the input parameters:
        - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
        - mu (float): the step size for ISTA
        - _lambda (float): the threshold for ISTA
        - K (int): the number of iterations for ISTA
        - device (str): the device to run the module on, default is "cpu"
        - initialize_randomly: if true, we initialize W1 and W2 randomly, otheriwse we use our knowledge of the problem to initialize them in a good way

        Since LISTA inherits from ISTAPrototype, it has all the functionalities of ISTAPrototype.
        We only need to specify the parameters that are specific to ISTA. which is the fact that W1 and W2 are learned parameters over the iterations.
        """

        # save mu and _lambda
        self.mu = mu
        self._lambda = _lambda

        if initialize_randomly:
            self.W1   = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.N, self.M, device=self.device)) for _ in range(K)])
            self.W2   = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.N, self.N, device=self.device)) for _ in range(K)])
            self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.N, device=self.device)) for _ in range(K)])

        else:
            # create initial W1 and W2 of Ista
            W1_initialization = self.mu*self.A.t()
            W2_initialization = torch.eye(self.N).to(self.device) - self.mu*self.A.t()@self.A

            # now create the W1 and W2 as torch.nn.Parameter, but do so over all the K iterations
            self.W1 = torch.nn.ParameterList([torch.nn.Parameter(W1_initialization.clone().detach()) for _ in range(K)])
            self.W2 = torch.nn.ParameterList([torch.nn.Parameter(W2_initialization.clone().detach()) for _ in range(K)])
            self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(self.N, device=self.device)) for _ in range(K)])

    # The prototype requires three functions to be implemented by the inherited modules:
    def get_W1_at_iteration(self, k):
        # return the W1 at the current iteration
        return self.W1[k]
    
    def get_W2_at_iteration(self, k):
        # return the W2 at the current iteration
        return self.W2[k]
    
    def get_bias_at_iteration(self, k):
        # return the bias at the current iteration
        return self.bias[k]
    
    def get_lambda_at_iteration(self, k):
        # simply return _lambda, LISTA does not change _lambda over iterations
        return self._lambda
    
# %% some helper functions
def create_random_matrix_with_good_singular_values(M: int, N: int):
    """
    This function creates a random matrix A with good singular values for ISTA.
    i.e. the largest singular value of A.T@A is 1.0, this makes the ISTA more stable.
    """

    # create a random matrix A
    A = torch.randn(M, N)
    
    # We are going to adapt measurement A, such that its largest eigenvalue is 1.0, this makes the ISTA more stable
    largest_eigenvalue = torch.svd(A.t() @ A).S[0]
    A = 1.0 * A / (largest_eigenvalue**0.5)

    return A

# %% take an (L)ISTA module and some plotting parameters, and analyze it visually
def visual_analysis_of_ista(ista: ISTA, K: int, nr_points_along_axis: int, margin: float, indices_of_projection: tuple[int,int,int], A: torch.tensor, 
                            save_folder: str = "test_figures", tqdm_position: int = 0, tqdm_leave: bool = True, verbose: bool = False,
                            magntiude: float = 1.0, magnitude_ood: float = None):
    """
    Creates a visual analysis of the ISTA module. This is done by visualizing the linear regions of the Jacobian, and the sparsity of the x-vector.
    We only visualize in part of the space, namely a hyperplane that passes through three anchor points. This hyperplane is embedded in y-space,
    which is the actual input space of the ISTA module. We project the Jacobian to a 2D space, and visualize the linear regions in this 2D space.

    inputs:
    - ista: the ISTA module
    - K: the number of iterations
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
    nr_regions_arrray = torch.zeros(K)

    # loop over the iterations
    for k in tqdm(range(K), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="visual analysis of ISTA, runnning over folds"):
        with torch.no_grad():
            x, jacobian = ista.forward_at_iteration(x, y, k, jacobian, jacobian_projection)

        # extract the linear regions from the jacobian
        nr_of_regions, norms, region_idx = extract_linear_regions_from_jacobian(jacobian)
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
        plt.savefig(f"{save_folder}/iteration_{k}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # save the number of regions
        nr_regions_arrray[k] = nr_of_regions
    
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

def random_analysis_of_ista(ista: ISTA, K: int, nr_points_total: int, nr_points_in_batch: int, max_magnitude: float, A: torch.tensor, save_name: str = "test_figures", save_folder: str = "random_analysis_figures", verbose: bool = False, tqdm_position: int = 0, tqdm_leave: bool = True):
    """
    Creates an analysis of the ISTA module. This is done by looking really into all the axis of y, not just two as specified by the linear projection.
    However, the computational cost of this is high, so we only do this for a subset of the space, namely a hypercube that is centered around the origin.
    Additionally, we do not look into a meshgrid of points, rather we look into a random subset of points.
    """
    # calculate how many batches we need to get the total number of points
    assert nr_points_total % nr_points_in_batch == 0, "nr_points_total should be divisible by nr_points_in_batch"
    nr_batches = nr_points_total // nr_points_in_batch

    # create the save folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # create the random y data
    M = A.shape[0]
    y = (torch.rand(nr_points_total, M, device="cpu") - 0.5) * 2 * max_magnitude
   
    # run the initials function to get the initial x and jacobian
    x, jacobian = ista.get_initial_x_and_jacobian(nr_points_total, calculate_jacobian = True)

    # put them back on the cpu, because memory is limited
    x = x.cpu()
    jacobian = jacobian.cpu()

    # create an array of nr regions over the iterations
    nr_regions_arrray = torch.zeros(K)

    # loop over the iterations
    for k in tqdm(range(K), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="random analysis of ISTA, runnning over folds"):
        unique_entries_current = None
        # loop over the batches
        for i in tqdm(range(nr_batches), position=tqdm_position+1, leave=tqdm_leave, disable= not verbose, desc="random analysis of ISTA, runnning over batches"):
            # extract the y and x data, and jacobian
            y_batch = y[i*nr_points_in_batch:(i+1)*nr_points_in_batch].to(ista.device)
            x_batch = x[i*nr_points_in_batch:(i+1)*nr_points_in_batch].to(ista.device)
            jacobian_batch = jacobian[i*nr_points_in_batch:(i+1)*nr_points_in_batch].to(ista.device)

            # do ista
            with torch.no_grad():
                x_batch, jacobian_batch = ista.forward_at_iteration(x_batch, y_batch, k, jacobian_batch)

            # extract the linear regions from the jacobian
            _, _, unique_entries_new = extract_linear_regions_from_jacobian(jacobian)

            # save the batches back to x and jacobian (on the cpu)
            x[i*nr_points_in_batch:(i+1)*nr_points_in_batch] = x_batch.cpu()
            jacobian[i*nr_points_in_batch:(i+1)*nr_points_in_batch] = jacobian_batch.cpu()

            # push the unqiue entries to the cpu
            unique_entries_new = unique_entries_new.cpu()

            # update the unique entries by comparing it to the current ones, and taking the union of them
            if unique_entries_current is None:
                unique_entries_current = unique_entries_new
            else:
                unique_entries_current = torch.cat((unique_entries_current, unique_entries_new), dim=0)

        # remove duplicates
        unique_entries_current, _ = torch.unique(unique_entries_current, dim=0, return_inverse=True)

        # save the number of regions
        nr_regions_arrray[k] = len(unique_entries_current)

    # plot the number of regions over the iterations
    plt.figure()
    plt.semilogy(nr_regions_arrray,'-', label = "number of linear regions", base = 2)
    plt.semilogy([0,len(nr_regions_arrray)], [nr_points_total,nr_points_total], 'r--', label = "total number of points used", base = 2)
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("number of linear regions")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{save_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return nr_regions_arrray

def knot_density_analysis(ista: ISTA, K: int, A: torch.tensor, 
                            nr_paths: int = 4,  anchor_point_std: float = 1, nr_points_along_path: int = 4000, path_delta: float = 0.001, 
                            save_name: str = "test_figures", save_folder: str = "knot_density_figures", verbose: bool = False, color: str = "black",
                            tqdm_position: int = 0, tqdm_leave: bool = True):
    """
    We sample random lines in the y-space, and see how many knots we get over the iterations.
    We can then divide the knots by the lenght of the line, to get a knot-density.
    
    Given many different lines, we get an expectation of the knot-density over the y-space (as well as a variance).

    inputs:
    - ista: the ISTA module
    - K: the number of iterations
    - A: the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
    - nr_paths: the number of random paths to sample
    - anchor_point_std: the standard deviation of the anchor point, which is the point gnerated from a gaussian, which defines the middle of the line
    - nr_points_along_path: the number of points along the path
    - path_delta: the length of each step along the path, should be small for a good approximation
    - save_name: the name to save the figure as
    - save_folder: the folder to save the figure in
    - verbose: if True, print a progress bar
    - color: the color of the plot
    """
    # empty the CUDA cache
    torch.cuda.empty_cache()
    
    # get M and N of the matrix A
    M = A.shape[0]
    N = A.shape[1]

    # create the save folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # create a knot density array of size (nr_lines, K)
    knot_density_array = torch.zeros(nr_paths, K+1)

    # precalculate the lenght of the path
    length_of_path = path_delta * nr_points_along_path

    # loop over the lines, with tqdm enabled if verbose is True
    for path_idx in tqdm(range(nr_paths), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="knot density analysis, runnning over paths"):
        y = generate_path(M, nr_points_along_path, path_delta, anchor_point_std, ista.device)

        # run the initials function to get the initial x and jacobian
        x, jacobian = ista.get_initial_x_and_jacobian(nr_points_along_path, calculate_jacobian = True)

        # loop over the iterations
        for k in range(K):
            with torch.no_grad():
                x, jacobian = ista.forward_at_iteration(x, y, k, jacobian)

            # extract the number of knots in the jacobian, along the batch dimension
            nr_of_knots =  extract_knots_from_jacobian(jacobian)
            knot_density = nr_of_knots.cpu() / length_of_path
            knot_density_array[path_idx, k+1] = knot_density

    # plot the knot density over the iterations
    plt.figure()
    folds = np.arange(0,K+1)
    knot_denity_mean = torch.mean(knot_density_array, dim=0)
    plt.plot(folds,knot_denity_mean,'-', label = "mean", color = color)
    plt.xlabel("fold")
    plt.ylabel("knot density")
    plt.grid()
    plt.title(save_name)
    plt.xlim([0,K])
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{save_name}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_folder}/{save_name}.svg", bbox_inches='tight')
    plt.close()

    # give the knot density array back
    return knot_denity_mean

def generate_path(M: int, nr_points_along_path: int, path_delta: float, anchor_point_std: float, device: str):
    """
    generate a random path, this is done by sampling two anchor points, and then creating a path between them with a step size of path_delta.
    If we have do not yet reach the nr_points_along_path, we sample a new anchor point, and continue our path towards that, etc. etc.
    This is done untill we hit nr_points_along_path. Note that we can thus quit early before reaching the final anchor point. As we always exactly use nr_points_along_path points
    with a fixed delta between the points. The length of the path is thus always the same.
    """

    # step one, sample anchor point 0
    anchor_point_zero = torch.randn(M, device = device) * anchor_point_std

    # initialze the current length of the path
    current_length = 0
    
    # initialize the y data
    y = torch.zeros(nr_points_along_path, M, device = device)

    # keep looping untill we hit current_length == nr_points_along_path
    while current_length < nr_points_along_path:
        # sample a new anchor point
        anchor_point_one = torch.randn(M, device = device) * anchor_point_std

        # calculate the direction vector
        direction_vector = (anchor_point_one - anchor_point_zero) / torch.linalg.norm(anchor_point_one - anchor_point_zero, ord=2)

        # calculate the distance between the two anchor points
        distance = torch.linalg.norm(anchor_point_one - anchor_point_zero, ord=2)

        # calculate the number of points we can still take
        nr_points_to_take = min(nr_points_along_path - current_length, int(distance / path_delta))

        # put the points in the y data
        y[current_length:current_length+nr_points_to_take] = anchor_point_zero + direction_vector * torch.arange(nr_points_to_take, device = device).float().unsqueeze(1) * path_delta

        # update the current length
        current_length += nr_points_to_take

        # update the anchor point
        anchor_point_zero = anchor_point_one

    # assert current length is equal to nr_points_along_path
    assert current_length == nr_points_along_path, "current_length should be equal to nr_points_along_path"

    return y

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

def extract_knots_from_jacobian(jacobian: torch.tensor):
    """
    Given a Jacobian matrix, extract the knots from it. This assumes the jacobian was created along a 1D-path.
    This makes it simpler that extracting the linear regions, as we can just use the differences between consecutive points, and count the nr of times it is non-zero.

    inputs:
    - jacobian (torch.tensor): the Jacobian of the forward function, of shape (nr_lines_in_batch, nr_points_in_line, N, M), where the nr_of_lines_in_batch is the number of lines we are looking at, and nr_points_in_line is the number of points along the line
                               the Jacobian can also be of shape (nr_points_in_line, N, M), in other words we only have one line, and we are looking at the points along the line
    outputs:
    - nr_of_knots (int): the number of knots we encounter along the path
    """     
    # assert the jacobian has 3 or 4 dimensions
    assert len(jacobian.shape) == 4 or len(jacobian.shape) == 3, "the jacobian should have 3 or 4 dimensions"

    # get the number of dimensions in the jacobian, because we have slightly different behavior for the two cases
    if len(jacobian.shape) == 4:
        # get the shape of the jacobian
        nr_lines_in_batch, nr_points_in_line, N, Z  = jacobian.shape # let's just call the last dimension Z for now
    
        # reshape jacobian to (batch, N*Z)
        jacobian = jacobian.view(nr_lines_in_batch, nr_points_in_line, N*Z)

        # calculate the differences between consecutive points
        nr_of_knots = torch.sum(torch.any((jacobian[:,1:,:] - jacobian[:,:-1,:])>0, dim=2), dim=1)
    else:
        # get the shape of the jacobian
        nr_points_in_line, N, Z  = jacobian.shape

        # reshape jacobian to (batch, N*Z)
        jacobian = jacobian.view(nr_points_in_line, N*Z)

        # calculate the differences between consecutive points
        nr_of_knots = torch.sum(torch.any((jacobian[1:,:] - jacobian[:-1,:])>0, dim=1))
    
    return nr_of_knots

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

# %% training of LISTA
def data_generator(A: torch.tensor, batch_size: int = 4, maximum_sparsity: int = 4, x_magnitude: tuple[float,float] = (0.5, 1.5), N: int = 16, device: str = "cpu", noise_std: float = 0.1):
    """
    Create some data for training LISTA. This data is simply a random x-vector, and the y-vector is created by multiplying A with x.

    There is however a constraint on the sparsity of x, it should be at most maximum_sparsity. We do this by multiplying the x-vector with a mask.

    inputs:
    - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
    - batch_size (int): the batch size
    - maximum_sparsity (int): the maximum sparsity of the x-vector
    - x_magnitude (tuple of floats): the magnitude of the x-vector, (min, max)
    - N (int): the signal dimension of x
    - device (str): the device to run the module on, default is "cpu"
    - noise_std (float): the standard deviation of the noise added to the y-vector

    outputs:
    - y (torch.tensor): the y-vector, of shape (batch_size, M)
    - x (torch.tensor): the x-vector, of shape (batch_size, N)
    """
    # push A to the correct device if it is not already there
    A = A.to(device)

    # create a random x-vector
    x = torch.rand(batch_size, N, device=device) * (x_magnitude[1] - x_magnitude[0]) + x_magnitude[0]

    # create a mask that makes sure the x-vector is at most maximum_sparsity, or even less than that
    mask = torch.zeros(batch_size, N, device=device)
    for i in range(batch_size):
        sparsity = torch.randint(0, maximum_sparsity+1, (1,))
        x_indices = torch.randperm(N)[:sparsity]
        mask[i, x_indices] = 1

    # apply the mask
    x = x * mask

    # create the y-vector
    y = torch.nn.functional.linear(x, A) + noise_std*torch.randn(batch_size, A.shape[0], device=device)

    return y, x  


def get_regularization_loss_smooth_jacobian(lista: LISTA, regularize_config: dict, loss_multiplier: torch.tensor, sum_of_loss_multiplier: float):
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
    nr_folds = lista.K
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

    # step 7, randomly select a k (fold index)
    k = torch.randint(0, nr_folds, (1,)).item()

    # get the nr of knots, and the location of the knots
    knot_locations = torch.nonzero(differences[k,:])[:,0]
    nr_of_knots = len(knot_locations)

    # step 8, loop over each region in the jacobian, except the two edge regions (first and last)
    regularization_loss = 0
    for region_idx in range(1,nr_of_knots):
        # get the indices of the region
        start_idx = knot_locations[region_idx-1].item()
        end_idx   = knot_locations[region_idx].item()

        # get the value of the jacobian of this region, as well as its left and right neighbour
        jacobian_region = jacobian_over_time[k, start_idx+1:end_idx+1, :]

        # calculate the loss, as the l1 loss to the jacobian of the closest neighbour
        if differences[k, start_idx] < differences[k, end_idx]:
            # the left neighbour is the closest
            regularization_loss += torch.abs(jacobian_region - jacobian_over_time[k, start_idx, :]).mean()
        else:
            # the right neighbour is the closest   
            regularization_loss += torch.abs(jacobian_region - jacobian_over_time[k, end_idx+1, :]).mean()

    # multiply the loss by the loss multiplier that corresponds with K
    regularization_loss = regularization_loss * loss_multiplier[k] * nr_folds / sum_of_loss_multiplier


    return regularization_loss

def get_regularization_loss_tv_jacobian(lista: LISTA, regularize_config: dict, loss_multiplier: torch.tensor, sum_of_loss_multiplier: float):
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
    nr_folds = lista.K
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

    # step 7, multiply each fold with the loss multiplier
    differences = differences * loss_multiplier.unsqueeze(1)

    # step 8, calculate the total variation loss as the mean of all the differences
    regularization_loss = torch.sum(torch.mean(differences, dim = 1))/sum_of_loss_multiplier

    return regularization_loss

def get_regularization_loss_tie_weights(lista: LISTA, regularize_config: dict):
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


def train_lista(lista: LISTA, data_generator, nr_iterations: int, forgetting_factor:float, learning_rate: float = 1e-4, patience: int = 100, #NOSONAR
                show_loss_plot: bool = False, loss_folder: str = None, verbose: bool = True, tqdm_position: int = 0, tqdm_leave: bool = True,
                regularize: bool = False, regularize_config: dict = None, save_name: str = "lista"): 
    """ 
    Train the LISTA module using the data generator.

    inputs:
    - lista (LISTA): the LISTA module
    - data_generator (function): a function that generates the data
    - nr_iterations (int): the number of iterations to train for
    - forgetting_factor (float): the forgetting factor for the moving average of the loss, 
      i.e. the last fold has a loss of forgetting_factor^0, the second last has a loss of forgetting_factor^1, etc.
    - learning_rate (float): the learning rate of the optimizer
    - patience (int): the patience for the early stopping
    - show_loss_plot (bool): if True, show the loss plot
    - loss_folder (str): the folder to save the loss plot in
    - verbose (bool): if True, print the loss
    - tqdm_position (int): the position of the tqdm bar
    - tqdm_leave (bool): if True, leave the tqdm bar
    - regularize (bool): if True, regularize the LISTA module -> RLISTA
    - regularize_config (dict): the configuration of the regularization
    - save_name (str): the name to save the loss plot as

    outputs:
    - lista (LISTA): the trained LISTA module
    """
    # create the optimizer
    optimizer = torch.optim.Adam(lista.parameters(), lr=learning_rate)
    
    # initialize the loss list
    losses = torch.zeros(nr_iterations)

    reconstruction_losses = torch.zeros(nr_iterations)
    regularization_losses = torch.zeros(nr_iterations)

    # calculate the loss_multiplier over the folds
    K = lista.K
    loss_multiplier = torch.tensor([forgetting_factor**i for i in range(K)], device=lista.device)
    # reverse the loss_multiplier, because we want the last fold to have the largest loss
    loss_multiplier = loss_multiplier.flip(0)
    sum_of_loss_multiplier = loss_multiplier.sum()

    # current best loss is infinity
    best_loss = float('inf')
    patience_counter = 0

    # loop over the iterations
    for i in tqdm(range(nr_iterations), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc=f"training {save_name}, runnning over batches"):
        # generate the data
        y, x = data_generator()

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        x_hat, _ = lista(y, verbose = False, return_intermediate = True, calculate_jacobian = False)

        # because x_hat has the intermediate x's, we need to expand the x to the same shape
        x = x.unsqueeze(2).expand_as(x_hat)

        # calculate the l1 loss over the K folds
        loss_per_fold = torch.abs(x_hat - x).mean((0,1))
        reconstruction_loss = torch.sum(loss_per_fold * loss_multiplier)/sum_of_loss_multiplier

        # now check if we need to regularize
        if regularize:
            if regularize_config["type"] == "smooth_jacobian":
                regularization_loss = get_regularization_loss_smooth_jacobian(lista, regularize_config, loss_multiplier, sum_of_loss_multiplier)
            elif regularize_config["type"] == "tv_jacobian":
                regularization_loss = get_regularization_loss_tv_jacobian(lista, regularize_config, loss_multiplier, sum_of_loss_multiplier)
            elif regularize_config["type"] == "tie_weights":
                regularization_loss = get_regularization_loss_tie_weights(lista, regularize_config)
            else:
                raise ValueError("regularize_config['type'] is not valid")
            
            regularization_loss *= regularize_config["weight"]
            loss = reconstruction_loss + regularization_loss    
        else:
            loss = reconstruction_loss

        # backprop
        loss.backward()

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(lista.parameters(), 1.0)

        # optimizer step
        optimizer.step()

        # save the loss
        losses[i] = loss.item()
        if regularize:
            reconstruction_losses[i] = reconstruction_loss.item()
            regularization_losses[i] = regularization_loss.item()

        # show the loss plot
        batches = np.arange(i+1)
        xmax = batches.max() if i > 0 else 1             # make sure the xmax is at least 2


        plt.figure()
        plt.plot(batches,losses[:i+1].cpu().numpy())
        if regularize:
            plt.plot(batches,reconstruction_losses[:i+1].cpu().numpy())
            plt.plot(batches,regularization_losses[:i+1].cpu().numpy())
            plt.legend(["total loss", "reconstruction loss", "regularization loss"], loc='best')

        plt.xlim(batches.min(), xmax)
        plt.ylim(0, 0.15)
        plt.grid()
        plt.title("loss over the batches")
        plt.xlabel("batch")
        plt.ylabel("loss")
        plt.tight_layout()
        
        if loss_folder is None:
            plt.savefig(f"loss_{save_name}.jpg", dpi=300, bbox_inches='tight')
            plt.savefig(f"loss_{save_name}.svg", bbox_inches='tight')
        else:
            plt.savefig(f"{loss_folder}/loss_{save_name}.jpg", dpi=300, bbox_inches='tight')
            plt.savefig(f"{loss_folder}/loss_{save_name}.svg", bbox_inches='tight')

        if show_loss_plot:
            plt.show()
        else:
            plt.close()       

        # check if this loss is the current best loss
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        # check if patience is reached, if so, stop
        if patience_counter == patience:
            break

    # save the state_dict
    state_dict = lista.state_dict()
    torch.save(state_dict, f"{loss_folder}/{save_name}_state_dict.tar")

    return lista, losses

# %% grid search for ISTA
def grid_search_ista(A, data_generator, mus: np.array, _lambdas: np.array, K: int, forgetting_factor:float = 1.0, device: str="cpu", verbose: bool = True, tqdm_position: int = 0, tqdm_leave: bool = True, use_accuracy: bool = False):
    """
    perfrom a grid search for the best mu and lambda for the ISTA module. using the data generator.
    """

    # step 1, generate the data for the grid search
    y, x = data_generator()

    # step 2, create the grid
    grid = torch.zeros(len(mus), len(_lambdas))

    # calculate the loss multiplier over the folds
    loss_multiplier = torch.tensor([forgetting_factor**i for i in range(K)], device=device)
    # reverse the loss_multiplier, because we want the last fold to have the largest loss
    loss_multiplier = loss_multiplier.flip(0)
    sum_of_loss_multiplier = loss_multiplier.sum()

    # step 3, loop over the grid
    for i, mu in enumerate(tqdm(mus, position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="grid search for ISTA, runnning over mus")):
        for j, _lambda in enumerate(tqdm(_lambdas, position=tqdm_position+1, leave=(tqdm_leave and (i+1)==len(mus)), disable=not verbose, desc="grid search for ISTA, runnning over lambdas")):
            # create the ISTA module
            ista = ISTA(A, mu = mu, _lambda = _lambda, K=K, device=device)

            # run the ISTA module
            x_hat,_ = ista(y, verbose = False, return_intermediate = True, calculate_jacobian = False)

            

            if use_accuracy:
                # calculate the support accuracy
                accuracy = 0
                for k in range(K):
                    accuracy += support_accuracy(x_hat[:,:,k], x) * loss_multiplier[k] * 100.0 / sum_of_loss_multiplier
                loss = 100 - accuracy
                
            else:
                # because x_hat has the intermediate x's, we need to expand the x to the same shape, if it does not yet have the same shape
                if len(x_hat.shape) != len(x.shape):
                    x = x.unsqueeze(2).expand_as(x_hat)

                # calculate the l1 loss over the K folds
                loss_per_fold = torch.abs(x_hat - x).mean((0,1))
                loss = torch.sum(loss_per_fold * loss_multiplier)/sum_of_loss_multiplier

            # save the loss
            grid[i,j] = loss

    # step 4, find the best mu and lambda
    best_idx = torch.argmin(grid)
    best_mu_idx = best_idx // len(_lambdas)
    best_lambda_idx = best_idx % len(_lambdas)

    # step 5, get the best mu and lambda
    best_mu = mus[best_mu_idx]
    best_lambda = _lambdas[best_lambda_idx]

    return best_mu, best_lambda

# %% get the support accuracy of a module
def support_accuracy(x1: torch.tensor, x2: torch.tensor):
    """
    calculates the accuracy of the support of the x1-vector, and comparing it to the support of the x2-vector.

    inputs:
    - x1: the x1-vector, of shape (batch_size, N)
    - x2: the x2-vector, of shape (batch_size, N)
    """

    # get the support as all values where x1 or x2 is non-zero
    support_x1 = (x1 != 0).float()
    support_x2 = (x2 != 0).float()

    # accuracy is simply the average number of times the support is the same times 100
    accuracy = torch.mean((support_x1 == support_x2)*100.0)

    return accuracy
    

def support_accuracy_analysis(ista: ISTA, K: int, A: torch.tensor, y: torch.tensor, x: torch.tensor,
                              save_name: str = "test_figures", save_folder: str = "knot_density_figures", verbose: bool = False, color: str = "black",
                              tqdm_position: int = 0, tqdm_leave: bool = True):
    """
    calculates the support accuracy of the ISTA module. This is done by looking at the support of the x-vector, and comparing it to the support of the x-hat vector.
    We calculate the support accuracy over the iterations.

    inputs:
    - ista: the ISTA module
    - K: the number of iterations
    - A: the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
    - y: the y-vector, of shape (batch_size, M)
    - x: the x-vector, of shape (batch_size, N)
    - save_name: the name to save the figure as
    - save_folder: the folder to save the figure in
    - verbose: if True, print a progress bar
    - color: the color of the plot
    - tqdm_position: the position of the tqdm bar
    - tqdm_leave: if True, leave the tqdm bar after finishing
    """ 
    # empty the CUDA cache
    torch.cuda.empty_cache()

    # create the save folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # create an accuracy array of size (K + 1)
    accuracy_array        = torch.zeros(K+1)
    reconstruction_losses = torch.zeros(K+1)

    # run the initials function to get the initial x and jacobian
    x_hat, jacobian = ista.get_initial_x_and_jacobian(x.shape[0], calculate_jacobian = False)

    # get the initial support accuracy
    accuracy_array[0] = support_accuracy(x, x_hat)

    # get the initial reconstruction loss
    reconstruction_losses[0] = torch.mean(torch.abs(x_hat - x))

    # loop over the iterations of K
    for k in tqdm(range(K), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="support accuracy analysis, runnning over folds"):
        with torch.no_grad():
            x_hat, jacobian = ista.forward_at_iteration(x_hat, y, k, jacobian)

        # calculate the support accuracy
        accuracy_array[k+1] = support_accuracy(x, x_hat)

        # calculate the reconstruction loss
        reconstruction_losses[k+1] = torch.mean(torch.abs(x_hat - x))

    # plot the support accuracy over the iterations
    plt.figure()
    folds = np.arange(0,K+1)
    plt.plot(folds,accuracy_array,'-', label = "accuracy", color = color)
    plt.xlabel("fold")
    plt.ylabel("support accuracy")
    plt.grid()
    plt.title(save_name)
    plt.xlim([0,K])
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{save_name}_support_accuracy.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_folder}/{save_name}_support_accuracy.svg", bbox_inches='tight')
    plt.close()

    # plot the reconstruction loss over the iterations
    plt.figure()
    folds = np.arange(0,K+1)
    plt.plot(folds,reconstruction_losses,'-', label = "reconstruction loss", color = color)
    plt.yscale("log", base=2)
    plt.xlabel("fold")
    plt.ylabel("validation loss")
    plt.grid()
    plt.title(save_name)
    plt.xlim([0,K])
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{save_name}_validation_loss.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_folder}/{save_name}_validation_loss.svg", bbox_inches='tight')
    plt.close()

    # give the knot density array back
    return accuracy_array, reconstruction_losses


# %% test the modules
if __name__ == "__main__":
    # set the seed
    torch.manual_seed(0)

    # params
    M = 8
    N = 16
    batch_size = 512
    K_ista =  2**11
    K_lista = 2**10
    mu = 1
    _lambda = 0.1
    device = "cuda:0"

    # grid search for ista?
    perform_grid_search_ista = True
    mus      = np.arange(201)/200 # from 0 to 2, with steps of 0.01
    _lambdas = np.arange(101)/100 # from 0 to 1, with steps of 0.01
    nr_points_for_grid_search = 2**12

    # visual analysis
    nr_points_along_axis = 2**10
    indices_of_projection = (0,1,2)
    margin = 0.5

    # random analysis
    max_magnitude = 2
    nr_points_total    = 2**24
    nr_points_in_batch = 2**21
    
    # knot density analysis
    nr_paths             = 1
    anchor_point_std     = 1
    nr_points_along_path = 2**20  
    path_delta           = 0.001

    # lista data and training
    maximum_sparsity = 4
    x_magnitude = (0, 2)
    nr_iterations_to_train_for = 2048
    end_forgetting_factor = 1
    forgetting_factor = end_forgetting_factor**(1/K_lista)

    # tests
    test_ista_visually      = False
    test_ista_randomly      = False
    test_ista_knot_density  = True

    test_lista_visually     = False
    test_lista_randomly     = False
    test_lista_knot_density = True

    # create a random matrix A
    A = create_random_matrix_with_good_singular_values(M, N)  

    # %% grid search for ISTA
    if perform_grid_search_ista:
        # create a data generator
        data_generator_initialized = lambda: data_generator(A, nr_points_for_grid_search, maximum_sparsity, x_magnitude, N, device)

        # find the best mu and lambda
        mu, _lambda = grid_search_ista(A, data_generator_initialized, mus, _lambdas, K_ista, forgetting_factor = forgetting_factor, device=device)

        # print the best mu and lambda
        print("\ngrid search for ISTA results in:")
        print(f"best mu: {mu}")
        print(f"best lambda: {_lambda}\n")


    # %% test the ISTA module
    if test_ista_visually or test_ista_randomly or test_ista_knot_density:
        # create the ISTA module
        ista = ISTA(A, mu = mu, _lambda = _lambda, K=K_ista, device=device)

        if test_ista_visually:
            # perform the visual analysis on it
            nr_regions_arrray_ista_visual = visual_analysis_of_ista(ista, K_ista, nr_points_along_axis, margin, indices_of_projection, A, save_folder = "ista_figures")
            make_gif_from_figures_in_folder("ista_figures", 5)

        if test_ista_randomly:
            # perform the full analysis on it
            nr_regions_arrray_ista_full = random_analysis_of_ista(ista, K_ista, nr_points_total,  nr_points_in_batch, max_magnitude, A, save_name = "nr_regions_ISTA_random_points")

        if test_ista_knot_density:
            # perform the knot density analysis
            knot_density_ista = knot_density_analysis(ista, K_ista, A, nr_paths = nr_paths,  anchor_point_std = anchor_point_std,
                                                        nr_points_along_path=nr_points_along_path, path_delta=path_delta,
                                                        save_name = "knot_density_ISTA", verbose = True, color = 'tab:blue')

    # %% test the LISTA module
    if test_lista_visually or test_lista_randomly or test_lista_knot_density:
        # create the ISTA module again random initialization
        lista = LISTA(A, mu = mu, _lambda = _lambda, K=K_lista, device=device, initialize_randomly = False)

        # we then train LISTA
        data_generator_initialized = lambda: data_generator(A, batch_size, maximum_sparsity, x_magnitude, N, device)
        lista,_ = train_lista(lista, data_generator_initialized, nr_iterations_to_train_for, forgetting_factor, show_loss_plot = False)

        if test_lista_visually:
            # perform the visual analysis on it
            nr_regions_arrray_lista_learned_visual =  visual_analysis_of_ista(lista, K_lista, nr_points_along_axis, margin, indices_of_projection, A, save_folder = "lista_figures_trained")
            make_gif_from_figures_in_folder("lista_figures_trained", 5)

        if test_lista_randomly:
            # perform the full analysis on it
            nr_regions_arrray_lista_full = random_analysis_of_ista(lista, K_lista, nr_points_total,  nr_points_in_batch, max_magnitude, A, save_name = "nr_regions_LISTA_random_points")

        if test_lista_knot_density:
            # perform the knot density analysis
            knot_density_lista = knot_density_analysis(lista, K_lista, A, nr_paths = nr_paths,  anchor_point_std = anchor_point_std,
                                                         nr_points_along_path=nr_points_along_path, path_delta=path_delta,
                                                         save_name = "knot_density_LISTA", verbose = True, color= 'tab:orange')


    # %% if we plotted both, make a united figure for the number of regions over the iterations
    if test_ista_visually and test_lista_visually:
        plt.figure()
        plt.plot(nr_regions_arrray_ista_visual,'-', label = "ISTA")
        plt.plot(nr_regions_arrray_lista_learned_visual,'-', label = "LISTA after training")
        plt.grid()
        plt.xlabel("iteration")
        plt.ylabel("number of linear regions")
        plt.legend()
        plt.tight_layout()
        plt.savefig("hyperplane_analysis_figures/united_nr_regions_over_iterations.png", dpi=300, bbox_inches='tight')
        plt.savefig("hyperplane_analysis_figures/united_nr_regions_over_iterations.svg", bbox_inches='tight')
        plt.close()

    if test_ista_randomly and test_lista_randomly:
        plt.figure()
        plt.semilogy(nr_regions_arrray_ista_full,'-', label = "ISTA", base = 2)
        plt.semilogy(nr_regions_arrray_lista_full,'-', label = "LISTA after training", base = 2)
        plt.semilogy([0,len(nr_regions_arrray_lista_full)], [nr_points_total,nr_points_total], 'r--', label = "total number of points used", base = 2)
        plt.grid()
        plt.xlabel("iteration")
        plt.ylabel("number of linear regions")
        plt.legend()
        plt.tight_layout()
        plt.savefig("random_analysis_figures/united_nr_regions_over_iterations_random.png", dpi=300, bbox_inches='tight')
        plt.savefig("random_analysis_figures/united_nr_regions_over_iterations_random.svg", bbox_inches='tight')
        plt.close()

    if test_ista_knot_density and test_lista_knot_density:
        K_max = max(K_ista, K_lista)
        folds_ista = np.arange(1,K_ista+1)
        knot_density_ista_mean = knot_density_ista.mean(dim=0)
        folds_lista = np.arange(1,K_lista+1)
        knot_density_lista_mean = knot_density_lista.mean(dim=0)

        plt.figure()
        plt.plot(folds_ista,knot_density_ista_mean,'-', label = "ISTA", c = 'tab:blue')
        plt.plot(folds_lista,knot_density_lista_mean,'-', label = "LISTA", c = 'tab:orange')
        plt.grid()
        plt.xlabel("fold")
        plt.ylabel("knot density")
        plt.legend(loc='best')
        plt.xlim([0,K_max])
        plt.tight_layout()
        plt.savefig("knot_density_figures/united_knot_density_over_iterations.png", dpi=300, bbox_inches='tight')
        plt.savefig("knot_density_figures/united_knot_density_over_iterations.svg", bbox_inches='tight')
        plt.close()
