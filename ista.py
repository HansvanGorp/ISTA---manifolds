"""
We here implement (R)(L)ISTA as a pytorch module
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

# %% create an ISTA prototype module
class ISTAPrototype(torch.nn.Module):
    def __init__(self, A: torch.tensor, nr_folds: int = 16, device: str = "cpu", train_inputs = None):
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
        - nr_folds (int): the number of iterations for ISTA
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
        self.nr_folds = nr_folds
        self.device = device
        self.train_inputs = train_inputs

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
        - x (torch.tensor): the output x, of shape (batch_size, N) or (batch_size, N, nr_folds) if return_intermediate is True
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
            x_intermediate = torch.zeros(y.shape[0], self.N, self.nr_folds, dtype=torch.float32, device=self.device)

        # Now start going over the iterations
        for fold_idx in tqdm(range(self.nr_folds), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="running ISTA folds"):
            # perform the forward function at this iteration
            x, jacobian = self.forward_at_iteration(x, y, fold_idx, jacobian, jacobian_projection)

            # if we are returning the intermediate x's, store them
            if return_intermediate:
                x_intermediate[:,:,fold_idx] = x

        # if we are returning the intermediate x's, return them
        if return_intermediate:
            return x_intermediate, jacobian
        else:
            return x, jacobian
    
    def forward_at_iteration(self, x:torch.tensor, y: torch.tensor, fold_idx: int, jacobian:torch.tensor = None, jacobian_projection: torch.tensor = None):
        """
        Implements the forward function of the prototype at a specific iteration.

        inputs:
        - x (torch.tensor): the current x, of shape (batch_size, N)
        - y (torch.tensor): the input y, of shape (batch_size, M)
        - fold_idx (int): the current iteration (this is needed to get the correct W1 and W2 matrices)
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M), or (batch_size, N, 2) if jacobian_projection is not None
        - jacobian_projection (torch.tensor): a projection matrix to project the Jacobian to a 2D space, for visualization, of shape (2, M)
        """
        # make sure things are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)
        jacobian = jacobian.to(self.device) if jacobian is not None else None
        jacobian_projection = jacobian_projection.to(self.device) if jacobian_projection is not None else None

        # step 1, perform data-consistency
        x, jacobian = self.data_consistency( x, y, fold_idx, jacobian, jacobian_projection)

        # step 2, perform thresholding
        x, jacobian = self.soft_thresholding(x,    fold_idx, jacobian)

        return x, jacobian

    def get_initial_x_and_jacobian(self, batch_size: int, calculate_jacobian: bool, jacobian_projection: torch.tensor = None, overwite_device: str = None):
        return self.get_initial_x(batch_size, overwite_device=overwite_device), self.initalize_jacobian(batch_size, calculate_jacobian, jacobian_projection,overwite_device=overwite_device)
    
    def get_initial_x(self, batch_size: int, overwite_device: str = None):
        """
        Initializes the x vector.

        inputs:
        - batch_size (int): the batch size
        """
        return torch.zeros(batch_size, self.N, dtype=torch.float32, device=self.device if overwite_device is None else overwite_device)
    
    def initalize_jacobian(self, batch_size: int, calculate_jacobian: bool, jacobian_projection: torch.tensor, overwite_device: str = None):
        """
        Initializes the Jacobian matrix.

        inputs:
        - batch_size (int): the batch size
        - calculate_jacobian (bool): if True, calculate the Jacobian of the forward function
        - jacobian_projection (torch.tensor): a projection matrix to project the Jacobian to a 2D space, for visualization, of shape (2, M)
        """
        if calculate_jacobian and jacobian_projection is None:
            # initialize the Jacobian with all zeros
            jacobian = torch.zeros(batch_size, self.N, self.M, dtype=torch.float32, device=self.device if overwite_device is None else overwite_device)

        elif calculate_jacobian and jacobian_projection is not None:
            # initialize the Jacobian with all zeros in the 2D space
            jacobian = torch.zeros(batch_size, self.N,       2, dtype=torch.float32, device=self.device if overwite_device is None else overwite_device)

        else:
            # put it to None
            jacobian = None

        return jacobian

    def data_consistency(self, x: torch.tensor, y: torch.tensor, fold_idx: int, jacobian: torch.tensor = None, jacobian_projection: torch.tensor = None):
        """
        Implements the data consistency step of the ISTA algorithm.

        inputs:
        - x (torch.tensor): the current x, of shape (batch_size, N)
        - y (torch.tensor): the input y, of shape (batch_size, M)
        - fold_idx (int): the current iteration (this is needed to get the correct W1 and W2 matrices)
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M) (if None, it is not calculated)
        - jacobian_projection (torch.tensor): a projection matrix to project the Jacobian to a 2D space, for visualization, of shape (2, M)
        """
        # get W1 and W2 at the current iteration
        W1   = self.get_W1_at_iteration(fold_idx)
        W2   = self.get_W2_at_iteration(fold_idx)
        bias = self.get_bias_at_iteration(fold_idx)

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
    
    def soft_thresholding(self, x: torch.tensor, fold_idx: int, jacobian: torch.tensor = None, max_clip: float = 10):
        """
        Implements the soft thresholding step of the ISTA algorithm.

        inputs:
        - x (torch.tensor): the current x, of shape (batch_size, N)
        - fold_idx (int): the current iteration (this is needed to get the correct lambda value)
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M) (if None, it is not calculated)
        - max_clip (float): the maximum magnitude value to clip x to (to prevent numerical instability)
        """
        # get lambda at the current iteration
        _lambda = self.get_lambda_at_iteration(fold_idx)

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
    def __init__(self, A: torch.tensor, mu: float = 0.5, _lambda: float = 0.5, nr_folds: int = 16, device: str = "cpu"):
        super(ISTA, self).__init__(A, nr_folds, device)
        """Create the ISTA module with the input parameters:
        - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
        - mu (float): the step size for ISTA
        - _lambda (float): the threshold for ISTA
        - nr_folds (int): the number of iterations for ISTA
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

    # reset params with new mu and lambda
    def reset_params_using_mu_and_lambda(self, mu: float, _lambda: float):
        self.mu = mu
        self._lambda = _lambda
        self.W1 = self.mu*self.A.t()
        self.W2 = torch.eye(self.N).to(self.device) - self.mu*self.A.t()@self.A

    # The prototype requires three functions to be implemented by the inherited modules:
    def get_W1_at_iteration(self, fold_idx):
        # simply return W1, ISTA does not change W1 over iterations
        return self.W1
    
    def get_W2_at_iteration(self, fold_idx):
        # simply return W2, ISTA does not change W2 over iterations
        return self.W2
    
    def get_bias_at_iteration(self, fold_idx):
        # simply return 0, ISTA does not have a bias
        return torch.zeros(1, device = self.device)
    
    def get_lambda_at_iteration(self, fold_idx):
        # simply return _lambda, ISTA does not change _lambda over iterations
        return self._lambda

# %% create a FISTA module that inherits from ISTAPrototype
class FISTA(ISTAPrototype):
    # For reference, see: https://epubs.siam.org/doi/pdf/10.1137/080716542?casa_token=FPitKygai98AAAAA:Hd0FPfroFaH8n5wqnPYaRCS2NRsAMq13au6XL60EpkO_KSRvvD9s8pzKUIelOGMumOrq88uj1w
    def __init__(self, A: torch.tensor, mu: float = 0.5, _lambda: float = 0.5, nr_folds: int = 16, device: str = "cpu"):
        super(FISTA, self).__init__(A, nr_folds, device)
        """Create the FISTA module with the input parameters:
        - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
        - mu (float): the step size for ISTA
        - _lambda (float): the threshold for ISTA
        - nr_folds (int): the number of iterations for ISTA
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

        self.t_prev = 1
        # x_prev is initialised in the forward_at_iteration function
        self.x_prev = None
        self.jacobian_prev = None

    def forward_at_iteration(self, x:torch.tensor, y: torch.tensor, fold_idx: int, jacobian:torch.tensor = None, jacobian_projection: torch.tensor = None):
        """
        Implements the forward function of the prototype at a specific iteration.

        inputs:
        - x (torch.tensor): the current x, of shape (batch_size, N)
        - y (torch.tensor): the input y, of shape (batch_size, M)
        - fold_idx (int): the current iteration (this is needed to get the correct W1 and W2 matrices)
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M), or (batch_size, N, 2) if jacobian_projection is not None
        - jacobian_projection (torch.tensor): a projection matrix to project the Jacobian to a 2D space, for visualization, of shape (2, M)
        """
        # make sure things are on the correct device
        x = x.to(self.device)
        y = y.to(self.device)
        jacobian = jacobian.to(self.device) if jacobian is not None else None
        jacobian_projection = jacobian_projection.to(self.device) if jacobian_projection is not None else None

        if fold_idx == 0 or self.x_prev is None:
            # intialize x_prev and jacobian_prev as initial x and initial jacobian
            self.x_prev = x
            self.jacobian_prev = jacobian

        # step 1, perform data-consistency
        x, jacobian = self.data_consistency( x, y, fold_idx, jacobian, jacobian_projection)

        # step 2, perform thresholding
        # x is now equal to x_k from the FISTA paper
        x, jacobian = self.soft_thresholding(x,    fold_idx, jacobian)
        
        # then compute t_{k+1}
        t = (1 + np.sqrt(1 + 4*(self.t_prev)**2)) / 2
        t_scaling = ((self.t_prev - 1) / t)
        
        # x_new is y_{k+1} from paper
        x_new = x + t_scaling * (x - self.x_prev) 
        
        # compute new jacobian if needed
        if jacobian is not None:
            jacobian = ((1 + t_scaling) * jacobian) - (t_scaling * self.jacobian_prev)

        self.x_prev = x
        self.t_prev = t
        self.jacobian_prev = jacobian

        return x_new, jacobian

    # reset params with new mu and lambda
    def reset_params_using_mu_and_lambda(self, mu: float, _lambda: float):
        self.mu = mu
        self._lambda = _lambda
        self.W1 = self.mu*self.A.t()
        self.W2 = torch.eye(self.N).to(self.device) - self.mu*self.A.t()@self.A

    # The prototype requires three functions to be implemented by the inherited modules:
    def get_W1_at_iteration(self, fold_idx):
        # simply return W1, ISTA does not change W1 over iterations
        return self.W1
    
    def get_W2_at_iteration(self, fold_idx):
        # simply return W2, ISTA does not change W2 over iterations
        return self.W2
    
    def get_bias_at_iteration(self, fold_idx):
        # simply return 0, ISTA does not have a bias
        return torch.zeros(1, device = self.device)
    
    def get_lambda_at_iteration(self, fold_idx):
        # simply return _lambda, ISTA does not change _lambda over iterations
        return self._lambda

# %% create a LISTA module that inherits from ISTAPrototype
class LISTA(ISTAPrototype):
    def __init__(self, A: torch.tensor, mu: float = 0.5, _lambda: float = 0.5, nr_folds: int = 16, device: str = "cpu", initialize_randomly: bool = True, share_weights: bool = False, train_inputs = None):
        super(LISTA, self).__init__(A, nr_folds, device, train_inputs=train_inputs)
        """Create the LISTA module with the input parameters:
        - A (torch.tensor): the matrix A in the equation y=Ax, of shape (M, N), with M<N, i.e. M is the measurement dimension and N is the signal dimension
        - mu (float): the step size for ISTA
        - _lambda (float): the threshold for ISTA
        - nr_folds (int): the number of iterations for ISTA
        - device (str): the device to run the module on, default is "cpu"
        - initialize_randomly: if true, we initialize W1 and W2 randomly, otheriwse we use our knowledge of the problem to initialize them in a good way

        Since LISTA inherits from ISTAPrototype, it has all the functionalities of ISTAPrototype.
        We only need to specify the parameters that are specific to ISTA. which is the fact that W1 and W2 are learned parameters over the iterations.
        """

        # save mu and _lambda
        self.mu = mu
        self._lambda = _lambda
        self.share_weights = share_weights

        if initialize_randomly:
            if share_weights:
                self.W1 = torch.nn.Parameter(torch.randn(self.N, self.M, device=self.device) / self.N**0.5)
                self.W2 = torch.nn.Parameter(torch.randn(self.N, self.M, device=self.device) / self.N**0.5)
                self.bias = torch.nn.Parameter(torch.randn(self.N, device=self.device))
            else:
                self.W1   = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.N, self.M, device=self.device) / self.N**0.5) for _ in range(nr_folds)])
                self.W2   = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.N, self.N, device=self.device) / self.N**0.5) for _ in range(nr_folds)])
                self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.N, device=self.device)) for _ in range(nr_folds)])
 
        else:
            # create initial W1 and W2 of Ista
            W1_initialization = self.mu*self.A.t()
            W2_initialization = torch.eye(self.N).to(self.device) - self.mu*self.A.t()@self.A

            if share_weights:
                self.W1 = torch.nn.Parameter(W1_initialization.clone().detach())
                self.W2 = torch.nn.Parameter(W2_initialization.clone().detach())
                self.bias = torch.nn.Parameter(torch.zeros(self.N, device=self.device))
            else:
                # now create the W1 and W2 as torch.nn.Parameter, but do so over all the nr_folds iterations
                self.W1 = torch.nn.ParameterList([torch.nn.Parameter(W1_initialization.clone().detach()) for _ in range(nr_folds)])
                self.W2 = torch.nn.ParameterList([torch.nn.Parameter(W2_initialization.clone().detach()) for _ in range(nr_folds)])
                self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(self.N, device=self.device)) for _ in range(nr_folds)])

    # The prototype requires three functions to be implemented by the inherited modules:
    def get_W1_at_iteration(self, fold_idx):
        if self.share_weights:
            return self.W1
        else:
            # return the W1 at the current iteration
            return self.W1[fold_idx]
    
    def get_W2_at_iteration(self, fold_idx):
        if self.share_weights:
            return self.W2
        else:
            # return the W2 at the current iteration
            return self.W2[fold_idx]
    
    def get_bias_at_iteration(self, fold_idx):
        if self.share_weights:
            return self.bias
        else:
            # return the bias at the current iteration
            return self.bias[fold_idx]
    
    def get_lambda_at_iteration(self, fold_idx):
        # simply return _lambda, LISTA does not change _lambda over iterations
        return self._lambda
    

class ToeplitzLISTA(ISTAPrototype):
    def __init__(self, A: torch.tensor, mu: float = 0.5, _lambda: float = 0.5, nr_folds: int = 16, device: str = "cpu", initialize_randomly: bool = True, share_weights: bool = False, train_inputs = None):
        super(ToeplitzLISTA, self).__init__(A, nr_folds, device, train_inputs=train_inputs)
        """
        Toeplitz LISTA
        https://ieeexplore.ieee.org/abstract/document/8835826
        """

        # save mu and _lambda
        self.mu = mu
        self._lambda = _lambda
        self.share_weights = share_weights

        if initialize_randomly:
            if share_weights:
                self.W1 = torch.nn.Parameter(torch.randn(self.N, self.M, device=self.device))
                self.conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(2*self.N - 1), stride=1, padding=(self.N-1), device=self.device)
                self.bias = torch.nn.Parameter(torch.randn(self.N, device=self.device))
            else:
                self.W1   = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.N, self.M, device=self.device)) for _ in range(nr_folds)])
                self.conv  = torch.nn.ParameterList([torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(2*self.N - 1), stride=1, padding=(self.N-1), device=self.device) for _ in range(nr_folds)])
                self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(self.N, device=self.device)) for _ in range(nr_folds)])
 
        else:
            # create initial W1 and W2 of Ista
            W1_initialization = self.mu*self.A.t()
            # W2_initialization = torch.eye(self.N).to(self.device) - self.mu*self.A.t()@self.A
            
            # The convs are still initialised randomly

            if share_weights:
                self.W1 = torch.nn.Parameter(W1_initialization.clone().detach())
                self.conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(2*self.N - 1), stride=1, padding=(self.N-1), device=self.device)
                self.bias = torch.nn.Parameter(torch.zeros(self.N, device=self.device))
            else:
                # now create the W1 and W2 as torch.nn.Parameter, but do so over all the nr_folds iterations
                self.W1 = torch.nn.ParameterList([torch.nn.Parameter(W1_initialization.clone().detach()) for _ in range(nr_folds)])
                self.conv = torch.nn.ParameterList([torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(2*self.N - 1), stride=1, padding=(self.N-1), device=self.device) for _ in range(nr_folds)])
                self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(self.N, device=self.device)) for _ in range(nr_folds)])

    # The prototype requires three functions to be implemented by the inherited modules:
    def get_W1_at_iteration(self, fold_idx):
        if self.share_weights:
            return self.W1
        else:
            # return the W1 at the current iteration
            return self.W1[fold_idx]
    
    def get_W2_at_iteration(self, fold_idx):
        if self.share_weights:
            return self.conv
        else:
            # return the W2 at the current iteration
            return self.conv[fold_idx]
    
    def get_bias_at_iteration(self, fold_idx):
        if self.share_weights:
            return self.bias
        else:
            # return the bias at the current iteration
            return self.bias[fold_idx]
    
    def get_lambda_at_iteration(self, fold_idx):
        # simply return _lambda, LISTA does not change _lambda over iterations
        return self._lambda
    
    # def get_conv_as_matrix(self, fold_idx):
    #     kernel = self.conv[fold_idx].weight[0, 0, :]
    
    def data_consistency(self, x: torch.tensor, y: torch.tensor, fold_idx: int, jacobian: torch.tensor = None, jacobian_projection: torch.tensor = None):
        """
        Implements the data consistency step of the ISTA algorithm.

        inputs:
        - x (torch.tensor): the current x, of shape (batch_size, N)
        - y (torch.tensor): the input y, of shape (batch_size, M)
        - fold_idx (int): the current iteration (this is needed to get the correct W1 and W2 matrices)
        - jacobian (torch.tensor): the Jacobian of the forward function, of shape (batch_size, N, M) (if None, it is not calculated)
        - jacobian_projection (torch.tensor): a projection matrix to project the Jacobian to a 2D space, for visualization, of shape (2, M)
        """
        # get W1 and W2 at the current iteration
        W1   = self.get_W1_at_iteration(fold_idx)
        W2_conv = self.get_W2_at_iteration(fold_idx)
        bias = self.get_bias_at_iteration(fold_idx)

        # perform data consistency
        W1_times_y = torch.nn.functional.linear(y, W1)
        W2_times_x = W2_conv(x[:, None, :])[:,0,:] # add and remove channel dim to keep conv happy 
        x = W1_times_y + W2_times_x + bias.unsqueeze(0)

        # calculate the Jacobian if needed
        if jacobian is not None:
            # the jacobian gets multiplied by W2, which is a linear operation
            # here we apply the conv to each column separately and stack back together, which should
            # implement the same operation as if we would multiply by the conv in matrix form.
            jacobian = torch.stack([W2_conv(jacobian_col_i[:, None, :])[:, 0, :] for jacobian_col_i in torch.unbind(jacobian, axis=2)], axis=2)

            # then W1 gets added to the result, which is also a linear operation
            additive_jacobian = W1
            if jacobian_projection is not None:
                # if we are projecting the Jacobian to a 2D space, we need to project the additive Jacobian as well
                additive_jacobian = torch.matmul(additive_jacobian, jacobian_projection)
                
            jacobian = jacobian + additive_jacobian.unsqueeze(0)

        # return the result
        return x, jacobian