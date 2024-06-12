"""
This file creates some functions usefull for the design of experiments.
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings

# %%
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

def create_random_matrix(M: int, N: int):
    """
    This function creates a random matrix A.
    """

    # create a random matrix A
    A = torch.randn(M, N) * 0.1

    return A


def sample_experiment(config: dict, max_tries: int = 1000):
    """
    This function will sample parameters that vary to create an experiment.
    """
    for _ in range(max_tries):
        # sample the parameters that vary
        M = torch.randint(config["data_that_varies"]["M"]["min"], config["data_that_varies"]["M"]["max"] + 1, (1,)).item()
        N = torch.randint(config["data_that_varies"]["N"]["min"], config["data_that_varies"]["N"]["max"] + 1, (1,)).item()
        K = torch.randint(config["data_that_varies"]["K"]["min"], config["data_that_varies"]["K"]["max"] + 1, (1,)).item()

        # check if the parameters are valid
        if M <= N and K <= M:
            break

    else:
        # for-else triggers if the for loop did not break (i.e. we did not find valid parameters after max_tries)
        raise ValueError("Could not find valid parameters after {} tries.".format(max_tries))

    # create the A matrix that belongs to these parameters
    if config["A_with_good_singular_values"]:
        A = create_random_matrix_with_good_singular_values(M, N)
    else:
        A = create_random_matrix(M, N)
    
    return M, N, K, A