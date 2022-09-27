# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import torch as th

"""
    The tri_cube kernel/ Epanechnikov/  The Gaussian kernel.

Args:
    t (float 2d array): The elem value v means relative distance between two point. v = norm(x1 - x2)/lam

Returns:
    float: The kernel weight.
"""
def K_Epanechnikov(t: th.tensor) -> th.tensor:
    indexes = t <= 1
    t[indexes] = 3 / 4 * (1. - t[indexes] ** 2)
    t[~indexes] = 0
    return t

def K_tri_cube(t: th.tensor) -> th.tensor:
    indexes = t <= 1
    t_select = t[indexes]
    t[indexes] = (1. - t_select**3)**3
    t[~indexes] = 0
    return t

def K_Gaussian(t: th.tensor) -> th.tensor:
    #return th.exp(-1 / 2 * t ** 2)
    return th.exp(-1 / 2 * t) / np.sqrt(2 * np.pi) # need to normalize to 1 if used by KDE.

"""
    X, x is all (N_samples, p) (N_samples2, p)
"""
def compute_pairwise_distance(X: th.tensor, x: th.tensor, isMultiDim=True) -> th.tensor:
    X_diff = x.unsqueeze(2).permute(1,0,2) - X.unsqueeze(2).permute(1,2,0)
    # a norm_2 function apply on axis 1 to compute pairwise distance.
    #X_diff = th.sqrt((X_diff ** 2).sum(axis=0))
    if isMultiDim:
        X_diff = th.square(X_diff)
        X_diff =X_diff.sum(axis=0)

        #(N1, N2)
    else:
        X_diff = th.square(X_diff)
        #(p, N1, N2)
    return X_diff


