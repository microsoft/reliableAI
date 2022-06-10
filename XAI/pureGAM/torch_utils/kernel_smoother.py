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

def compute_pairwise_distance_2nd(X1: th.tensor, X2: th.tensor, x1: th.tensor, x2: th.tensor) -> th.tensor:
    X1_diff = x1.unsqueeze(1) - X1.unsqueeze(0)
    X2_diff = x2.unsqueeze(1) - X2.unsqueeze(0)
    # a norm_2 function apply on axis 1 to compute pairwise distance.
    #X_diff = th.sqrt((X_diff ** 2).sum(axis=0))
    X_diff = X1_diff ** 2 + X2_diff ** 2
    return X_diff

# Deprecated, in Numerical Transformer, you should use
#         if lam:
#             X_diff_divlam = compute_pairwise_distance(X, x)/lam
#         else:
#             X_diff_divlam = compute_pairwise_distance(X, x)/self.lam
#         kij = self.K_transform_func(X_diff_divlam)
#         if is_norm:
#             kij = kij/kij.sum(axis=-1, keepdim=True)
#         #np.apply_along_axis(lambda k: k / k.sum(), -1, kij)
#         return kij
class Kernel_Weigther:
    def __init__(self, lam: float, K_transform_func=K_Epanechnikov):
        self.lam = lam
        self.K_transform_func = K_transform_func

    '''
        be aware that kw.S() is low cost in time complexity but relatively high in space complexity.
        So it's preferred to use kw.S() inside of SGD training than pre-calculate kw.S() before training.
    '''
    def S(self, X, x, is_norm=True, lam=None):
        if lam:
            X_diff_divlam = compute_pairwise_distance(X, x)/np.square(self.lam) # FIX: should be squared lambda.
        else:
            X_diff_divlam = compute_pairwise_distance(X, x)/np.square(self.lam) # FIX: should be squared lambda.
        kij = self.K_transform_func(X_diff_divlam)
        if is_norm:
            kij = kij/kij.sum(axis=-1, keepdim=True)
        #np.apply_along_axis(lambda k: k / k.sum(), -1, kij)
        return kij

    '''def S_batch(self, X, x, batch_size = 2048):
        x_list = x.split(batch_size, dim=0)
        ret_list = []
        for x_batch in x_list:
            ret_list.append(self.S(X, x_batch).cpu())
        return ret_list'''

    def S_train(self, X):
        return self.S(X, X)
"""
    examples
    from kernel_smoother_tensor import Kernel_Weigther
    kernel_weighter = Kernel_Weigther(lam=0.1)
    t0 = time()
    s2 = kernel_weighter.S(X, X2)
    t1 = time()
    print("Time using broadcasting:{} seconds.".format(t1-t0))
    print(s2)
"""
