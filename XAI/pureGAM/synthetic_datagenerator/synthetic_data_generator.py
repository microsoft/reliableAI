import numpy as np
import matplotlib.pyplot as plt
import pandas
from scipy.stats import ortho_group
from scipy.stats import norm
from pureGAM_model.utils import generate_pairwise_idxes
import random
from pureGAM_model.categorical_encoder import Categorical_encoder

"""
Numerical Data
"""
def generate_x(name, n, p, seed=42, cov_mat=None, alpha=None, unif_range=None):
    np.random.seed(seed)
    if name == "unif":
        if unif_range is not None:
            assert len(unif_range) == 2
        else:
            unif_range = [-1., 1.]
        data = np.random.uniform(unif_range[0], unif_range[1], size=(n, p))
    elif name == "gaussian":
        if cov_mat is not None:
            assert cov_mat.shape[0] == p and cov_mat.shape[1] == p
        else:
            cov_mat = np.identity(p)    
        data = np.random.multivariate_normal(np.zeros(p), cov_mat, n)
    elif name == "dirichlet":
        if alpha is not None:
            assert len(alpha) == p + 1
        else:
            alpha = np.ones(p + 1)
        data = np.random.dirichlet(alpha, n)[:, :-1] # drop the last column which has perfect correlation with the others.
    else:
        raise ValueError(str(name) + " is not a valid distribution!")
    return data

## round power to prevent e.g. (-1)^0.5. Assume the coef is in (1.5, 2.5), transform to {1,2,3}.
def round_pow(coef, lbd=1.5, ubd=2.5, ubd_to=3):
    coef_new = np.round((coef - lbd) / (ubd - lbd) * ubd_to, 0) + 1
    return coef_new

def function_main_effects(name, coef=None, round_power=True, abs_power=True):
    if name == "x":
        return lambda x : x
    if name == "x^c":
        if coef is None:
            coef = 2.
        elif round_power:
            coef = round_pow(coef)
        return lambda x: np.power(x, coef)
    if name == "c^x":
        if coef is None:
            coef = 2.
        elif abs_power:
            coef = np.abs(coef)
        return lambda x: np.power(coef, x)
    if name == "log(x)": # careful when it is not defined!
        return lambda x: np.log(x)
    if name == "sin(x)":
        if coef is None:
            coef = np.pi
        return lambda x: np.sin(coef * x)
    if name == "cos(x)":
        if coef is None:
            coef = np.pi
        return lambda x: np.cos(coef * x)

def function_interaction(name, coef1=None, coef2=None, round_power=True, abs_power=True):
    if name == "x^cy^d":
        if coef1 is None or coef2 is None:
            coef1 = 1.
            coef2 = 1.
        elif round_power:
            coef1 = round_pow(coef1)
            coef2 = round_pow(coef2)
        return lambda x, y: np.power(x, coef1) * np.power(y, coef2)
    if name == "c^xd^y":
        if coef1 is None or coef2 is None:
            coef1 = 2.
            coef2 = 2.
        elif abs_power:
            coef1 = np.abs(coef1)
            coef2 = np.abs(coef2)
        return lambda x, y: np.power(coef1, x) * np.power(coef2, y)
    if name == "log(cx+dy)": # careful when it is not defined!
        if coef1 is None or coef2 is None:
            coef1 = 1.
            coef2 = 1.
        return lambda x, y: np.log(coef1 * x + coef2 * y)
    if name == "sin(cx+dy)":
        if coef1 is None or coef2 is None:
            coef1 = np.pi
            coef2 = np.pi
        return lambda x, y: np.sin(coef1 * x + coef2 * y)
    if name == "cos(cx+dy)":
        if coef1 is None or coef2 is None:
            coef1 = np.pi
            coef2 = np.pi
        return lambda x, y: np.cos(coef1 * x + coef2 * y)
    if name == "sin(cxy)":
        if coef1 is None or coef2 is None:
            coef1 = np.pi
            coef2 = np.pi
        return lambda x, y: np.sin(coef1 * x  * y)
    if name == "cos(cxy)":
        if coef1 is None or coef2 is None:
            coef1 = np.pi
            coef2 = np.pi
        return lambda x, y: np.cos(coef1 * x * y)
    if name == "xsin(y)":
        return lambda x, y: x * np.sin(y)
    if name == "x2^y":
        return lambda x,y : x * np.power(2, y)
    if name == "None":
        return lambda x, y: 0.


"""
Num data gen
"""
def num_gen_gauss(N, p, seed=2142543, truncate=True, bound=5, return_cov=False, return_X_only=False, new_cov=True, cov_for_cat=False):
    np.random.seed(seed)
    Int_Idx_df = pandas.DataFrame([[0 for _ in range(p)] for _ in range(p)], columns=list(range(p)),
                                  index=list(range(p)))
    if truncate:
        len_total = 0
        Xs = []
        last_len_totol = 0
        while len_total < N:
            #print(len_total, N)
            dist = "gaussian"
            if new_cov:
                U = ortho_group.rvs(dim=p)
                d = np.arange(1, p + 1) ** 1
                s = d.sum()
                d = d / s
                if cov_for_cat:
                    d = d * s
                cov_mat = U @ np.diag(d) @ U.T
            else:
                A = np.random.rand(p, p) - 0.5
                cov_mat = (A @ A.T + np.identity(p)) / p
            main_names_all = ["x", "x^c", "c^x", "sin(x)"] # did not add cos cuz EBM and GAMInet learn poorly; no log lest log(-1)
            int_names_all = ["x^cy^d", "c^xd^y", "sin(cx+dy)"] #+ ['None'] * int( 3*p -9 + random.randint(-1,1)) # did not add cos or xsiny or x2y cuz EBM and GAMInet learn poorly.
            main_funcs = np.random.choice(main_names_all, p)
            int_funcs = np.random.choice(int_names_all, (p, p)) # only the lower diagonal will be used.
            main_coefs = [None] * p
            int_coefs1 = [None] * p
            int_coefs2 = [None] * p
            main_times = np.random.rand(p)
            int_times = 1e2 * np.random.rand(p, p)
            for i in range(p):
                for j in range(i+1):
                    int_times[j, i] =0
            def ground_truth(x):
                res = 0.
                # for i in range(p):
                #     res += main_times[i] * function_main_effects(main_funcs[i], main_coefs[i])(x[i]) # dont need this cuz interactions arent pure anyway.
                for i in range(p):
                    for j in range(i):
                        if random.random() > 2/(p-1):
                            int_times[i, j] = 0
                        res += int_times[i, j] * function_interaction(int_funcs[i, j], int_coefs1[i], int_coefs2[j])(x[i], x[j])
                return res
            X = generate_x(dist, N * p, p, seed=seed, cov_mat=cov_mat) # generate a bit more.
            X = X[(X < bound).all(axis=1) & (X > - bound).all(axis=1), :]
            Xs.append(X)
            len_total = len_total + X.shape[0]
            if len_total == last_len_totol:
                assert False, "Cant generate data"
            else:
                last_len_totol = len_total
        X = np.concatenate(Xs)[:N]
    else:
        dist = "gaussian"
        A = np.random.rand(p, p) - 0.5
        cov_mat = (A @ A.T) / p

        main_names_all = ["x", "x^c", "c^x", "sin(x)"] # did not add cos cuz EBM and GAMInet learn poorly; no log lest log(-1)
        int_names_all = ["x^cy^d", "c^xd^y", "sin(cx+dy)"] #+ ['None'] * int( 3*p -9 + random.randint(-1,1)) # did not add cos or xsiny or x2y cuz EBM and GAMInet learn poorly.
        main_funcs = np.random.choice(main_names_all, p)
        int_funcs = np.random.choice(int_names_all, (p, p)) # only the lower diagonal will be used.
        main_coefs = [None] * p
        int_coefs1 = [None] * p
        int_coefs2 = [None] * p
        main_times = np.random.rand(p)
        int_times = np.random.rand(p, p)
        for i in range(p):
            for j in range(i+1):
                int_times[j, i] =0
        def ground_truth(x):
            res = 0.
            for i in range(p):
                res += main_times[i] * function_main_effects(main_funcs[i], main_coefs[i])(x[i])
            for i in range(p):
                for j in range(i):
                    if random.random() > 2/(p-1):
                        int_times[i, j] = 0
                    res += int_times[i, j] * function_interaction(int_funcs[i, j], int_coefs1[i], int_coefs2[j])(x[i], x[j])
            return res
        X = generate_x(dist, N, p, seed=seed, cov_mat=cov_mat)
    if return_X_only:
        return X
    y = ground_truth(X.T)
    if return_cov:
        return X, y, int_times, cov_mat,
    else:
        return X, y, int_times




def gen_cat_data(N, cards, probabs=None, seed=345891): # if probabs is none, randomly generate one.
    assert probabs is None, "Currently not support given probabs."
    p = len(cards)
    A = np.random.rand(p, p) - 0.5
    cov_mat = (A @ A.T + np.identity(p)) / p
    bound = 0.5
    X = num_gen_gauss(N, p, seed=seed, truncate=True, bound=bound, return_cov=False, return_X_only=True, new_cov=True, cov_for_cat=True)
    def bin_func(x, n, bound):
        cut_quans = np.linspace(0.2, 0.8, n + 1)
        cut_pts = norm.ppf(q=cut_quans[1: -1])
        return np.digitize(x, cut_pts)
    X = np.array([bin_func(X[:, i], cards[i], bound) for i in range(X.shape[1])]).T
    print(X.shape)
    # Generate full-grid data
    p = len(cards)
    n0 = 0
    for i in range(p):
        for j in range(i):
            data_ij = np.array(np.meshgrid(np.arange(cards[i]), np.arange(cards[j]))).T.reshape(-1, 2)
            n = data_ij.shape[0]
            try:
                for k in range(3):
                    X[n0:n0 + n, [i,j]] = data_ij
                    n0 = n0 + n
                    # print(n0)
            except ValueError:
                raise Exception("Not enough #records to satisfy full_grid!")
    print("{} data added to satisty full-grid.".format(n0 / X.shape[0]))
    # assert full grid
    for i in range(X.shape[1]):
        for j in range(i):
            x1, x2 = X[:, i], X[:, j]
            u1, u2 = np.unique(x1), np.unique(x2)
            raw_oh1 = (x1[:,None] == u1).astype(int)
            raw_oh2 = (x2[:,None] == u2).astype(int)
            cmbn_counts = raw_oh1.T @ raw_oh2
            assert (cmbn_counts > 0).all(), "Data not full-grid!\n{}".format(cmbn_counts)
    np.random.shuffle(X)
    return X


def gen_cat_y(X, scale_univ=1, scale_biv=1, seed=42, noise_scale=1):
    np.random.seed(seed)
    bivariate_ids = []
    p = X.shape[1]
    Int_Idx_df = pandas.DataFrame( [[0 for _ in range(p)] for _ in range(p)], columns= list(range(p)), index = list(range(p)))
    for pair_idx1, pair_idx2 in generate_pairwise_idxes(p):
        if random.random() <= 2/(p-1):
            bivariate_ids.append((pair_idx1, pair_idx2))
            Int_Idx_df[pair_idx1][pair_idx2] = 1
    print(bivariate_ids, X.shape)
    enc = Categorical_encoder(X.shape[1], bivariate_ids=bivariate_ids)
    X_enc = enc.fit_transform(X)
    print("X_enc shape", X_enc.shape)
    num_enc_univ = enc.univariate_idx_list[-1][1]
    num_enc_biv = enc.bivariate_idx_list[-1][1] - num_enc_univ
    coefs_univ = np.random.rand(num_enc_univ) * scale_univ
    coefs_biv = np.random.rand(num_enc_biv) * scale_biv
    # mask some interaction to avoid full interaction:


    coefs = np.concatenate((coefs_univ, coefs_biv))
    y_gen =  X_enc @ coefs
    y_gen += noise_scale * np.random.rand(y_gen.shape[0])
    return y_gen, Int_Idx_df