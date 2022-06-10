import math
import os
import time
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import torch as th
import tensorflow as tf
from sgd_solver.utils import safe_norm
from torch_utils.kernel_smoother import compute_pairwise_distance
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

epsilon = 1e-15

'''def score_pureGAM(pureGAM_model, int_pred, X_num, h_map, bandwidths, epsilon=1e-13, save_folder=None):
    pureGAM_int_list = [(pair[0], pair[1]) for pair in pureGAM_model.pairwise_idxes_num]
    print(pureGAM_int_list)
    print(int_pred.shape, X_num.shape)

    X_num = th.tensor(X_num).to(device)'''

def compute_proj(X_num, INT, h):
    k = 1 / (math.sqrt(2 * math.pi) * h) * torch.exp(- compute_pairwise_distance(X_num, X_num, isMultiDim=False).squeeze(0) / (2 * h * h))
    # todo: safe_norm or  1// N_subset???
    res = th.mm(safe_norm(k), INT)  # todo / N_subset ???
    # todo: why square for nom and denom ???

    torch.cuda.empty_cache()
    return th.square(res).mean().detach().item()

def score_pureness(int_pred:th.tensor, X_num:th.tensor, interaction_list:th.tensor, h_map, bandwidths, save_folder=None):
    t1 = time.time()
    # scale_lam_list = [1 / 10, 1, 10]
    scale_lam_list = [1]
    scale_lam_list_names = ['lam' + str(scale_lam) for scale_lam in scale_lam_list]
    print('* Interaction_list', interaction_list)
    pure_score_pureGAM1, pure_score_pureGAM2 = pd.DataFrame(index=interaction_list,
                                                            columns=bandwidths.tolist() + scale_lam_list_names), \
                                               pd.DataFrame(index=interaction_list,
                                                            columns=bandwidths.tolist() + scale_lam_list_names)
    for j, h in enumerate(bandwidths):
        for i, pair in enumerate(interaction_list):
            # (n1, n2)
            #print('Hi', i)
            with torch.no_grad():
                INT = int_pred[i].unsqueeze(dim=-1)
                denom = th.square(th.tensor(INT)).mean().item()
                pure_score_pureGAM1.iloc[i, j] = compute_proj(X_num[:, pair[0]].unsqueeze(-1), INT, h)/(denom + epsilon)
                torch.cuda.empty_cache()

                pure_score_pureGAM2.iloc[i, j] = compute_proj(X_num[:, pair[1]].unsqueeze(-1), INT, h) / (denom + epsilon)
                del denom
                torch.cuda.empty_cache()
                # X1_dist, X2_dist, INT = X_dist_square[pair[0]].to(device), X_dist_square[pair[1]].to(device), int_pred[i].unsqueeze(dim=-1)
                # k1 = th.tensor(1 / (math.sqrt(2 * math.pi) * h) * torch.exp(- X_dist_square[pair[0]].to(device) / (2 * h * h)))
                # k2 = th.tensor(1 / (math.sqrt(2 * math.pi) * h) * torch.exp(- X_dist_square[pair[1]].to(device) / (2 * h * h)))

    if h_map is not None:
        for k, scale_lam in enumerate(scale_lam_list):
            for i, pair in enumerate(interaction_list):
                INT = int_pred[i].unsqueeze(dim=-1)
                denom = th.square(th.tensor(int_pred[i])).mean().item()
                h1, h2 = h_map[pair[0]], h_map[pair[1]]

                pure_score_pureGAM1.iloc[i, len(bandwidths) + k] = compute_proj(X_num[:, pair[0]].unsqueeze(-1), INT, h1)/(denom + epsilon)
                torch.cuda.empty_cache()

                pure_score_pureGAM2.iloc[i, len(bandwidths) + k] = compute_proj(X_num[:, pair[1]].unsqueeze(-1), INT, h2)/(denom + epsilon)
                del denom
                torch.cuda.empty_cache()
                # X1_dist, X2_dist, INT = X_dist_square[pair[0]].to(device), X_dist_square[pair[1]].to(device), int_pred[
                #    i].unsqueeze(dim=-1)

    t2 = time.time()
    print("compute pureness score :", t2 - t1)
    pure_score_pureGAM1, pure_score_pureGAM2 = pure_score_pureGAM1.loc[sorted(pure_score_pureGAM1.index)], \
                                               pure_score_pureGAM2.loc[sorted(pure_score_pureGAM2.index)]
    print(pure_score_pureGAM1)
    pure_score_pureGAM1.loc['avg_log'], pure_score_pureGAM2.loc['avg_log'] = np.log10(pure_score_pureGAM1.astype('float')).replace([-np.inf], np.nan).dropna().mean(0),\
                                                                             np.log10(pure_score_pureGAM2.astype('float')).replace([-np.inf], np.nan).dropna().mean(0)
    print(pure_score_pureGAM1)
    #print(pure_score_pureGAM2)

    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pure_score_pureGAM1.to_csv(os.path.join(save_folder, "pure_score1.csv"))
        pure_score_pureGAM2.to_csv(os.path.join(save_folder, "pure_score2.csv"))
    return pure_score_pureGAM1, pure_score_pureGAM2

"""def true_pureness_score_gaussian_pureGAM(int_pred:th.tensor, interaction_list:list, cov_mat, num_sigmas, N, sy, normalize=True, epsilon=1e-13, save_folder=None, device=None):
    pure_score = pd.DataFrame(index=interaction_list, columns=["X1", "X2"])
    for i, pair in enumerate(interaction_list):
        cov_submat = cov_mat[[pair[0], pair[1]], :][:, [pair[0], pair[1]]]
        '''if device:
            cov_submat = th.tensor(cov_submat).to(device)'''
        def func(x):
            return th.from_numpy((predict_one_int_pureGAM(model, x, i).reshape(-1, 1)).reshape(-1))
        right = (np.sqrt(cov_submat[0, 0]) + np.sqrt(cov_submat[1, 1])) * num_sigmas / 2
        left = - right
        score1, score2 = true_pureness_scores(model_func=func, cov=cov_submat, N=N, xl=left, xr=right, epsilon=epsilon)
        pure_score.iloc[i, 0] = score1
        pure_score.iloc[i, 1] = score2
    pure_score = pure_score.loc[sorted(pure_score.index)]
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pure_score.to_csv(os.path.join(save_folder, "pure_score_true.csv"))
    return pure_score

def true_pureness_scores(model_func, cov, N, xl, xr, epsilon=1e-13):
    s11 = cov[0, 0]
    s22 = cov[1, 1]
    s12 = cov[0, 1]
    det_sig = (s11*s22-s12**2)
    a1 = s22/det_sig
    a12 = - s12/det_sig
    a2 = s11/det_sig
    xs = th.linspace(xl, xr, N)
    ys = th.linspace(xl, xr, N)
    xxs, yys = th.meshgrid(xs, ys)
    xy = th.stack([xxs.T, yys.T]).T.reshape(-1, 2)
    def p(x, y):
        return 1/(2*np.pi*np.sqrt(det_sig)) * th.exp(-1/(2*det_sig) * (s22*x*x - 2*s12*x*y + s11*y*y))
    preds = model_func(xy)
    probabs = p(*xy.T)
    zx, zy = (preds * probabs).reshape(N, N).mean(axis=1) * (xr - xl), (preds * probabs).reshape(N, N).mean(axis=0) * (xr - xl)
    def p(x):
        return 1/(np.sqrt(2*np.pi*s11)) * th.exp(-1/(2*s11) * x*x)
    def p(y):
        return 1/(np.sqrt(2*np.pi*s22)) * th.exp(-1/(2*s22) * y*y)
    probabs_x = p(xs)
    nomi_x = (th.square(zx) * probabs_x).mean() * (xr - xl)
    probabs_y = p(ys)
    nomi_y = (th.square(zy) * probabs_y).mean() * (xr - xl)
    denom = (th.square(preds) * probabs).mean() * (xr - xl) * (xr - xl)
    return (nomi_x + epsilon) / (denom + epsilon), (nomi_y + epsilon) / (denom + epsilon)
"""
