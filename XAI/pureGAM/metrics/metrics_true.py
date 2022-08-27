import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from metrics.metrics import true_pureness_scores, predict_one_int_ebm, predict_one_int_gami, predict_one_int_pureGAM


def true_pureness_score_gaussian_pureGAM(model, cov_mat, num_sigmas, N, normalize=True, epsilon=1e-13, save_folder=None, device=None):
    int_list = [(pair[0], pair[1]) for pair in model.pairwise_idxes_num]
    pure_score = pd.DataFrame(index=int_list, columns=["X1", "X2"])
    for i, pair in enumerate(int_list):
        cov_submat = cov_mat[[pair[0], pair[1]], :][:, [pair[0], pair[1]]]
        def func(x):
            return th.from_numpy((predict_one_int_pureGAM(model, x, i).reshape(-1, 1)).reshape(-1))
        right = (np.sqrt(cov_submat[0, 0]) + np.sqrt(cov_submat[1, 1])) * num_sigmas / 2
        left = - right
        score1, score2 = true_pureness_scores(model_func=func, cov=cov_submat, N=N, xl=left, xr=right, epsilon=epsilon)
        pure_score.iloc[i, 0] = score1.item()
        pure_score.iloc[i, 1] = score2.item()
    pure_score = pure_score.loc[sorted(pure_score.index)]
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pure_score.to_csv(os.path.join(save_folder, "pure_score_true.csv"))
    return pure_score


def true_pureness_score_gaussian_ebm(model, cov_mat, num_sigmas, N, normalize=True, epsilon=1e-13, save_folder=None):
    int_list = []
    for i, gp in enumerate(model.feature_groups_):
        if len(gp) == 2:
            int_list.append((gp[0], gp[1]))
    pure_score = pd.DataFrame(index=int_list, columns=["X1", "X2"])
    for i, pair in enumerate(int_list):
        cov_submat = cov_mat[[pair[0], pair[1]], :][:, [pair[0], pair[1]]]
        def func(x):
            return th.from_numpy((predict_one_int_ebm(model, x, i).reshape(-1, 1)).reshape(-1)) # return a tensor.
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


def true_pureness_score_gaussian_gami(model, cov_mat, num_sigmas, N, normalize=True, epsilon=1e-13, save_folder=None):
    int_list = model.interaction_list
    pure_score = pd.DataFrame(index=int_list, columns=["X1", "X2"])
    for i, pair in enumerate(int_list):
        cov_submat = cov_mat[[pair[0], pair[1]], :][:, [pair[0], pair[1]]]
        def func(x):
            return th.from_numpy((predict_one_int_gami(model, x, i).reshape(-1, 1)).reshape(-1)) # return a tensor.
        right = (np.sqrt(cov_submat[0, 0]) + np.sqrt(cov_submat[1, 1])) * num_sigmas / 2
        left = - right
        score1, score2 = true_pureness_scores(model_func=func, cov=cov_submat, N=N, xl=left, xr=right, epsilon=epsilon)
        pure_score.iloc[i, 0] = score1.item()
        pure_score.iloc[i, 1] = score2.item()
    pure_score = pure_score.loc[sorted(pure_score.index)]
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pure_score.to_csv(os.path.join(save_folder, "pure_score_true.csv"))
    return pure_score