# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import os
import time
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import torch as th
import tensorflow as tf
from pureGAM_model.utils import safe_norm
from torch_utils.kernel_smoother import compute_pairwise_distance
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

epsilon = 1e-15

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
            with torch.no_grad():
                INT = int_pred[i].unsqueeze(dim=-1)
                denom = th.square(th.tensor(INT)).mean().item()
                pure_score_pureGAM1.iloc[i, j] = compute_proj(X_num[:, pair[0]].unsqueeze(-1), INT, h)/(denom + epsilon)
                torch.cuda.empty_cache()

                pure_score_pureGAM2.iloc[i, j] = compute_proj(X_num[:, pair[1]].unsqueeze(-1), INT, h) / (denom + epsilon)
                del denom
                torch.cuda.empty_cache()

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
    pure_score_pureGAM1.loc['avg_log'], pure_score_pureGAM2.loc['avg_log'] = np.log10(pure_score_pureGAM1.astype('float')).replace([-np.inf], np.nan).dropna().mean(0),\
                                                                             np.log10(pure_score_pureGAM2.astype('float')).replace([-np.inf], np.nan).dropna().mean(0)

    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pure_score_pureGAM1.to_csv(os.path.join(save_folder, "pure_score1.csv"))
        pure_score_pureGAM2.to_csv(os.path.join(save_folder, "pure_score2.csv"))
    return pure_score_pureGAM1, pure_score_pureGAM2

