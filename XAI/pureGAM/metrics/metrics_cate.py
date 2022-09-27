# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from metrics.metrics import pureness_score_categorical_normalized, predict_int_GAMI, predict_vec_ebm, \
    predict_int_pureGAM_cat


def score_gami_cat(GAMInet_model, test_x, N_subset=None, save_folder=None):
    gami_int_list = GAMInet_model.interaction_list
    gami_int_pred = predict_int_GAMI(GAMInet_model, test_x)
    for i in range(len(gami_int_pred)):
        gami_int_pred[i] = (gami_int_pred[i])
    pure_score_gami = pd.DataFrame(index=gami_int_list, columns=["X1", "X2"])
    for i, pair in enumerate(gami_int_list):
        X1, X2 = th.from_numpy(test_x[:, pair[0]]).double().numpy(), th.from_numpy(test_x[:, pair[1]]).double().numpy()
        pred_int = gami_int_pred[i]
        score1, score2 = pureness_score_categorical_normalized(X1, X2, pred_int)
        pure_score_gami.iloc[i, 0] = score1
        pure_score_gami.iloc[i, 1] = score2
    pure_score_gami = pure_score_gami.loc[sorted(pure_score_gami.index)]
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_x).to_csv(os.path.join(save_folder, "test_x.csv"), index=None)
        pd.DataFrame(np.array([i.flatten() for i in gami_int_pred]).T, columns=gami_int_list).to_csv(os.path.join(save_folder, "int_preds.csv"), index=None)
        pure_score_gami.to_csv(os.path.join(save_folder, "pure_score.csv"))
    return pure_score_gami


def score_ebm_cat(ebm, test_x, N_subset=None, save_folder=None):
    ebm_int_list = []
    ebm_int_pred = []
    ebm_outputs = predict_vec_ebm(ebm, test_x)
    for i, gp in enumerate(ebm.feature_groups_):
        if len(gp) == 2:
            ebm_int_list.append((gp[0], gp[1]))
            ebm_int_pred.append(ebm_outputs[i])
    pure_score_ebm = pd.DataFrame(index=ebm_int_list, columns=["X1", "X2"])
    for i, pair in enumerate(ebm_int_list):
        X1, X2 = test_x[:, pair[0]], test_x[:, pair[1]]
        pred_int = (ebm_int_pred[i].reshape(-1, 1)).reshape(-1)
        score1, score2 = pureness_score_categorical_normalized(X1, X2, pred_int)
        pure_score_ebm.iloc[i, 0] = score1
        pure_score_ebm.iloc[i, 1] = score2
    pure_score_ebm = pure_score_ebm.loc[sorted(pure_score_ebm.index)]
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_x).to_csv(os.path.join(save_folder, "test_x.csv"), index=None)
        pd.DataFrame(np.array([i for i in ebm_int_pred]).T, columns=ebm_int_list).to_csv(os.path.join(save_folder, "int_preds.csv"), index=None)
        pure_score_ebm.to_csv(os.path.join(save_folder, "pure_score.csv"))
    return pure_score_ebm


def score_pureGAM_cat(pureGAM_model, test_x, N_subset=None, save_folder=None):
    pureGAM_int_list = [(pair[0], pair[1]) for pair in pureGAM_model.pairwise_idxes_cate]
    pureGAM_int_pred = predict_int_pureGAM_cat(pureGAM_model, th.from_numpy(test_x))
    pure_score_pureGAM = pd.DataFrame(index=pureGAM_int_list, columns=["X1", "X2"])
    for i, pair in enumerate(pureGAM_int_list):
        X1, X2 = test_x[:, pair[0]], test_x[:, pair[1]]
        pred_int = (pureGAM_int_pred[i].reshape(-1, 1).detach().cpu().numpy()).reshape(-1)
        score1, score2 = pureness_score_categorical_normalized(X1, X2, pred_int)
        pure_score_pureGAM.iloc[i, 0] = score1
        pure_score_pureGAM.iloc[i, 1] = score2
    pure_score_pureGAM = pure_score_pureGAM.loc[sorted(pure_score_pureGAM.index)]
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_x).to_csv(os.path.join(save_folder, "train_x.csv"), index=None)
        #print(pureGAM_int_pred)
        #print(pureGAM_int_list)
        #pd.DataFrame(np.array([i.detach().numpy() for i in pureGAM_int_pred]).T, columns=pureGAM_int_list).to_csv(os.path.join(save_folder, "int_preds.csv"), index=None)
        pure_score_pureGAM.to_csv(os.path.join(save_folder, "pure_score.csv"))
    return pure_score_pureGAM
