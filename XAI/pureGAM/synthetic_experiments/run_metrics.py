import os
import time
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import torch as th
import tensorflow as tf
from sgd_solver.utils import safe_norm
from experiment_metrics.metrics import pureness_score2_normalized, pureness_score_categorical_normalized, pureness_score2_normalized_single
from experiment_metrics.metrics import empirical_estimation_y, empirical_estimation_x, true_pureness_scores,pureness_loss_est2, myKW
from interpret.utils import unify_data
from interpret.glassbox.ebm.utils import EBMUtils
from torch_utils.kernel_smoother import compute_pairwise_distance
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

"""
Get the interaction array predicted by model.
"""

"""GAMI-Net"""
def predict_int_GAMI(model, inputs):
    res = []
    model.interact_outputs = model.interact_blocks(inputs, sample_weight=None, training=False)
    for i, (k1, k2) in enumerate(model.interaction_list):
        interaction_weights = tf.multiply(model.output_layer.interaction_switcher, model.output_layer.interaction_weights)
        b = tf.multiply(tf.gather(model.interact_outputs, [i], axis=1), tf.gather(interaction_weights, [i], axis=0))
        res.append(b.numpy())
    return res

def predict_one_int_gami(model, x, pair_id):
    int_list = model.interaction_list
    max_features = len(model.feature_list_)
    idx, idy = int_list[pair_id][0], int_list[pair_id][1]
    test_x = np.zeros((x.shape[0], max_features))
    test_x[:, [idx, idy]] = x
    pred = predict_int_GAMI(model, test_x)[pair_id].reshape(-1)
    # pred_vec = predict_vec_ebm(model, test_x)
    # ebm_int_pred = []
    # for i, gp in enumerate(ebm.feature_groups_):
    #     if len(gp) == 2:
    #         ebm_int_pred.append(pred_vec[i])
    # pred = ebm_int_pred[pair_id]
    # pred = th.from_numpy(pred_vec[max_features + pair_id])
    return pred

"""EBM"""
def predict_vec_ebm(model, X):
    """ Predicts on provided samples.

    Args:
        X: Numpy array for samples.

    Returns:
        Predicted class label per sample.
    """
    X_orig, _, _, _ = unify_data(X, None, model.feature_names, model.feature_types)
    X = model.preprocessor_.transform(X_orig)
    X = np.ascontiguousarray(X.T)

    if model.interactions != 0:
        X_pair = model.pair_preprocessor_.transform(X_orig)
        X_pair = np.ascontiguousarray(X_pair.T)
    else:
        X_pair = None
    scores_gen = EBMUtils.scores_by_feature_group(
            X, X_pair, model.feature_groups_, model.additive_terms_
        )
    scores_vec = []
    for _, _, scores in scores_gen:
        scores_vec.append(scores)
    return scores_vec

def predict_one_int_ebm(model, x, pair_id):
    int_list = []
    max_features = 0
    for i, gp in enumerate(model.feature_groups_):
        if len(gp) == 1:
            max_features += 1
        if len(gp) == 2:
            int_list.append((gp[0], gp[1]))
    idx, idy = int_list[pair_id][0], int_list[pair_id][1]
    test_x = np.zeros((x.shape[0], max_features))
    test_x[:, [idx, idy]] = x
    pred_vec = predict_vec_ebm(model, test_x)
    # ebm_int_pred = []
    # for i, gp in enumerate(ebm.feature_groups_):
    #     if len(gp) == 2:
    #         ebm_int_pred.append(pred_vec[i])
    # pred = ebm_int_pred[pair_id]
    pred = pred_vec[max_features + pair_id]
    return pred

"""PureGAM"""
def predict_vec_pureGAM(pureGAM_model, X):
    batch_N_X = X
    _, batch_S_biv_tmp, _ = pureGAM_model.num_enc.cpu().transform(batch_N_X, is_norm=False)
    # todo: conditional pureness is on what axis and what points?? 3choices: on test points\sampled 1-d points\sampled 2-points using 1-d distance
    # norm
    batch_S_biv = safe_norm(batch_S_biv_tmp)
    return th.einsum("ijk,ikt->ijt", batch_S_biv, pureGAM_model.model_smoothing.eta_biv.cpu()).squeeze(axis=-1)

def predict_one_int_pureGAM(pureGAM_model, X_sub, pair_id): # X_sub contains just the two columns
    i = pair_id
    id_biv = pureGAM_model.num_enc.bivariate_ids_[i]
    X_diff_divlam = compute_pairwise_distance((pureGAM_model.num_enc.X_memo_biv[:, [id_biv[0], id_biv[1]]]).cpu(), X_sub.cpu()) / (th.square(pureGAM_model.num_enc.get_lam()[1][i]) + pureGAM_model.num_enc.eps).cpu()
    kij = pureGAM_model.num_enc.K_transform_func(X_diff_divlam)
    kij = safe_norm(kij)
    eta_biv = pureGAM_model.model_smoothing.get_eta()[1]
    if len(eta_biv.squeeze().shape) == 1:
        return th.matmul(kij, eta_biv.reshape(-1, 1).cpu()).detach().numpy()
    else:
        return th.matmul(kij, eta_biv.squeeze().T[:, i].cpu()).detach().numpy()

def predict_int_pureGAM_cat(pureGAM_model, X):
    batch_C_X_encoded = th.tensor(pureGAM_model.cate_enc.transform(X)).to(pureGAM_model.device)

    univ_cardi = pureGAM_model.cate_enc.cardinality_univ
    #print(univ_cardi)
    eta_biv = pureGAM_model.model_categorical.get_eta()[1]
    contri_mat_biv = th.mul(batch_C_X_encoded[:,univ_cardi:], eta_biv.unsqueeze(dim=0)[0]) #(m, p)
    idx_list = pureGAM_model.cate_enc.bivariate_idx_list
    res = []
    for ids in idx_list:
        yh = contri_mat_biv[:, ids[0]-univ_cardi:ids[1]-univ_cardi].sum(axis=1).detach().cpu()
        res.append(yh)
    return res

"""
Calculate the scores.
"""
def score_gami2(GAMInet_model, test_x_tr, h_map, bandwidths, epsilon=1e-13, N_subset=None, save_folder=None):
    test_x = test_x_tr
    gami_int_list = GAMInet_model.interaction_list
    gami_int_pred = predict_int_GAMI(GAMInet_model, test_x)
    for i in range(len(gami_int_pred)):
        gami_int_pred[i] = (gami_int_pred[i])
    #scale_lam_list = [1/10, 1, 10]
    scale_lam_list = [1]
    scale_lam_list_names = ['lam'+str(scale_lam) for scale_lam in scale_lam_list]
    pure_score_gami1, pure_score_gami2 = pd.DataFrame(index=gami_int_list, columns=bandwidths.tolist() + scale_lam_list_names), pd.DataFrame(index=gami_int_list, columns=bandwidths.tolist() + scale_lam_list_names)
    for i, pair in enumerate(gami_int_list):
        # X1, X2 = th.from_numpy(test_x[:, pair[0]]).double(), th.from_numpy(test_x[:, pair[1]]).double()

        t1 = time.time()
        X1, X2 = th.from_numpy(test_x[:, pair[0]]), th.from_numpy(test_x[:, pair[1]])
        # pred_int = th.from_numpy(gami_int_pred[i].numpy()).double()
        pred_int = th.from_numpy((gami_int_pred[i].reshape(-1,1)).reshape(-1).numpy()).double()
        for j, h in enumerate(bandwidths):
            pure_score_gami1.iloc[i, j], pure_score_gami2.iloc[i, j] = pureness_score2_normalized(X1, X2, pred_int, h, epsilon=epsilon, N_subset=N_subset)
        if h_map is not None:
            for k, scale_lam in enumerate(scale_lam_list):
                pure_score_gami1.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X1, pred_int, scale_lam*h_map[pair[0]], epsilon=epsilon, N_subset=N_subset)
                pure_score_gami2.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X2, pred_int, scale_lam*h_map[pair[1]], epsilon=epsilon, N_subset=N_subset)

        t2 = time.time()
        print(t2-t1)
    pure_score_gami1, pure_score_gami2 = pure_score_gami1.loc[sorted(pure_score_gami1.index)], pure_score_gami2.loc[sorted(pure_score_gami2.index)]
    pure_score_gami1.loc['avg_log'], pure_score_gami2.loc['avg_log'] = np.log10(
        pure_score_gami1.astype('float')).mean(0), np.log10(pure_score_gami2.astype('float')).mean(0)

    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_x).to_csv(os.path.join(save_folder, "test_x.csv"), index=None)
        pd.DataFrame(np.array([i.numpy().flatten() for i in gami_int_pred]).T, columns=gami_int_list).to_csv(os.path.join(save_folder, "int_preds.csv"), index=None)
        pure_score_gami1.to_csv(os.path.join(save_folder, "pure_score1.csv"))
        pure_score_gami2.to_csv(os.path.join(save_folder, "pure_score2.csv"))
    return pure_score_gami1, pure_score_gami2

def score_gami(GAMInet_model, test_x_tr, sx, sy, h_map, bandwidths, epsilon=1e-13, N_subset=None, save_folder=None):
    test_x = sx.inverse_transform(test_x_tr)
    gami_int_list = GAMInet_model.interaction_list
    gami_int_pred = predict_int_GAMI(GAMInet_model, test_x)
    for i in range(len(gami_int_pred)):
        gami_int_pred[i] = (gami_int_pred[i])
    scale_lam_list = [1/10, 1, 10]
    scale_lam_list_names = ['lam'+str(scale_lam) for scale_lam in scale_lam_list]
    pure_score_gami1, pure_score_gami2 = pd.DataFrame(index=gami_int_list, columns=bandwidths.tolist() + scale_lam_list_names), pd.DataFrame(index=gami_int_list, columns=bandwidths.tolist() + scale_lam_list_names)
    for i, pair in enumerate(gami_int_list):
        # X1, X2 = th.from_numpy(test_x[:, pair[0]]).double(), th.from_numpy(test_x[:, pair[1]]).double()
        X1, X2 = th.from_numpy(test_x[:, pair[0]]), th.from_numpy(test_x[:, pair[1]])
        # pred_int = th.from_numpy(gami_int_pred[i].numpy()).double()
        pred_int = th.from_numpy((gami_int_pred[i].reshape(-1,1)).reshape(-1)).double()
        for j, h in enumerate(bandwidths):
            pure_score_gami1.iloc[i, j], pure_score_gami2.iloc[i, j] = pureness_score2_normalized(X1, X2, pred_int, h, epsilon=epsilon, N_subset=N_subset)
        if h_map is not None:
            for k, scale_lam in enumerate(scale_lam_list):
                pure_score_gami1.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X1, pred_int, scale_lam*h_map[pair[0]], epsilon=epsilon, N_subset=N_subset)
                pure_score_gami2.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X2, pred_int, scale_lam*h_map[pair[1]], epsilon=epsilon, N_subset=N_subset)
        '''# todo: add predict using GAMINET
        est_xtrue = true_pureness_est_gaussian_x_gami(GAMInet_model, 0, 0, 1, cov_submat, -0.75, 1, 1000, 1000,
                                                 save_folder=None)
        est_ytrue = ??
        denom = th.square(pred_int).mean().numpy()
        score_xtrue = th.square(est_xtrue)/(denom + eps)
        score_ytrue = th.square(est_ytrue) / (denom + eps)
        # todo: nomalize? denom?
        pure_score_gami1.iloc[i, len(bandwidths)+1], pure_score_gami2.iloc[i, len(bandwidths)+1] = score_xtrue, score_ytrue'''

    pure_score_gami1, pure_score_gami2 = pure_score_gami1.loc[sorted(pure_score_gami1.index)], pure_score_gami2.loc[sorted(pure_score_gami2.index)]
    pure_score_gami1.loc['avg'], pure_score_gami2.loc['avg'] = pure_score_gami1.mean(0), pure_score_gami2.mean(0)

    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_x).to_csv(os.path.join(save_folder, "test_x.csv"), index=None)
        pd.DataFrame(np.array([i.numpy().flatten() for i in gami_int_pred]).T, columns=gami_int_list).to_csv(os.path.join(save_folder, "int_preds.csv"), index=None)
        pure_score_gami1.to_csv(os.path.join(save_folder, "pure_score1.csv"))
        pure_score_gami2.to_csv(os.path.join(save_folder, "pure_score2.csv"))
    return pure_score_gami1, pure_score_gami2

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

def score_ebm(ebm, test_x_tr, sx, sy, h_map, bandwidths, epsilon=1e-13, N_subset=None, save_folder=None):
    test_x = sx.inverse_transform(test_x_tr)
    ebm_int_list = []
    ebm_int_pred = []
    ebm_outputs = predict_vec_ebm(ebm, test_x)
    for i, gp in enumerate(ebm.feature_groups_):
        if len(gp) == 2:
            ebm_int_list.append((gp[0], gp[1]))
            ebm_int_pred.append(ebm_outputs[i])

    scale_lam_list = [1/2, 3/4, 7/8, 1, 8/7, 4/3, 2/1]
    scale_lam_list_names = ['lam'+str(scale_lam) for scale_lam in scale_lam_list]
    pure_score_ebm1, pure_score_ebm2 = pd.DataFrame(index=ebm_int_list, columns=bandwidths.tolist() + scale_lam_list_names), pd.DataFrame(index=ebm_int_list, columns=bandwidths.tolist() + scale_lam_list_names)
    for i, pair in enumerate(ebm_int_list):
        X1, X2 = th.from_numpy(test_x[:, pair[0]]).double(), th.from_numpy(test_x[:, pair[1]]).double()
        pred_int = th.from_numpy((ebm_int_pred[i].reshape(-1, 1)).reshape(-1)).double()
        for j, h in enumerate(bandwidths):
            pure_score_ebm1.iloc[i, j], pure_score_ebm2.iloc[i, j] = pureness_score2_normalized(X1, X2, pred_int, h, epsilon=epsilon, N_subset=N_subset)

        if h_map is not None:
            for k, scale_lam in enumerate(scale_lam_list):
                pure_score_ebm1.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X1, pred_int, scale_lam*h_map[pair[0]], epsilon=epsilon, N_subset=N_subset)
                pure_score_ebm2.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X2, pred_int, scale_lam*h_map[pair[1]], epsilon=epsilon, N_subset=N_subset)

        '''# todo:
        pure_score_gami1.iloc[i, len(bandwidths)], pure_score_gami2.iloc[
            i, len(bandwidths)] = pureness_score2_normalized(X1, X2, pred_int, pair_h_map[(pair[0], pair[1])],
                                                             epsilon=epsilon, N_subset=N_subset)

        # todo: add predict using GAMINET
        est_xtrue = true_pureness_est_gaussian_x_gami(GAMInet_model, 0, 0, 1, cov_submat, -0.75, 1, 1000, 1000,
                                                      save_folder=None)
        est_ytrue = ??
        denom = th.square(pred_int).mean().numpy()
        score_xtrue = th.square(est_xtrue) / (denom + eps)
        score_ytrue = th.square(est_ytrue) / (denom + eps)
        # todo: nomalize? denom?
        pure_score_gami1.iloc[i, len(bandwidths) + 1], pure_score_gami2.iloc[i, len(bandwidths) + 1] = score_xtrue, score_ytrue'''

    pure_score_ebm1, pure_score_ebm2 = pure_score_ebm1.loc[sorted(pure_score_ebm1.index)], pure_score_ebm2.loc[sorted(pure_score_ebm2.index)]
    pure_score_ebm1.loc['avg'], pure_score_ebm2.loc['avg'] = pure_score_ebm1.mean(0), pure_score_ebm2.mean(0)

    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_x).to_csv(os.path.join(save_folder, "test_x.csv"), index=None)
        pd.DataFrame(np.array([i for i in ebm_int_pred]).T, columns=ebm_int_list).to_csv(os.path.join(save_folder, "int_preds.csv"), index=None)
        pure_score_ebm1.to_csv(os.path.join(save_folder, "pure_score1.csv"))
        pure_score_ebm2.to_csv(os.path.join(save_folder, "pure_score2.csv"))
    return pure_score_ebm1, pure_score_ebm2


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
import math
def score_pureGAM3(pureGAM_model, int_pred, X_num, h_map, bandwidths, epsilon=1e-13, save_folder=None):
    pureGAM_int_list = [(pair[0], pair[1]) for pair in pureGAM_model.pairwise_idxes_num]
    print(pureGAM_int_list)
    print(int_pred.shape, X_num.shape)

    #scale_lam_list = [1 / 10, 1, 10]
    scale_lam_list = [1]
    scale_lam_list_names = ['lam' + str(scale_lam) for scale_lam in scale_lam_list]
    pure_score_pureGAM1, pure_score_pureGAM2 = pd.DataFrame(index=pureGAM_int_list,columns=bandwidths.tolist() + scale_lam_list_names),\
                                               pd.DataFrame(index=pureGAM_int_list, columns=bandwidths.tolist() + scale_lam_list_names)
    t1 = time.time()
    device = th.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_num = th.tensor(X_num).to(device)

    for j, h in enumerate(bandwidths):
        for i, pair in enumerate(pureGAM_int_list):
            #(n1, n2)
            X1 = X_num[:, pair[0]].unsqueeze(-1)
            X2 = X_num[:, pair[1]].unsqueeze(-1)
            X1_dist, X2_dist, INT = compute_pairwise_distance(X1, X1), compute_pairwise_distance(X2, X2), int_pred[i].unsqueeze(dim=-1)
            #X1_dist, X2_dist, INT = X_dist_square[pair[0]].to(device), X_dist_square[pair[1]].to(device), int_pred[i].unsqueeze(dim=-1)
            #k1 = th.tensor(1 / (math.sqrt(2 * math.pi) * h) * torch.exp(- X_dist_square[pair[0]].to(device) / (2 * h * h)))
            #k2 = th.tensor(1 / (math.sqrt(2 * math.pi) * h) * torch.exp(- X_dist_square[pair[1]].to(device) / (2 * h * h)))

            k1 = 1 / (math.sqrt(2 * math.pi) * h) * torch.exp(- X1_dist / (2 * h * h))
            k2 = 1 / (math.sqrt(2 * math.pi) * h) * torch.exp(- X2_dist / (2 * h * h))

            #todo: safe_norm or  1// N_subset???
            k1_norm = safe_norm(k1)
            k2_norm = safe_norm(k2)

            # (n1, n2) * (n2, 1)
            res1 = th.mm(k1_norm, INT) # todo / N_subset ???
            res2 = th.mm(k2_norm, INT)

            # todo: why square for nom and denom ???
            denom = th.square(th.tensor(int_pred[i])).mean().item()

            pure_score_pureGAM1.iloc[i, j] = ((th.square(res1).mean() + epsilon)).item() / (denom + epsilon)
            pure_score_pureGAM2.iloc[i, j] = ((th.square(res2).mean() + epsilon)).item() / (denom + epsilon)
    if h_map is not None:
        for k, scale_lam in enumerate(scale_lam_list):
            for i, pair in enumerate(pureGAM_int_list):
                X1 = X_num[:, pair[0]].unsqueeze(-1)
                X2 = X_num[:, pair[1]].unsqueeze(-1)
                X1_dist, X2_dist, INT = compute_pairwise_distance(X1, X1), compute_pairwise_distance(X2, X2), int_pred[
                    i].unsqueeze(dim=-1)
                #X1_dist, X2_dist, INT = X_dist_square[pair[0]].to(device), X_dist_square[pair[1]].to(device), int_pred[
                #    i].unsqueeze(dim=-1)
                h1, h2 = h_map[pair[0]], h_map[pair[1]]

                k1 = 1 / (math.sqrt(2 * math.pi) * h1) * torch.exp(- X1_dist / (2 * h1 * h1))
                k2 = 1 / (math.sqrt(2 * math.pi) * h2) * torch.exp(- X2_dist / (2 * h2 * h2))

                # todo: safe_norm or  1// N_subset???
                k1_norm = safe_norm(k1)
                k2_norm = safe_norm(k2)

                res1 = th.mm(k1_norm, INT)  # todo / N_subset ???
                res2 = th.mm(k2_norm, INT)

                # todo: why square for nom and denom ???
                denom = th.square(th.tensor(int_pred[i])).mean().item()

                pure_score_pureGAM1.iloc[i, len(bandwidths) + k] = ((th.square(res1).mean() + epsilon)).item() / (denom + epsilon)
                pure_score_pureGAM2.iloc[i, len(bandwidths) + k] = ((th.square(res2).mean() + epsilon)).item() / (denom + epsilon)
    t2 = time.time()
    print("compute pureness score :", t2 - t1)
    pure_score_pureGAM1, pure_score_pureGAM2 = pure_score_pureGAM1.loc[sorted(pure_score_pureGAM1.index)], \
                                               pure_score_pureGAM2.loc[sorted(pure_score_pureGAM2.index)]
    print(pure_score_pureGAM1)
    pure_score_pureGAM1.loc['avg_log'], pure_score_pureGAM2.loc['avg_log'] = np.log10(
        pure_score_pureGAM1.astype('float')).mean(0), np.log10(pure_score_pureGAM2.astype('float')).mean(0)
    print(pure_score_pureGAM2)

    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pure_score_pureGAM1.to_csv(os.path.join(save_folder, "pure_score1.csv"))
        pure_score_pureGAM2.to_csv(os.path.join(save_folder, "pure_score2.csv"))
    return pure_score_pureGAM1, pure_score_pureGAM2


def score_pureGAM2(pureGAM_model, test_x_tr, h_map, bandwidths, epsilon=1e-13, N_subset=None, save_folder=None):
    test_x = test_x_tr
    pureGAM_int_list = [(pair[0], pair[1]) for pair in pureGAM_model.pairwise_idxes_num]
    pureGAM_outputs = predict_vec_pureGAM(pureGAM_model, th.from_numpy(test_x))
    pureGAM_int_pred = [pureGAM_outputs[i, :].detach().numpy() for i in range(len(pureGAM_int_list))]

    scale_lam_list = [1/10, 1, 10]
    scale_lam_list_names = ['lam'+str(scale_lam) for scale_lam in scale_lam_list]
    pure_score_pureGAM1, pure_score_pureGAM2 = pd.DataFrame(index=pureGAM_int_list, columns=bandwidths.tolist() + scale_lam_list_names), pd.DataFrame(index=pureGAM_int_list, columns=bandwidths.tolist() + scale_lam_list_names)

    for i, pair in enumerate(pureGAM_int_list):
        X1, X2 = th.from_numpy(test_x[:, pair[0]]).double(), th.from_numpy(test_x[:, pair[1]]).double()
        pred_int = th.from_numpy((pureGAM_int_pred[i].reshape(-1, 1)).reshape(-1)).double()
        t1 = time.time()
        for j, h in enumerate(bandwidths):
            pure_score_pureGAM1.iloc[i, j], pure_score_pureGAM2.iloc[i, j] = pureness_score2_normalized(X1, X2, pred_int, h, epsilon=epsilon, N_subset=N_subset)
        if h_map is not None:
            for k, scale_lam in enumerate(scale_lam_list):
                pure_score_pureGAM1.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X1, pred_int, scale_lam*h_map[pair[0]], epsilon=epsilon, N_subset=N_subset)
                pure_score_pureGAM2.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X2, pred_int, scale_lam*h_map[pair[1]], epsilon=epsilon, N_subset=N_subset)
        t2 = time.time()
        print('time2', t2-t1)
    pure_score_pureGAM1, pure_score_pureGAM2 = pure_score_pureGAM1.loc[sorted(pure_score_pureGAM1.index)], pure_score_pureGAM2.loc[sorted(pure_score_pureGAM2.index)]
    print(pure_score_pureGAM1)
    pure_score_pureGAM1.loc['avg_log'], pure_score_pureGAM2.loc['avg_log'] = np.log10(pure_score_pureGAM1.astype('float')).mean(0), np.log10(pure_score_pureGAM2.astype('float')).mean(0)
    print(pure_score_pureGAM2)
    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_x).to_csv(os.path.join(save_folder, "test_x.csv"), index=None)
        pd.DataFrame(pureGAM_outputs.detach().numpy().T, columns=pureGAM_int_list).to_csv(os.path.join(save_folder, "int_preds.csv"), index=None)
        pure_score_pureGAM1.to_csv(os.path.join(save_folder, "pure_score1.csv"))
        pure_score_pureGAM2.to_csv(os.path.join(save_folder, "pure_score2.csv"))
    return pure_score_pureGAM1, pure_score_pureGAM2

def score_pureGAM(pureGAM_model, test_x_tr, sx, sy, h_map, bandwidths, epsilon=1e-13, N_subset=None, save_folder=None):
    test_x = sx.inverse_transform(test_x_tr)
    pureGAM_int_list = [(pair[0], pair[1]) for pair in pureGAM_model.pairwise_idxes_num]
    pureGAM_outputs = predict_vec_pureGAM(pureGAM_model, th.from_numpy(test_x))
    pureGAM_int_pred = [pureGAM_outputs[i, :].detach().numpy() for i in range(len(pureGAM_int_list))]

    scale_lam_list = [1/2, 3/4, 7/8, 1, 8/7, 4/3, 2/1]
    scale_lam_list_names = ['lam'+str(scale_lam) for scale_lam in scale_lam_list]
    pure_score_pureGAM1, pure_score_pureGAM2 = pd.DataFrame(index=pureGAM_int_list, columns=bandwidths.tolist() + scale_lam_list_names), pd.DataFrame(index=pureGAM_int_list, columns=bandwidths.tolist() + scale_lam_list_names)
    for i, pair in enumerate(pureGAM_int_list):
        X1, X2 = th.from_numpy(test_x[:, pair[0]]).double(), th.from_numpy(test_x[:, pair[1]]).double()
        pred_int = th.from_numpy((pureGAM_int_pred[i].reshape(-1, 1)).reshape(-1)).double()
        for j, h in enumerate(bandwidths):
            t1 = time.time()
            pure_score_pureGAM1.iloc[i, j], pure_score_pureGAM2.iloc[i, j] = pureness_score2_normalized(X1, X2, pred_int, h, epsilon=epsilon, N_subset=N_subset)
            t2 = time.time()
            print(t2-t1)
        if h_map is not None:
            for k, scale_lam in enumerate(scale_lam_list):
                pure_score_pureGAM1.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X1, pred_int, scale_lam*h_map[pair[0]], epsilon=epsilon, N_subset=N_subset)
                pure_score_pureGAM2.iloc[i, len(bandwidths)+k] = pureness_score2_normalized_single(X2, pred_int, scale_lam*h_map[pair[1]], epsilon=epsilon, N_subset=N_subset)

        '''# todo: add predict using GAMINET
        est_xtrue = true_pureness_est_gaussian_x_gami(GAMInet_model, 0, 0, 1, cov_submat, -0.75, 1, 1000, 1000,
                                                      save_folder=None)
        est_ytrue = ??
        denom = th.square(pred_int).mean().numpy()
        score_xtrue = th.square(est_xtrue) / (denom + eps)
        score_ytrue = th.square(est_ytrue) / (denom + eps)
        # todo: nomalize? denom?
        pure_score_pureGAM1.iloc[i, len(bandwidths) + 1], pure_score_pureGAM2.iloc[i, len(bandwidths) + 1] = score_xtrue, score_ytrue'''

    pure_score_pureGAM1, pure_score_pureGAM2 = pure_score_pureGAM1.loc[sorted(pure_score_pureGAM1.index)], pure_score_pureGAM2.loc[sorted(pure_score_pureGAM2.index)]
    pure_score_pureGAM1.loc['avg'], pure_score_pureGAM2.loc['avg'] = pure_score_pureGAM1.mean(0), pure_score_pureGAM2.mean(0)

    if save_folder is not None:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(test_x).to_csv(os.path.join(save_folder, "test_x.csv"), index=None)
        pd.DataFrame(pureGAM_outputs.detach().numpy().T, columns=pureGAM_int_list).to_csv(os.path.join(save_folder, "int_preds.csv"), index=None)
        pure_score_pureGAM1.to_csv(os.path.join(save_folder, "pure_score1.csv"))
        pure_score_pureGAM2.to_csv(os.path.join(save_folder, "pure_score2.csv"))
    return pure_score_pureGAM1, pure_score_pureGAM2

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

def true_pureness_est_gaussian_y_pureGAM(pureGAM_model, pair_id, cov_submat, yleft, yright, N_ys, N_sample):
    def func(x):
        return predict_one_int_pureGAM(pureGAM_model, th.from_numpy(x), pair_id)
    ys = np.linspace(yleft, yright, num=N_ys)
    zs = empirical_estimation_y(ys, func, cov_submat, N_sample)
    return ys, zs

def true_pureness_est_gaussian_x_pureGAM(pureGAM_model, pair_id, cov_submat, xleft, xright, N_xs, N_sample):
    def func(x):
        return predict_one_int_pureGAM(pureGAM_model, th.from_numpy(x), pair_id)
    xs = np.linspace(xleft, xright, num=N_xs)
    zs = empirical_estimation_x(xs, func, cov_submat, N_sample)
    return xs, zs

def true_pureness_score_gaussian_pureGAM(model, cov_mat, num_sigmas, N, normalize=True, epsilon=1e-13, save_folder=None, device=None):
    int_list = [(pair[0], pair[1]) for pair in model.pairwise_idxes_num]
    pure_score = pd.DataFrame(index=int_list, columns=["X1", "X2"])
    for i, pair in enumerate(int_list):
        cov_submat = cov_mat[[pair[0], pair[1]], :][:, [pair[0], pair[1]]]
        '''if device:
            cov_submat = th.tensor(cov_submat).to(device)'''
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