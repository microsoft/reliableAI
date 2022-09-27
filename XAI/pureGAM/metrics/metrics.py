# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import tensorflow as tf
import torch as th
from interpret.glassbox.ebm.utils import EBMUtils
from interpret.utils import unify_data

from pureGAM_model.utils import safe_norm
from torch_utils.kernel_smoother import compute_pairwise_distance
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

def rmse(label, pred, scaler):
    pred = scaler.inverse_transform(pred.reshape([-1, 1]))
    label = scaler.inverse_transform(label.reshape([-1, 1]))
    return np.sqrt(np.mean((pred - label)**2))

def metric_wrapper(metric, scaler):
    def wrapper(label, pred):
        return metric(label, pred, scaler=scaler)
    return wrapper

def pureness_score_categorical_normalized(X1, X2, pred_int, N_subset=None):
    assert len(X1) == len(X2) and len(X1) == len(pred_int), "Not same length!"
    if N_subset is None:
        N_subset = len(X1)
    indices = np.random.choice(len(X1), size=N_subset, replace=False)
    X1_, X2_, Int_ = X1[indices].reshape(-1), X2[indices].reshape(-1), pred_int[indices].reshape(-1)
    denom = np.square(Int_).sum() # not divided by N here, do it in the end.
    X1u, X2u = np.unique(X1_), np.unique(X2_)
    id1 = X1_ == X1u.reshape(-1, 1)
    id2 = X2_ == X2u.reshape(-1, 1)
    sum1 = id1 @ Int_
    sum2 = id2 @ Int_
    n1 = id1.sum(axis=1)
    n2 = id2.sum(axis=1)
    return (np.square(sum1) * n1).sum() / (N_subset * N_subset * denom), (np.square(sum2) * n2).sum() / (N_subset * N_subset * denom)


def true_pureness_scores(model_func, cov, N, xl, xr, epsilon=1e-13):
    s11 = cov[0, 0]
    s22 = cov[1, 1]
    s12 = cov[0, 1]
    det_sig = (s11*s22-s12**2)
    xs = th.linspace(xl, xr, N)
    ys = th.linspace(xl, xr, N)
    xxs, yys = th.meshgrid(xs, ys)
    xy = th.stack([xxs.T, yys.T]).T.reshape(-1, 2)
    def p(x, y):
        return 1/(2*np.pi*np.sqrt(det_sig)) * th.exp(-1/(2*det_sig) * (s22*x*x - 2*s12*x*y + s11*y*y))
    preds = model_func(xy)
    probabs = p(*xy.T)
    zx, zy = (preds * probabs).reshape(N, N).mean(axis=1) * (xr - xl), (preds * probabs).reshape(N, N).mean(axis=0) * (xr - xl)
    def pX(x):
        return 1/(np.sqrt(2*np.pi*s11)) * th.exp(-1/(2*s11) * x*x)
    def pY(y):
        return 1/(np.sqrt(2*np.pi*s22)) * th.exp(-1/(2*s22) * y*y)
    probabs_x = pX(xs)
    nomi_x = (th.square(zx) * probabs_x).mean() * (xr - xl)
    probabs_y = pY(ys)
    nomi_y = (th.square(zy) * probabs_y).mean() * (xr - xl)
    denom = (th.square(preds) * probabs).mean() * (xr - xl) * (xr - xl)
    return (nomi_x + epsilon) / (denom + epsilon), (nomi_y + epsilon) / (denom + epsilon)


def predict_int_GAMI(model, inputs):
    res = []
    model.interact_outputs = model.interact_blocks(inputs, sample_weight=None, training=False)
    for i, (k1, k2) in enumerate(model.interaction_list):
        interaction_weights = tf.multiply(model.output_layer.interaction_switcher, model.output_layer.interaction_weights)
        b = tf.multiply(tf.gather(model.interact_outputs, [i], axis=1), tf.gather(interaction_weights, [i], axis=0))
        res.append(b.numpy())
    return res


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
    pred = pred_vec[max_features + pair_id]
    return pred


def predict_one_int_gami(model, x, pair_id):
    int_list = model.interaction_list
    max_features = len(model.feature_list_)
    idx, idy = int_list[pair_id][0], int_list[pair_id][1]
    test_x = np.zeros((x.shape[0], max_features))
    test_x[:, [idx, idy]] = x
    pred = predict_int_GAMI(model, test_x)[pair_id].reshape(-1)
    return pred


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
