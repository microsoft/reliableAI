import os
from pathlib import Path
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from benchmarks.synthetic_data_generator import generate_x, function_main_effects, function_interaction, num_gen_gauss
from benchmarks.synthetic_data_generator import check_hist, check_func_main, check_func_int
from experiment_metrics.metrics import pureness_loss_est, pureness_loss_est2, pureness_score2, pureness_score2_normalized
from synthetic_experiments.run_metrics import true_pureness_score_gaussian_pureGAM, true_pureness_score_gaussian_ebm, true_pureness_score_gaussian_gami
from synthetic_experiments.run_metrics2 import score_pureness
import traceback
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from interpret import show
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve, accuracy_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler

"""PureGAM"""
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.join(path.dirname(path.dirname(path.abspath(__file__))), "test_sgd_solve_new"))

import pandas as pd
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pickle as pkl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sgd_solver.utils import generate_pairwise_idxes, _print_metrics, safe_norm, save_model, load_model
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from optimizer.AugLagSGD import Adam_AugLag
from torch_utils.dataset_util import PureGamDataset, PureGamDataset_smoothingInTraining
from torch_utils.readwrite import make_dir
from sgd_solver.pureGam import PureGam
from optimizer.AugLagSGD import Adam_AugLag
from sklearn.metrics import r2_score
import time
from run_metrics import predict_vec_pureGAM, score_pureGAM, score_pureGAM_cat
import numpy

batch_scale = 4
N_param_scale = 0.5

isTest=False

def run(train_x, train_y, test_x, test_y, cov_mat, results_folder):
    """
    pureGAM
    """
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    model_output_dir = "model/save/"+results_folder+'/'
    X_num, y = train_x, train_y.reshape(-1)
    X_cate = np.zeros((X_num.shape[0], 0))
    X_num_test, y_test = test_x, test_y.reshape(-1)
    X_cate_test = np.zeros((X_num_test.shape[0], 0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = {}

    train_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        th.tensor(X_cate).to(device),th.tensor(X_num).to(device), th.tensor(y).to(device)),
        batch_size=batch_scale*128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        th.tensor(X_cate_test).to(device), th.tensor(X_num_test).to(device), th.tensor(y_test).to(device)),
        batch_size=batch_scale*128*2, shuffle=False, **kwargs)
    train_loader_all = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        th.tensor(X_cate).to(device), th.tensor(X_num).to(device), th.tensor(y).to(device)),
        batch_size=batch_scale*128*2, shuffle=False, **kwargs)

    model = PureGam(X_num.shape[1], X_cate.shape[1],
                    N_param_univ=int(N_param_scale * 512),
                    N_param_biv=int(N_param_scale * 256),
                    # init_kw_lam_univ=0.15, init_kw_lam_biv=0.1,
                    init_kw_lam_univ=0.5, init_kw_lam_biv=0.7,
                    #init_kw_lam_univ=0.4, init_kw_lam_biv=0.5,
                    #init_kw_lam_univ=0.3, init_kw_lam_biv=0.4,
                    # init_kw_lam_univ=0.2, init_kw_lam_biv=0.3,
                    isInteraction=True, model_output_dir=model_output_dir, device=device,
                    bias=y.mean(), isPurenessConstraint=True, isAugLag=False, isInnerBatchPureLoss=False,
                    isLossEnhanceForDenseArea=True, dropout_rate=0,  # dropout_rate=0.2,
                    pure_lam_scale=1,
                    adaptiveInfoPerEpoch=10,
                    pairwise_idxes_cate=[],
                    is_balancer=False,
                    epoch4balancer_change=20,
                    isLossMean=False)

    '''pureGAM_model = PureGam(X_num.shape[1], X_cate.shape[1], N_param_univ = N_param_scale*512, N_param_biv = N_param_scale*512, init_kw_lam_univ = 0.15, init_kw_lam_biv = 0.15,
                    isInteraction=True, model_output_dir = model_output_dir, device=device,
                    bias = y.mean(), isPurenessConstraint=True, isAugLag=False, isInnerBatchPureLoss=False, isLossEnhanceForDenseArea=False, dropout_rate=0, pure_lam_scale=1)'''
    model.init_param_points(X_num)
    model.fit_cate_encoder(X_cate)

    # Prepare the optimizer for training, declare the learning rate of all model paramaters
    '''optimizer = Adam_AugLag([
        {'params': pureGAM_model.model_categorical.parameters(), 'lr': 0},#6e-3},
        {'params': pureGAM_model.model_smoothing.eta_univ, 'lr': 1e-2/X_num.shape[1]},
        {'params': pureGAM_model.model_smoothing.eta_biv, 'lr': 2*1e-2/X_num.shape[1]/(X_num.shape[1]-1)},
        {'params': pureGAM_model.model_smoothing.bias, 'lr': 0},

        # Adapative Kernel learning rate
        {'params': pureGAM_model.num_enc.lam_univ, 'lr': 1e-3/X_num.shape[1]},
        {'params': pureGAM_model.num_enc.lam_biv, 'lr': 2*1e-2/X_num.shape[1]/(X_num.shape[1]-1)},
        {'params': pureGAM_model.num_enc.X_memo_univ, 'lr': 0},#1e-2
        {'params': pureGAM_model.num_enc.X_memo_biv, 'lr': 0},#1e-2
    ], amsgrad=True)'''

    over_all_lr_rate = 1 * 1e-3
    optimizer = Adam_AugLag([
        {'params': [model.model_categorical.w1_univ, model.model_categorical.w2_univ, model.model_categorical.b_univ],
         'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate},  # 6e-3},
        {'params': [model.model_categorical.w1_biv, model.model_categorical.w2_biv, model.model_categorical.b_biv],
         'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate},

        {'params': [model.model_smoothing.w1_univ, model.model_smoothing.w2_univ, model.model_smoothing.b_univ],
         'lr': over_all_lr_rate},
        {'params': [model.model_smoothing.w1_biv, model.model_smoothing.w2_biv, model.model_smoothing.b_biv],
         'lr': over_all_lr_rate},
        {'params': model.model_smoothing.bias, 'lr': 0},

        # Adapative Kernel learning rate
        # todo: revisit the exp version of num_enc
        {'params': [model.num_enc.lam_w1_univ, model.num_enc.lam_w2_univ, model.num_enc.lam_b_univ],
         'lr': 1 * over_all_lr_rate},
        {'params': [model.num_enc.lam_w1_biv, model.num_enc.lam_w2_biv, model.num_enc.lam_b_biv],
         'lr': over_all_lr_rate},
        {'params': model.num_enc.X_memo_univ, 'lr': 0},  # 1e-2
        {'params': model.num_enc.X_memo_biv, 'lr': 0},  # 1e-2
    ], amsgrad=True)

    print("Dropout !:!: ", model.model_smoothing.dropout_univ.p, model.model_smoothing.dropout_biv.p)
    try:
        train_time = 0
        model.load_model(results_folder +"/model_best")
        print('[z]', model.num_enc.get_lam()[0])
        print('[s]', model.num_enc.get_lam()[1])
        assert False
    except:
        t0 = time.time()
        '''model.train(train_loader, test_loader, optimizer=optimizer, num_epoch=int(8 * 1e5/len(y)), tolerent_level=int(4 * 1e5/len(y)),
            # Pureness Loss ratio of total loss
            ratio_part2=4e1,  lmd1=1e0, lmd2=1e0, lmd3=2e0)'''
        model.train(train_loader, test_loader, optimizer=optimizer, num_epoch=1 if isTest else int(10 * 200 * math.sqrt(1e4 / len(y))),
                    tolerent_level=1 if isTest else int(10 * math.sqrt(1e4 / len(y))),
                    # Pureness Loss ratio of total loss
                    # ratio_part2=2e1,
                    # ratio_part2=1e-1, #todo : with balancer
                    ratio_part2=0.5e-2,  # todo: no balancer
                    lmd1=1e0, lmd2=1e0, lmd3=1e0, test_data_loader=test_loader)
        t1 = time.time()
        train_time = t1-t0
        model.save_model(results_folder +"/model_best")

    #todo
    h_map = model.num_enc.get_lam()[0].detach().cpu().numpy().tolist()

    t0 = time.time()
    y_hat_test = model.predict(test_loader)
    t1 = time.time()
    test_time = t1-t0
    y_hat_train = model.predict(train_loader_all)

    r2_train = r2_score(y, y_hat_train)
    r2_test= r2_score(y_test, y_hat_test)
    mse_train = mean_squared_error(y, y_hat_train)
    mse_test = mean_squared_error(y_test, y_hat_test)

    pred_int = model.predict_biv_contri_mat(train_loader_all)
    try:
        assert False
        pure_score1, pure_score2 = score_pureness(pred_int, X_num=th.tensor(X_num).to(device),
                                                  interaction_list=[(pair[0], pair[1]) for pair in
                                                                    model.pairwise_idxes_num],
                                                  h_map=h_map, bandwidths=np.array([0.1]),
                                                  save_folder=results_folder)
        enum_pure_score = (pure_score1.loc['avg_log'].loc['lam1'] + pure_score2.loc['avg_log'].loc['lam1']) / 2

    except:
        traceback.print_exc()
        enum_pure_score = 0
        # print('[%%%%]')
    # score_pureGAM(model, test_x, sxx, syy, h_map, bandwidths=np.array([1, 0.5, 0.1, 0.05, 0.01]), epsilon=0, N_subset=None, save_folder=results_folder)
    true_pure_df = true_pureness_score_gaussian_pureGAM(model=model, cov_mat=cov_mat, num_sigmas=3, N=200,
                                                        normalize=True, epsilon=0, save_folder=results_folder, device=device)
    print(true_pure_df)
    true_pure_df = np.log10(true_pure_df.astype('float')).dropna()
    print(true_pure_df)
    enum_true_score = true_pure_df.mean().mean()

    acc_df = pd.DataFrame(
        [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time], [0, enum_pure_score], [0, enum_true_score]],
        index=["r2", "mse", "time", "log_pure_score", "true_pure_score"], columns=["train", "test"])

    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    print(acc_df)
    """
    pureGAM pureness
    """
    #print(h_map)


    return h_map

batch_scale = 4
N_param_scale = 0.5

import math
def run_cat(train_x, train_y, test_x, test_y, results_folder, avg_cardi=3):
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    model_output_dir = "model/save/"+results_folder+'/'
    # X_num, y = train_x, train_y.reshape(-1)
    # X_cate = np.zeros((X_num.shape[0], 0))
    # X_num_test, y_test = test_x, test_y.reshape(-1)
    # X_cate_test = np.zeros((X_num_test.shape[0], 0))
    X_cate, y = train_x, train_y.reshape(-1)
    X_num = np.zeros((X_cate.shape[0], 0))
    # X_num = np.zeros((X_cate.shape[0], 2)) # FIXME Need to debug when there is no num, I for the moment introduce a zero num feature.
    X_cate_test, y_test = test_x, test_y.reshape(-1)
    X_num_test = np.zeros((X_cate_test.shape[0], 0))
    # X_num_test = np.zeros((X_cate_test.shape[0], 2)) # FIXME Need to debug when there is no num, I for the moment introduce a zero num feature.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = {}

    train_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        th.tensor(X_cate),th.tensor(X_num).to(device), th.tensor(y).to(device)),
        batch_size=batch_scale*128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        th.tensor(X_cate_test), th.tensor(X_num_test).to(device), th.tensor(y_test).to(device)),
        batch_size=2*batch_scale*128, shuffle=False, **kwargs)
    train_loader_all = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        th.tensor(X_cate), th.tensor(X_num).to(device), th.tensor(y).to(device)),
        batch_size=2*batch_scale*128, shuffle=False, **kwargs)


    '''pureGAM_model = PureGam(X_num.shape[1], X_cate.shape[1], N_param_univ = N_param_scale*512, N_param_biv = N_param_scale*512, init_kw_lam_univ = 0.1, init_kw_lam_biv = 0.15,
                    isInteraction=True, model_output_dir = model_output_dir, device=device,
                    bias = y.mean(), isPurenessConstraint=False, isAugLag=False, isInnerBatchPureLoss=False, isLossEnhanceForDenseArea = False)'''
    model = PureGam(X_num.shape[1], X_cate.shape[1],
                    N_param_univ=int(N_param_scale * 512),
                    N_param_biv=int(N_param_scale * 256),
                    # init_kw_lam_univ=0.15, init_kw_lam_biv=0.1,
                    init_kw_lam_univ=0.5, init_kw_lam_biv=0.7,
                    #init_kw_lam_univ=0.4, init_kw_lam_biv=0.5,
                    #init_kw_lam_univ=0.3, init_kw_lam_biv=0.4,
                    # init_kw_lam_univ=0.2, init_kw_lam_biv=0.3,
                    isInteraction=True, model_output_dir=model_output_dir, device=device,
                    bias=y.mean(), isPurenessConstraint=False, isAugLag=False, isInnerBatchPureLoss=False,
                    isLossEnhanceForDenseArea=False, dropout_rate=0,  # dropout_rate=0.2,
                    pure_lam_scale=1,
                    adaptiveInfoPerEpoch=10,
                    pairwise_idxes_cate=None,
                    is_balancer=False,
                    epoch4balancer_change=20,
                    isLossMean=False)
    model.init_param_points(X_num)
    model.fit_cate_encoder(X_cate)

    # Prepare the optimizer for training, declare the learning rate of all model paramaters
    '''optimizer = Adam_AugLag([
        {'params': pureGAM_model.model_categorical.eta_univ, 'lr': 1e-3/X_cate.shape[1]/(avg_cardi-1)**2 * 10000/X_cate.shape[0]},#6e-3},
        #{'params': pureGAM_model.model_categorical.eta_biv, 'lr': 1e-4},#)
        {'params': pureGAM_model.model_categorical.eta_biv, 'lr': 2 * 1e-3/(X_cate.shape[1]-1)/X_cate.shape[1]/(avg_cardi-1)**2 * 10000/X_cate.shape[0]},#/(avg_cardi-1)},
        {'params': pureGAM_model.model_smoothing.eta_univ, 'lr': 0},
        {'params': pureGAM_model.model_smoothing.eta_biv, 'lr': 0},
        {'params': pureGAM_model.model_smoothing.bias, 'lr': 0},

        # Adapative Kernel learning rate
        {'params': pureGAM_model.num_enc.lam_univ, 'lr': 0},
        {'params': pureGAM_model.num_enc.lam_biv, 'lr': 0},
        {'params': pureGAM_model.num_enc.X_memo_univ, 'lr': 0},#1e-2
        {'params': pureGAM_model.num_enc.X_memo_biv, 'lr': 0},#1e-2
    ], amsgrad=True)'''
    over_all_lr_rate = 1 * 1e-3
    """{'params': [model.model_categorical.w1_univ, model.model_categorical.w2_univ, model.model_categorical.b_univ],
     # 'lr': 0},
     'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate / (batch_scale * 128) * avg_cardi * X_cate.shape[
         0]},  # 6e-3},
    # 'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate },
    {'params': [model.model_categorical.w1_biv, model.model_categorical.w2_biv, model.model_categorical.b_biv],
     'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate / (batch_scale * 128) * avg_cardi ** 2},
    # 'lr': 0},"""
    print("&^ biv param", 1/ (batch_scale * 128) / (avg_cardi ** 2) * X_cate.shape[1] )
    optimizer = Adam_AugLag([


        {'params': [model.model_categorical.w1_univ, model.model_categorical.w2_univ, model.model_categorical.b_univ],
         'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate  / ((avg_cardi - 1) ** 2)},
        #'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate / (batch_scale * 128) * X_cate.shape[0]/ ((avg_cardi-1) ** 2) },
        #'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate / avg_cardi**2 * X_cate.shape[1] },
         #'lr': 1e-3 / X_cate.shape[1] / (avg_cardi - 1) },  # 6e-3},
        # {'params': pureGAM_model.model_categorical.eta_biv, 'lr': 1e-4},#)
        # 'lr': 0},
        {'params': [model.model_categorical.w1_biv, model.model_categorical.w2_biv, model.model_categorical.b_biv],
         'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate  / ((avg_cardi - 1) ** 2)/ (batch_scale * 128)},
         #'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate / (batch_scale * 128) /  ((avg_cardi-1) ** 2)  },
         #'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate/ (batch_scale * 128) / avg_cardi**2 * X_cate.shape[1] },
         #'lr': 1e-3 / (X_cate.shape[1] - 1) / X_cate.shape[1] / (avg_cardi - 1) ** 2 * 10000 / X_cate.shape[0]},
        #'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate / (batch_scale * 128) ** 2/avg_cardi},
        # 'lr': 0},
        # /(avg_cardi-1)},

        {'params': [model.model_smoothing.w1_univ, model.model_smoothing.w2_univ, model.model_smoothing.b_univ],
         'lr': 0},
        {'params': [model.model_smoothing.w1_biv, model.model_smoothing.w2_biv, model.model_smoothing.b_biv],
         'lr': 0},
        {'params': model.model_smoothing.bias, 'lr': 0},

        # Adapative Kernel learning rate
        # todo: revisit the exp version of num_enc
        {'params': [model.num_enc.lam_w1_univ, model.num_enc.lam_w2_univ, model.num_enc.lam_b_univ],
         'lr': 1 * 0},
        {'params': [model.num_enc.lam_w1_biv, model.num_enc.lam_w2_biv, model.num_enc.lam_b_biv],
         'lr': 0},
        {'params': model.num_enc.X_memo_univ, 'lr': 0},  # 1e-2
        {'params': model.num_enc.X_memo_biv, 'lr': 0},  # 1e-2
    ], amsgrad=True)

    try:
        assert False
        train_time = 0
        model.load_model(results_folder +"/model_best")
    except:
        t0 = time.time()
        '''model.train(train_loader, test_loader, optimizer=optimizer, num_epoch=int(2 * 1e2 * math.sqrt(1e4/len(y))), tolerent_level=int(2 * 1e2 * math.sqrt(1e4/len(y))),
            # Pureness Loss ratio of total loss
            ratio_part2=0e0,  lmd1=1e0, lmd2=1e0, lmd3=2e0)'''
        model.train(train_loader, test_loader, optimizer=optimizer,
                    num_epoch=1 if isTest else int(10 * 200 * math.sqrt(1e4 / len(y))),
                    tolerent_level=1 if isTest else int(20 * math.sqrt(1e4 / len(y))),
                    ratio_part2=0e-2,  # todo: no balancer
                    lmd1=1e0, lmd2=1e0, lmd3=1e0, test_data_loader=test_loader)
        t1 = time.time()
        train_time = t1-t0
        model.save_model(results_folder +"/model_best")

    t0 = time.time()
    y_hat_test = model.predict(test_loader)
    t1 = time.time()
    test_time = t1-t0
    y_hat_train = model.predict(train_loader_all)

    r2_pureGAM_train = r2_score(y, y_hat_train)
    r2_pureGAM = r2_score(y_test, y_hat_test)
    mse_train = mean_squared_error(y, y_hat_train)
    mse_test = mean_squared_error(y_test, y_hat_test)

    pure_score_df = score_pureGAM_cat(model, train_x, N_subset=None, save_folder=results_folder)
    print(pure_score_df)
    pure_score_df = np.log10(pure_score_df.astype('float')).dropna()
    print(pure_score_df)
    enum_true_score = pure_score_df.mean().mean()

    acc_df = pd.DataFrame([[r2_pureGAM_train, r2_pureGAM], [mse_train, mse_test], [train_time, test_time],
                           [0, enum_true_score]], index=["r2", "mse", "time", "true_pure_score"], columns=["train", "test"])

    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    print(acc_df)
    """
    pureGAM pureness
    """

