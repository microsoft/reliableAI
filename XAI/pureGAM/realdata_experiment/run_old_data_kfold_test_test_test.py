import pandas as pd
from sklearn.model_selection import train_test_split
import os

# only for multi gpu, if cpu, please delete this line
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import time
import pickle as pkl
import torch
import torch as th
import torch.utils.data as data
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from torch_utils.dataset_util import PureGamDataset, PureGamDataset_smoothingInTraining
from torch_utils.readwrite import make_dir
from sgd_solver.pureGam import PureGam
from optimizer.AugLagSGD import Adam_AugLag
from sklearn.metrics import r2_score, mean_squared_error
import math
from sklearn.preprocessing import power_transform
from pathlib import Path
from experiment_metrics.metrics import metric_wrapper, rmse
from sklearn.preprocessing import MinMaxScaler
from gaminet import GAMINet
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from synthetic_experiments.run_metrics import predict_vec_pureGAM, score_pureGAM, score_pureGAM_cat, score_pureGAM3, score_gami2, predict_int_GAMI
from synthetic_experiments.run_metrics2 import score_pureness
import traceback

np_config.enable_numpy_behavior()
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
isLittleTest = False

def process_gami_info(X_num, y):
    # PowerTransform
    X_num = power_transform(X_num)
    #y = power_transform(y)
    task_type = "Regression"
    meta_info = {"X" + str(i + 1): {'type': 'continuous'} for i in range(X_num.shape[1])}
    meta_info.update({'Y': {'type': 'target'}})
    print(meta_info)
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            sy = MinMaxScaler((0, 1))
            y = sy.fit_transform(y)
            meta_info[key]['scaler'] = sy
        else:
            sx = MinMaxScaler((0, 1))
            sx.fit([[0], [1]])
            X_num[:, [i]] = sx.transform(X_num[:, [i]])
            meta_info[key]['scaler'] = sx
    get_metric = metric_wrapper(rmse, sy)
    return task_type, meta_info, get_metric, X_num, y

def run_pureGam_gami_kfold(X_num, X_cate, y, results_folder, isPureScore=True, seed=42):
    #preprocess minmax/power
    task_type, meta_info, get_metric, X_num, y = process_gami_info(X_num, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf.split(X_num)):
        X_num_train, X_num_test = X_num[train_index], X_num[test_index]
        X_cate_train, X_cate_test = X_cate[train_index], X_cate[test_index]
        y_train, y_test = y[train_index], y[test_index]

        pureGam = run_pureGam(X_num_train, X_cate_train, y_train, X_num_test, X_cate_test, y_test,
                              results_folder=results_folder + '/' + 'pureGam' + '_fold' + str(i), isPureScore=isPureScore)
        _ = run_gami(X_num_train, X_cate_train, y_train, X_num_test, X_cate_test, y_test, task_type, meta_info,
                     results_folder=results_folder + '/' + 'Gami' + '_fold' + str(i), isPureScore=isPureScore,
                     h_map=pureGam.num_enc.get_lam()[0].detach().cpu().numpy().tolist())

def run_pureGam(X_num, X_cate, y_train, X_num_test, X_cate_test, y_test, results_folder, isPureScore=True):
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]
    print("!!Data ", X_num.shape, y.shape)
    '''# train_test split
    X_num, X_num_test, X_cate, X_cate_test, y_train, y_test = \
        train_test_split(X_num, X_cate, y, test_size=0.2, random_state=seed)
    # print("yTest!", y_test)'''
    X_num_ttrain, X_num_valid, X_cate_ttrain, X_cate_valid, y_ttrain, y_valid = \
        train_test_split(X_num, X_cate, y_train, test_size=0.2, random_state=int(math.sqrt(seed)) + 1)

    '''_, X_num_ttrain_smp, _, X_cate_ttrain_smp, _, y_ttrain_smp = \
        train_test_split(X_num_ttrain, X_cate_ttrain, y_ttrain, test_size=0.25, random_state=seed * 2 + 101)'''

    batch_scale = 4#2
    N_param_scale = 0.5# todo: 0.5
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        X_cate_ttrain, th.tensor(X_num_ttrain).to(device), th.tensor(y_ttrain).to(device)),
        batch_size=int(batch_scale * 128), shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        X_cate_valid, th.tensor(X_num_valid).to(device), th.tensor(y_valid).to(device)),
        batch_size=int(batch_scale * 128 * 2), shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        X_cate_test, th.tensor(X_num_test).to(device), th.tensor(y_test).to(device)),
        batch_size=int(batch_scale * 128 * 2), shuffle=False, **kwargs)
    train_loader_all = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        X_cate, th.tensor(X_num).to(device), th.tensor(y_train).to(device)),
        batch_size=int(batch_scale * 128 * 2), shuffle=False, **kwargs)

    '''N_param_univ=int(N_param_scale * 512),
    N_param_biv=int(N_param_scale * 256),  # todo:#256),''',
    model = PureGam(X_num.shape[1], X_cate.shape[1],
                    N_param_univ = int(N_param_scale * 512),
                   N_param_biv = int(N_param_scale * 256),
                    # init_kw_lam_univ=0.15, init_kw_lam_biv=0.1,
                    init_kw_lam_univ=0.5, init_kw_lam_biv=0.7,
                    #init_kw_lam_univ=0.4, init_kw_lam_biv=0.5,
                    #init_kw_lam_univ=0.3, init_kw_lam_biv=0.4,
                    #init_kw_lam_univ=0.2, init_kw_lam_biv=0.3,
                    isInteraction=True, model_output_dir=model_output_dir, device=device,
                    bias=y_train.mean(), isPurenessConstraint=True, isAugLag=False, isInnerBatchPureLoss=False,
                    isLossEnhanceForDenseArea=True, dropout_rate=0,  # dropout_rate=0.2,
                    pure_lam_scale=1,
                    adaptiveInfoPerEpoch=10,
                    pairwise_idxes_cate=[],
                    is_balancer=False,
                    epoch4balancer_change=0,
                    isLossMean=False)
                    #is_balancer=False) # todo: revisit

    model.init_param_points(X_num_ttrain)
    model.fit_cate_encoder(X_cate_ttrain)
    over_all_lr_rate = 1 * 1e-4
    avg_cardi = 3

    # Prepare the optimizer for training, declare the learning rate of all model paramaters

    '''optimizer = Adam_AugLag([
        {'params': [model.model_categorical.w1_univ,model.model_categorical.w2_univ,model.model_categorical.b_univ],
         'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate / X_cate.shape[1] / (avg_cardi - 1) ** 2 * 10000 /
                                              X_cate.shape[0]},  # 6e-3},
        {'params': [model.model_categorical.w1_biv, model.model_categorical.w2_biv, model.model_categorical.b_biv],
         'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate / (X_cate.shape[1] - 1) / X_cate.shape[1] / (
                 avg_cardi - 1) ** 2 * 10000 / X_cate.shape[0]},

        {'params': [model.model_smoothing.w1_univ, model.model_smoothing.w2_univ, model.model_smoothing.b_univ],
         'lr': over_all_lr_rate / X_num.shape[1]},
        {'params': [model.model_smoothing.w1_biv, model.model_smoothing.w2_biv, model.model_smoothing.b_biv],
         'lr': over_all_lr_rate / X_num.shape[1] / (X_num.shape[1] - 1)},
        {'params': model.model_smoothing.bias, 'lr': 0},

        # Adapative Kernel learning rate
        #todo: revisit the exp version of num_enc
        {'params': [model.num_enc.lam_w1_univ, model.num_enc.lam_w2_univ, model.num_enc.lam_b_univ],
         'lr': 1 * over_all_lr_rate / X_num.shape[1]},
        {'params': [model.num_enc.lam_w1_biv, model.num_enc.lam_w2_biv, model.num_enc.lam_b_biv],
         'lr': over_all_lr_rate / X_num.shape[1]},
        {'params': model.num_enc.X_memo_univ, 'lr': 0},  # 1e-2
        {'params': model.num_enc.X_memo_biv, 'lr': 0},  # 1e-2
    ], amsgrad=True)'''

    optimizer = Adam_AugLag([
        {'params': [model.model_categorical.w1_univ, model.model_categorical.w2_univ, model.model_categorical.b_univ],
         'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate },  # 6e-3},
        {'params': [model.model_categorical.w1_biv, model.model_categorical.w2_biv, model.model_categorical.b_biv],
         'lr': 0 if X_cate.shape[1] == 0 else over_all_lr_rate},

        {'params': [model.model_smoothing.w1_univ, model.model_smoothing.w2_univ, model.model_smoothing.b_univ],
         'lr': over_all_lr_rate },
        {'params': [model.model_smoothing.w1_biv, model.model_smoothing.w2_biv, model.model_smoothing.b_biv],
         'lr': over_all_lr_rate },
        {'params': model.model_smoothing.bias, 'lr': 0},

        # Adapative Kernel learning rate
        # todo: revisit the exp version of num_enc
        {'params': [model.num_enc.lam_w1_univ, model.num_enc.lam_w2_univ, model.num_enc.lam_b_univ],
         'lr': 1 * over_all_lr_rate },
        {'params': [model.num_enc.lam_w1_biv, model.num_enc.lam_w2_biv, model.num_enc.lam_b_biv],
         'lr': over_all_lr_rate },
        {'params': model.num_enc.X_memo_univ, 'lr': 0},  # 1e-2
        {'params': model.num_enc.X_memo_biv, 'lr': 0},  # 1e-2
    ], amsgrad=True)

    print("{*} Model Size", model.get_model_size())
    import torch.nn as nn

    # init points
    '''print(model.model_smoothing.eta_univ[:10])
    for p in model.model_smoothing.parameters():
        #print(p.size(), p.numel())
        if p.dim() > 1 and p.numel() > 0:
            nn.init.xavier_uniform_(p)
    for p in model.model_categorical.parameters():
        if p.dim() > 1 and p.numel() > 0:
            nn.init.xavier_uniform_(p)

    print(model.model_smoothing.eta_univ[:10])'''

    try:
        assert False
        train_time = 0
        model.load_model(results_folder + "/model_best")
    except:
        t0 = time.time()
        model.train(train_loader, valid_loader, optimizer=optimizer, num_epoch=2 if isLittleTest else int(5*200 * math.sqrt(1e4 / len(y))),
                    tolerent_level=int(30 * math.sqrt(1e4 / len(y))),
                    # Pureness Loss ratio of total loss
                     #ratio_part2=2e1,
                    #ratio_part2=1e-1, #todo : with balancer
                    ratio_part2=0.5e-2, #todo: no balancer
                    lmd1=1e0, lmd2=1e0, lmd3=1e0, test_data_loader=test_loader)
        '''ratio_part2=2e1,
        lmd1=2e0, lmd2=2e0, lmd3=1e0, test_data_loader=test_loader)'''

        '''model.train(train_loader, valid_loader, optimizer=optimizer, num_epoch=int(160 * math.sqrt(1e4 / len(y))),
                    tolerent_level=int(10 * math.sqrt(1e4 / len(y))),
                    # Pureness Loss ratio of total loss
                    ratio_part2=1e2, lmd1=0e0, lmd2=0e0, lmd3=1e0, test_data_loader=test_loader)'''
        # ratio_part2=1e2, lmd1=1e0, lmd2=1e0, lmd3=2e0)
        t1 = time.time()

        train_time = t1 - t0
        model.save_model(results_folder + "/model_best")

    h_map = model.num_enc.get_lam()[0].detach().cpu().numpy().tolist()

    t0 = time.time()
    y_hat_test = model.predict(test_loader)
    t1 = time.time()
    test_time = t1 - t0
    y_hat_train = model.predict(train_loader_all)

    print(len(y_train), len(y_test))
    print(len(y_hat_train), len(y_hat_test))
    r2_train = r2_score(y_train, y_hat_train)
    r2_test = r2_score(y_test, y_hat_test)
    mse_train = mean_squared_error(y_train, y_hat_train)
    mse_test = mean_squared_error(y_test, y_hat_test)
    """
    pureGAM pureness
    """
    print(model.num_enc.get_lam()[0])
    print(model.num_enc.get_lam()[1])
    if isPureScore:
        pred_int = model.predict_biv_contri_mat(train_loader_all)
        try:
            pure_score1, pure_score2 = score_pureness(pred_int, X_num=th.tensor(X_num).to(device),
                                                      interaction_list=[(pair[0], pair[1]) for pair in
                                                                        model.pairwise_idxes_num],
                                                      h_map=h_map, bandwidths=np.array([0.1]),
                                                      save_folder=results_folder)
            enum_pure_score = (pure_score1.loc['avg_log'].loc['lam1'] + pure_score2.loc['avg_log'].loc['lam1']) / 2

        except:
            traceback.print_exc()
            enum_pure_score = 0
            assert False
            # print('[%%%%]')

        acc_df = pd.DataFrame(
            [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time], [0, enum_pure_score]],
            index=["r2", "mse", "time", "log_pure_score"], columns=["train", "test"])
    else:
        acc_df = pd.DataFrame(
            [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time]],
            index=["r2", "mse", "time"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    print(acc_df)
    return model


def run_gami(X_num, X_cate, y_train, X_num_test, X_cate_test, y_test, task_type, meta_info, results_folder, h_map, isPureScore=True):
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    '''X_num, X_num_test, X_cate, X_cate_test, y_train, y_test = \
        train_test_split(X_num, X_cate, y, test_size=0.2, random_state=seed)'''
    # print("yTest!", y_test)

    model = GAMINet(meta_info=meta_info, interact_num=X_num.shape[1],
                    # X_num.shape[1]*X_num.shape[1]-1/2, # X_num.shape[1], #
                    interact_arch=[40] * 5, subnet_arch=[40] * 5,
                    batch_size=200, task_type=task_type, activation_func=tf.nn.relu,
                    main_effect_epochs=2 if isLittleTest else int(2*500 * math.sqrt(1e4 / X_num.shape[0])),
                    interaction_epochs=2 if isLittleTest else int(2*500 * math.sqrt(1e4 / X_num.shape[0])),
                    tuning_epochs=2 if isLittleTest else int(500 * math.sqrt(1e4 / X_num.shape[0])),

                    lr_bp=[0.0001, 0.0001, 0.0001],
                    early_stop_thres=[int(30 * math.sqrt(1e4 / X_num.shape[0])),
                                      int(30 * math.sqrt(1e4 / X_num.shape[0])),
                                      int(20 * math.sqrt(1e4 / X_num.shape[0]))],
                    heredity=True, loss_threshold=0.01, reg_clarity=0.1,
                    mono_increasing_list=[], mono_decreasing_list=[],  ## the indices list of features
                    verbose=False, val_ratio=0.2, random_state=int(math.sqrt(seed)) + 1)  # seed*2)

    try:
        assert False
        train_time = 0
        model.load(results_folder + '/', "model_best")
        # model.load(results_folder , "gami_best")
    except:
        t0 = time.time()
        model.fit(X_num, y_train)
        t1 = time.time()
        train_time = t1 - t0
        try:
            model.save(results_folder + '/', "model_best")
        except:
            traceback.print_exc()
            print("Cannont save model")
    # model.plot_numerical(X_num_test)

    t0 = time.time()
    y_hat_test = model.predict(X_num_test)[:, 0]
    t1 = time.time()
    test_time = t1 - t0
    # y_hat_val = model.predict(X_num[model.val_idx, :])[:, 0]
    # y_hat_train = model.predict(X_num[model.tr_idx, :])[:, 0]
    y_hat_train = model.predict(X_num)[:, 0]

    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    print(len(y_train), len(y_test))
    print(len(y_hat_train), len(y_hat_test))
    r2_train = r2_score(y_train, y_hat_train)
    r2_test = r2_score(y_test, y_hat_test)
    mse_train = mean_squared_error(y_train, y_hat_train)
    mse_test = mean_squared_error(y_test, y_hat_test)
    """
    pureGAM pureness
    """

    '''_, X_num_ttrain_smp, _, X_cate_ttrain_smp, _, y_ttrain_smp = \
        train_test_split(X_num[model.tr_idx, :], X_cate[model.tr_idx, :], y_train[model.tr_idx], test_size=0.25,
                         random_state=seed * 2 + 101)'''
    if isPureScore:
        gami_int_pred = predict_int_GAMI(model, X_num)
        try:
            gami_int_pred = th.tensor(gami_int_pred, dtype=th.double).squeeze(2).to(device)

            pure_score1, pure_score2 = score_pureness(gami_int_pred, X_num=th.tensor(X_num).to(device),
                                                      interaction_list=model.interaction_list,
                                                      h_map=h_map, bandwidths=np.array([0.1]),
                                                      save_folder=results_folder)
            enum_pure_score = (pure_score1.loc['avg_log'].loc['lam1'] + pure_score2.loc['avg_log'].loc['lam1']) / 2
        except:
            traceback.print_exc()
            enum_pure_score = 0
            #print('[%%%%]')

        acc_df = pd.DataFrame(
            [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time], [0, enum_pure_score]],
            index=["r2", "mse", "time", "log_pure_score"], columns=["train", "test"])
    else:
        acc_df = pd.DataFrame(
            [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time]],
            index=["r2", "mse", "time"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "gami_accuracy.csv"))
    print(acc_df)

if __name__ == "__main__":
    #seed = 2050, 2061, 2072
    seed = 42
    base_dir = "../"

    # todo: cali house
    dataset_name = "cal_house_processed"
    # dataset_name = "flat_dirichlet_generate"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name

    # Load Data from dataset
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")

    X_num = df.loc[:,
            ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]].values
    X_cate = df.loc[:, []].values
    y = df.loc[:, ["MedHouseVal"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)



