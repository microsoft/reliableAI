import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import time
import torch as th
import torch.utils.data as data
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from torch_utils.dataset_util import PureGamDataset_smoothingInTraining
from pureGAM_model.pureGAM import PureGam
from torch.optim import Adam
from sklearn.metrics import r2_score, mean_squared_error
import math
from sklearn.preprocessing import power_transform
from pathlib import Path
from metrics.metrics import metric_wrapper, rmse, predict_int_GAMI
from sklearn.preprocessing import MinMaxScaler
from gaminet import GAMINet
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
from metrics.metrics_torch import score_pureness
import traceback
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor

np_config.enable_numpy_behavior()
# Device
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

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

        #todo:
        run_ebm(X_num_train, X_cate_train, y_train, X_num_test, X_cate_test, y_test,
                              results_folder=results_folder + '/' + 'ebm' + '_fold' + str(i))
        run_xgb(X_num_train, X_cate_train, y_train, X_num_test, X_cate_test, y_test,
                              results_folder=results_folder + '/' + 'xgb' + '_fold' + str(i))

def run_ebm(X_num, X_cate, y_train, X_num_test, X_cate_test, y_test, results_folder):
    Path(results_folder).mkdir(parents=True, exist_ok=True)

    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    # run_gami.run(X_num, y, X_cate_test, y_test, "res/", int_num=10)

    ebm = ExplainableBoostingRegressor(interactions=X_num.shape[0])
    print(X_num.shape)

    t0 = time.time()
    ebm.fit(X_num, y_train)
    t1 = time.time()
    train_time = t1 - t0

    t0 = time.time()
    y_hat_test = ebm.predict(X_num_test)
    t1 = time.time()
    test_time = t1 - t0
    y_hat_val = ebm.predict(X_num)
    # y_hat_train = model.predict(X_num[model.tr_idx, :])[:, 0]
    y_hat_train = ebm.predict(X_num)


    r2_train = r2_score(y_train, y_hat_train)
    r2_test = r2_score(y_test, y_hat_test)
    mse_train = mean_squared_error(y_train, y_hat_train)
    mse_test = mean_squared_error(y_test, y_hat_test)

    acc_df = pd.DataFrame(
        [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time]],
        index=["r2", "mse", "time"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    print(acc_df)

def run_xgb(X_num, X_cate, y_train, X_num_test, X_cate_test, y_test, results_folder):
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    xgbm = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=7, min_child_weight=5)
    # xgbm = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=15, min_child_weight=10)
    # "xgb_mid":xgb.XGBRegressor(n_estimators = 500, learning_rate = 0.3, max_depth = 7, min_child_weight=5),
    # "xgb_slow":xgb.XGBRegressor(n_estimators = 1000, learning_rate = 0.1, max_depth = 7, min_child_weight=5),

    t0 = time.time()
    xgbm.fit(X_num, y_train)
    t1 = time.time()
    train_time = t1 - t0

    t0 = time.time()
    y_hat_test = xgbm.predict(X_num_test)
    t1 = time.time()
    test_time = t1 - t0
    # y_hat_train = model.predict(X_num[model.tr_idx, :])[:, 0]
    y_hat_train = xgbm.predict(X_num)

    r2_train = r2_score(y_train, y_hat_train)
    r2_test = r2_score(y_test, y_hat_test)
    mse_train = mean_squared_error(y_train, y_hat_train)
    mse_test = mean_squared_error(y_test, y_hat_test)

    acc_df = pd.DataFrame(
        [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time]],
        index=["r2", "mse", "time"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    print(acc_df)

def run_pureGam(X_num, X_cate, y_train, X_num_test, X_cate_test, y_test, results_folder, isPureScore=True):
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]
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
    train_loader = th.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        X_cate_ttrain, th.tensor(X_num_ttrain).to(device), th.tensor(y_ttrain).to(device)),
        batch_size=int(batch_scale * 128), shuffle=True, **kwargs)
    valid_loader = th.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        X_cate_valid, th.tensor(X_num_valid).to(device), th.tensor(y_valid).to(device)),
        batch_size=int(batch_scale * 128 * 2), shuffle=False, **kwargs)
    test_loader = th.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
        X_cate_test, th.tensor(X_num_test).to(device), th.tensor(y_test).to(device)),
        batch_size=int(batch_scale * 128 * 2), shuffle=False, **kwargs)
    train_loader_all = th.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
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
                    epoch4balancer_change=20,
                    isLossMean=False)
                    #is_balancer=False) # todo: revisit

    model.init_param_points(X_num_ttrain)
    model.fit_cate_encoder(X_cate_ttrain)
    over_all_lr_rate = 1 * 1e-4
    avg_cardi = 3

    # Prepare the optimizer for training, declare the learning rate of all model paramaters
    optimizer = Adam([
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

    try:
        assert False
        train_time = 0
        model.load_model(results_folder + "/model_best")
    except:
        t0 = time.time()
        model.train(train_loader, valid_loader, optimizer=optimizer, num_epoch=int(5*200 * math.sqrt(1e4 / len(y))),
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
                    main_effect_epochs=int(10*500 * math.sqrt(1e4 / X_num.shape[0])),
                    interaction_epochs=int(10*500 * math.sqrt(1e4 / X_num.shape[0])),
                    tuning_epochs=int(500 * math.sqrt(1e4 / X_num.shape[0])),

                    lr_bp=[0.0001, 0.0001, 0.0001],
                    early_stop_thres=[int(50 * math.sqrt(1e4 / X_num.shape[0])),
                                      int(50 * math.sqrt(1e4 / X_num.shape[0])),
                                      int(50 * math.sqrt(1e4 / X_num.shape[0]))],
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
    seed = 10100
    base_dir = "../"

    # todo: AirplaneCompanies
    dataset_name = "AirplaneCompanies"
    data_file_name = 'stock'
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_table(base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".data", sep='\t',
                       header=None).dropna(axis=1, how='all')
    print("load data from ::", base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".data")
    columns_names = \
        pd.read_table(base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".domain",
                      header=None).values.T.tolist()[0]
    columns_names = [name.split(' : ')[0] for name in columns_names]
    df.columns = columns_names
    X_num = df.loc[:, ["company1", "company2", "company3", "company4", "company5", "company6", "company7",
                       "company8", "company9"]].values
    X_cate = df.loc[:, []].values
    # X_cate = df.loc[:, ["WeekStatus", "Day_of_week", "Load_Type"]].values
    y = df.loc[:, ["company10"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: bank
    dataset_name = "Bank"
    data_file_name = 'bank8FM'
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_table(base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".data", sep=' ',
                       header=None).dropna(axis=1, how='all')
    print("load data from ::", base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".data")
    columns_names = \
        pd.read_table(base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".domain",
                      header=None).values.T.tolist()[0]
    columns_names = [name.split('  : ')[0] for name in columns_names]
    df.columns = columns_names
    #print(df)
    X_num = df.loc[:, ["a1cx", "a1cy", "b2x", "b2y", "a2pop", "a3pop", "temp", "mxql"]].values
    X_cate = df.loc[:, []].values
    # X_cate = df.loc[:, ["WeekStatus", "Day_of_week", "Load_Type"]].values
    y = df.loc[:, ["rej"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: Elevators delta
    dataset_name = "deltaElevators"
    data_file_name = 'delta_elevators'
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_table(base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".data", sep=' ',
                       header=None)
    print("load data from ::", base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".data")
    columns_names = \
    pd.read_table(base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".domain",
                  header=None).values.T.tolist()[0]
    columns_names = [name.split(' : ')[0] for name in columns_names]
    df.columns = columns_names
    X_num = df.loc[:, ["climbRate","Altitude","RollRate","curRoll","diffClb","diffDiffClb"]].values
    X_cate = df.loc[:, []].values
    # X_cate = df.loc[:, ["WeekStatus", "Day_of_week", "Load_Type"]].values
    y = df.loc[:, ["Se"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)


    # todo: Kinematics
    dataset_name = "Kinematics"
    data_file_name = 'kin8nm'
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_table(base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".data", sep=',', header=None)
    print("load data from ::", base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".data")
    columns_names = pd.read_table(base_dir + "realdata/ga2m/" + dataset_name + '/' + data_file_name + ".domain", sep=',', header=None).values.T.tolist()[0]
    columns_names = [name.split(' : ')[0] for name in columns_names]
    df.columns = columns_names
    #print(df)
    X_num = df.loc[:, ["theta1", "theta2", "theta3", "theta4", "theta5", "theta6", "theta7", "theta8"]].values
    X_cate = df.loc[:, []].values
    # X_cate = df.loc[:, ["WeekStatus", "Day_of_week", "Load_Type"]].values
    y = df.loc[:, ["y"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: seoul bike
    dataset_name = "SeoulBikeData"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/new/" + dataset_name + ".csv", sep=',')
    print("load data from ::", base_dir + "realdata/new/" + dataset_name + ".csv")
    df = df.dropna(axis=0)
    #print(df)
    X_num = df.loc[:,
            ["Temperature", "Humidity", "Wind speed", "Visibility", "Dew point temperature",
             "Solar Radiation", "Rainfall", "Snowfall"]].values
    X_cate = df.loc[:, []].values
    # X_cate = df.loc[:, ["Seasons","Holiday","Functioning Day"]].values
    y = df.loc[:, ["Rented Bike Count"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: wine white
    dataset_name = "winequality-white"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/new/" + dataset_name + ".csv", sep=';')
    print("load data from ::", base_dir + "realdata/new/" + dataset_name + ".csv")
    X_num = df.loc[:,
            ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]].values
    X_cate = df.loc[:, []].values
    # "L3","L4","L5"
    y = df.loc[:, ["quality"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: Steel
    dataset_name = "Steel_industry_data"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/new/" + dataset_name + ".csv", sep=',')
    print("load data from ::", base_dir + "realdata/new/" + dataset_name + ".csv")
    df = df.dropna(axis=0)
    X_num = df.loc[:,
            ["Usage_kWh", "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
             "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "NSM"]].values
    X_cate = df.loc[:, []].values
    # X_cate = df.loc[:, ["WeekStatus", "Day_of_week", "Load_Type"]].values
    y = df.loc[:, ["CO2(tCO2)"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: Air quality UCI
    dataset_name = "AirQualityUCI"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/new/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/new/" + dataset_name + ".csv")
    df = df.dropna(axis=1, how='all').dropna(axis=0)
    X_num = df.loc[:,
            ["CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)",
             "PT08.S5(O3)", "T", "RH", "AH"]].values
    X_cate = df.loc[:, []].values
    # X_cate = df.loc[:, ["WeekStatus", "Day_of_week", "Load_Type"]].values
    y = df.loc[:, ["PT08.S4(NO2)"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: treasury
    dataset_name = "treasury"
    # dataset_name = "flat_dirichlet_generate"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name

    # Load Data from dataset
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")

    # "HoursPerWeek","TotalHours", carry ?"Age","APM"

    X_num = df.loc[:, ["1Y-CMaturityRate", "30Y-CMortgageRate", "3M-Rate-AuctionAverage", "3M-Rate-SecondaryMarket",
                       "3Y-CMaturityRate", "5Y-CMaturityRate", "bankCredit", "currency", "demandDeposits",
                       "federalFunds", "moneyStock", "checkableDeposits", "loansLeases", "savingsDeposits",
                       "tradeCurrencies"]].values
    X_cate = df.loc[:, []].values
    # "L3","L4","L5"
    y = df.loc[:, ["1MonthCDRate"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)



    #todo : CCCP
    dataset_name = "CCCP"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")

    X_num = df.loc[:, ["AT","V","AP","RH"]].values
    X_cate = df.loc[:, []].values

    print(X_num)
    # X_cate = df.loc[:, ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]].values
    # "L3","L4","L5"
    y = df.loc[:, ["PE"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo : skill craft
    dataset_name = "skill_craft"
    # dataset_name = "flat_dirichlet_generate"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name

    # Load Data from dataset
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")

    # "HoursPerWeek","TotalHours", carry ?"Age","APM"

    X_num = df.loc[:,
            ["SelectByHotkeys", "AssignToHotkeys", "UniqueHotkeys", "MinimapAttacks", "MinimapRightClicks",
             "NumberOfPACs", "GapBetweenPACs", "ActionLatency", "ActionsInPAC", "TotalMapExplored",
             "WorkersMade", "UniqueUnitsMade", "ComplexUnitsMade"]].values
    X_cate = df.loc[:, []].values
    # "L3","L4","L5"
    y = df.loc[:, ["ComplexAbilitiesUsed"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: airfoil
    base_dir = "../"
    dataset_name = "airfoil"
    # dataset_name = "flat_dirichlet_generate"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name

    # Load Data from dataset
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")

    # "HoursPerWeek","TotalHours", carry ?"Age","APM"

    X_num = df.loc[:, ["in1", "in2", "in3", "in4", "in5"]].values
    X_cate = df.loc[:, []].values
    # "L3","L4","L5"
    y = df.loc[:, ["out"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    #todo: abalone
    dataset_name = "abalone"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_'+ str(seed) +'/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")
    X_num = df.loc[:,
            ["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]].values
    #X_cate = df.loc[:, ["Sex"]].values
    X_cate = df.loc[:, []].values
    # "L3","L4","L5"
    y = df.loc[:, ["Class_number_of_rings"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: electrical grid
    dataset_name = "electrical_grid"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")

    X_num = df.loc[:, ["tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4"]].values
    X_cate = df.loc[:, []].values
    # "L3","L4","L5"
    y = df.loc[:, ["stab"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    #todo: wine red
    dataset_name = "wine_red"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_'+ str(seed) +'/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")
    # "HoursPerWeek","TotalHours", carry ?"Age","APM"

    X_num = df.loc[:,
            ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide",
             "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]].values
    X_cate = df.loc[:, []].values
    # "L3","L4","L5"
    y = df.loc[:, ["class"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    #todo:wind
    dataset_name = "wind"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")

    X_num = df.loc[:, ["year", "month", "day", "RPT", "VAL", "ROS", "KIL", "SHA", "BIR", "DUB", "CLA", "MUL", "CLO",
                       "BEL"]].values
    X_cate = df.loc[:, []].values
    # "L3","L4","L5"
    y = df.loc[:, ["MAL"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo: weather_wizmir
    dataset_name = "weather_wizmir"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name

    X_num = df.loc[:, ["Max_temperature", "Min_temperature", "Dewpoint", "Precipitation", "Sea_level_pressure",
                       "Standard_pressure", "Visibility", "Wind_speed", "Max_wind_speed"]].values
    X_cate = df.loc[:, []].values
    y = df.loc[:, ["Mean_temperature"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

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

    # todo : elev
    dataset_name = "elevators"
    # dataset_name = "flat_dirichlet_generate"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name

    # Load Data from dataset
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")

    X_num = df.loc[:,
            ["climbRate", "Sgz", "p", "q", "curRoll", "absRoll", "diffClb", "diffRollRate", "diffDiffClb", "SaTime1",
             "SaTime2", "SaTime3", "SaTime4", "diffSaTime1", "diffSaTime2", "diffSaTime3", "diffSaTime4", "Sa"]].values
    X_cate = df.loc[:, []].values
    y = df.loc[:, ["Goal"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)

    # todo:ailerons
    dataset_name = "ailerons"
    model_output_dir = "../model_save/realdata/" + dataset_name + "/"
    base_results_folder = results_folder = 'results_' + str(seed) + '/' + dataset_name
    df = pd.read_csv(base_dir + "realdata/" + dataset_name + ".csv")
    print("load data from ::", base_dir + "realdata/" + dataset_name + ".csv")

    X_num = df.loc[:,
            ["climbRate", "Sgz", "p", "q", "curPitch", "curRoll", "absRoll", "diffClb", "diffRollRate", "diffDiffClb",
             "SeTime1", "SeTime2", "SeTime3", "SeTime4", "SeTime5", "SeTime6", "SeTime7", "SeTime8", "SeTime9",
             "SeTime10", "SeTime11", "SeTime12", "SeTime13", "SeTime14", "diffSeTime1", "diffSeTime2", "diffSeTime3",
             "diffSeTime4", "diffSeTime5", "diffSeTime6", "diffSeTime7", "diffSeTime8", "diffSeTime9", "diffSeTime10",
             "diffSeTime11", "diffSeTime12", "diffSeTime13", "diffSeTime14", "alpha", "Se"]].values
    X_cate = df.loc[:, []].values
    y = df.loc[:, ["goal"]].values
    run_pureGam_gami_kfold(X_num, X_cate, y, base_results_folder, seed=seed)








