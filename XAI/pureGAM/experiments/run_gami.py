# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
from pathlib import Path
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import time
from metrics.metrics_true import true_pureness_score_gaussian_pureGAM, true_pureness_score_gaussian_gami
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve, accuracy_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler, OrdinalEncoder

# GAMI-NET
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split

from gaminet import GAMINet
from gaminet.utils import local_visualize
from gaminet.utils import global_visualize_density
from gaminet.utils import feature_importance_visualize
from gaminet.utils import plot_trajectory
from gaminet.utils import plot_regularization

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# def metric_wrapper(metric, scaler):
#     def wrapper(label, pred):
#         return metric(label, pred, scaler=scaler)
#     return wrapper

# def rmse(label, pred, scaler):
#     pred = scaler.inverse_transform(pred.reshape([-1, 1]))
#     label = scaler.inverse_transform(label.reshape([-1, 1]))
#     return np.sqrt(np.mean((pred - label)**2))

from metrics.metrics import predict_int_GAMI
from metrics.metrics_cate import score_gami_cat
from metrics.metrics_torch import score_pureness
import math
import torch as th
import traceback
## default int_num of gami is 20:
## https://github.com/xingzhis/gaminet/blob/1816efd49c81fe8f797bbb49b1537b4993d172c2/gaminet/gaminet.py#L14
#todo: revisit
def run(train_x, train_y, test_x, test_y, cov_mat, results_folder, h_map, int_num=20, heredity=False):
    """
    GAMI-Net
    """

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    Path(results_folder).mkdir(parents=True, exist_ok=True)
    x, y = train_x, train_y.reshape(-1, 1)
    xtest, ytest = test_x, test_y.reshape(-1, 1)
    task_type = "Regression"
    p = train_x.shape[1]
    meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(p)}
    meta_info.update({'Y':{'type':'target'}})  
    # transform data
    """
    Here we assume the input data has already been through minmaxscaler, and sx and sy are indentity maps.
    (for the syntax of GAMI-Net.)
    """

    sy = None
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            sy = MinMaxScaler((0, 1))
            y = sy.fit_transform(y)
            ytest = sy.transform(ytest)
            meta_info[key]['scaler'] = sy
        else:
            sx = MinMaxScaler((0, 1))
            x[:,[i]] = sx.fit_transform(x[:,[i]])
            xtest[:,[i]] = sx.transform(xtest[:,[i]])
            meta_info[key]['scaler'] = sx

    # get_metric = metric_wrapper(rmse, sy)
    # x, xt, y, yt = train_test_split(x, y)

    random_state = 42
    ## Had to turn off the heredity, for fairness of comparison?
    '''GAMInet_model = GAMINet(meta_info=meta_info, interact_num=int_num, 
                    interact_arch=[40] * 5, subnet_arch=[40] * 5, 
                    batch_size=200, task_type=task_type, activation_func=tf.nn.relu, 
                    main_effect_epochs=500, interaction_epochs=500, tuning_epochs=500,
                    lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                    heredity=heredity, loss_threshold=0.01, reg_clarity=1,
                    verbose=True, val_ratio=0.2, random_state=random_state)'''

    GAMInet_model = GAMINet(meta_info=meta_info, interact_num=train_x.shape[1],
                    # X_num.shape[1]*X_num.shape[1]-1/2, # X_num.shape[1], #
                    interact_arch=[40] * 5, subnet_arch=[40] * 5,
                    batch_size=200, task_type=task_type, activation_func=tf.nn.relu,
                    main_effect_epochs=int(10 * 500 * math.sqrt(1e4 / y.shape[0])),
                    interaction_epochs=int(10 * 500 * math.sqrt(1e4 / y.shape[0])),
                    tuning_epochs=int(500 * math.sqrt(1e4 / y.shape[0])),

                    lr_bp=[0.0001, 0.0001, 0.0001],
                    early_stop_thres=[int(50 * math.sqrt(1e4 / y.shape[0])),
                                      int(50 * math.sqrt(1e4 / y.shape[0])),
                                      int(50 * math.sqrt(1e4 / y.shape[0]))],
                    heredity=True, loss_threshold=0.01, reg_clarity=1,
                    mono_increasing_list=[], mono_decreasing_list=[],  ## the indices list of features
                    verbose=True, val_ratio=0.2, random_state=int(math.sqrt(random_state)) + 1)


    try:
        train_time = 0
        GAMInet_model.load(results_folder+ '/', "model_best")
    except:
        t0 = time.time()
        GAMInet_model.fit(x, y)
        t1 = time.time()
        train_time = t1-t0
        GAMInet_model.save(results_folder + '/', "model_best")

    pred_train = sy.inverse_transform(GAMInet_model.predict(x))
    t0 = time.time()
    pred_test = sy.inverse_transform(GAMInet_model.predict(xtest))
    t1 = time.time()
    test_time = t1-t0

    print(train_y)
    print(pred_train)
    print(test_y)
    print(pred_test)
    r2_train = r2_score(train_y, pred_train)
    r2_test = r2_score(test_y, pred_test)
    mse_train = mean_squared_error(train_y, pred_train)
    mse_test = mean_squared_error(test_y, pred_test)

    gami_int_pred = predict_int_GAMI(GAMInet_model, x)

    try:
        gami_int_pred = th.tensor(gami_int_pred, dtype=th.double).squeeze(2).to(device)

        pure_score1, pure_score2 = score_pureness(gami_int_pred, X_num=th.tensor(x).to(device),
                                                  interaction_list=GAMInet_model.interaction_list,
                                                  h_map=h_map, bandwidths=np.array([0.1]),
                                                  save_folder=results_folder)
        enum_pure_score = (pure_score1.loc['avg_log'].loc['lam1'] + pure_score2.loc['avg_log'].loc['lam1']) / 2
    except:
        traceback.print_exc()
        enum_pure_score = 0
    """
    GAMI pureness
    """

    true_pure_df = true_pureness_score_gaussian_gami(model=GAMInet_model, cov_mat=cov_mat, num_sigmas=3, N=200, normalize=True, epsilon=0, save_folder=results_folder)
    #todo: true_pure_df = true_pureness_score_gaussian_gami(model=GAMInet_model, cov_mat=cov_mat, num_sigmas=4, N=200, sy=syy, normalize=True, epsilon=0, save_folder=results_folder)
    true_pure_df = np.log10(true_pure_df.astype('float')).dropna()
    print(true_pure_df)
    enum_true_score = true_pure_df.mean().mean()

    acc_df = pd.DataFrame(
        [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time], [0, enum_pure_score], [0, enum_true_score]],
        index=["r2", "mse", "time", "log_pure_score", "true_pure_score"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    print(acc_df)

    # GAMInet_model.save(folder="./synth_gami_net_model")

    # val_x = train_x[GAMInet_model.val_idx, :]
    # val_y = train_y[GAMInet_model.val_idx, :]
    # tr_x = train_x[GAMInet_model.tr_idx, :]
    # tr_y = train_y[GAMInet_model.tr_idx, :]
    # pred_train = GAMInet_model.predict(tr_x)
    # pred_val = GAMInet_model.predict(val_x)
    # pred_test = GAMInet_model.predict(test_x)
    # gaminet_stat = np.hstack([np.round(get_metric(tr_y, pred_train),5), 
    #                       np.round(get_metric(val_y, pred_val),5),
    #                       np.round(get_metric(test_y, pred_test),5)])
    # print(gaminet_stat)

    # print(r2_score(tr_y, pred_train))
    # print(r2_score(val_y, pred_val))
    # print(r2_score(test_y, pred_test))

    # r2_gami = r2_score(test_y, pred_test)

    # simu_dir = results_folder
    # if not os.path.exists(simu_dir):
    #     os.makedirs(simu_dir)
    # data_dict_logs = GAMInet_model.summary_logs(save_dict=False)
    # plot_trajectory(data_dict_logs, folder=simu_dir, name="s1_traj_plot", log_scale=True, save_png=True)
    # plot_regularization(data_dict_logs, folder=simu_dir, name="s1_regu_plot", log_scale=True, save_png=True)
    # data_dict = GAMInet_model.global_explain(save_dict=False)
    # global_visualize_density(data_dict, save_png=True, folder=simu_dir, name='s1_global')


def run_cat(train_x, train_y, test_x, test_y, results_folder, int_num=20, heredity=False):
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    p = train_x.shape[1]
    x, y = train_x, train_y.reshape(-1, 1)
    xtest, ytest = test_x, test_y.reshape(-1, 1)
    task_type = "Regression"
    meta_info = {"X" + str(i + 1):{'type':'categorical'} for i in range(p)}
    meta_info.update({'Y':{'type':'target'}})  
    # for i, (key, item) in enumerate(meta_info.items()):
    #     if item['type'] == 'target':
    #         sy = MinMaxScaler((0, 1))
    #         y = sy.fit_transform(y)
    #         meta_info[key]['scaler'] = sy
    #     else:
    #         sx = MinMaxScaler((0, 1))
    #         sx.fit([[0], [1]])
    #         x[:,[i]] = sx.transform(x[:,[i]])
    #         meta_info[key]['scaler'] = sx
    xx = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
    for i, (key, item) in enumerate(meta_info.items()):
        # if item['type'] == 'target':
        #     enc = OrdinalEncoder()
        #     enc.fit(y)
        #     y = enc.transform(y)
        #     meta_info[key]['values'] = enc.categories_[0].tolist()
        if item['type'] == 'target':
            sy = MinMaxScaler((0, 1))
            y = sy.fit_transform(y)
            meta_info[key]['scaler'] = sy
        elif item['type'] == 'categorical':
            enc = OrdinalEncoder()
            xx[:,[i]] = enc.fit_transform(x[:,[i]])
            meta_info[key]['values'] = []
            for item in enc.categories_[0].tolist():
                try:
                    if item == int(item):
                        meta_info[key]['values'].append(str(int(item)))
                    else:
                        meta_info[key]['values'].append(str(item))
                except ValueError:
                    meta_info[key]['values'].append(str(item))
        else:
            sx = MinMaxScaler((0, 1))
            xx[:,[i]] = sx.fit_transform(x[:,[i]])
            meta_info[key]['scaler'] = sx

    random_state = 42
    ## Had to turn off the heredity otherwise it cannot learn anything for our data.
    ##  Our data has mostly interactions and almost no main effects.
    GAMInet_model = GAMINet(meta_info=meta_info, interact_num=int_num, 
                    interact_arch=[40] * 5, subnet_arch=[40] * 5,
                    batch_size=200, task_type=task_type, activation_func=tf.nn.relu, 
                    main_effect_epochs=5000, interaction_epochs=5000, tuning_epochs=500,
                    lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                    heredity=heredity, loss_threshold=0.01, reg_clarity=0.1,
                    mono_increasing_list=[], mono_decreasing_list=[], ## the indices list of features
                    verbose=True, val_ratio=0.2, random_state=random_state)

    if path.exists(path.join(results_folder, "model_best")):
        train_time = 0
        GAMInet_model.load(results_folder+ '/', "model_best")
    else:
        t0 = time.time()
        GAMInet_model.fit(x, y)
        t1 = time.time()
        train_time = t1 - t0
        GAMInet_model.save(results_folder+ '/', "model_best")
        """
          File "/Users/xingzhi/Projects/temp_repos/pureGAM-experiment/synthetic_experiments/run_gami.py", line 293, in run_cat
            GAMInet_model.save(results_folder+ '/', "model_best")
          File "/Users/xingzhi/anaconda3/lib/python3.8/site-packages/gaminet/gaminet.py", line 888, in save
            model_dict["active_indice"] = self.active_indice
          AttributeError: 'GAMINet' object has no attribute 'active_indice'
        """


    pred_train = sy.inverse_transform(GAMInet_model.predict(x))
    t0 = time.time()
    pred_test = sy.inverse_transform(GAMInet_model.predict(xtest))
    t1 = time.time()
    test_time = t1 - t0

    r2_gamitrain = r2_score(train_y, pred_train)
    r2_gami = r2_score(test_y, pred_test)
    mse_train = mean_squared_error(train_y, pred_train)
    mse_test = mean_squared_error(test_y, pred_test)

    if len(GAMInet_model.interaction_list) == 0:
        with open(os.path.join(results_folder, "no_interaction.log"), "w") as f:
            f.write("No interactions learned.")
        enum_true_score = 0
    else:
        pure_score_df = score_gami_cat(GAMInet_model, train_x, N_subset=None, save_folder=results_folder)
        pure_score_df = np.log10(pure_score_df.astype('float')).dropna()
        print(pure_score_df)
        enum_true_score = pure_score_df.mean().mean()

    acc_df = pd.DataFrame([[r2_gamitrain, r2_gami], [mse_train, mse_test], [train_time, test_time], [0, enum_true_score]],
                          index=["r2", "mse", "time", "true_pure_score"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    print(acc_df)
    """
    GAMI pureness
    """
