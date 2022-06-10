import pandas as pd
from sklearn.model_selection import train_test_split

import os
# only for multi gpu, if cpu, please delete this line
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import time
import pickle as pkl
import torch
import torch as th
import torch.utils.data as data
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from torch_utils.dataset_util import PureGamDataset, PureGamDataset_smoothingInTraining
from torch_utils.readwrite import make_dir
from sgd_solver.pureGam import PureGam
from optimizer.AugLagSGD import Adam_AugLag
from sklearn.metrics import r2_score, mean_squared_error
import math
from sklearn.preprocessing import power_transform
from benchmarks.synthetic_data_generator import num_gen_gauss, num_gen_gauss_3rd_int, gen_3nd_order_data, gen_4th_order_data
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from gaminet import GAMINet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import synthetic_experiments.run_gami as run_gami

from experiment_metrics.metrics import metric_wrapper, rmse
from pathlib import Path

if __name__ == "__main__":
    #base_dir = "../../purendata/"
    '''base_dir = "../"
    dataset_name = "flat_dirichlet"
    #dataset_name = "flat_dirichlet_generate"
    model_output_dir = "../model_save/synthetic/" + dataset_name + "/"

    # Load Data from dataset
    df = pd.read_csv(base_dir + "synthetic/" + dataset_name + ".csv")'''

    base_dir = ""

    model_output_dir = "../model_save/synthetic/" + "order3" + "/"
    # generate  data
    N = 10000
    seed = 43
    #X0, y0 = num_gen_gauss_3rd_int(N, 4, seed = 267534)

    base_results_folder = results_folder = 'results_order3/'





    '''X0, y0 = gen_4th_order_data(N, seed=42)
    df = pd.DataFrame(np.c_[X0, y0], columns=["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10",  "y"])
    X_num = df.loc[:, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"]].values'''

    '''X0, y0 = gen_3nd_order_data(N, seed=42)
    df = pd.DataFrame(np.c_[X0, y0], columns=["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8",  "y"])
    X_num = df.loc[:, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values'''


    X0, y0 = num_gen_gauss_3rd_int(N, 4, seed = 267534)
    df = pd.DataFrame(np.c_[X0, y0], columns=["X1", "X2", "X3", "X4", "y"])
    X_num = df.loc[:, ["X1", "X2", "X3", "X4"]].values
    X_cate = df.loc[:, []].values
    y = df.loc[:, ['y']].values

    # PowerTransform
    X_num = power_transform(X_num)
    y = power_transform(y)

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


    def run_gami(X_num, X_cate, y, results_folder, h_map):
        Path(results_folder).mkdir(parents=True, exist_ok=True)



        X_num, X_num_test, X_cate, X_cate_test, y_train, y_test = \
            train_test_split(X_num, X_cate, y, test_size=0.2, random_state=seed)
        #print("yTest!", y_test)


        model = GAMINet(meta_info=meta_info, interact_num=X_num.shape[1],
                # X_num.shape[1]*X_num.shape[1]-1/2, # X_num.shape[1], #
                interact_arch=[40] * 5, subnet_arch=[40] * 5,
                batch_size=256, task_type=task_type, activation_func=tf.nn.relu,
                main_effect_epochs=int(200 * math.sqrt(1e4 / X_num.shape[0])),
                interaction_epochs=int(200 * math.sqrt(1e4 / X_num.shape[0])),
                tuning_epochs=int(100 * math.sqrt(1e4 / X_num.shape[0])),

                lr_bp=[0.0001, 0.0001, 0.0001],
                early_stop_thres=[int(30 * math.sqrt(1e4 / X_num.shape[0])), int(30 * math.sqrt(1e4 / X_num.shape[0])),
                                  int(20 * math.sqrt(1e4 / X_num.shape[0]))],
                heredity=True, loss_threshold=0.01, reg_clarity=1,
                mono_increasing_list=[], mono_decreasing_list=[],  ## the indices list of features
                verbose=True, val_ratio=0.2, random_state=int(math.sqrt(seed)) + 1)  # seed*2)


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
            model.save(results_folder + '/', "model_best")
        # model.plot_numerical(X_num_test)


        t0 = time.time()
        y_hat_test = model.predict(X_num_test)[:, 0]
        t1 = time.time()
        test_time = t1 - t0
        y_hat_val = model.predict(X_num[model.val_idx, :])[:, 0]
        # y_hat_train = model.predict(X_num[model.tr_idx, :])[:, 0]
        y_hat_train = model.predict(X_num)[:, 0]

        y_train = y_train[:, 0]
        y_test = y_test[:, 0]

        r2_train = r2_score(y_train, y_hat_train)
        r2_test = r2_score(y_test, y_hat_test)
        mse_train = mean_squared_error(y_train, y_hat_train)
        mse_test = mean_squared_error(y_test, y_hat_test)

        print(len(y_train), len(y_test))
        print(len(y_hat_train), len(y_hat_test))
        """
        pureGAM pureness
        """
        '''pure_score1, pure_score2 = score_gami2(model, X_num_test, h_map,
                                               bandwidths=np.array([0.1, 0.01]), epsilon=0,
                                               N_subset=None, save_folder=results_folder)
        # print(pure_score_pureGAM1)
        enum_pure_score = pure_score1.loc['avg_log'].loc['lam1'] + pure_score2.loc['avg_log'].loc[
            'lam1'] / 2

        acc_df = pd.DataFrame(
            [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time], [0, enum_pure_score]],
            index=["r2", "mse", "time", "log_pure_score"], columns=["train", "test"])
        acc_df.to_csv(os.path.join(results_folder, "gami_accuracy.csv"))'''

        acc_df = pd.DataFrame(
            [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time]],
            index=["r2", "mse", "time"], columns=["train", "test"])
        print(acc_df)

    _ = run_gami(X_num, X_cate, y, results_folder=base_results_folder+'/' + 'Gami', h_map= None)
    y = y[:, 0]
    # train_test split
    X_num, X_num_test, X_cate, X_cate_test, y_train, y_test = \
        train_test_split(X_num, X_cate, y, test_size=0.2, random_state=seed)

    #run_gami.run(X_num, y, X_cate_test, y_test, "res/", int_num=10)

    ebm = ExplainableBoostingRegressor(interactions=10)
    '''ebm.fit(X_num, y)
    yt_ebm = ebm.predict(X_num)
    r2_ebmt = r2_score(y, yt_ebm)
    yh_ebm = ebm.predict(X_num_test)
    r2_ebm = r2_score(y_test, yh_ebm)
    acc_df = pd.DataFrame([r2_ebmt, r2_ebm], index=["train", "test"], columns=["r2"])
    print(acc_df)'''
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
    print(acc_df)

    xgbm = xgb.XGBRegressor(n_estimators = 2000, learning_rate = 0.05, max_depth = 7, min_child_weight=5)
    #xgbm = xgb.XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=15, min_child_weight=10)
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
    print(acc_df)

    '''task_type = "Regression"
    p = X_num.shape[1]
    meta_info = {"X" + str(i + 1): {'type': 'continuous'} for i in range(p)}
    meta_info.update({'Y': {'type': 'target'}})
    # transform data
    sy = None
    for i, (key, item) in enumerate(meta_info.items()):
        if item['type'] == 'target':
            sy = MinMaxScaler((0, 1))
            y = sy.fit_transform(y)
            y_test = sy.transform(y_test)
            meta_info[key]['scaler'] = sy
        else:
            sx = MinMaxScaler((0, 1))
            X_num[:, [i]] = sx.fit_transform(X_num[:, [i]])
            X_num_test[:, [i]] = sx.transform(X_num_test[:, [i]])
            meta_info[key]['scaler'] = sx
    # get_metric = metric_wrapper(rmse, sy)
    # x, xt, y, yt = train_test_split(x, y)

    #y = y[:, 0]
    #y_test = y_test[:, 0]
    print("!!Data ", X_num.shape, y)

    random_state = 42
    ## Had to turn off the heredity, for fairness of comparison?
    GAMInet_model = GAMINet(meta_info=meta_info, interact_num=10,
                            interact_arch=[40] * 5, subnet_arch=[40] * 5,
                            batch_size=256, task_type=task_type, activation_func=tf.nn.relu,
                            main_effect_epochs=500, interaction_epochs=500, tuning_epochs=500,
                            lr_bp=[0.0001, 0.0001, 0.0001], early_stop_thres=[50, 50, 50],
                            heredity=True, loss_threshold=0.01, reg_clarity=1,
                            verbose=True, val_ratio=0.2, random_state=random_state)

    print(X_num.shape, y.shape)
    GAMInet_model.fit(X_num, y)
    y_hat = GAMInet_model.predict(X_num).reshape([-1, 1])
    y_hat_test = GAMInet_model.predict(X_num_test).reshape([-1, 1])
    pred_train = sy.inverse_transform(y_hat)
    pred_test = sy.inverse_transform(y_hat_test)
    r2_gamitrain = r2_score(y, y_hat)
    r2_gami = r2_score(y_test, y_hat_test)

    acc_df = pd.DataFrame([r2_gamitrain, r2_gami], index=["train", "test"], columns=["r2"])
    print(acc_df)'''