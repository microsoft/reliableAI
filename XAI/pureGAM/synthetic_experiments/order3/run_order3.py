import pandas as pd
from sklearn.model_selection import train_test_split

import os
# only for multi gpu, if cpu, please delete this line
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from benchmarks.synthetic_data_generator import num_gen_gauss, num_gen_gauss_3rd_int, gen_3nd_order_data
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
    #X0, y0 = num_gen_gauss_3rd_int(N, 4, seed = 267534)
    seed = 42


    base_results_folder = results_folder = 'results_order3_time2/'

    X0, y0 = num_gen_gauss_3rd_int(N, 4, seed = 267534)
    df = pd.DataFrame(np.c_[X0, y0], columns=["X1", "X2", "X3", "X4", "y"])
    X_num = df.loc[:, ["X1", "X2", "X3", "X4"]].values
    '''X0, y0 = gen_3nd_order_data(N, seed=42)
    df = pd.DataFrame(np.c_[X0, y0], columns=["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8",  "y"])
    X_num = df.loc[:, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]].values'''
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
    print("!!Data ", X_num.shape, y)


    def run_pureGam(X_num, X_cate, y, results_folder):
        Path(results_folder).mkdir(parents=True, exist_ok=True)
        y = y[:, 0]
        print("!!Data ", X_num.shape, y)

        # train_test split
        '''X_num, X_num_test, X_cate, X_cate_test, y_train, y_test = \
            train_test_split(X_num, X_cate, y, test_size=0.2, random_state=41)'''


        X_num, X_num_test, X_cate, X_cate_test, y_train, y_test = \
            train_test_split(X_num, X_cate, y, test_size=0.2, random_state=seed)
        #print("yTest!", y_test)

        X_num_ttrain, X_num_valid, X_cate_ttrain, X_cate_valid, y_ttrain, y_valid = \
            train_test_split(X_num, X_cate, y_train, test_size=0.2, random_state=int(math.sqrt(seed))+1 )

        _, X_num_ttrain_smp, _, X_cate_ttrain_smp, _, y_ttrain_smp = \
            train_test_split(X_num_ttrain, X_cate_ttrain, y_ttrain, test_size=0.25, random_state=seed*2+101)

        # Device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        batch_scale = 1
        N_param_scale = 0.5
        kwargs = {}
        train_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
            th.tensor(X_cate).to(device), th.tensor(X_num).to(device), th.tensor(y_train).to(device)),
            batch_size=int(batch_scale * 128), shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
            X_cate_valid, th.tensor(X_num_valid).to(device), th.tensor(y_valid).to(device)),
            batch_size=int(batch_scale * 128 * 4), shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
            th.tensor(X_cate_test).to(device), th.tensor(X_num_test).to(device), th.tensor(y_test).to(device)),
            batch_size=int(batch_scale * 128 * 2), shuffle=False, **kwargs)
        train_loader_all = torch.utils.data.DataLoader(PureGamDataset_smoothingInTraining(
            th.tensor(X_cate).to(device), th.tensor(X_num).to(device), th.tensor(y_train).to(device)),
            batch_size=int(batch_scale * 128 * 2), shuffle=False, **kwargs)

        model = PureGam(X_num.shape[1], X_cate.shape[1], N_param_univ=int(N_param_scale * 512),
                        N_param_biv=int(N_param_scale * 512),
                        # init_kw_lam_univ=0.15, init_kw_lam_biv=0.1,

                        init_kw_lam_univ=0.4, init_kw_lam_biv=0.2,
                        isInteraction=True, model_output_dir=model_output_dir, device=device,
                        bias=y_train.mean(), isPurenessConstraint=True, isAugLag=False, isInnerBatchPureLoss=False,
                        isLossEnhanceForDenseArea=True, dropout_rate=0, pure_lam_scale=1,
                        adaptiveInfoPerEpoch=2,
                        #init_kw_lam_univ=0.2, init_kw_lam_biv=0.1,
                        # order 3
                        N_param_triv=int(N_param_scale * 512), init_kw_lam_triv=0.2,
                        #trivpair_idxes_num=[],
                        trivpair_idxes_num=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
                        )
        '''model = PureGam(X_num.shape[1], X_cate.shape[1], N_param_univ=512,
                        N_param_biv=512, init_kw_lam_univ=0.2, init_kw_lam_biv=0.1,
                        isInteraction=True, model_output_dir=model_output_dir, device=device, adaptiveInfoPerEpoch=1,
                        bias=y.mean(), isPurenessConstraint=False, isAugLag=False,
                        # loss term variation
                        isInnerBatchPureLoss=False, isLossEnhanceForDenseArea=False,
                        # order 3
                        N_param_triv=512, init_kw_lam_triv=0.1,
                        # trivpair_idxes_num=[],
                        trivpair_idxes_num=[[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
                        )'''

        model.init_param_points(X_num)
        model.fit_cate_encoder(X_cate)

        # Prepare the optimizer for training, declare the learning rate of all model paramaters

        '''# Adapative Kernel learning rate
        {'params': model.num_enc.lam_univ, 'lr': 1e-2 / X_num.shape[1]},
        {'params': model.num_enc.lam_biv, 'lr': 1e-2 / X_num.shape[1] / (X_num.shape[1] - 1)},
        {'params': model.num_enc.X_memo_univ, 'lr': 0},  # 1e-2
        {'params': model.num_enc.X_memo_biv, 'lr': 0},  # 1e-2

        # order 3
        {'params': model.model_smoothing.eta_triv,
         'lr': 3 * 2 * 1e-2/ X_num.shape[1] / (X_num.shape[1] - 1) / (X_num.shape[1] - 2)},
        {'params': model.num_enc.lam_triv, 'lr': 3 * 2 * 1e-2 / X_num.shape[1] / (X_num.shape[1] - 1) / (X_num.shape[1] - 2)},
        {'params': model.num_enc.X_memo_triv, 'lr': 0}'''

        optimizer = Adam_AugLag([
            {'params': model.model_categorical.parameters(), 'lr': 0},  # 6e-3},
            {'params': model.model_smoothing.eta_univ, 'lr': 1e-2 / X_num.shape[1]},
            {'params': model.model_smoothing.eta_biv, 'lr': 1e-2 / X_num.shape[1] / (X_num.shape[1] - 1)},
            {'params': model.model_smoothing.bias, 'lr': 0},


            {'params': model.num_enc.lam_univ, 'lr': 1e-2 / X_num.shape[1]},
            {'params': model.num_enc.lam_biv, 'lr': 0.5* 1e-2 / X_num.shape[1] },
            {'params': model.num_enc.X_memo_univ, 'lr': 0},  # 1e-2
            {'params': model.num_enc.X_memo_biv, 'lr': 0},  # 1e-2

            # order 3
            {'params': model.model_smoothing.eta_triv,
             'lr': 3 * 2 * 1e-2 / X_num.shape[1] / (X_num.shape[1] - 1) / (X_num.shape[1] - 2)},
            {'params': model.num_enc.lam_triv,
             'lr':  1e-2 / X_num.shape[1] },
            {'params': model.num_enc.X_memo_triv, 'lr': 0}
        ], amsgrad=True)

        '''optimizer = Adam_AugLag([
            {'params': model.model_categorical.parameters(), 'lr': 0e-3},
            {'params': model.model_smoothing.eta_univ, 'lr': 1e-3},
            {'params': model.model_smoothing.eta_biv, 'lr': 2 * 1e-3 / (X_num.shape[1] - 1)},
            {'params': model.model_smoothing.bias, 'lr': 0e-4},

            # Adapative Kernel learning rate
            {'params': model.num_enc.lam_univ, 'lr': 1e-3},
            {'params': model.num_enc.lam_biv, 'lr': 2 * 1e-3 / (X_num.shape[1] - 1)},
            {'params': model.num_enc.X_memo_univ, 'lr': 0},  # 1e-2
            {'params': model.num_enc.X_memo_biv, 'lr': 0},  # 1e-2

            # order 3
            {'params': model.model_smoothing.eta_triv,
             'lr': 3 * 2 * 1e-3 / (X_num.shape[1] - 1) / (X_num.shape[1] - 2)},
            {'params': model.num_enc.lam_triv, 'lr': 3 * 2 * 1e-3 / (X_num.shape[1] - 1) / (X_num.shape[1] - 2)},
            {'params': model.num_enc.X_memo_triv, 'lr': 0}
        ], amsgrad=True)'''

        print("{*} Model Size", model.get_model_size())

        try:
            assert False
            train_time = 0
            model.load_model(results_folder + "/model_best")
        except:
            t0 = time.time()
            model.train(train_loader, valid_loader, optimizer=optimizer, num_epoch=int(50 * math.sqrt(1e4 / len(y))),
                        tolerent_level=int(7 * math.sqrt(1e4 / len(y))),
                        # Pureness Loss ratio of total loss
                        ratio_part2=2e2, lmd1=0e0, lmd2=0e0, lmd3=1e0, test_data_loader=test_loader)

            '''model.train(train_loader, test_loader, optimizer=optimizer,
                        num_epoch=int(8 * 1e5 / len(y)), tolerent_level=int(4 * 1e5 / len(y)),
                        ratio_part2=0, lmd1=1e0, lmd2=1e0, lmd3=5e0, )

            y_hat_test, _, _, _, _, _ = model.predict_batch(X_num_test, X_cate_test)
            print(math.sqrt(mean_squared_error(y_test, y_hat_test.cpu())))
            print(r2_score(y_test, y_hat_test.cpu()))
            model.plot_numerical(X_num_test)'''

            t1 = time.time()
            train_time = t1 - t0
            model.save_model(results_folder + "/model_best")

        h_map = model.num_enc.lam_univ.detach().cpu().numpy().tolist()

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

        print(model.num_enc.lam_univ)
        print(model.num_enc.lam_biv)
        '''pure_score1, pure_score2 = score_pureGAM2(model, X_num_test, h_map, bandwidths=np.array([0.1, 0.01]), epsilon=0,
                                                  N_subset=None, save_folder=results_folder)
        # print(pure_score_pureGAM1)
        enum_pure_score = pure_score1.loc['avg_log'].loc['lam1'] + pure_score2.loc['avg_log'].loc['lam1'] / 2

        acc_df = pd.DataFrame(
            [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time], [0, enum_pure_score]],
            index=["r2", "mse", "time", "log_pure_score"], columns=["train", "test"])'''
        acc_df = pd.DataFrame(
            [[r2_train, r2_test], [mse_train, mse_test], [train_time, test_time]],
            index=["r2", "mse", "time"], columns=["train", "test"])
        acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
        print(acc_df)
        return model

    pureGam = run_pureGam(X_num, X_cate, y, results_folder=base_results_folder+'/' + 'pureGam')







