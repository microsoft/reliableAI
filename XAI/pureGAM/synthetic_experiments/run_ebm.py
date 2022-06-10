import os
from pathlib import Path
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from benchmarks.synthetic_data_generator import generate_x, function_main_effects, function_interaction, num_gen_gauss
from benchmarks.synthetic_data_generator import check_hist, check_func_main, check_func_int
from experiment_metrics.metrics import pureness_loss_est, pureness_loss_est2, pureness_score2, pureness_score2_normalized
from synthetic_experiments.run_metrics import true_pureness_score_gaussian_pureGAM, true_pureness_score_gaussian_ebm, true_pureness_score_gaussian_gami

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

from interpret.glassbox.ebm.utils import EBMUtils
from interpret.utils import unify_data

from run_metrics import predict_vec_ebm, score_ebm, score_ebm_cat
import time

## default int_num of ebm is 10:
## https://interpret.ml/docs/ebm.html
def run(train_x, train_y, test_x, test_y, sxx, syy, cov_mat, results_folder, h_map, int_num=10): # default int_num
    """
    EBM
    """
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    ebm = ExplainableBoostingRegressor(interactions=int_num)
    t0 = time.time()
    ebm.fit(train_x, train_y)
    t1 = time.time()
    train_time = t1-t0

    t0 = time.time()
    yh_ebm = ebm.predict(test_x)
    t1 = time.time()
    test_time = t1-t0
    yt_ebm = ebm.predict(train_x)

    mse_train = mean_squared_error(train_y, yt_ebm)
    mse_test = mean_squared_error(train_y, yt_ebm)
    r2_ebmt = r2_score(train_y, yt_ebm)
    r2_ebm = r2_score(test_y, yh_ebm)
    acc_df = pd.DataFrame([[r2_ebmt, r2_ebm], [mse_train, mse_test], [train_time, test_time]],
                          index=["r2", "mse", "time"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    print(acc_df)
    """
    EBM Pureness
    """
    #score_ebm(ebm, test_x, sxx, syy, bandwidths=np.array([1, 0.5, 0.1, 0.05, 0.01]), h_map=h_map, epsilon=0, N_subset=None, save_folder=results_folder)
    #true_pureness_score_gaussian_ebm(model=ebm, cov_mat=cov_mat, num_sigmas=4, N=200, sy=syy, normalize=True, epsilon=0, save_folder=results_folder)


'''def run_main_effects(train_x, train_y, test_x, test_y, results_folder):
    """
    EBM main effects
    """
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    ebm = ExplainableBoostingRegressor(interactions=0)
    ebm.fit(train_x, train_y)
    yt_ebm = ebm.predict(train_x)
    r2_ebmt = r2_score(train_y, yt_ebm)
    yh_ebm = ebm.predict(test_x)
    r2_ebm = r2_score(test_y, yh_ebm)
    acc_df = pd.DataFrame([r2_ebmt, r2_ebm], index=["train", "test"], columns=["r2"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))'''


def run_cat(train_x, train_y, test_x, test_y, syy, results_folder, int_num=10): # default int_num
    """
    EBM
    """
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    ebm = ExplainableBoostingRegressor(interactions=int_num)
    t0 = time.time()
    ebm.fit(train_x, train_y)
    t1 = time.time()
    train_time = t1-t0

    t0 = time.time()
    yh_ebm = ebm.predict(test_x)
    t1 = time.time()
    test_time = t1-t0
    yt_ebm = ebm.predict(train_x)

    mse_train = mean_squared_error(train_y, yt_ebm)
    r2_ebmt = r2_score(train_y, yt_ebm)
    r2_ebm = r2_score(test_y, yh_ebm)
    mse_test = mean_squared_error(train_y, yt_ebm)
    acc_df = pd.DataFrame([[r2_ebmt, r2_ebm], [mse_train, mse_test], [train_time, test_time]], index=["r2", "mse", "time"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    """
    EBM Pureness
    """
    score_ebm_cat(ebm, test_x, syy, N_subset=None, save_folder=results_folder)

'''def run_cat_main_effects(train_x, train_y, test_x, test_y, results_folder):
    """
    EBM main effects
    """
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    ebm = ExplainableBoostingRegressor(interactions=0)
    ebm.fit(train_x, train_y)
    yt_ebm = ebm.predict(train_x)
    r2_ebmt = r2_score(train_y, yt_ebm)
    yh_ebm = ebm.predict(test_x)
    r2_ebm = r2_score(test_y, yh_ebm)
    acc_df = pd.DataFrame([r2_ebmt, r2_ebm], index=["train", "test"], columns=["r2"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))'''