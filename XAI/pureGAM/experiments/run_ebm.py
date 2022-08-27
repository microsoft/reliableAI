import os
from pathlib import Path
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from metrics.metrics_cate import score_ebm_cat
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
    mse_test = mean_squared_error(test_y, yh_ebm)
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
    mse_test = mean_squared_error(test_y, yh_ebm)
    r2_ebmt = r2_score(train_y, yt_ebm)
    r2_ebm = r2_score(test_y, yh_ebm)
    acc_df = pd.DataFrame([[r2_ebmt, r2_ebm], [mse_train, mse_test], [train_time, test_time]], index=["r2", "mse", "time"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
    """
    EBM Pureness
    """
    score_ebm_cat(ebm, test_x, syy, N_subset=None, save_folder=results_folder)
