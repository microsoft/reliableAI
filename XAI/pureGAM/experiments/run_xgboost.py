import os
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
import time

# not calculating pureness of xgb.
def run_pw_int(train_x, train_y, test_x, test_y, results_folder): # default int_num
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    pw_ints = [[j, i] for i in range(train_x.shape[1]) for j in range(i)]
    interaction_constraints = str(pw_ints)
    xgbm = xgb.XGBRegressor(interaction_constraints=interaction_constraints)

    t0 = time.time()
    xgbm.fit(train_x, train_y)
    t1 = time.time()
    train_time = t1-t0

    t0 = time.time()
    yh_xgbm = xgbm.predict(test_x)
    t1 = time.time()
    test_time = t1-t0
    yt_xgbmt = xgbm.predict(train_x)

    mse_train = mean_squared_error(train_y, yt_xgbmt)
    mse_test = mean_squared_error(train_y, yh_xgbm)
    r2_xgbmt = r2_score(train_y, yt_xgbmt)
    r2_xgbm = r2_score(test_y, yh_xgbm)

    acc_df = pd.DataFrame([[r2_xgbmt, r2_xgbm], [mse_train, mse_test], [train_time, test_time]],
                          index=["r2", "mse", "time"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))

def run_max_int(train_x, train_y, test_x, test_y, results_folder): # default int_num
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    xgbm = xgb.XGBRegressor()

    t0 = time.time()
    xgbm.fit(train_x, train_y)
    t1 = time.time()
    train_time = t1-t0

    t0 = time.time()
    yh_xgbm = xgbm.predict(test_x)
    t1 = time.time()
    test_time = t1-t0
    yt_xgbmt = xgbm.predict(train_x)


    mse_train = mean_squared_error(train_y, yt_xgbmt)
    mse_test = mean_squared_error(test_y, yh_xgbm)
    r2_xgbmt = r2_score(train_y, yt_xgbmt)
    r2_xgbm = r2_score(test_y, yh_xgbm)

    acc_df = pd.DataFrame([[r2_xgbmt, r2_xgbm], [mse_train, mse_test], [train_time, test_time]],
                          index=["r2", "mse", "time"], columns=["train", "test"])
    acc_df.to_csv(os.path.join(results_folder, "accuracy.csv"))
