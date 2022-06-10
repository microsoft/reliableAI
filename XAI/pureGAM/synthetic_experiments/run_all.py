import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer,  MinMaxScaler
from pathlib import Path
import pandas as pd
import run_pureGAM2
import run_gami2
import run_ebm
import run_xgboost

import torch
from sgd_solver.pureGam import PureGam
from sklearn.preprocessing import  power_transform

from numerical_gen import generate_data, get_names, generate_data_test,get_names_test, get_names1, get_names2, get_names3

class IdentityTransform:
    def __init__(self):
        pass
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

'''def process_gami_info(X_num, y):
    # PowerTransform

    #X_num = power_transform(X_num)
    # y = power_transform(y)
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
            #sx.fit([[0], [1]])
            #print(i, (key, item))
            #X_num[:, i] = sx.transform(X_num[:, i])
            #X_num.iloc[:, [i]] = sx.fit_transform(X_num.iloc[:, [i]])
            meta_info[key]['scaler'] = sx
    print(X_num.shape)
    return task_type, meta_info, X_num, y'''

def main(seed):
    int_num_gami = 40
    int_num_ebm = 40
    t0 = time.time()

    #names = get_names()
    #names = get_names_test()
    names = get_names2()

    ## DEBUG
    # names = ["debug"]
    ## END DEBUG
    print("")
    for name in names:
        t1 = time.time()
        print("Running on dataset {}:".format(name))
        fpath = os.path.join("data3/", name)
        #fpath = os.path.join("data/", name)
        rpath = os.path.join("results3/", name)
        X, y, cov = pd.read_csv(os.path.join(fpath, "X.csv")), pd.read_csv(os.path.join(fpath, "y.csv")), pd.read_csv(os.path.join(fpath, "cov.csv"))
        cov = np.asarray(cov)

        #task_type, meta_info, X, y = process_gami_info(X, y)
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed)
        ## DEBUG
        # train_x, test_x, train_y, test_y = X, X, y, y
        ## ENG DEBUG
        train_x, test_x, train_y, test_y = np.asarray(train_x), np.asarray(test_x), np.asarray(train_y), np.asarray(test_y)
        # pt = PowerTransformer(method='yeo-johnson', standardize=True)
        pt = IdentityTransform() # on this data MinMaxScaler is way better than PowerTransformer.
        # pt = pt.fit(train_x)
        train_x = pt.transform(train_x)
        test_x = pt.transform(test_x)
        # pty = PowerTransformer(method='yeo-johnson', standardize=True)
        pty = MinMaxScaler()
        pty = pty.fit(train_y)
        train_y = pty.transform(train_y)
        test_y = pty.transform(test_y)

        t6 = time.time()
        print("Running EBM...")
        run_ebm.run(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, sxx=pt, h_map=None,
                    syy=pty, cov_mat=cov, results_folder=os.path.join(rpath, "ebm"), int_num=int_num_ebm)
        t7 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t7 - t6),
                                                                                                  np.round(t7 - t1),
                                                                                                  np.round(t7 - t0)))

        '''print("Running pureGAM...")
        adaptive_kernel = run_pureGAM2.run(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, cov_mat=cov, results_folder=os.path.join(rpath, "pureGAM2"))
        t2 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t2 - t1), np.round(t2 - t1), np.round(t2 - t0)))'''

        '''print("Running pureGAM main effects only...")
        run_pureGAM.run_main_effects(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, results_folder=os.path.join(rpath, "pureGAM_main"))
        t3 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t3 - t2), np.round(t3 - t1), np.round(t3 - t0)))'''

        '''t3 = time.time()
        print("Running GAMI-Net...")
        #run_gami.run(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, sxx=pt, syy=pty, h_map=None, cov_mat=cov, results_folder=os.path.join(rpath, "gami"), int_num=int_num_gami, heredity=False)

        run_gami2.run(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, h_map=None, cov_mat=cov, results_folder=os.path.join(rpath, "gami2"), int_num=int_num_gami, heredity=False)
        t4 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t4 - t3), np.round(t4 - t1), np.round(t4 - t0)))'''


        # load lam
        '''device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        N_param_scale = 2
        pureGAM_model = PureGam(train_x.shape[1], 0, N_param_univ = N_param_scale*512, N_param_biv = N_param_scale*512, init_kw_lam_univ = 0.15, init_kw_lam_biv = 0.15,
                isInteraction=True, model_output_dir = "tmp/", device=device,
                bias = y.mean(), isPurenessConstraint=True, isAugLag=False, isInnerBatchPureLoss=False, isLossEnhanceForDenseArea=False, dropout_rate=0, pure_lam_scale=1)
        pureGAM_model.load_model(os.path.join(rpath, "pureGAM") + "/model_best")
        adaptive_kernel = pureGAM_model.num_enc.lam_univ.detach().cpu().numpy().tolist()'''

        '''print("Running GAMI-Net main effects only...")
        # run_gami.run_main_effects(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, results_folder=os.path.join(rpath, "gami_main"))
        t6 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t6 - t5), np.round(t6 - t1), np.round(t6 - t0)))'''
        '''t6 = time.time()

        
        print("Running EBM main effects only...")
        # run_ebm.run_main_effects(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, results_folder=os.path.join(rpath, "ebm_main"))
        t8 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t8 - t7), np.round(t8 - t1), np.round(t8 - t0)))
        print("Running XGBoost...")
        run_xgboost.run_max_int(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, results_folder=os.path.join(rpath, "xgboost_max_int"))'''

        # t8 = time.time()
        # print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t8 - t7), np.round(t8 - t1), np.round(t8 - t0)))
        # print("Running XGBoost pairwise interactions...")
        # run_xgboost.run_pw_int(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, results_folder=os.path.join(rpath, "xgboost_pw_int"))
        # t9 = time.time()
        # print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t9 - t8), np.round(t9 - t1), np.round(t9 - t0)))
        # print("Running XGboost main effects...")
        # run_xgboost.run_main_effects(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, results_folder=os.path.join(rpath, "xgboost_main"))
        # t10 = time.time()
        # print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t10 - t9), np.round(t10 - t1), np.round(t10 - t0)))
        # print("")
if __name__ == '__main__':
    seed = 3453218
    main(seed)