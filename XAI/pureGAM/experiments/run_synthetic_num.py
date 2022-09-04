import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import run_pureGAM
import run_gami
import run_ebm
from synthetic_datagenerator.numerical_gen import get_names


class IdentityTransform:
    def __init__(self):
        pass
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

def main(seed):
    int_num_gami = 40
    int_num_ebm = 40
    t0 = time.time()

    #names = get_names()
    #names = get_names_test()
    names = get_names()

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
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t7 - t6),np.round(t7 - t1),np.round(t7 - t0)))

        print("Running pureGAM...")
        adaptive_kernel = run_pureGAM.run(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, cov_mat=cov, results_folder=os.path.join(rpath, "pureGAM2"))
        t2 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t2 - t1), np.round(t2 - t1), np.round(t2 - t0)))

        t3 = time.time()
        print("Running GAMI-Net...")
        run_gami.run(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, h_map=None, cov_mat=cov, results_folder=os.path.join(rpath, "gami2"), int_num=int_num_gami, heredity=False)
        t4 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t4 - t3), np.round(t4 - t1), np.round(t4 - t0)))


if __name__ == '__main__':
    seed = 3453218
    main(seed)