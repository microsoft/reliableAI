import os
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import run_pureGAM
import run_gami

from synthetic_datagenerator.categorical_gen import get_name_test


def main(seed):
    int_num_gami = 40
    int_num_ebm = 40
    t0 = time.time()
    #TODO
    #names = get_name_new()
    names = get_name_test()
    ## DEBUG
    # names = ["debug"]
    ## END DEBUG
    print("")
    for name in names:
        print("@@@ ", name,"@@@ ")
        t1 = time.time()
        print("Running on dataset {}:".format(name))
        #TODO
        #fpath = os.path.join("cat_new_data/", name.split('_')[1])
        fpath = os.path.join("cat_data/", name)

        rpath = os.path.join("cat_results_test_test5/", name)

        X, y = pd.read_csv(os.path.join(fpath, "X.csv")), pd.read_csv(os.path.join(fpath, "y.csv"))
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed)
        ## DEBUG
        # train_x, test_x, train_y, test_y = X, X, y, y
        ## ENG DEBUG
        train_x, test_x, train_y, test_y = np.asarray(train_x), np.asarray(test_x), np.asarray(train_y), np.asarray(test_y)
        # pt = PowerTransformer(method='yeo-johnson', standardize=True)
        # pt = MinMaxScaler() # on this data MinMaxScaler is way better than PowerTransformer.
        # pt = pt.fit(train_x)
        # train_x = pt.transform(train_x)
        # test_x = pt.transform(test_x)
        # pty = PowerTransformer(method='yeo-johnson', standardize=True)
        pty = MinMaxScaler()
        pty = pty.fit(train_y)

        avg_cardi = int(name.split('_')[-1])

        train_y = pty.transform(train_y)
        test_y = pty.transform(test_y)

        #TODO
        print("Running pureGAM...")
        run_pureGAM.run_cat(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, results_folder=os.path.join(rpath, "pureGAM2"), avg_cardi=avg_cardi)
        t2 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t2 - t1), np.round(t2 - t1), np.round(t2 - t0)))
        t3 = time.time()
        print("Running GAMI-Net...")
        run_gami.run_cat(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
                         results_folder=os.path.join(rpath, "gami2"), int_num=int_num_gami, heredity=True)
        t4 = time.time()
        print("Finished. Time used: {}. Time elapsed in this dataset: {}. Total time: {}.".format(np.round(t4 - t3), np.round(t4 - t1), np.round(t4 - t0)))

if __name__ == '__main__':
    seed = 3453218
    main(seed)