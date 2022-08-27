import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from synthetic_data_generator import gen_cat_data, gen_cat_y

def generate_data(seed=24536475):

    names = []
    out_folder = ("cat_data/")

    # Ns = [10000]
    # p1 = 5
    # c1 = 3

    # N1 = 1000
    # ps = []
    # c1 = 2
    
    # N1 = 10000
    # p1 = 5
    # cs = []

    # Ns = [1000, 10000]
    # p1 = 10
    # c1 = 5

    # N1 = 1000
    # ps = [5, 10]
    # c1 = 2
    
    # N1 = 10000
    # p1 = 5
    # cs = [2, 5]
    t0 = time.time()
    """
    Cannot have too large cardinality or dimension, 
    or full-grid is impossible with this amount of data, 
    or the features would be highly correlated!
    N > 2*c^2*p
    """

    '''
    Ns = [10000, 20000, 40000]#, 60000, 80000, 100000] #[3000, 4000]#]
    p1 = 5
    c1 = 5

    for N in Ns:
        t1 = time.time()
        p = p1
        c = c1
        cards = [c] * p
        Path(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c))).mkdir(parents=True, exist_ok=True)
        filename_X = "{}_{}_{}/X.csv".format(N, p, c)
        filename_y = "{}_{}_{}/y.csv".format(N, p, c)
        X = gen_cat_data(N, cards)
        y, int_df = gen_cat_y(X, scale_univ=1, scale_biv=1e-2, noise_scale=0)
        int_df.to_csv(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c) + 'Interaction.csv'))
        xpath = os.path.join(out_folder, filename_X)
        ypath = os.path.join(out_folder, filename_y)
        pd.DataFrame(X).to_csv(xpath, index=None)
        pd.DataFrame(y).to_csv(ypath, index=None)
        names.append("{}_{}_{}".format(N, p, c))
        t2 = time.time()
        print("Data {}x{}x{} generated. Time used: {}. Total time: {}.".format(N, p, c, np.round(t2 - t1), np.round(t2 - t0)))'''

    N1 = 20000
    ps = [2]  # [10, 20] , 6, 7
    c1 = 5

    for p in ps:
        t1 = time.time()
        N = N1
        c = c1
        cards = [c] * p
        Path(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c))).mkdir(parents=True, exist_ok=True)
        filename_X = "{}_{}_{}/X.csv".format(N, p, c)
        filename_y = "{}_{}_{}/y.csv".format(N, p, c)
        X = gen_cat_data(N, cards)
        y, int_df = gen_cat_y(X, scale_univ=1, scale_biv=1e-2, noise_scale=0)
        int_df.to_csv(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c) + 'Interaction.csv'))
        xpath = os.path.join(out_folder, filename_X)
        ypath = os.path.join(out_folder, filename_y)
        pd.DataFrame(X).to_csv(xpath, index=None)
        pd.DataFrame(y).to_csv(ypath, index=None)
        names.append("{}_{}_{}".format(N, p, c))
        t2 = time.time()
        print("Data {}x{}x{} generated. Time used: {}. Total time: {}.".format(N, p, c, np.round(t2 - t1), np.round(t2 - t0)))

    N1 = 20000
    p1 = 5
    cs = [2]  # , 8] #, 5, 6, 7] , 6, 7
    for c in cs:
        t1 = time.time()
        N = N1
        p = p1
        cards = [c] * p
        Path(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c))).mkdir(parents=True, exist_ok=True)
        filename_X = "{}_{}_{}/X.csv".format(N, p, c)
        filename_y = "{}_{}_{}/y.csv".format(N, p, c)
        X = gen_cat_data(N, cards)
        y, int_df = gen_cat_y(X, scale_univ=1, scale_biv=1e-2, noise_scale=0)
        int_df.to_csv(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c) + 'Interaction.csv'))
        xpath = os.path.join(out_folder, filename_X)
        ypath = os.path.join(out_folder, filename_y)
        pd.DataFrame(X).to_csv(xpath, index=None)
        pd.DataFrame(y).to_csv(ypath, index=None)
        names.append("{}_{}_{}".format(N, p, c))
        t2 = time.time()
        print("Data {}x{}x{} generated. Time used: {}. Total time: {}.".format(N, p, c, np.round(t2 - t1), np.round(t2 - t0)))
    return names


def generate_data_test(seed=24536475):
    names = []
    out_folder = ("cat_data/")

    # Ns = [10000]
    # p1 = 5
    # c1 = 3

    # N1 = 1000
    # ps = []
    # c1 = 2

    # N1 = 10000
    # p1 = 5
    # cs = []

    # Ns = [1000, 10000]
    # p1 = 10
    # c1 = 5

    # N1 = 1000
    # ps = [5, 10]
    # c1 = 2

    # N1 = 10000
    # p1 = 5
    # cs = [2, 5]

    """
    Cannot have too large cardinality or dimension, 
    or full-grid is impossible with this amount of data, 
    or the features would be highly correlated!
    N > 2*c^2*p
    """

    t0 = time.time()
    Ns = [40000, 60000]  # , 60000, 80000, 100000] #[3000, 4000]#]
    p1 = 5
    c1 = 5

    for N in Ns:
        t1 = time.time()
        p = p1
        c = c1
        cards = [c] * p
        Path(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c))).mkdir(parents=True, exist_ok=True)
        filename_X = "{}_{}_{}/X.csv".format(N, p, c)
        filename_y = "{}_{}_{}/y.csv".format(N, p, c)
        X = gen_cat_data(N, cards)
        y, int_df = gen_cat_y(X, scale_univ=1, scale_biv=1e-2, noise_scale=0)
        int_df.to_csv(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c) + 'Interaction.csv'))
        xpath = os.path.join(out_folder, filename_X)
        ypath = os.path.join(out_folder, filename_y)
        pd.DataFrame(X).to_csv(xpath, index=None)
        pd.DataFrame(y).to_csv(ypath, index=None)
        names.append("{}_{}_{}".format(N, p, c))
        t2 = time.time()
        print("Data {}x{}x{} generated. Time used: {}. Total time: {}.".format(N, p, c, np.round(t2 - t1),
                                                                               np.round(t2 - t0)))

    '''N1 = 20000
    ps = [2, 3, 4, 5 ,6]  # [10, 20] , 6, 7
    c1 = 3

    for p in ps:
        t1 = time.time()
        N = N1
        c = c1
        cards = [c] * p
        Path(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c))).mkdir(parents=True, exist_ok=True)
        filename_X = "{}_{}_{}/X.csv".format(N, p, c)
        filename_y = "{}_{}_{}/y.csv".format(N, p, c)
        X = gen_cat_data(N, cards)
        y, int_df = gen_cat_y(X, scale_univ=1, scale_biv=1e-2, noise_scale=0)
        int_df.to_csv(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c) + 'Interaction.csv'))
        xpath = os.path.join(out_folder, filename_X)
        ypath = os.path.join(out_folder, filename_y)
        pd.DataFrame(X).to_csv(xpath, index=None)
        pd.DataFrame(y).to_csv(ypath, index=None)
        names.append("{}_{}_{}".format(N, p, c))
        t2 = time.time()
        print("Data {}x{}x{} generated. Time used: {}. Total time: {}.".format(N, p, c, np.round(t2 - t1),
                                                                               np.round(t2 - t0)))

    N1 = 20000
    p1 = 3
    cs = [2, 3, 4 ,5 ,6]  # , 8] #, 5, 6, 7] , 6, 7
    for c in cs:
        t1 = time.time()
        N = N1
        p = p1
        cards = [c] * p
        Path(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c))).mkdir(parents=True, exist_ok=True)
        filename_X = "{}_{}_{}/X.csv".format(N, p, c)
        filename_y = "{}_{}_{}/y.csv".format(N, p, c)
        X = gen_cat_data(N, cards)
        y, int_df = gen_cat_y(X, scale_univ=1, scale_biv=1e-2, noise_scale=0)
        int_df.to_csv(os.path.join(out_folder, "{}_{}_{}/".format(N, p, c) + 'Interaction.csv'))
        xpath = os.path.join(out_folder, filename_X)
        ypath = os.path.join(out_folder, filename_y)
        pd.DataFrame(X).to_csv(xpath, index=None)
        pd.DataFrame(y).to_csv(ypath, index=None)
        names.append("{}_{}_{}".format(N, p, c))
        t2 = time.time()
        print("Data {}x{}x{} generated. Time used: {}. Total time: {}.".format(N, p, c, np.round(t2 - t1),
                                                                               np.round(t2 - t0)))'''
    return names

def generate_name():

    names = []

    '''Ns = [10000, 20000, 40000, 60000, 80000]
    p1 = 5
    c1 = 5

    for N in Ns:
        p = p1
        c = c1
        names.append("{}_{}_{}".format(N, p, c))'''

    N1 = 20000
    ps = [2, 3, 4, 5, 6]#[2, 3, 4, 5, 6]
    c1 = 5
    for p in ps:
        N = N1
        c = c1
        names.append("{}_{}_{}".format(N, p, c))

    N1 = 20000
    p1 = 5
    cs = [2, 3, 4, 5, 6]
    for c in cs:
        N = N1
        p = p1
        names.append("{}_{}_{}".format(N, p, c))
    return names

def get_name_test():

    names = []

    Ns = [20000, 40000, 60000, 80000]
    p1 = 5
    c1 = 5

    for N in Ns:
        p = p1
        c = c1
        names.append("{}_{}_{}".format(N, p, c))

    '''N1 = 20000
    ps = [2, 3, 4, 5, 6]#[2, 3, 4, 5, 6]
    c1 = 5
    for p in ps:
        N = N1
        c = c1
        names.append("{}_{}_{}".format(N, p, c))

    N1 = 20000
    p1 = 5
    cs = [2,3,4,5, 6]
    for c in cs:
        N = N1
        p = p1
        names.append("{}_{}_{}".format(N, p, c))'''
    return names


def get_name_new():
    names = []

    N1 = 20000
    ps = [10, 15, 20]  # [2, 3, 4, 5, 6]
    c1 = 2
    for p in ps:
        N = N1
        c = c1
        names.append("{}_{}_{}".format(N, p, c))

    return names

if __name__ == "__main__":
    print("Generating data...")
    names = generate_data()