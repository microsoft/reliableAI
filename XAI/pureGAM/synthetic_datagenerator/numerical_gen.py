# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import time
from pathlib import Path
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import numpy as np
import pandas as pd
from synthetic_datagenerator.synthetic_data_generator import num_gen_gauss


def generate_data(seed = 214254):
    names = []
    out_folder = ("../experiments/data3/")
    Ns = [20000, 40000, 60000, 80000]#[10000,20000, 40000, 60000, 80000]#[2000, 3000] ##, 100000]10000,[1500,
    p1 = 10
    N1 = 20000
    ps = [20, 30, 40] #,  [3,4]
    #ps = [40]
    
    # Ns = [10000]
    # p1 = 3
    # N1 = 1000
    # ps = []

    # Ns = [10000]
    # p1 = 5
    # ps = [5]
    # N1 = 20000

    t0 = time.time()
    for N in Ns:
        t1 = time.time()
        p = p1
        Path(os.path.join(out_folder, "{}_{}/".format(N, p))).mkdir(parents=True, exist_ok=True)
        filename_X = "{}_{}/X.csv".format(N, p)
        filename_y = "{}_{}/y.csv".format(N, p)
        filename_cov = "{}_{}/cov.csv".format(N, p)
        X, y, int_mat, cov = num_gen_gauss(N, p, seed, return_cov=True)
        xpath = os.path.join(out_folder, filename_X)
        ypath = os.path.join(out_folder, filename_y)
        covpath = os.path.join(out_folder, filename_cov)
        pd.DataFrame(X).to_csv(xpath, index=None)
        pd.DataFrame(y).to_csv(ypath, index=None)
        pd.DataFrame(cov).to_csv(covpath, index=None)
        pd.DataFrame(int_mat).to_csv(os.path.join(out_folder, "{}_{}/".format(N, p) + 'Interaction.csv'), index=None)
        names.append("{}_{}".format(N, p))
        t2 = time.time()
        print("Data {}x{} generated. Time used: {}. Total time: {}.".format(N, p, np.round(t2 - t1), np.round(t2 - t0)))
    for p in ps:
        t1 = time.time()
        N = N1
        Path(os.path.join(out_folder, "{}_{}/".format(N, p))).mkdir(parents=True, exist_ok=True)
        filename_X = "{}_{}/X.csv".format(N, p)
        filename_y = "{}_{}/y.csv".format(N, p)
        filename_cov = "{}_{}/cov.csv".format(N, p)
        X, y, int_mat, cov = num_gen_gauss(N, p, seed, return_cov=True)
        xpath = os.path.join(out_folder, filename_X)
        ypath = os.path.join(out_folder, filename_y)
        covpath = os.path.join(out_folder, filename_cov)
        pd.DataFrame(X).to_csv(xpath, index=None)
        pd.DataFrame(y).to_csv(ypath, index=None)
        pd.DataFrame(cov).to_csv(covpath, index=None)
        pd.DataFrame(int_mat).to_csv(os.path.join(out_folder, "{}_{}/".format(N, p) + 'Interaction.csv'), index=None)
        names.append("{}_{}".format(N, p))
        t2 = time.time()
        print("Data {}x{} generated. Time used: {}. Total time: {}.".format(N, p, np.round(t2 - t1), np.round(t2 - t0)))
    return names

def get_names1():
    names = []
    Ns = [40000, 60000, 80000] #[2000, 3000] ##, 100000]10000,[1500,[10000,20000, 40000, 60000, 80000, 100000]
    p1 = 10
    for N in Ns:
        p = p1
        names.append("{}_{}".format(N, p))

    return names

def get_names2():
    names = []
    N1 = 20000
    ps = [40] #,  [3,4]
    for p in ps:
        N = N1
        names.append("{}_{}".format(N, p))

    return names

def get_names3():
    names = []
    Ns = [20000] #[2000, 3000] ##, 100000]10000,[1500,[10000,20000, 40000, 60000, 80000, 100000]
    p1 = 10
    for N in Ns:
        p = p1
        names.append("{}_{}".format(N, p))
    return names

def get_names():
    names = []
    Ns = [20000, 40000, 60000, 80000] #[2000, 3000] ##, 100000]10000,[1500,[10000,20000, 40000, 60000, 80000, 100000]
    p1 = 10
    N1 = 20000
    ps = [20, 30, 40] #,  [3,4]

    for N in Ns:
        p = p1
        names.append("{}_{}".format(N, p))
    for p in ps:
        N = N1
        names.append("{}_{}".format(N, p))
    return names


def generate_data_test(seed = 267534):

    names = []
    out_folder = ("data4/")
    Ns = [20000]  # [2000, 3000] ##, 100000]10000,[1500,
    p1 = 40

    t0 = time.time()
    for N in Ns:
        t1 = time.time()
        p = p1
        Path(os.path.join(out_folder, "{}_{}/".format(N, p))).mkdir(parents=True, exist_ok=True)
        filename_X = "{}_{}/X.csv".format(N, p)
        filename_y = "{}_{}/y.csv".format(N, p)
        filename_cov = "{}_{}/cov.csv".format(N, p)
        X, y, int_mat, cov = num_gen_gauss(N, p, seed, return_cov=True)
        xpath = os.path.join(out_folder, filename_X)
        ypath = os.path.join(out_folder, filename_y)
        covpath = os.path.join(out_folder, filename_cov)
        pd.DataFrame(X).to_csv(xpath, index=None)
        pd.DataFrame(y).to_csv(ypath, index=None)
        pd.DataFrame(cov).to_csv(covpath, index=None)
        pd.DataFrame(int_mat).to_csv(os.path.join(out_folder, "{}_{}/".format(N, p) + 'Interaction.csv'), index=None)
        names.append("{}_{}".format(N, p))
        t2 = time.time()
        print("Data {}x{} generated. Time used: {}. Total time: {}.".format(N, p, np.round(t2 - t1), np.round(t2 - t0)))

    return names

def get_names_test():

    names = []
    Ns = [20000]  # [2000, 3000] ##, 100000]10000,[1500,
    p1 = 10


    for N in Ns:
        p = p1
        names.append("{}_{}".format(N, p))

    return names


if __name__ == "__main__":
    generate_data()
    print("Generating data...")
