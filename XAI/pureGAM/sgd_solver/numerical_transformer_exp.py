import numpy as np
import torch.autograd

'''from pure_smoother.kernel_smoother_tensor import Kernel_Weigther
from pure_smoother.kernel_smoother_tensor import K_Epanechnikov as K_E
from pure_smoother.kernel_smoother_tensor import K_Gaussian as K_G
from pure_smoother.kernel_smoother_tensor import K_tri_cube as K_t'''
from torch_utils.kernel_smoother import compute_pairwise_distance
from torch_utils.kernel_smoother import K_Epanechnikov as K_E
from torch_utils.kernel_smoother import K_Gaussian as K_G
from torch_utils.kernel_smoother import K_tri_cube as K_t
from sgd_solver.utils import generate_pairwise_idxes
from torch_utils.synthetic import numerical_generator
import torch as th
import torch.nn as nn
'''
class Numerical_transformer:
    def __init__(self, feature_p, univariate_ids=None, bivariate_ids=None,
                 kw_univ: Kernel_Weigther = Kernel_Weigther(0.1, K_G),
                 kw_biv: Kernel_Weigther = Kernel_Weigther(0.1, K_G)):
        self.p_ = feature_p
        self.univariate_ids_ = univariate_ids
        if univariate_ids is None:
            self.univariate_ids_ = np.arange(self.p_)
        self.bivariate_ids_ = bivariate_ids
        if bivariate_ids is None:
            self.bivariate_ids_ = np.array(generate_pairwise_idxes(self.p_))

        self.kw_univ = kw_univ
        self.kw_biv = kw_biv

        self.X_memo = None
        self.is_fit_ = False

    def fit(self, X):
        if isinstance(X, th.Tensor):
            self.X_memo = X.clone().detach()
        else:
            self.X_memo = th.tensor(X)
        self.is_fit_ = True

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, is_norm=True):
        assert self.is_fit_, "The encoder hasn't been fit yet!"
        univariate_encoded_data_ = []
        bivariate_encoded_data_ = []
        for id_univ in self.univariate_ids_:
            S_univ = self.kw_univ.S(self.X_memo[:, [id_univ]], X[:, [id_univ]], is_norm=is_norm)
            univariate_encoded_data_.append(S_univ)
        for id_biv in self.bivariate_ids_:
            S_biv = self.kw_biv.S(self.X_memo[:, [id_biv[0], id_biv[1]]], X[:, [id_biv[0], id_biv[1]]], is_norm=is_norm)
            bivariate_encoded_data_.append(S_biv)
        #return np.array(univariate_encoded_data_ + bivariate_encoded_data_)
        if len(univariate_encoded_data_ + bivariate_encoded_data_) > 0:
            return th.stack(univariate_encoded_data_ + bivariate_encoded_data_, dim=0)
        else:
            return th.zeros((X.shape[0],0))'''

class Numerical_transformer_adapt(nn.Module):
    def __init__(self, feature_p, num_points_univ, num_points_biv, num_points_triv,
                 univariate_ids=None, bivariate_ids=None, trivariate_ids=None,
                 default_lam_univ=0.1, default_lam_biv=0.1, default_lam_triv=0.1):
        super(Numerical_transformer_adapt, self).__init__()
        self.p_ = feature_p
        self.num_points_univ = num_points_univ
        self.num_points_biv = num_points_biv
        self.num_points_triv = num_points_triv

        self.univariate_ids_ = univariate_ids
        if univariate_ids is None:
            self.univariate_ids_ = np.arange(feature_p)
        self.bivariate_ids_ = bivariate_ids
        if bivariate_ids is None:
            self.bivariate_ids_ = np.array(generate_pairwise_idxes(feature_p))
        self.trivariate_ids_ = trivariate_ids
        if trivariate_ids is None:
            self.trivariate_ids_ = []

        self.lam_w1_univ = nn.Parameter(th.ones([len(self.univariate_ids_)], dtype=th.double)*default_lam_univ)
        self.lam_w1_biv = nn.Parameter(th.ones([len(self.bivariate_ids_)], dtype=th.double)*default_lam_biv)
        self.lam_w1_triv = nn.Parameter(th.ones([len(self.trivariate_ids_)], dtype=th.double)*default_lam_triv)

        self.lam_w2_univ = nn.Parameter(th.zeros([len(self.univariate_ids_)], dtype=th.double))
        self.lam_w2_biv = nn.Parameter(th.zeros([len(self.bivariate_ids_)], dtype=th.double))
        self.lam_w2_triv = nn.Parameter(th.zeros([len(self.trivariate_ids_)], dtype=th.double))

        self.lam_b_univ = nn.Parameter(th.zeros([len(self.univariate_ids_)], dtype=th.double))
        self.lam_b_biv = nn.Parameter(th.zeros([len(self.bivariate_ids_)], dtype=th.double))
        self.lam_b_triv = nn.Parameter(th.zeros([len(self.trivariate_ids_)], dtype=th.double))
        self.K_transform_func = K_G

        self.eps = 1e-12
        self.X_memo_univ = nn.Parameter(th.zeros([num_points_univ, feature_p], dtype=th.double))
        self.X_memo_biv = nn.Parameter(th.zeros([num_points_biv, feature_p], dtype=th.double))
        self.X_memo_triv = nn.Parameter(th.zeros([num_points_triv, feature_p], dtype=th.double))
        self.is_fit_ = True

    def init_param_points(self, init_points_univ, init_points_biv, init_points_triv):
        if init_points_univ is not None:
            self.X_memo_univ = nn.Parameter(init_points_univ+self.eps)
        if init_points_biv is not None:
            self.X_memo_biv = nn.Parameter(init_points_biv + self.eps)
        if init_points_triv is not None:
            self.X_memo_triv = nn.Parameter(init_points_triv + self.eps)

    def transform(self, X, is_norm=True, lam_scale = 1):
        assert self.is_fit_, "The encoder hasn't been fit yet!"
        univariate_encoded_data_ = []
        bivariate_encoded_data_ = []
        trivariate_encoded_data_ = []

        lam_univ = self.lam_w1_univ*th.exp(self.lam_w2_univ)+self.lam_b_univ
        lam_biv = self.lam_w1_biv*th.exp(self.lam_w2_biv)+self.lam_b_biv
        lam_triv = self.lam_w1_triv*th.exp(self.lam_w2_triv)+self.lam_b_triv

        for i, id_univ in enumerate(self.univariate_ids_):
            #S_univ = self.kw_univ.S(self.X_memo_univ[:, [id_univ]], X[:, [id_univ]], is_norm=is_norm, lam=self.lam_univ[i])
            X_diff_divlam = compute_pairwise_distance(self.X_memo_univ[:, [id_univ]], X[:, [id_univ]]) / (th.square(lam_scale * lam_univ[i]) + self.eps) # fix for square of bandwidth
            kij = self.K_transform_func(X_diff_divlam)
            if is_norm:
                kij = kij / kij.sum(axis=-1, keepdim=True)
            univariate_encoded_data_.append(kij)
        for i, id_biv in enumerate(self.bivariate_ids_):
            #print("what", self.lam_biv[i])
            X_diff_divlam = compute_pairwise_distance(self.X_memo_biv[:, [id_biv[0], id_biv[1]]], X[:, [id_biv[0], id_biv[1]]]) / (th.square(lam_scale * lam_biv[i]) + self.eps) # fix for square of bandwidth
            #X_diff_divlam = compute_pairwise_distance_2nd(self.X_memo_biv[:, id_biv[0]], self.X_memo_biv[:, id_biv[1]], X[:, id_biv[0]] , X[:, id_biv[1]]) / self.lam_biv[i]
            kij = self.K_transform_func(X_diff_divlam)
            if is_norm:
                kij = kij / kij.sum(axis=-1, keepdim=True)
            bivariate_encoded_data_.append(kij)
        # triple
        for i, id_triv in enumerate(self.trivariate_ids_):
            X_diff_divlam = compute_pairwise_distance(self.X_memo_triv[:, [id_triv[0], id_triv[1], id_triv[2]]], X[:, [id_triv[0], id_triv[1], id_triv[2]]]) / (th.square(lam_scale * lam_triv[i]) + self.eps) # fix for square of bandwidth
            kij = self.K_transform_func(X_diff_divlam)
            if is_norm:
                kij = kij / kij.sum(axis=-1, keepdim=True)
            trivariate_encoded_data_.append(kij)


        #return np.array(univariate_encoded_data_ + bivariate_encoded_data_)
        univariate_encoded_data = th.stack(univariate_encoded_data_, dim=0) if \
            len(univariate_encoded_data_) > 0 else th.zeros((0, X.shape[0], self.num_points_univ), device=X.device)
        bivariate_encoded_data = th.stack(bivariate_encoded_data_, dim=0) if \
            len(bivariate_encoded_data_) > 0 else th.zeros((0, X.shape[0], self.num_points_biv), device=X.device)
        trivariate_encoded_data = th.stack(trivariate_encoded_data_, dim=0) if \
            len(trivariate_encoded_data_) > 0 else th.zeros((0, X.shape[0], self.num_points_triv), device=X.device)

        '''if th.isnan(univariate_encoded_data).any():
            print("haha da")
            print(univariate_encoded_data[-4])
            print(univariate_encoded_data[-2])'''

        return univariate_encoded_data, bivariate_encoded_data, trivariate_encoded_data

    # only to compute smoothing matrix between X and X
    def transform_inner(self, X, lam_scale=1):
        assert self.is_fit_, "The encoder hasn't been fit yet!"
        univariate_encoded_data_ = []
        bivariate_encoded_data_ = []
        trivariate_encoded_data_ = []

        lam_univ = self.lam_w1_univ*th.exp(self.lam_w2_univ)+self.lam_b_univ
        lam_biv = self.lam_w1_biv*th.exp(self.lam_w2_biv)+self.lam_b_biv
        lam_triv = self.lam_w1_triv*th.exp(self.lam_w2_triv)+self.lam_b_triv

        for i, id_univ in enumerate(self.univariate_ids_):
            X_diff_divlam = compute_pairwise_distance(X[:, [id_univ]], X[:, [id_univ]]) / \
                            (th.square(lam_scale * lam_univ[i]) + self.eps) # fix for square of bandwidth
            kij = self.K_transform_func(X_diff_divlam)
            univariate_encoded_data_.append(kij)
        for i, id_biv in enumerate(self.bivariate_ids_):
            X_diff_divlam = compute_pairwise_distance(X[:, [id_biv[0], id_biv[1]]], X[:, [id_biv[0], id_biv[1]]] ) / \
                            (th.square(lam_scale * lam_biv[i]) + self.eps) # fix for square of bandwidth
            kij = self.K_transform_func(X_diff_divlam)
            bivariate_encoded_data_.append(kij)
        # triple
        for i, id_triv in enumerate(self.trivariate_ids_):
            X_diff_divlam = compute_pairwise_distance(X[:, [id_triv[0], id_triv[1], id_triv[2]]], X[:, [id_triv[0], id_triv[1], id_triv[2]]]) \
                            / (th.square(lam_scale * lam_triv[i]) + self.eps) # fix for square of bandwidth
            kij = self.K_transform_func(X_diff_divlam)
            trivariate_encoded_data_.append(kij)

        #return np.array(univariate_encoded_data_ + bivariate_encoded_data_)
        univariate_encoded_data = th.stack(univariate_encoded_data_, dim=0) if \
            len(univariate_encoded_data_) > 0 else th.zeros((0, X.shape[0], self.num_points_univ), device=X.device)
        bivariate_encoded_data = th.stack(bivariate_encoded_data_, dim=0) if \
            len(bivariate_encoded_data_) > 0 else th.zeros((0, X.shape[0], self.num_points_biv), device=X.device)
        trivariate_encoded_data = th.stack(trivariate_encoded_data_, dim=0) if \
            len(trivariate_encoded_data_) > 0 else th.zeros((0, X.shape[0], self.num_points_triv), device=X.device)
        return univariate_encoded_data, bivariate_encoded_data, trivariate_encoded_data

    def get_lam(self):
        with th.no_grad():
            lam_univ = self.lam_w1_univ*th.exp(self.lam_w2_univ)+self.lam_b_univ
            lam_biv = self.lam_w1_biv*th.exp(self.lam_w2_biv)+self.lam_b_biv
            lam_triv = self.lam_w1_triv*th.exp(self.lam_w2_triv)+self.lam_b_triv
        return lam_univ, lam_biv, lam_triv

    '''def transform_seperate(self, X, is_norm=True):
        assert self.is_fit_, "The encoder hasn't been fit yet!"
        univariate_encoded_data_ = []
        bivariate_encoded_data_ = []
        for i, id_univ in enumerate(self.univariate_ids_):
            S_univ = self.kw_univ.S(self.X_memo_univ[:, [id_univ]], X[:, [id_univ]], is_norm=is_norm, lam=self.lam_univ[i])
            univariate_encoded_data_.append(S_univ)
        for i, id_biv in enumerate(self.bivariate_ids_):
            S_biv = self.kw_biv.S(self.X_memo_biv[:, [id_biv[0], id_biv[1]]], X[:, [id_biv[0], id_biv[1]]], is_norm=is_norm, lam=self.lam_biv[i])
            bivariate_encoded_data_.append(S_biv)
        #return np.array(univariate_encoded_data_ + bivariate_encoded_data_)
        univariate_encoded_data = th.stack(univariate_encoded_data_, dim=0) if \
            len(univariate_encoded_data_) > 0 else th.zeros((0, X.shape[0], self.num_points_univ), device=X.device)
        bivariate_encoded_data = th.stack(bivariate_encoded_data_, dim=0) if \
            len(bivariate_encoded_data_) > 0 else th.zeros((0, X.shape[0], self.num_points_biv), device=X.device)
        return univariate_encoded_data, bivariate_encoded_data'''

if __name__ == "__main__":
    N, p = 1000, 3
    X, y = numerical_generator(N)
    X_test, y_test = numerical_generator(1000)
    encoder = Numerical_transformer(p)
    S_train = encoder.fit_transform(X)
    S_test = encoder.transform(X_test)
    print(S_train.shape)
    print(S_test.shape)