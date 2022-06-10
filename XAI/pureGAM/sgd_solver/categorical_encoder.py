"""
July 12 2021
@ Ziyu Wang
"""

import numpy as np
## for the moment temprorarily use the old pure encoder univariate to be compatible with the explain functions.
#from torch_utils.pure_encoder import pure_encoder_univariate, pure_encoder_bivariate
from pure_coding.pure_encoder import pure_encoder_bivariate, pure_encoder_univariate
from sgd_solver.utils import generate_pairwise_idxes
from sklearn.preprocessing import LabelEncoder


class Categorical_encoder:
    def __init__(self, feature_p, univariate_ids=None, bivariate_ids=None):
        self.p_ = feature_p
        self.univariate_ids_ = univariate_ids
        if univariate_ids is None:  # Did not specify features to learn with, by default uses all.
            self.univariate_ids_ = np.arange(self.p_)
        self.bivariate_ids_ = bivariate_ids
        if bivariate_ids is None:  # Did not specify features to learn with, by default u
            self.bivariate_ids_ = np.array(generate_pairwise_idxes(self.p_))
        self.univariate_encoders_ = []
        self.bivariate_encoders_ = []

        self.univariate_idx_list = []
        self.bivariate_idx_list = []

        self.cardinality_univ = 0
        self.cardinality_biv = 0

        self.is_fit_ = False

    def fit(self, X):
        pass

    def fit_transform(self, X):
        # clear all the params gained in last fit
        self.univariate_encoders_ = []
        self.bivariate_encoders_ = []
        self.univariate_idx_list = []
        self.bivariate_idx_list = []

        univariate_encoded_data_ = []
        bivariate_encoded_data_ = []

        for id_univ in self.univariate_ids_:
            enc = pure_encoder_univariate()
            dat = enc.fit_transform(X[:, id_univ])
            self.univariate_encoders_.append(enc)
            univariate_encoded_data_.append(dat)
        for id_biv in self.bivariate_ids_:
            enc = pure_encoder_bivariate()
            dat = enc.fit_transform(X[:, id_biv[0]], X[:, id_biv[1]])
            self.bivariate_encoders_.append(enc)
            bivariate_encoded_data_.append(dat)
        # todo: triv enc
        '''for id_triv in self.trivariate_ids_:
            enc = pure_encoder_trivariate()
            dat = enc.fit_transform(X[:, id_triv[0]], X[:, id_triv[1]], X[:, id_triv[2])
            self.trivariate_encoders_.append(enc)
            trivariate_encoded_data_.append(dat)'''
        self.separate_coef()
        self.is_fit_ = True
        if len(univariate_encoded_data_ + bivariate_encoded_data_) > 0:
            return np.concatenate(univariate_encoded_data_ + bivariate_encoded_data_, axis=1)
        else:
            return np.zeros((X.shape[0],0))

    def transform(self, X):
        assert self.is_fit_, "The encoder hasn't been fit yet!"
        univariate_encoded_data_ = []
        bivariate_encoded_data_ = []
        # Do pure encoding.
        for i, id_univ in enumerate(self.univariate_ids_):
            enc = self.univariate_encoders_[i]
            dat = enc.transform(X[:, id_univ])
            univariate_encoded_data_.append(dat)
        for i, id_biv in enumerate(self.bivariate_ids_):
            enc = self.bivariate_encoders_[i]
            dat = enc.transform(X[:, id_biv[0]], X[:, id_biv[1]])
            bivariate_encoded_data_.append(dat)

        # todo: triv enc
        '''trivariate_encoded_data_ = []
        for i, id_triv in enumerate(self.trivariate_ids_):
            enc = self.trivariate_encoders_[i]
            dat = enc.transform(X[:, id_biv[0]], X[:, id_biv[1]])
            trivariate_encoded_data_.append(dat)'''
        if len(univariate_encoded_data_ + bivariate_encoded_data_) > 0:
            return np.concatenate(univariate_encoded_data_ + bivariate_encoded_data_, axis=1)
        else:
            return np.zeros((X.shape[0],0))
            
    def separate_coef(self):
        """
        Must call inside self.fit_transform()
        """
        starting_id = 0
        for i, _ in enumerate(self.univariate_ids_):
            enc_number = self.univariate_encoders_[i].get_encoded_number()
            # add start and end idx of univ in encoded columns
            self.univariate_idx_list.append((starting_id, starting_id + enc_number))
            self.cardinality_univ += enc_number
            starting_id += enc_number
        for i, _ in enumerate(self.bivariate_ids_):
            enc_number = self.bivariate_encoders_[i].get_encoded_number()
            # add start and end idx of biv in encoded columns
            self.bivariate_idx_list.append((starting_id, starting_id + enc_number))
            self.cardinality_biv += enc_number
            starting_id += enc_number

    def explain(self, cate_coef, feature_names=None):
        explanation = dict()
        for i, id_univ in enumerate(self.univariate_ids_):
            start_idx, end_idx = self.univariate_idx_list[i]
            if feature_names is None:
                feature_names = [str(i) for i in self.univariate_ids_]
            else:
                assert len(feature_names) == len(self.univariate_ids_), "feature_name inconsistent with univariate_ids!"
            explanation[feature_names[id_univ]] = self.univariate_encoders_[i]\
                .explain(cate_coef[start_idx: end_idx])
        for i, id_biv in enumerate(self.bivariate_ids_):
            start_idx, end_idx = self.bivariate_idx_list[i]
            explanation["({}, {})".format(feature_names[id_biv[0]], feature_names[id_biv[1]])] = \
            self.bivariate_encoders_[i].explain(cate_coef[start_idx: end_idx])
        return explanation