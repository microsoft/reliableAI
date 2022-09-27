# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
July 9 2021
@ Xingzhi Sun
Provides endoder for pure encoding:
main effects, interaction.
"""
import numpy as np

class pure_encoder_univariate:
    def __init__(self):
        self.unique_levels_ = None
        self.encoding_weights_ = None
        self.is_fit_ = False
    def fit(self, col):
        col_ = np.asarray(col)
        self.unique_levels_ = np.unique(col_)
        raw_oh = (col_[:,None] == self.unique_levels_).astype(int) # The raw one-hot encoding matrix.
        self.encoding_weights_ = (np.einsum("ij->j", raw_oh[:,:-1]) / raw_oh[:,-1].sum()).reshape(1, -1)
        self.is_fit_ = True
    def fit_transform(self, col):
        col_ = np.asarray(col)
        self.unique_levels_ = np.unique(col_)
        raw_oh = (col_[:,None] == self.unique_levels_).astype(int) # The raw one-hot encoding matrix.
        self.encoding_weights_ = (np.einsum("ij->j", raw_oh[:,:-1]) / raw_oh[:,-1].sum()).reshape(1, -1)
        encoded_mat = (raw_oh[:,:-1]
            - raw_oh[:,-1].reshape(-1,1)
            @ self.encoding_weights_
        )
        self.is_fit_ = True
        return encoded_mat
    def transform(self, col):
        assert self.is_fit_, "The encoder hasn't been fit yet!"
        col_ = np.asarray(col)
        raw_oh = (col_[:,None] == self.unique_levels_).astype(int) # The raw one-hot encoding matrix.
        encoded_mat = (raw_oh[:,:-1] - raw_oh[:,-1].reshape(-1,1) @ self.encoding_weights_ )
        return encoded_mat
    def explain(self, coefficients):
        coef_ = np.asarray(coefficients)
        assert coef_.shape == (self.unique_levels_.shape[0] - 1,), "Cannot match shape of coefficients with shape of unique levels!"
        last_coef = - np.dot(coef_, self.encoding_weights_[0])
        coef_all = (np.r_[coef_, last_coef])
        explain_axis = self.unique_levels_
        return coef_all, explain_axis # Returns the values and corresponding axis.
        # return (np.r_[self.unique_levels_.reshape(1, -1), coef_all.reshape(1, -1)]).T # returns a 2D array, of which each row is a level and its value.
    def get_encoded_number(self):
        """
        Gets the number of columns of the encoded matrix.

        Returns:
            int: the number of columns of the encoded matrix.
        """        
        return self.unique_levels_.shape[0] - 1

class pure_encoder_bivariate:
    def __init__(self):
        self.eps_ = 1e-8 # Numerical zero.
        self.unique_levels1_ = None
        self.unique_levels2_ = None
        self.n_levels1_ = None
        self.n_levels2_ = None
        self.cmbn_freq_ = None
        self.encoding_weights_ = None
        self.is_fit_ = False
    def fit(self, col1, col2):
        col1_, col2_ = np.asarray(col1), np.asarray(col2)
        assert col1_.shape == col2_.shape, "The two column have different shapes!"
        self.unique_levels1_, self.unique_levels2_ = np.unique(col1_), np.unique(col2_)
        self.n_levels1_, self.n_levels2_ = self.unique_levels1_.shape[0], self.unique_levels2_.shape[0]
        # Get raw one-hot encoding.
        raw_oh1 = (col1_[:,None] == self.unique_levels1_).astype(int)
        raw_oh2 = (col2_[:,None] == self.unique_levels2_).astype(int)
        # Get counts and frequencies.
        cmbn_counts = raw_oh1.T @ raw_oh2
        n = col1_.shape[0]
        self.cmbn_freq_ = cmbn_counts / n
        # Get product array for faster computation using np.einsum().
        cmbn_prod = np.einsum("ij,ik->jki", raw_oh1, raw_oh2)
        # Encode when full-grid -- each combination has at least one data point.
        if (self.cmbn_freq_ > self.eps_).all():
            self.encoding_weights_ = self.get_weights_full_grid()
        else:
            raise NotImplementedError("Currently only supports full-grid data!")
            # TODO for non-full-grid case, after ruling out the non-grid-closed case,
            #      solve equation for a basis of weights.
        self.is_fit_ = True
    
    def fit_transform(self, col1, col2):
        col1_, col2_ = np.asarray(col1), np.asarray(col2)
        assert col1_.shape == col2_.shape, "The two column have different shapes!"
        self.unique_levels1_, self.unique_levels2_ = np.unique(col1_), np.unique(col2_)
        self.n_levels1_, self.n_levels2_ = self.unique_levels1_.shape[0], self.unique_levels2_.shape[0]
        # Get raw one-hot encoding.
        raw_oh1 = (col1_[:,None] == self.unique_levels1_).astype(int)
        raw_oh2 = (col2_[:,None] == self.unique_levels2_).astype(int)
        # Get counts and frequencies.
        cmbn_counts = raw_oh1.T @ raw_oh2
        n = col1_.shape[0]
        self.cmbn_freq_ = cmbn_counts / n
        # Get product array for faster computation using np.einsum().
        cmbn_prod = np.einsum("ij,ik->jki", raw_oh1, raw_oh2)
        # Encode when full-grid -- each combination has at least one data point.
        if (self.cmbn_freq_ > self.eps_).all():
            self.encoding_weights_ = self.get_weights_full_grid()
            encoded_mat_raw_shape = self.pure_encode_helper(cmbn_prod)
        else:
            raise NotImplementedError("Currently only supports full-grid data!\n{}\n{}".format(cmbn_counts, self.cmbn_freq_))
            # TODO for non-full-grid case, after ruling out the non-grid-closed case,
            #      solve equation for a basis of weights.
        self.is_fit_ = True
        return encoded_mat_raw_shape.reshape((self.n_levels1_ - 1) * (self.n_levels2_ - 1), n).T # reshape to proper shape.

    def transform(self, col1, col2):
        assert self.is_fit_, "The encoder hasn't been fit yet!"
        col1_, col2_ = np.asarray(col1), np.asarray(col2)
        assert col1_.shape == col2_.shape, "The two column have different shapes!"
        # Get raw one-hot encoding.
        raw_oh1 = (col1_[:,None] == self.unique_levels1_).astype(int)
        raw_oh2 = (col2_[:,None] == self.unique_levels2_).astype(int)
        n = col1_.shape[0]
        # Get product array for faster computation using np.einsum().
        cmbn_prod = np.einsum("ij,ik->jki", raw_oh1, raw_oh2)
        # Encode when full-grid -- each combination has at least one data point.
        encoded_mat_raw_shape = self.pure_encode_helper(cmbn_prod)
        return encoded_mat_raw_shape.reshape((self.n_levels1_ - 1) * (self.n_levels2_ - 1), n).T # reshape to proper shape.

    def explain(self, coefficients):
        assert coefficients.shape == ((self.n_levels1_ - 1) * (self.n_levels2_ - 1),), "Coefficient shape mismatch with encoding!"
        coef_r = coefficients.reshape((self.n_levels1_ - 1), (self.n_levels2_ - 1))
        explain_value = np.einsum("st,stjk->jk", coef_r, self.encoding_weights_)
        explain_axis_1 = self.unique_levels1_
        explain_axis_2 = self.unique_levels2_
        return explain_value, explain_axis_1, explain_axis_2 # Returns the values and corresponding axes.
                                                             # explain_axis_1 is the row index, and explain_axis_2 is the column names.

    def get_weights_full_grid(self):
        """
        Gets encoding weights for full-grid data.
        The weights has shape: (n_levels1 - 1, n_levels2 - 1, n_levels1, n_levels2),
        The first two dimensions constitute a double index of the basis,
        and each 2Darray element in the (n_levels1 - 1, n_levels2 - 1) "matrix" specifies weights
        to multiply on each column of the 2-feature-combination feature for each basis element.
        """        
        cmbn_signs = np.zeros((self.n_levels1_ - 1, self.n_levels2_ - 1, self.n_levels1_, self.n_levels2_)) # Initiallize summation sign matrix.
        for i in range(self.n_levels1_ - 1):
            for j in range(self.n_levels2_ - 1):
                i1, j1 = i + 1, j + 1 # Position of the right bottom element.
                cmbn_signs[i, j, 0, 0] = 1.
                cmbn_signs[i, j, i1, 0] = -1.
                cmbn_signs[i, j, 0, j1] = -1.
                cmbn_signs[i, j, i1, j1] = 1.
        cmbn_freq_reciprocal = 1 / self.cmbn_freq_
        weights = np.einsum("stjk,jk->stjk", cmbn_signs, cmbn_freq_reciprocal)
        return weights

    def pure_encode_helper(self, cmbn_prod):
        """
        Does pure encoding given weights.
        For full-grid data, weights are defined by the reciprocal of cmbn_freq.
        For non-full-grid data, weights are solved from a linear equation.
        """        
        encoded_mat = np.einsum("jki,stjk->sti", cmbn_prod, self.encoding_weights_)
        return encoded_mat

    def get_encoded_number(self):
        """
        Gets the number of columns of the encoded matrix.

        Returns:
            int: the number of columns of the encoded matrix.
        """   
        return (self.n_levels1_ - 1) * (self.n_levels2_ - 1)
