import numpy as np
import torch as th
import torch.functional as F

class pure_encoder_univariate:
    def __init__(self):
        self.unique_levels_ = None
        self.encoding_weights_ = None
        self.is_fit_ = False

    def fit(self, col):
        #col_ = th.tensor(col)
        col_ = col
        self.unique_levels_ = np.unique(col_)

        n = col_.shape[0]
        raw_oh = th.tensor(col_[:, None] == self.unique_levels_).int() # The raw one-hot encoding matrix.
        # Get counts and frequencies.
        self.cmbn_freq_ = raw_oh.sum(axis=0)/ n
        self.is_fit_ = True

    def fit_transform(self, col):
        self.fit(col)
        return self.transform(col)

    def transform(self, col):
        assert self.is_fit_, "The encoder hasn't been fit yet!"
        #col_ = th.tensor(col)
        col_ = col
        n = col.shape[0]
        raw_oh = th.tensor(col_[:, None] == self.unique_levels_).int()  # The raw one-hot encoding matrix.

        encoded_mat = th.zeros([n, self.unique_levels_.shape[0] - 1])
        indexes_part1 = raw_oh[:, 0] == 1
        encoded_mat[indexes_part1, :] = -1 / self.cmbn_freq_[0]

        indexes_part2 = raw_oh[:, 0] != 1
        encoded_mat[indexes_part2, :] = raw_oh[indexes_part2, 1:] * 1 / self.cmbn_freq_[1:].unsqueeze(0)

        return encoded_mat

    def explain(self, coefficients):
        coef_ = th.tensor(coefficients)
        assert coef_.shape == (
            self.unique_levels_.shape[0] - 1,), "Cannot match shape of coefficients with shape of unique levels!"
        explain_value = th.zeros([self.unique_levels_.shape[0]])
        explain_value[0] = -(coef_ / self.cmbn_freq_[0]).sum()
        explain_value[1:] = coef_ / self.cmbn_freq_[1:]

        explain_axis = self.unique_levels_
        return explain_value, explain_axis  # Returns the values and corresponding axis.

    def get_encoded_number(self):
        return self.unique_levels_.shape[0] - 1


class pure_encoder_bivariate:
    def __init__(self):
        self.eps_ = 1e-8  # Numerical zero.
        self.unique_levels1_ = None
        self.unique_levels2_ = None
        self.n_levels1_ = None
        self.n_levels2_ = None
        self.cmbn_freq_ = None
        self.is_fit_ = False

    def fit(self, col1, col2):
        #col1_, col2_ = th.tensor(col1), th.tensor(col2)
        col1_, col2_ = col1, col2
        assert col1_.shape == col2_.shape, "The two column have different shapes!"
        n = col1_.shape[0]
        self.unique_levels1_, self.unique_levels2_ = np.unique(col1_), np.unique(col2_)
        self.n_levels1_, self.n_levels2_ = self.unique_levels1_.shape[0], self.unique_levels2_.shape[0]
        # Get raw one-hot encoding.
        raw_oh1 = th.tensor(col1_[:, None] == self.unique_levels1_).int()
        raw_oh2 = th.tensor(col2_[:, None] == self.unique_levels2_).int()
        # Get counts and frequencies.
        cmbn_counts = th.mm(raw_oh1.T, raw_oh2)
        self.cmbn_freq_ = cmbn_counts / n

        # Encode when full-grid -- each combination has at least one data point. else raise error
        if not ((self.cmbn_freq_ > self.eps_).all()):
            raise NotImplementedError("Currently only supports full-grid data!")
            # TODO for non-full-grid case, after ruling out the non-grid-closed case,
            #      solve equation for a basis of weights.
        self.is_fit_ = True

    def fit_transform(self, col1, col2):
        self.fit(col1, col2)
        return self.transform(col1, col2)

    def transform(self, col1, col2):
        assert self.is_fit_, "The encoder hasn't been fit yet!"
        #col1_, col2_ = th.tensor(col1), th.tensor(col2)
        col1_, col2_ = col1, col2
        assert col1_.shape == col2_.shape, "The two column have different shapes!"
        # Get raw one-hot encoding.
        raw_oh1 = th.tensor(col1_[:, None] == self.unique_levels1_).int()
        raw_oh2 = th.tensor(col2_[:, None] == self.unique_levels2_).int()

        n = col1_.shape[0]

        # cmbn_prod shape (n, p1, p2), Get product array
        cmbn_prod = th.bmm(raw_oh1.unsqueeze(2), raw_oh2.unsqueeze(1))
        # Encode when full-grid -- each combination has at least one data point.
        if (self.cmbn_freq_ > self.eps_).all():
            # (n, p1-1, p2-1)
            encoded_mat = th.zeros([n, self.n_levels1_ - 1 , self.n_levels2_ - 1])

            indexes_part1 = (raw_oh1[:, 0] == 1) & (raw_oh2[:, 0] == 1)
            encoded_mat[indexes_part1, :] = 1/self.cmbn_freq_[0, 0]

            indexes_part2 = (raw_oh1[:, 0] == 1) & (raw_oh2[:, 0] != 1)
            # (n,1,p2-1)/ (1,p2-1)
            encoded_mat[indexes_part2, :] = -cmbn_prod[indexes_part2, :1, 1:] / self.cmbn_freq_[:1, 1:].unsqueeze(0)

            indexes_part3 = (raw_oh1[:, 0] != 1) & (raw_oh2[:, 0] == 1)
            # (n,p1-1,1)/ (p1-1,1)
            encoded_mat[indexes_part3, :] = -cmbn_prod[indexes_part3, 1:, :1]/self.cmbn_freq_[1:, :1].unsqueeze(0)

            indexes_part4 = (raw_oh1[:, 0] != 1) & (raw_oh2[:, 0] != 1)
            encoded_mat[indexes_part4, :] = cmbn_prod[indexes_part4, 1:, 1:] * 1/self.cmbn_freq_[1:, 1:].unsqueeze(0)
        else:
            raise NotImplementedError("Currently only supports full-grid data!")
            # TODO for non-full-grid case, after ruling out the non-grid-closed case,
            #      solve equation for a basis of weights.
        self.is_fit_ = True
        return encoded_mat.reshape(n, (self.n_levels1_ - 1) * (self.n_levels2_ - 1))  # reshape to proper shape.

    def explain(self, coefficients):
        # we should map from (p1-1, p2-1) to (p1, p2)
        assert coefficients.shape == (
            (self.n_levels1_ - 1) * (self.n_levels2_ - 1),), "Coefficient shape mismatch with encoding!"
        coef_r = coefficients.reshape((self.n_levels1_ - 1), (self.n_levels2_ - 1))

        explain_value = th.zeros([self.n_levels1_, self.n_levels2_])
        explain_value[0, 0] = (coef_r/self.cmbn_freq_[0, 0]).sum()
        explain_value[:1, 1:] = -(coef_r / self.cmbn_freq_[:1, 1:]).sum(axis=0, keepdim=True)

        explain_value[1:, :1] = -(coef_r / self.cmbn_freq_[1:, :1]).sum(axis=1, keepdim=True)
        explain_value[1:, 1:] = coef_r / self.cmbn_freq_[1:, 1:]

        return explain_value, self.unique_levels1_, self.unique_levels2_  # Returns the values and corresponding axes.
        # explain_axis_1 is the row index, and explain_axis_2 is the column names.

    # Gets the number of columns of the encoded matrix.
    def get_encoded_number(self):
        return (self.n_levels1_ - 1) * (self.n_levels2_ - 1)