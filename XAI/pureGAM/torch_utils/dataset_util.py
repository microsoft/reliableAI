import torch.utils.data as data
import torch as th
from sklearn.model_selection import train_test_split
#from datasets.Synthetic.data_generator import categorical_generator, numerical_generator

class PureGamDataset(data.Dataset):
    # X_pe means X after pureness encoding
    def __init__(self, X_pe, S, y):
        self.X_pe = X_pe
        self.S = S
        self.y = y

    def __getitem__(self, index):
        #todo : S_inv should be regularize on cnt of index
        X_pe, S, S_inv, y = \
            self.X_pe[index, :], self.S[:, index, :], self.S[:, :, index],self.y[index]
        return X_pe, S, S_inv, y

    def __len__(self):
        return self.S.shape[1]

class PureGamDataset_smoothingInTraining(data.Dataset):
    # X_pe means X after pureness encoding
    def __init__(self, X_pe, X_num, y):
        self.X_pe = X_pe
        self.X_num = X_num
        self.y = y

    def __getitem__(self, index):
        #todo : S_inv should be regularize on cnt of index
        X_pe, X_num, y = \
            self.X_pe[index, :], self.X_num[index, :],self.y[index]
        return X_pe, X_num, y

    def __len__(self):
        return self.X_pe.shape[0]

if __name__ == "__main__":
    pass
    '''N = 2000
    X_cate, y_cate, eta, eta_inter = categorical_generator(N)
    X_num, y_num = numerical_generator(N)
    y = y_cate + y_num

    bias = y.mean()
    y_mean_centered = y - bias
    y_mean_centered = th.tensor(y_mean_centered)

    X_cate_train, X_cate_test, X_num_train, X_num_test, y_train, y_test =\
        train_test_split(X_cate, X_num, y_mean_centered, test_size=0.5, random_state=42)'''