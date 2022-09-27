# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.utils.data as data

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
