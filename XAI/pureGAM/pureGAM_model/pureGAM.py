from time import time
import torch
import torch as th
import torch.nn as nn
from pureGAM_model.categorical_encoder import Categorical_encoder
from pureGAM_model.numerical_smoother import Numerical_transformer_adapt
from sklearn.model_selection import train_test_split
from pureGAM_model.utils import generate_pairwise_idxes, _print_metrics, safe_norm, safe_norm_alter, save_model, load_model
from pureGAM_model.submodels import Smoother_additive_model_pairwise, Categorical_additive_model
import pandas as pd
from torch_utils.readwrite import make_dir
from sklearn.metrics import r2_score, mean_squared_error

class PureGam:
    def __init__(self, p_univ_num, p_univ_cate, N_param_univ, N_param_biv, init_kw_lam_univ, init_kw_lam_biv,
                 isInteraction, model_output_dir, bias, trivpair_idxes_num = [], N_param_triv = 128, init_kw_lam_triv=0.1,
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), adaptiveInfoPerEpoch=10,
                 isPurenessConstraint=True, isAugLag = False, AugLag_startepoch = 60, AugLag_interepoch = 50,
                 isInnerBatchPureLoss=False, isLossEnhanceForDenseArea = False, dropout_rate=0.1, pure_lam_scale=1,
                 pairwise_idxes_cate=None, is_balancer=False, epoch4balancer_change=5, isLossMean=True, verbose=False):
        self.model_output_dir = model_output_dir
        make_dir(dir_path=model_output_dir)
        '''self.p_univ_num = X_num.shape[1]
        self.p_univ_cate = X_cate.shape[1]
        self.bias = y.mean.item()'''

        self.p_univ_num = p_univ_num
        self.p_univ_cate = p_univ_cate
        self.N_param_univ = N_param_univ
        self.N_param_biv = N_param_biv
        self.N_param_triv = N_param_triv

        self.bias = bias
        self.device = device

        self.adaptiveInfoPerEpoch = adaptiveInfoPerEpoch

        self.isPurenessConstraint = isPurenessConstraint
        self.isAugLag = isAugLag
        self.AugLag_startepoch = AugLag_startepoch
        self.AugLag_interepoch = AugLag_interepoch

        self.isInnerBatchPureLoss = isInnerBatchPureLoss
        self.isLossEnhanceForDenseArea = isLossEnhanceForDenseArea

        print("&&&& Device USED ::", self.device)
        # generate Interaction pairs
        if isInteraction:
            pairwise_idxes_num = generate_pairwise_idxes(self.p_univ_num)
            if pairwise_idxes_cate is None:
                pairwise_idxes_cate = generate_pairwise_idxes(self.p_univ_cate)
            print("Full Interaction", pairwise_idxes_num)
            print("Full Interaction", pairwise_idxes_cate)
        else:
            print("No Interaction")
            pairwise_idxes_num = []
            pairwise_idxes_cate = []


        if pairwise_idxes_num:
            pairwise_idx1, pairwise_idx2 = zip(*pairwise_idxes_num)
            pairwise_idx1, pairwise_idx2 = list(pairwise_idx1), list(pairwise_idx2)
        else:
            pairwise_idx1, pairwise_idx2 = [], []

        # triple
        self.trivpair_idxes_num = trivpair_idxes_num
        if self.trivpair_idxes_num:
            self.pairwise_biv_idx1, self.pairwise_biv_idx2, self.pairwise_biv_idx3 = [], [], []
            # todo: assume the sub-biv pair idx of triv pairs are included in pairwise_idxes_num
            for (univ_idx1, univ_idx2, univ_idx3) in self.trivpair_idxes_num:
                self.pairwise_biv_idx1.append(pairwise_idxes_num.index([univ_idx1, univ_idx2]) )
                self.pairwise_biv_idx2.append(pairwise_idxes_num.index([univ_idx1, univ_idx3]) )
                self.pairwise_biv_idx3.append(pairwise_idxes_num.index([univ_idx2, univ_idx3]) )
            print("TRIV!!", self.pairwise_biv_idx1, self.pairwise_biv_idx2, self.pairwise_biv_idx3)
        else:
            self.pairwise_biv_idx1, self.pairwise_biv_idx2, self.pairwise_biv_idx3 = [], [], []


        self.pairwise_idxes_num = pairwise_idxes_num
        self.pairwise_idxes_cate = pairwise_idxes_cate
        self.pairwise_idx1, self.pairwise_idx2 = pairwise_idx1, pairwise_idx2

        self.cate_enc = Categorical_encoder(self.p_univ_cate, bivariate_ids=self.pairwise_idxes_cate)
        # kernel weighter for california housing
        self.num_enc = Numerical_transformer_adapt(p_univ_num, N_param_univ, N_param_biv, N_param_triv,
                                              bivariate_ids=pairwise_idxes_num, trivariate_ids=trivpair_idxes_num,
                                               default_lam_univ=init_kw_lam_univ,
                                               default_lam_biv=init_kw_lam_biv,
                                               default_lam_triv = init_kw_lam_triv)
        self.num_enc = self.num_enc.to(self.device)

        self.model_smoothing = Smoother_additive_model_pairwise(p_univ_num, len(pairwise_idxes_num), len(trivpair_idxes_num),
                                                                N_param_univ, N_param_biv, N_param_triv, bias=bias, dropout_rate=dropout_rate)
        # self.model_smoothing = init_model(self.model_smoothing)
        self.model_smoothing = self.model_smoothing.to(self.device)

        # todo : Categorical_additive_model must be determined after fit data_preENcoder.
        self.model_categorical = Categorical_additive_model(0, 0, 0)
        # self.model_categorical = init_model(self.model_categorical)
        self.model_categorical = self.model_categorical.to(self.device)

        # augLagrange parameters init with 0
        self.lmd1_auglag_tensor = torch.zeros([self.p_univ_num], dtype=th.double, device=self.device)
        self.lmd2_auglag_tensor = torch.zeros([len(self.pairwise_idxes_num)], dtype=th.double, device=self.device)
        self.lmd3_auglag_tensor = torch.zeros([len(self.pairwise_idxes_num) * 2, self.N_param_univ], dtype=th.double, device=self.device)

        self.pure_lam_scale = pure_lam_scale
        self.is_balancer=is_balancer
        self.balancer_alpha = 0
        self.epoch4balancer_change = epoch4balancer_change

        self.isLossMean = isLossMean
        self.verbose = verbose

    def get_model_size(self):
        return sum(p.numel() for p in self.model_smoothing.parameters() if p.requires_grad) +\
               sum(p.numel() for p in self.model_categorical.parameters() if p.requires_grad) +\
               sum(p.numel() for p in self.num_enc.parameters() if p.requires_grad)

    def fit_cate_encoder(self, Cate_tabu_data):
        _ = self.cate_enc.fit_transform(Cate_tabu_data)
        self.model_categorical = Categorical_additive_model(self.cate_enc.cardinality_univ, self.cate_enc.cardinality_biv, 0)
        # self.model_smoothing = init_model(self.model_smoothing)
        # self.model_categorical = init_model(self.model_categorical)
        self.model_categorical = self.model_categorical.to(self.device)

        #print("Model Params Shape :: ", self.cate_enc.cardinality_univ, self.cate_enc.cardinality_biv, self.model_smoothing.eta_univ.shape, self.model_smoothing.eta_biv.shape)
        if self.verbose:
            print("Model Params Shape :: ", self.cate_enc.cardinality_univ, self.cate_enc.cardinality_biv,
              self.model_smoothing.w1_univ.shape, self.model_smoothing.w1_biv.shape)

    def init_param_points(self, X_num):
        # sample X_num_param as params of Smoothing Predictive model
        _, X_num_param_univ = \
            train_test_split(X_num, test_size=self.N_param_univ, random_state=41*1)
        _, X_num_param_biv = \
            train_test_split(X_num, test_size=self.N_param_biv, random_state=41*2+1)
        _, X_num_param_triv = \
            train_test_split(X_num, test_size=self.N_param_triv, random_state=41*3+2)
        X_num_param_univ = th.tensor(X_num_param_univ).to(self.device)
        X_num_param_biv = th.tensor(X_num_param_biv).to(self.device)
        X_num_param_triv = th.tensor(X_num_param_triv).to(self.device)
        self.num_enc.init_param_points(X_num_param_univ, X_num_param_biv, X_num_param_triv)
        # sample X_num_sample in X_num, according to N_param, to using as model parameters in Numerical_transformer

    def __batch_tranX(self, batch_N_X):
        batch_S_univ_tmp, batch_S_biv_tmp, batch_S_triv_tmp = self.num_enc.transform(batch_N_X, is_norm=False)
        # norm
        # todo: batch_S_tmp.sum(axis=-1, keepdim=True) is likly to be zero, so we should judge here
        try:
            batch_S_univ = safe_norm(batch_S_univ_tmp)
            batch_S_biv = safe_norm(batch_S_biv_tmp)
            batch_S_triv = safe_norm(batch_S_triv_tmp)
        except:
            assert False

        with torch.no_grad():
            batch_S_univ_tmp_nograd, batch_S_biv_tmp_nograd, batch_S_triv_tmp_nograd = self.num_enc.transform(batch_N_X, is_norm=False, lam_scale=self.pure_lam_scale)

        if self.isInnerBatchPureLoss:
            with torch.no_grad():
                batch_S_transpose_univ, batch_S_transpose_biv , _ = self.num_enc.transform_inner(batch_N_X, lam_scale=self.pure_lam_scale)
        else:
            # todo: whether ??? batch_S_transpose = batch_S_univ.clone().permute(0, 2, 1)
            batch_S_transpose_univ = batch_S_univ_tmp_nograd.clone().permute(0, 2, 1)
            batch_S_transpose_biv = batch_S_biv_tmp_nograd.clone().permute(0, 2, 1)

        # norm
        batch_S_univ_nograd = safe_norm(batch_S_univ_tmp_nograd)
        batch_S_biv_nograd = safe_norm(batch_S_biv_tmp_nograd)
        batch_S_triv_nograd = safe_norm(batch_S_triv_tmp_nograd)

        # trans norm
        if self.isLossEnhanceForDenseArea:
            batch_S_transpose = safe_norm_alter(batch_S_transpose_univ)
            batch_S_transpose_biv = safe_norm_alter(batch_S_transpose_biv)
        else:
            batch_S_transpose = safe_norm(batch_S_transpose_univ)
            batch_S_transpose_biv = safe_norm(batch_S_transpose_biv)


        return batch_S_univ, batch_S_biv, batch_S_triv, \
               batch_S_univ_nograd, batch_S_biv_nograd, batch_S_triv_nograd, \
               batch_S_transpose, batch_S_transpose_biv



    # batch_N_X means the numerical data of a batch, while batch_C_X means cate data of a batch
    # if you want to run epoch w/o batch gradient, please call this func like :::
    # with torch.no_grad:
    #   self. run_epoch(xxx)
    def __run_epoch(self, batch_N_X, batch_C_X, batch_y, criterion, ratio_part2=0, lmd1=0, lmd2=0, lmd3=0, lmd4=0, lmd5=0, ss_y=None):
        #######################
        # Use encoder to transform data into another format to use by pure GAM
        # shape, (p, n_batch, n_training points)

        batch_C_X_encoded = th.tensor(self.cate_enc.transform(batch_C_X)).to(self.device)
        batch_S_univ, batch_S_biv, batch_S_triv, \
            batch_S_univ_nograd, batch_S_biv_nograd, batch_S_triv_nograd, \
            batch_S_transpose, batch_S_transpose_biv =  self.__batch_tranX(batch_N_X)

        y_hat_S = self.model_smoothing.forward(batch_S_univ, batch_S_biv, batch_S_triv)
        y_hat_C = self.model_categorical.forward(batch_C_X_encoded[:, :self.cate_enc.cardinality_univ], batch_C_X_encoded[:, self.cate_enc.cardinality_univ:])

        if th.isnan(y_hat_S).any():
            print(y_hat_S)
            assert False
        y_hat = y_hat_S + y_hat_C

        # from base regression loss to more pure loss
        loss_regression = criterion(y_hat, batch_y)
        loss = loss_regression

        if self.isPurenessConstraint:
            order1_mean, order2_mean, order3_mean, conditional_mean, conditional_mean_triv = self.model_smoothing. \
                compute_pureness_constrain(batch_S_univ_nograd, batch_S_biv_nograd, batch_S_triv_nograd, # batch_S_univ, batch_S_biv, batch_S_triv,
                                           batch_S_transpose[self.pairwise_idx1],
                                           batch_S_transpose[self.pairwise_idx2],
                                           batch_S_transpose_biv[self.pairwise_biv_idx1],
                                           batch_S_transpose_biv[self.pairwise_biv_idx2],
                                           batch_S_transpose_biv[self.pairwise_biv_idx3],
                                           )
            # todo: alternative loss computation
            if self.isLossMean:
                order1_loss = (order1_mean ** 2).mean()
                order2_loss = (order2_mean ** 2).mean()
                conditional_loss = (conditional_mean ** 2).mean()

                #todo : sum of empty tensor to avoid nan
                order3_loss = (order3_mean ** 2).mean()
                conditional_loss_triv = (conditional_mean_triv ** 2).sum()/conditional_mean_triv.shape[1]
            else:
                order1_loss = (order1_mean ** 2).sum()
                order2_loss = (order2_mean ** 2).sum()
                conditional_loss = (conditional_mean ** 2).sum()
                order3_loss = (order3_mean ** 2).sum()
                conditional_loss_triv = (conditional_mean_triv ** 2).sum()

            loss_pure = (lmd1 * order1_loss + lmd2 * order2_loss + lmd3 * conditional_loss + lmd4*order3_loss + lmd5 * conditional_loss_triv)

            if self.is_balancer and (loss_pure.detach().item() > 0):
                #balancer = loss_regression.detach().item()/loss_pure.detach().item()
                balancer = self.balancer_alpha
            else:
                balancer = 1

            #print('Loss!', loss_regression.detach().item(), (ratio_part2 * balancer * loss_pure).detach().item(), balancer)
            loss = loss_regression + ratio_part2 * balancer * loss_pure
            # print(conditional_mean.abs())


            if self.isAugLag:
                order1_loss_auglag = (order1_mean * self.lmd1_auglag_tensor).mean()
                order2_loss_auglag = (order2_mean * self.lmd2_auglag_tensor).mean()
                conditional_loss_auglag = (conditional_mean * self.lmd3_auglag_tensor).mean()

                loss += order1_loss_auglag + order2_loss_auglag + conditional_loss_auglag

            # save info
            #ss_y = ((batch_y - batch_y.mean())**2).mean().item()
            loss_tensor = th.tensor(
                [loss_regression, ratio_part2  * lmd1 * order1_loss, ratio_part2 * lmd2 * order2_loss, ratio_part2 * lmd3 * conditional_loss]).detach()
                #[loss_regression, ratio_part2 *balancer*lmd1 * order1_loss, ratio_part2 *balancer* lmd2 * order2_loss, ratio_part2 *balancer* lmd3 * conditional_loss]).detach()
            metric_tensor = th.tensor(
                [1 - loss_regression / ss_y, order1_mean.mean(), order2_mean.mean(), conditional_mean.abs().mean(),
                 conditional_mean.abs().max()]).detach()


        else:
            loss_tensor = th.tensor([loss_regression]).detach()
            metric_tensor = th.tensor([1 - loss_regression / ss_y])

        return y_hat, loss, loss_tensor, metric_tensor

    # This function can be directly used to predict all testing data
    def predict_batch_numerical(self, batch_N_X):
        # Use encoder to transform data into another format to use by pure GAM
        # shape, (p, n_batch, n_training points)
        if not isinstance(batch_N_X, th.Tensor):
            batch_N_X = th.tensor(batch_N_X).to(self.device)
        batch_S_univ, batch_S_biv, batch_S_triv, \
        batch_S_univ_nograd, batch_S_biv_nograd, batch_S_triv_nograd, \
        batch_S_transpose, batch_S_transpose_biv = self.__batch_tranX(batch_N_X)


        with torch.no_grad():
            y_hat_S = self.model_smoothing.forward(batch_S_univ, batch_S_biv, batch_S_triv)
            if th.isnan(y_hat_S).any():
                print(y_hat_S)
                assert False
            order1_mean, order2_mean, order3_mean, conditional_mean, conditional_mean_triv  = self.model_smoothing. \
                compute_pureness_constrain(batch_S_univ_nograd, batch_S_biv_nograd, batch_S_triv_nograd, # batch_S_univ, batch_S_biv, batch_S_triv,
                                           batch_S_transpose[self.pairwise_idx1],
                                           batch_S_transpose[self.pairwise_idx2],
                                           batch_S_transpose_biv[self.pairwise_biv_idx1],
                                           batch_S_transpose_biv[self.pairwise_biv_idx2],
                                           batch_S_transpose_biv[self.pairwise_biv_idx3],
                                           )
            contri_mat_univ, contri_mat_biv, contri_mat_triv = self.model_smoothing.get_smoothing_contri_mat(batch_S_univ, batch_S_biv, batch_S_triv)
        return y_hat_S.detach(), contri_mat_univ.detach(), contri_mat_biv.detach(), order1_mean.detach(), order2_mean.detach(), conditional_mean.detach()

    def predict_batch_categorical(self, batch_C_X):
        batch_C_X_encoded = th.tensor(self.cate_enc.transform(batch_C_X)).to(self.device)

        with torch.no_grad():
            y_hat_C = self.model_categorical.forward(batch_C_X_encoded[:, :self.cate_enc.cardinality_univ],
                                                     batch_C_X_encoded[:, self.cate_enc.cardinality_univ:])
        return y_hat_C.detach()

    def predict_batch(self, batch_N_X, batch_C_X):
        y_hat_C = self.predict_batch_categorical(batch_C_X)
        y_hat_S, contri_mat_univ, contri_mat_biv, order1_mean, order2_mean, conditional_mean = self.predict_batch_numerical(batch_N_X)
        with torch.no_grad():
            y_hat = y_hat_S + y_hat_C
        return y_hat.detach(), contri_mat_univ.detach(), contri_mat_biv.detach(), order1_mean.detach(), order2_mean.detach(), conditional_mean.detach()


    def save_model(self, output_path):
        save_model(self.cate_enc, self.num_enc, self.model_categorical, self.model_smoothing, output_path, isInfo=False)

    def load_model(self, in_path):
        self.cate_enc, self.num_enc, self.model_categorical, self.model_smoothing= load_model(self.cate_enc, self.num_enc,
                                                                                      self.model_categorical, self.model_smoothing, in_path)

    # according to the loader of inputdata and optimizer to optimizer model
    def train(self, train_data_loader, valid_data_loader, optimizer, num_epoch=500, accumulation_steps=1, tolerent_level=100,
              ratio_part2=0, lmd1=0, lmd2=0, lmd3=0, criterion=nn.MSELoss(), test_data_loader = None):
        criterion = criterion.to(self.device)
        optimizer.zero_grad()
        last_loss = 9e10
        best_epoch = -1
        t1 = time()
        t3 = time()

        print("Param::", self.num_enc.X_memo_univ[:10], self.num_enc.X_memo_biv[:10])

        # compute_ss,
        ss_y_train = ((train_data_loader.dataset.y - train_data_loader.dataset.y.mean())**2).mean()
        ss_y_valid = ((valid_data_loader.dataset.y - valid_data_loader.dataset.y.mean())**2).mean()

        #todo : fit the data preprocessor
        for epoch in range(num_epoch):
            total_loss_train, total_metrics_train, total_train_cnt = 0, 0, 0
            total_loss_valid, total_metrics_valid, total_valid_cnt = 0, 0, 0

            #if (epoch) % self.adaptiveInfoPerEpoch == 0:

            t4 = time()
            if t4 - t3 > 1000:
            #if epoch % 10 == 0:
                print("time : ", t4-t3)
                t3 = time()
                # print("Param::", num_enc.X_memo_univ)
                print("### {AS} Params of Adaptive Smoothing")
                print("lam_univ::=", self.num_enc.get_lam()[0])
                print("lam_biv::=", self.num_enc.get_lam()[1])
                print()

            for ith_batch, (batch_C_X, batch_N_X, batch_y) in enumerate(train_data_loader):
                y_hat_train, loss_train, loss_tensor_train, metric_tensor_train = self.__run_epoch(batch_N_X, batch_C_X, batch_y, criterion, ratio_part2, lmd1, lmd2, lmd3, ss_y = ss_y_train)

                # loss to train model
                loss_train.backward()
                if (ith_batch + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                #todo: test test
                if th.isnan(self.num_enc.get_lam()[0]).any():
                    assert False, "Error: nan in numerical smoother"

                # save intermediate metrics and loss
                total_loss_train += loss_tensor_train
                total_metrics_train += metric_tensor_train
                total_train_cnt += 1

            optimizer.step()
            optimizer.zero_grad()

            for batch_C_X, batch_N_X, batch_y in valid_data_loader:
                with torch.no_grad():
                    y_hat_valid, loss_valid, loss_tensor_valid, metric_tensor_valid = self.__run_epoch(batch_N_X, batch_C_X, batch_y, criterion, ratio_part2, lmd1, lmd2, lmd3, ss_y = ss_y_valid)
                    total_loss_valid += loss_tensor_valid
                    total_metrics_valid += metric_tensor_valid
                    total_valid_cnt += 1

            if self.is_balancer:
                if total_loss_train[1:].sum().detach().item() == 0:
                    tmp_balancer_alpha = 1
                else:
                    tmp_balancer_alpha = ratio_part2*total_loss_train[0].detach().item()/total_loss_train[1:].sum().detach().item()
                    print('* tmp_balancer_alpha ', tmp_balancer_alpha)

                total_loss_train[1:] *= self.balancer_alpha
                total_loss_valid[1:] *= self.balancer_alpha
                if ((epoch+1) % self.epoch4balancer_change == 0) and epoch <= self.epoch4balancer_change:
                    self.balancer_alpha = tmp_balancer_alpha
                    print("[$$] balancer_alpha ::", self.balancer_alpha)

            t2 = time()
            if t2 - t1 > 20:
            #if epoch % 10 == 0:
                print("time : ", t2-t1)
                t1 = time()
                print("Epoch", epoch, " is ::: ")#, total_loss_test)
                print("     train:", _print_metrics(total_loss_train.numpy()/total_train_cnt), _print_metrics(total_loss_train.sum().item()/total_train_cnt) )
                print("         ", _print_metrics(total_metrics_train.numpy()/total_train_cnt))
                '''print("     valid:", _print_metrics(total_loss_valid_all.numpy()/total_valid_cnt_all), _print_metrics(total_loss_valid_all.sum().item()/total_valid_cnt_all) )
                print("         ", _print_metrics(total_metrics_valid_all.numpy()/total_valid_cnt_all))'''
                print("     valid:", _print_metrics(total_loss_valid.numpy()/total_valid_cnt), _print_metrics(total_loss_valid.sum().item()/total_valid_cnt) )
                print("         ", _print_metrics(total_metrics_valid.numpy()/total_valid_cnt))
                #print("     Categorical eta", model_categorical.eta.squeeze().detach().cpu().numpy())
                #print("     Exp Categorical eta", cate_enc.explain(model_categorical.eta.squeeze().detach().cpu().numpy()) )
                #print(self.model_categorical.get_eta())

                #todo: compare
                isEpochPrintTestRes = False
                if isEpochPrintTestRes:
                    y_hat_valid, y_valid = self.predict_with_y(valid_data_loader)
                    no_batch_res = [mean_squared_error(y_valid, y_hat_valid), r2_score(y_valid, y_hat_valid)]
                    if test_data_loader is not None:
                        y_hat_test, y_test = self.predict_with_y(test_data_loader)
                        no_batch_res += [mean_squared_error(y_test, y_hat_test), r2_score(y_test, y_hat_test)]
                    print('!## [result]', _print_metrics(no_batch_res))

            ### save the metrics and models.
            if epoch > self.epoch4balancer_change:
                if (total_loss_valid[0].item()/total_valid_cnt <= last_loss):
                    last_loss = total_loss_valid[0].item()/total_valid_cnt
                    tolerent = 0
                    best_epoch = epoch
                    best_saved_metrics = (t2, total_loss_train, total_metrics_train, total_loss_valid, total_metrics_valid)
                    self.save_model(self.model_output_dir + 'model_' + str(epoch))
                # Else, the performance drops, early stopped
                else:
                    tolerent += 1
                    if tolerent > tolerent_level or epoch >= num_epoch-1:
                        print("Early Stopped")

                        '''#todo: save the lastly trained model
                        best_epoch = epoch
                        best_saved_metrics = (t2, total_loss_train, total_metrics_train, total_loss_valid, total_metrics_valid)
                        self.save_model(self.model_output_dir + 'model_' + str(epoch))'''

                        break

        self.load_model(self.model_output_dir + 'model_' + str(best_epoch))
        ###   PRINT BEST MODEL PERF    ###
        print("Best Model Is IN :: " , self.model_output_dir + 'model_' + str(best_epoch))
        t2, total_loss_train, total_metrics_train, total_loss_valid, total_metrics_valid = best_saved_metrics
        print("@@@ Best Epoch", best_epoch, " is ::: ")  # , total_loss_test)
        print("time", t2 - t1)
        print("     train:", _print_metrics(total_loss_train.numpy() / total_train_cnt),
              _print_metrics(total_loss_train.sum().item() / total_train_cnt))
        print("         ", _print_metrics(total_metrics_train.numpy() / total_train_cnt))
        print("     valid:", _print_metrics(total_loss_valid.numpy() / total_valid_cnt),
              _print_metrics(total_loss_valid.sum().item() / total_valid_cnt))
        print("         ", _print_metrics(total_metrics_valid.numpy() / total_valid_cnt))

        print("### {AS} Params of Adaptive Smoothing")
        print("lam_univ::=", self.num_enc.get_lam()[0])
        print("lam_biv::=", self.num_enc.get_lam()[1])

    # in test data loader , there should not be label y in data. And , its better have only one batch that contains all the test data
    def predict(self, test_data_loader):

        t1 = time()
        # # Notice ::! in test data loader , there should not be label y in data.
        all_y_hat_test = []
        for batch_C_X, batch_N_X, _ in test_data_loader:
            with torch.no_grad():
                y_hat_test, _, _, _, _, _ = self.predict_batch(batch_N_X, batch_C_X)
                all_y_hat_test += y_hat_test.cpu().numpy().tolist()
        t2 = time()
        print("Total predict time", t2 - t1)
        return all_y_hat_test

    def predict_with_y(self, test_data_loader):
        t1 = time()
        # # Notice ::! in test data loader , there should not be label y in data.
        all_y_hat_test = []
        all_y_test = []
        for batch_C_X, batch_N_X, tmp_y in test_data_loader:
            with torch.no_grad():
                y_hat_test, _, _, _, _, _ = self.predict_batch(batch_N_X, batch_C_X)
                all_y_hat_test += y_hat_test.cpu().numpy().tolist()
                all_y_test += tmp_y.cpu().numpy().tolist()
        t2 = time()
        #print("time", t2 - t1)
        return all_y_hat_test, all_y_test

    def predict_biv_contri_mat(self, train_data_loader):
        t1 = time()
        # # Notice ::! in test data loader , there should not be label y in data.
        all_contri_mat_biv = []
        for batch_C_X, batch_N_X, _ in train_data_loader:
            with torch.no_grad():
                _, _, contri_mat_biv, _, _, _ = self.predict_batch(batch_N_X, batch_C_X)
                all_contri_mat_biv.append(contri_mat_biv)
        t2 = time()
        #print("time", t2 - t1)
        return torch.cat(all_contri_mat_biv, dim=-1)# concat through (p, n_batch)

    def plot_numerical(self, train_N_X):
        NotImplemented

    def plot_categorical(self, train_C_X):
        NotImplemented


if __name__ == "__main__":
    pass





