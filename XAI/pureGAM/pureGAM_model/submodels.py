import torch
import torch as th
import torch.nn as nn

class Smoother_additive_model_pairwise(nn.Module):
    def __init__(self, feature_dims_univ, feature_dims_biv, feature_dims_triv, num_points_univ, num_points_biv, num_points_triv, bias=None, dropout_rate=0.1):#, pairwise_list=None):
        super(Smoother_additive_model_pairwise, self).__init__()
        #self.num_points = num_points
        #self.feature_dims = feature_dims
        # n1 and n2 is (p, n, 1) tensor of params with n training samples and p feats
        # Eta = w1 * exp(w2) + b
        self.w1_univ = nn.Parameter(th.zeros([num_points_univ, feature_dims_univ], dtype=torch.double).T.unsqueeze(dim=-1))
        self.w1_biv = nn.Parameter(th.zeros([num_points_biv, feature_dims_biv], dtype=torch.double).T.unsqueeze(dim=-1))
        self.w1_triv = nn.Parameter(th.zeros([num_points_triv, feature_dims_triv], dtype=torch.double).T.unsqueeze(dim=-1))

        self.w2_univ = nn.Parameter(th.zeros([num_points_univ, feature_dims_univ], dtype=torch.double).T.unsqueeze(dim=-1))
        self.w2_biv = nn.Parameter(th.zeros([num_points_biv, feature_dims_biv], dtype=torch.double).T.unsqueeze(dim=-1))
        self.w2_triv = nn.Parameter(th.zeros([num_points_triv, feature_dims_triv], dtype=torch.double).T.unsqueeze(dim=-1))

        '''self.w2_univ = nn.Parameter(th.ones([num_points_univ, feature_dims_univ], dtype=torch.double).T.unsqueeze(dim=-1))
        self.w2_biv = nn.Parameter(th.ones([num_points_biv, feature_dims_biv], dtype=torch.double).T.unsqueeze(dim=-1))
        self.w2_triv = nn.Parameter(th.ones([num_points_triv, feature_dims_triv], dtype=torch.double).T.unsqueeze(dim=-1))'''

        #todo: now b is (p,n,1), maybe can be (p,1,1) or (1,n,1)?
        self.b_univ = nn.Parameter(th.zeros([num_points_univ, feature_dims_univ], dtype=torch.double).T.unsqueeze(dim=-1))
        self.b_biv = nn.Parameter(th.zeros([num_points_biv, feature_dims_biv], dtype=torch.double).T.unsqueeze(dim=-1))
        self.b_triv = nn.Parameter(th.zeros([num_points_triv, feature_dims_triv], dtype=torch.double).T.unsqueeze(dim=-1))


        self.dropout_univ = nn.Dropout(p=dropout_rate, inplace=False)
        self.dropout_biv = nn.Dropout(p=dropout_rate, inplace=False)
        self.dropout_triv = nn.Dropout(p=dropout_rate, inplace=False)

        self.is_bias = False
        self.bias = nn.Parameter(th.tensor(0.0))
        if isinstance(bias, int) or isinstance(bias, float):
            self.is_bias = True
            # TODO: should not be init before training?
            self.bias = nn.Parameter(th.tensor(float(bias)))

        '''if pairwise_list is None:
            pairwise_list = [-1] * feature_dims
        assert len(pairwise_list) == feature_dims, "len(pairwise_list) == feature_dims"
        self.pairwise_list = pairwise_list
        self.main_feats_cnt = [elem==-1 for elem in self.pairwise_list].sum()'''


    ## X is (m, p) tensor of batched train/test input with batch size m and p feats,
    # S is (p, m, n)the corresponding smoother weight of X
    # return y_hat is (m) tensor of predicted value by additive model
    def forward(self, S_univ, S_biv, S_triv):
        # contri_mat is (p, m) tensor of

        eta_univ = self.dropout_univ(self.w1_univ) * self.dropout_univ(th.exp(self.w2_univ)) + self.dropout_univ(self.b_univ)
        eta_biv = self.dropout_biv( self.w1_biv) * self.dropout_biv(th.exp(self.w2_biv)) + self.dropout_biv(self.b_biv)
        eta_triv = self.dropout_triv( self.w1_triv) * self.dropout_triv(th.exp(self.w2_triv)) + self.dropout_triv(self.b_triv)
        contri_mat_univ = th.bmm(S_univ, eta_univ).squeeze(axis=-1)
        contri_mat_biv = th.bmm(S_biv, eta_biv).squeeze(axis=-1)
        contri_mat_triv = th.bmm(S_triv, eta_triv).squeeze(axis=-1)

        #print("shape contri_mat:", contri_mat.shape)
        y_hat = contri_mat_univ.sum(axis=0) + contri_mat_biv.sum(axis=0) + contri_mat_triv.sum(axis=0)
        if self.is_bias:
            y_hat += self.bias
            #print('wat', y_hat)
        return y_hat

    def get_eta(self):

        eta_univ = self.w1_univ*th.exp(self.w2_univ) + self.b_univ
        eta_biv = self.w1_biv*th.exp(self.w2_biv) + self.b_biv
        eta_triv = self.w1_triv*th.exp(self.w2_triv) + self.b_triv
        return eta_univ, eta_biv, eta_triv

    def get_smoothing_contri_mat(self, S_univ, S_biv, S_triv):
        '''contri_mat_univ = th.bmm(S_univ, self.dropout_univ(self.eta_univ)).squeeze(axis=-1)
        contri_mat_biv = th.bmm(S_biv, self.dropout_biv( self.eta_biv)).squeeze(axis=-1)
        contri_mat_triv = th.bmm(S_triv, self.dropout_triv( self.eta_triv)).squeeze(axis=-1)'''
        eta_univ = self.w1_univ*th.exp(self.w2_univ) + self.b_univ
        eta_biv = self.w1_biv*th.exp(self.w2_biv) + self.b_biv
        eta_triv = self.w1_triv*th.exp(self.w2_triv) + self.b_triv
        contri_mat_univ = th.bmm(S_univ, eta_univ).squeeze(axis=-1)
        contri_mat_biv = th.bmm(S_biv, eta_biv).squeeze(axis=-1)
        contri_mat_triv = th.bmm(S_triv,  eta_triv).squeeze(axis=-1)

        return contri_mat_univ.detach(), contri_mat_biv.detach(), contri_mat_triv.detach()

    # batch_mean_along_features is (p) tensor of predicted value by additive model
    # S is (p  (main effect), m, n), project n points into m points
    # and S_inv is (p_pairwise(2 order), n, m), project m points into n points,

    # and for test samples, this constrain dont apply anymore
    def compute_pureness_constrain(self, S_univ, S_biv, S_triv,
                                   S_inv_projectmaineffect1, S_inv_projectmaineffect2,
                                   S_inv_projectbiv1, S_inv_projectbiv2, S_inv_projectbiv3):
        """
            Keep the rule that all main feats should be listed at front of tensor S, and pairwise feats listed followed.
        """
        pairwise_feats_cnt = S_inv_projectmaineffect1.shape[0]
        assert pairwise_feats_cnt == S_inv_projectmaineffect1.shape[0], "Two project S must be in same shape"
        assert pairwise_feats_cnt == S_inv_projectmaineffect2.shape[0], "Two project S must be in same shape"

        #(p, m, 1)
        '''contri_mat_univ = th.bmm(S_univ, self.dropout_univ(self.eta_univ)).squeeze(axis=-1)
        contri_mat_biv_unsqueeze = th.bmm(S_biv, self.dropout_biv(self.eta_biv))
        contri_mat_triv_unsqueeze = th.bmm(S_triv, self.dropout_triv(self.eta_triv))'''

        eta_univ = self.w1_univ*th.exp(self.w2_univ) + self.b_univ
        eta_biv = self.w1_biv*th.exp(self.w2_biv) + self.b_biv
        eta_triv = self.w1_triv*th.exp(self.w2_triv) + self.b_triv

        contri_mat_univ = th.bmm(S_univ, eta_univ).squeeze(axis=-1)
        contri_mat_biv_unsqueeze = th.bmm(S_biv,  eta_biv)
        contri_mat_triv_unsqueeze = th.bmm(S_triv,  eta_triv)

        #(p, m, 1)
        batch_mean_along_features_univ = contri_mat_univ.mean(axis=1)
        batch_mean_along_features_biv = contri_mat_biv_unsqueeze.squeeze(axis=-1).mean(axis=1)
        batch_mean_along_features_triv = contri_mat_triv_unsqueeze.squeeze(axis=-1).mean(axis=1)

        #contri_mat_unsqueeze_pairwise = contri_mat_unsqueeze[[]]
        # (p', n, m) project m points on n points, result in (p', n)

        # proj 2-dim eta on 1-dim points and seek to be zero
        conditional_project1 = th.bmm(S_inv_projectmaineffect1, contri_mat_biv_unsqueeze).squeeze(axis=-1)
        conditional_project2 = th.bmm(S_inv_projectmaineffect2, contri_mat_biv_unsqueeze).squeeze(axis=-1)
        # cat
        conditional_project = th.cat((conditional_project1, conditional_project2), axis=0)


        # proj 3-dim eta on 2-dim points and seek to be zero
        conditional_project1_trivbiv = th.bmm(S_inv_projectbiv1, contri_mat_triv_unsqueeze).squeeze(axis=-1)
        conditional_project2_trivbiv = th.bmm(S_inv_projectbiv2, contri_mat_triv_unsqueeze).squeeze(axis=-1)
        conditional_project3_trivbiv = th.bmm(S_inv_projectbiv3, contri_mat_triv_unsqueeze).squeeze(axis=-1)
        # cat
        conditional_project_trivbiv = th.cat((conditional_project1_trivbiv, conditional_project2_trivbiv, conditional_project3_trivbiv), axis=0)

        # compute conditional pureness

        return batch_mean_along_features_univ, batch_mean_along_features_biv, batch_mean_along_features_triv,\
                conditional_project, conditional_project_trivbiv

    def explain(self):
        NotImplemented

class Categorical_additive_model(nn.Module):
    def __init__(self, feature_dims_univ, feature_dims_biv, feature_dims_triv, dropout_rate=0.1):#, pairwise_list=None):
        super(Categorical_additive_model, self).__init__()
        #self.feature_dims = feature_dims
        # Eta is (p) tensor of params with p feats
        #self.bias = nn.Parameter(0)
        self.w1_univ = nn.Parameter(th.zeros([feature_dims_univ], dtype=torch.double))
        self.w1_biv = nn.Parameter(th.zeros([feature_dims_biv], dtype=torch.double))
        self.w1_triv = nn.Parameter(th.zeros([feature_dims_triv], dtype=torch.double))

        self.w2_univ = nn.Parameter(th.zeros([feature_dims_univ], dtype=torch.double))
        self.w2_biv = nn.Parameter(th.zeros([feature_dims_biv], dtype=torch.double))
        self.w2_triv = nn.Parameter(th.zeros([feature_dims_triv], dtype=torch.double))

        self.b_univ = nn.Parameter(th.zeros([feature_dims_univ], dtype=torch.double))
        self.b_biv = nn.Parameter(th.zeros([feature_dims_biv], dtype=torch.double))
        self.b_triv = nn.Parameter(th.zeros([feature_dims_triv], dtype=torch.double))
        '''self.dropout_univ = nn.Dropout(p=dropout_rate, inplace=False)
        self.dropout_biv = nn.Dropout(p=dropout_rate, inplace=False)
        self.dropout_triv = nn.Dropout(p=dropout_rate, inplace=False)'''

    def forward(self, X_encoded_univ, X_encoded_biv):#, X_encoded_triv): #X_encoded):
        #contri_mat = torch.mul(X_encoded, self.dropout_univ(self.eta).unsqueeze(dim=0)) #(m, p)
        # no dropout

        eta_univ = self.w1_univ*th.exp(self.w2_univ) + self.b_univ
        eta_biv = self.w1_biv*th.exp(self.w2_biv) + self.b_biv
        #eta_triv = self.dropout_triv( self.w1_triv) * self.dropout_triv(th.exp(self.w2_triv)) + self.dropout_triv(self.b_triv)

        contri_mat_univ = torch.mul(X_encoded_univ, eta_univ.unsqueeze(dim=0) )
        contri_mat_biv = torch.mul(X_encoded_biv, eta_biv.unsqueeze(dim=0))
        #contri_mat_triv = torch.mul(X_encoded_triv, self.dropout_triv(self.eta_triv).unsqueeze(dim=0))

        #y_hat is (m) vector of categorical prediction
        y_hat = contri_mat_univ.sum(axis=1) + contri_mat_biv.sum(axis=1) #+ contri_mat_triv.sum(axis=1)
        return y_hat

    def get_eta(self):
        eta_univ = self.w1_univ*th.exp(self.w2_univ) + self.b_univ
        eta_biv = self.w1_biv*th.exp(self.w2_biv) + self.b_biv
        return eta_univ, eta_biv