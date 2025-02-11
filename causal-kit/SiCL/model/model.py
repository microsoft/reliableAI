from collections.abc import Iterable
import math

import torch
import torch.nn as nn
import numpy as np


class ThreeLayerMLP(nn.Module):
    """
    Three Layer MLP with ReLU activation on first layer and no activation on final layer
    """

    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.linear0 = nn.Linear(input_dim, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear0(x)
        x = nn.functional.relu(x)
        x = self.linear1(x)
        return x


class AffineLinear(nn.Module):

    def __init__(self, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AffineLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return input * torch.exp(self.weight) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Model(nn.Module):
    def __init__(self,
                 layers=6,
                 dim=128,
                 key_size=32,
                 num_heads=8,
                 widening_factor=4,
                 dropout=0.1,
                 out_dim=None,
                 logit_bias_init=-3.0,
                 cosine_temp_init=0.0,
                 ln_axis=-1,
                 name="BaseModel",
                 pairwise=True,
                 graph_prediction=False,
                 v_tensor_prediction=False,
                 continuous_data=True,
                 num_of_classes=None,
                 input_embedding_dim=None,
                 num_of_nodes=None,
                 environ_nodes=0,
                 unify_environ_embedding=False
                 ):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim
        self.layers = 2 * layers
        self.dropout = dropout
        self.ln_axis = ln_axis
        self.widening_factor = widening_factor
        self.num_heads = num_heads
        self.key_size = key_size
        self.logit_bias_init = logit_bias_init
        self.cosine_temp_init = cosine_temp_init
        self.environ_nodes = environ_nodes
        self.best_threshold = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        if environ_nodes:
            self.system_nodes = num_of_nodes - environ_nodes
        else:
            self.system_nodes = num_of_nodes
        self.unify_environ_embedding = unify_environ_embedding

        self.continuous_data = continuous_data
        if continuous_data:
            self.input_linear = nn.Linear(1, self.dim)
        else:
            if not input_embedding_dim:
                input_embedding_dim = self.dim
            if self.unify_environ_embedding:
                self.input_linears = nn.ModuleList(
                    [nn.Linear(input_embedding_dim, self.dim) for _ in range(self.system_nodes)])
                if isinstance(num_of_classes, Iterable):
                    self.embedding_layers = nn.ModuleList([nn.Embedding(
                        num_embeddings=num_of_classes[i], embedding_dim=input_embedding_dim) for i in
                        range(self.system_nodes)])
                else:
                    assert isinstance(num_of_classes, np.integer) or isinstance(num_of_classes, int)
                    self.embedding_layers = nn.ModuleList([nn.Embedding(
                        num_embeddings=num_of_classes, embedding_dim=input_embedding_dim) for i in
                        range(self.system_nodes)])
            else:
                self.input_linears = nn.ModuleList(
                    [nn.Linear(input_embedding_dim, self.dim) for _ in range(num_of_nodes)])
                if isinstance(num_of_classes, Iterable):
                    self.embedding_layers = nn.ModuleList([nn.Embedding(
                        num_embeddings=num_of_classes[i], embedding_dim=input_embedding_dim) for i in
                        range(num_of_nodes)])
                else:
                    assert isinstance(num_of_classes, np.integer) or isinstance(num_of_classes, int)
                    self.embedding_layers = nn.ModuleList([nn.Embedding(
                        num_embeddings=num_of_classes, embedding_dim=input_embedding_dim) for i in range(num_of_nodes)])

        if unify_environ_embedding:
            self.environ_embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=self.dim)

        self.q_layer_norms = nn.ModuleList([nn.LayerNorm(self.dim) for i in range(self.layers)])
        self.k_layer_norms = nn.ModuleList([nn.LayerNorm(self.dim) for i in range(self.layers)])
        self.v_layer_norms = nn.ModuleList([nn.LayerNorm(self.dim) for i in range(self.layers)])
        self.attns = nn.ModuleList([nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads) for i in
                                    range(self.layers)])  # different with Lorch
        self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for i in range(self.layers)])
        self.res_layer_norms = nn.ModuleList([nn.LayerNorm(self.dim) for i in range(self.layers)])
        self.ffn_linears1 = nn.ModuleList(
            [nn.Linear(in_features=self.dim, out_features=self.widening_factor * self.dim) for i in range(self.layers)])
        self.ffn_linears2 = nn.ModuleList(
            [nn.Linear(in_features=self.widening_factor * self.dim, out_features=self.dim) for i in range(self.layers)])
        self.dropout_layers2 = nn.ModuleList([nn.Dropout(self.dropout) for i in range(self.layers)])

        self.final_layer_norm = nn.LayerNorm(self.dim)
        self.u_layer_norm = nn.LayerNorm(self.dim)
        self.v_layer_norm = nn.LayerNorm(self.dim)
        self.u_linear = nn.Linear(self.dim, self.out_dim)
        self.v_linear = nn.Linear(self.dim, self.out_dim)

        self.final_affine = AffineLinear()
        self.pairwise = pairwise
        self.graph_prediction = graph_prediction
        self.v_tensor_prediction = v_tensor_prediction
        if self.pairwise:
            self.pairwise_init_MLP = ThreeLayerMLP(self.dim * 2, self.dim * 4, self.dim)
            self.pairwise_q_layer_norm = nn.LayerNorm(self.dim)
            self.pairwise_k_layer_norm = nn.LayerNorm(self.dim)
            self.pairwise_v_layer_norm = nn.LayerNorm(self.dim)
            self.pair_z_attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads)
            self.pair_dropout_layer = nn.Dropout(self.dropout)
            self.pair_res_layer_norm = nn.LayerNorm(self.dim)
            self.pair_ffn_linear1 = nn.Linear(in_features=self.dim, out_features=self.widening_factor * self.dim)
            self.pair_ffn_linear2 = nn.Linear(in_features=self.widening_factor * self.dim, out_features=self.dim)
            self.pair_dropout_layer2 = nn.Dropout(self.dropout)
            self.final_graph_prediction = nn.Linear(self.dim, 1)
            self.pair_final_layer_norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        x_system = x[..., :self.system_nodes]
        x_environ = x[..., self.system_nodes:]
        if self.continuous_data:
            if self.unify_environ_embedding:
                z = self.input_linear(x_system.unsqueeze(-1))
            else:
                z = self.input_linear(x.unsqueeze(-1))
        else:
            if self.unify_environ_embedding:
                z = [self.input_linears[i](embedding(x_system[:, :, i])) for i, embedding in
                     enumerate(self.embedding_layers)]
                z = torch.stack(z, dim=-2)
            else:
                z = [self.input_linears[i](embedding(x[:, :, i])) for i, embedding in enumerate(self.embedding_layers)]
                z = torch.stack(z, dim=-2)
        if self.unify_environ_embedding:
            z_environ = self.environ_embedding_layer(x_environ.type(torch.int))
            z = torch.concat([z, z_environ], dim=-2)
        batch_size, dataset_size, num_of_nodes, hidden_dim = z.shape

        for l in range(self.layers):
            z = torch.transpose(z, -3, -2)
            # mha
            q_in = self.q_layer_norms[l](z)
            k_in = self.k_layer_norms[l](z)
            v_in = self.v_layer_norms[l](z)

            # self.attns[l].eval()
            # z_attn1 = torch.stack([self.attns[l](i, j, k)[0] for i, j, k in zip(q_in, k_in, v_in)])
            qshape, kshape, vshape = q_in.shape, k_in.shape, v_in.shape
            assert qshape[0] == batch_size
            assert qshape[3] == hidden_dim

            q_inr = q_in.permute(1, 0, 2, 3).reshape(qshape[1], qshape[0] * qshape[2], qshape[3])
            k_inr = k_in.permute(1, 0, 2, 3).reshape(kshape[1], kshape[0] * kshape[2], kshape[3])
            v_inr = v_in.permute(1, 0, 2, 3).reshape(vshape[1], vshape[0] * vshape[2], vshape[3])
            z_attn = self.attns[l](q_inr, k_inr, v_inr)[0]
            z_attn = z_attn.reshape(qshape[1], qshape[0], qshape[2], qshape[3]).permute(1, 0, 2, 3)
            # assert torch.allclose(z_attn, z_attn1)
            z = z + self.dropout_layers[l](z_attn)

            # ffn
            z_in = self.res_layer_norms[l](z)
            z_ffn = self.ffn_linears2[l](torch.relu(self.ffn_linears1[l](z_in)))
            z = z + self.dropout_layers2[l](z_ffn)

        node_feature = z
        self.hidden_representation = z
        if self.pairwise:
            pair_concat_z = torch.cat(
                [z.unsqueeze(-2).expand(batch_size, dataset_size, num_of_nodes, num_of_nodes, hidden_dim),
                 z.unsqueeze(-3).expand(batch_size, dataset_size, num_of_nodes, num_of_nodes, hidden_dim)], dim=-1)
            pair_concat_z = pair_concat_z.reshape(batch_size, dataset_size, num_of_nodes * num_of_nodes, hidden_dim * 2)
            pair_z = self.pairwise_init_MLP(pair_concat_z).transpose(-3, -2)
            z = z.transpose(-3, -2)
            q_in = self.pairwise_q_layer_norm(pair_z)
            k_in = self.pairwise_k_layer_norm(z)
            v_in = self.pairwise_v_layer_norm(z)

            qshape, kshape, vshape = q_in.shape, k_in.shape, v_in.shape
            assert qshape[0] == batch_size
            assert qshape[3] == hidden_dim

            q_inr = q_in.permute(1, 0, 2, 3).reshape(qshape[1], qshape[0] * qshape[2], qshape[3])
            k_inr = k_in.permute(1, 0, 2, 3).reshape(kshape[1], kshape[0] * kshape[2], kshape[3])
            v_inr = v_in.permute(1, 0, 2, 3).reshape(vshape[1], vshape[0] * vshape[2], vshape[3])
            pair_z_attn = self.pair_z_attn(q_inr, k_inr, v_inr)[0]
            pair_z_attn = pair_z_attn.reshape(qshape[1], qshape[0], qshape[2], qshape[3]).permute(1, 0, 2, 3)

            pair_z = pair_z + self.pair_dropout_layer(pair_z_attn)

            pair_z_in = self.pair_res_layer_norm(pair_z)
            pair_z_ffn = self.pair_ffn_linear2(torch.relu(self.pair_ffn_linear1(pair_z_in)))
            pair_z = pair_z + self.pair_dropout_layer2(pair_z_ffn)

            pair_z = self.pair_final_layer_norm(pair_z)
            final_pairwise_feature = pair_z.transpose(-2, -3).reshape(batch_size, dataset_size, num_of_nodes,
                                                                      num_of_nodes, hidden_dim)
            pair_z = pair_z.max(dim=-2)[0]  # 64 * 32

            pair_z = pair_z.reshape(batch_size, num_of_nodes, num_of_nodes, hidden_dim)
            pairwise_feature = pair_z
            permute_pair_z = pair_z.permute(0, 2, 1, 3)
            skeleton = self.final_graph_prediction(pair_z).squeeze(-1)
            skeleton = torch.sigmoid(self.final_affine(skeleton))
            skeleton = skeleton - torch.diag_embed(torch.diagonal(skeleton, dim1=-2, dim2=-1))
            if self.graph_prediction:
                graph = self.final_graph_prediction(pair_z).squeeze(-1)
                graph = torch.sigmoid(self.final_affine(graph))
                graph = graph - torch.diag_embed(torch.diagonal(graph, dim1=-2, dim2=-1))
            else:
                graph = None

            if self.v_tensor_prediction:
                pass
            else:
                v_tensor = None
            return {"skeleton": skeleton, "graph": graph, "v-tensor": v_tensor, "node_feature": node_feature,
                    "pairwise_feature": final_pairwise_feature}
        else:  # not pairwise
            z = self.final_layer_norm(z)
            # assert z.shape[-2] == x.shape[-1] and z.shape[-3] == x.shape[-2], "Do we have an odd number of layers?"

            # [..., n_vars, dim]
            z = torch.max(z, dim=-3)[0]

            # u, v dibs embeddings for edge probabilities

            u = self.u_linear(self.u_layer_norm(z))
            v = self.v_linear(self.v_layer_norm(z))

            # edge logits
            # [..., n_vars, dim], [..., n_vars, dim] -> [..., n_vars, n_vars]
            u = u / torch.linalg.norm(u, dim=-1, ord=2, keepdim=True)
            v = v / torch.linalg.norm(v, dim=-1, ord=2, keepdim=True)
            logit_ij = torch.einsum("...id,...jd->...ij", u, v)
            logit_ij = torch.sigmoid(self.final_affine(logit_ij))

            graph = logit_ij - torch.diag_embed(torch.diagonal(logit_ij, dim1=-2, dim2=-1))

            if not self.graph_prediction:
                skeleton = graph
            else:
                skeleton = None
            return {"skeleton": skeleton, "graph": graph, "v-tensor": None}

    def save(self, filename='checkpoint.pt'):
        state = {
            'state_dict': self.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state['state_dict'])


if __name__ == '__main__':
    model = Model().cuda()
    x = torch.randn(20, 200, 5).cuda()
    y = model(x)
