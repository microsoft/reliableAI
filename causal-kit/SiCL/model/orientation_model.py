import torch
import torch.nn as nn

from model.model import Model
from model.model import AffineLinear, ThreeLayerMLP


class OriModel(nn.Module):
    
    def __init__(self, hidden_dim=128) -> None:
        super().__init__()
        # self.pairwise_feature_linear = nn.Linear(hidden_dim, hidden_dim)
        # self.node_feature_linear = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = ThreeLayerMLP(2 * hidden_dim, 4 * hidden_dim, 1)
        self.final_affine = AffineLinear()
        
    def forward(self, node_feature, pairwise_feature):
        """
        node_feature: ..., num_of_nodes, hidden_dim
        pairwise_feature: ..., num_of_nodes, num_of_nodes, hidden_dim
        output: num_of_nodes, num_of_nodes, num_of_nodes, 1
        """
        batch_size, dataset_size, num_of_nodes, hidden_dim = node_feature.shape
        node_feature = torch.max(node_feature, dim=-3)[0]
        pairwise_feature = torch.max(pairwise_feature, dim=-4)[0]
        # node_feature = node_feature.reshape(batch_size, 1, num_of_nodes, 1, hidden_dim).expand(
        #     batch_size, num_of_nodes, num_of_nodes, num_of_nodes, hidden_dim)
        # pairwise_feature = pairwise_feature.reshape(batch_size, num_of_nodes, 1, num_of_nodes, hidden_dim).expand(
        #     batch_size, num_of_nodes, num_of_nodes, num_of_nodes, hidden_dim)
        node_feature = node_feature.reshape(batch_size, num_of_nodes, 1, 1, hidden_dim).expand(
            batch_size, num_of_nodes, num_of_nodes, num_of_nodes, hidden_dim)
        pairwise_feature = pairwise_feature.reshape(batch_size, 1, num_of_nodes, num_of_nodes, hidden_dim).expand(
            batch_size, num_of_nodes, num_of_nodes, num_of_nodes, hidden_dim)
        concat_feature = torch.concat([node_feature, pairwise_feature], dim=-1)
        vstruc_feature = self.mlp(concat_feature)
        vstruc_cube_result = torch.sigmoid(self.final_affine(vstruc_feature)).squeeze()
        return vstruc_cube_result


class WholeModel(nn.Module):
    def __init__(self,continuous_data=True,
                num_of_classes=None,
                input_embedding_dim=None,
                num_of_nodes=None):
        super().__init__()
        self.feature_extractor = Model(graph_prediction=False, continuous_data=continuous_data,
                num_of_classes=num_of_classes,
                input_embedding_dim=input_embedding_dim,
                num_of_nodes=num_of_nodes,
                pairwise=True)
        self.orientation_model = OriModel()
        self.best_threshold = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
    def forward(self, x, freeze=True):
        if freeze:
            with torch.no_grad():
                results = self.feature_extractor(x)
        else:
            results = self.feature_extractor(x)
        node_feature = results["node_feature"]
        pairwise_feature = results["pairwise_feature"]
        vstruc_cube_result = self.orientation_model(node_feature, pairwise_feature)
        return {'vstruc': vstruc_cube_result}
    def save(self, filename='checkpoint.pt'):
        state = {
            'state_dict': self.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state['state_dict'])

    def load_feature_extrator(self, filename):
        state = torch.load(filename)
        self.feature_extractor.load_state_dict(state['state_dict'])


class OriNodewiseModel(nn.Module):

    def __init__(self, hidden_dim=128) -> None:
        super().__init__()
        self.mlp = ThreeLayerMLP(3 * hidden_dim, 4 * hidden_dim, 1)
        self.final_affine = AffineLinear()

    def forward(self, node_feature):
        """
        node_feature: ..., num_of_nodes, hidden_dim
        output: num_of_nodes, num_of_nodes, num_of_nodes, 1
        """
        batch_size, dataset_size, num_of_nodes, hidden_dim = node_feature.shape
        node_feature = torch.max(node_feature, dim=-3)[0]
        # node_feature = node_feature.reshape(batch_size, 1, num_of_nodes, 1, hidden_dim).expand(
        #     batch_size, num_of_nodes, num_of_nodes, num_of_nodes, hidden_dim)
        # pairwise_feature = pairwise_feature.reshape(batch_size, num_of_nodes, 1, num_of_nodes, hidden_dim).expand(
        #     batch_size, num_of_nodes, num_of_nodes, num_of_nodes, hidden_dim)
        node_featureT = node_feature.reshape(batch_size, num_of_nodes, 1, 1, hidden_dim).expand(
            batch_size, num_of_nodes, num_of_nodes, num_of_nodes, hidden_dim)
        node_featureX = node_feature.reshape(batch_size, 1, num_of_nodes, 1, hidden_dim).expand(
            batch_size, num_of_nodes, num_of_nodes, num_of_nodes, hidden_dim)
        node_featureY = node_feature.reshape(batch_size, 1, 1, num_of_nodes, hidden_dim).expand(
            batch_size, num_of_nodes, num_of_nodes, num_of_nodes, hidden_dim)

        concat_feature = torch.concat([node_featureT, node_featureX, node_featureY], dim=-1)
        vstruc_feature = self.mlp(concat_feature)
        vstruc_cube_result = torch.sigmoid(self.final_affine(vstruc_feature)).squeeze()
        return vstruc_cube_result

class WholeNodewiseModel(nn.Module):
    def __init__(self, continuous_data=True,
                 num_of_classes=None,
                 input_embedding_dim=None,
                 num_of_nodes=None,
                 ):
        super().__init__()
        self.feature_extractor = Model(graph_prediction=False, continuous_data=continuous_data,
                                            num_of_classes=num_of_classes,
                                            input_embedding_dim=input_embedding_dim,
                                            num_of_nodes=num_of_nodes,
                                            pairwise=False, layers=8)
        self.orientation_model = OriNodewiseModel()

    def forward(self, x, freeze=True):
        if freeze:
            with torch.no_grad():
                results = self.feature_extractor(x)
        else:
            results = self.feature_extractor(x)
        node_feature = results["node_feature"]
        vstruc_cube_result = self.orientation_model(node_feature)
        return {'vstruc': vstruc_cube_result}

    def save(self, filename='checkpoint.pt'):
        state = {
            'state_dict': self.state_dict(),
        }
        torch.save(state, filename)

    def load(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state['state_dict'])

    def load_feature_extrator(self, filename):
        state = torch.load(filename)
        self.feature_extractor.load_state_dict(state['state_dict'])


if __name__ == "__main__":
    a = 8
    node_feature = torch.randn(20, a, 64)
    pairwise_feature = torch.randn(20, a, a, 64)
    om = OriModel()
    om(node_feature, pairwise_feature)
