import random
import numpy as np
# import igraph as ig
from scipy.stats import truncnorm
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


class DagSimulator:
    def __init__(self):
        pass

    def simulate_graph(self, node_num, edge_num):
        """Simulate random DAG with some expected number of edges.
        
        Args:
            node_num (int): num of nodes
            edge_num (int): expected num of edges
            graph_type (str): Different strategies ER / SF for generate random DAG
        
        Returns:
            adj_matrix (np.ndarray): [node_num, node_num] binary adjacency matrix of DAG
        """
        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)
        
        def _random_permutation(M):
            P = np.random.permutation(np.eye(M.shape[0]))
            return P.T @ M @ P
        
        def _acyclic_orientation(B_und):
            return np.tril(B_und, k=-1)
        
        def _remove_isolating_node(B):
            non_iso_index = np.logical_or(B.any(axis=0), B.any(axis=1))
            return B[non_iso_index][:, non_iso_index]
    
        # First, generate random DAG use specified strategy
        G = ig.Graph.Barabasi(n=node_num, m=int(round(edge_num / node_num)), directed=True, outpref=True, power=-3)
        
        # Second, get adjacency matrix of DAG, resolve cyclic, remove isolating node, makes sure direction is not always from high to low, and check whether DAG is legal
        B = _graph_to_adjmat(G)
        B_perm = _random_permutation(B).astype(int)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag() , 'Generated graph is not a dag!'
        return B_perm
    

    def _simulate_cards(self, B, card_param=None):
        if card_param is None:
            card_param = {'lower': 2, 'upper': 6, 'mu': 2, 'basesigma': 1.5} # truncated normal distribution
        def _max_peers():
            '''
            why we need this: to calculate cpd of a node with k parents,
                the conditions to be enumerated is the production of these k parents' cardinalities
                which will be very exponentially slow w.r.t. k.
                so we want that, if a node has many parents (large k), these parents' cardinalities should be small
            i also tried to restrict each node's indegree at the graph sampling step,
                but i think that selection-bias on graph structure is worse than that on cardinalities
            an alternative you can try:
                use SEM to escape from slow forwards simulation, and then discretize.

            denote peers_num: peers_num[i, j] = k (where k>0),
                means that there are k parents pointing to node i, and j is among these k parents.
            max_peers = peers_num.max(axis=0): the larger max_peers[j], the smaller card[j] should be.
            :return:
            '''
            in_degrees = B.sum(axis=0)
            peers_num = in_degrees[:, None] * B.T
            return peers_num.max(axis=0)

        lower, upper, mu, basesigma = card_param['lower'], card_param['upper'], card_param['mu'], card_param['basesigma']
        sigma = basesigma / np.exp2(_max_peers()) ########## simply _max_peers() !
        cards = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(size=B.shape[0]).round().astype(int)
        return cards

    def simulate_discrete_bn(self, B, card_param=None, alpha_param=None):
        """Get a synthetic bayesian network from an adjacency matrix

        Args:
            B (np.ndarray): which adjacency matrix we use to generate bayesian network
            card_param (dict, optional): truncated normal distribution parameter for cardinality of all nodes. Defaults to None.
            uniform_alpha_param (dict, optional): use for generate random CPD of all nodes. Defaults to None.
        
        Returns:
            bn (BayesianNetwork): return bayesian network.
        """
        if alpha_param is None:
            alpha_param = {'lower': 0.1, 'upper': 1.0}
        
        def _random_alpha():
            return np.random.uniform(alpha_param['lower'], alpha_param['upper'])
        
        def _dirichlet(alpha, size):
            probs = np.random.dirichlet(np.ones(size) * alpha)
            probs[-1] = 1 - probs[:-1].sum() # to prevent numerical issues (not sum to 1)
            return probs
        
        # First, create BN with given adjacency matrix
        bn = BayesianNetwork(list(map(lambda x: (str(x[0]), str(x[1])), np.argwhere(B == 1))))

        # Second, simulates cardinalities of all nodes by truncating normal distribution
        cards = self._simulate_cards(B, card_param=card_param)

        # Third, generate random CPD of all nodes by cardinalities
        for node in range(len(cards)):
            parents = np.where(B[:, node] == 1)[0].tolist()
            parents_card = [cards[prt] for prt in parents]
            rand_ps = np.array([_dirichlet(_random_alpha(), cards[node]) for _ in range(int(np.prod(parents_card)))]).T.tolist()
            cpd = TabularCPD(str(node), cards[node], rand_ps, evidence=list(map(str, parents)), evidence_card=parents_card)
            bn.add_cpds(cpd)
        
        return bn