#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, argparse
import yaml
import igraph as ig
import numpy as np
from scipy.stats import truncnorm
from pgmpy.models.BayesianModel import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import networkx as nx
import matplotlib.pyplot as plt
from Tools import Graph
from collections import namedtuple

sys.path.append(os.path.dirname(os.getcwd()))

benchmark_names = [
    "asia", "cancer", "earthquake", "sachs", "survey",
    "alarm", "barley", "child", "insurance", "mildew", "water",
    "hailfinder", "hepar2", "win95pts",
    "andes", "diabetes", "link", "munin1", "pathfinder", "pigs",
    "munin", "munin2", "munin3", "munin4",
]

############### for transferability experiment ###############
SYN_NUM = {'train': 50, 'test': 5}
NodeNum = 50
Sparsity = 1.5  # sparsity, avg in degree
GraphType = 'ER'
SampleSize = 10000

NodeNum_controls = [10, 50, 100, 1000]
Sparsity_controls = [1, 2, 3, 4]
GraphType_controls = ['ER', 'SF']
SampleSize_controls = [1000, 5000, 10000, 15000, 20000]
controls_range = {
    'nodenum': NodeNum_controls,
    'sparsity': Sparsity_controls,
    'graphtype': GraphType_controls,
    'samplesize': SampleSize_controls
}


############### for transferability experiment ###############


class DagSimulator:
    def __init__(self):
        pass

    def simulate(self, d, s0, graph_type):
        """Simulate random DAG with some expected number of edges.
           partly stolen from https://github.com/xunzheng/notears
        Args:
            d (int): num of nodes
            s0 (int): expected num of edges
            graph_type (str): ER, SF
        Returns:
            B (np.ndarray): [d, d] binary adj matrix of DAG
        """

        if graph_type == 'ER':
            undirected_graph = ig.Graph.Erdos_Renyi(n=d, m=s0, directed=False)
        elif graph_type == 'SF':
            undirected_graph = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False, outpref=True, power=-3)
        else:
            raise ValueError('unknown graph type')
        adj_matrix = self._graph_to_adjmat(undirected_graph)

        # resolve cyclic
        adj_matrix = self._acyclic_orientation(adj_matrix)
        adj_matrix = self._remove_isolating_node(adj_matrix)

        # this permutation makes sure the direction is not always from high to low
        adj_matrix = self._random_permutation(adj_matrix).astype(int)
        assert ig.Graph.Adjacency(adj_matrix.tolist()).is_dag(), 'Generated graph is not a dag!'
        return adj_matrix

    @staticmethod
    def is_dag(w):
        g = ig.Graph.Weighted_Adjacency(w.tolist())
        return g.is_dag()

    @staticmethod
    def _random_permutation(matrix):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(matrix.shape[0]))
        # here permutes both row and column
        return P.T @ matrix @ P

    @staticmethod
    def _acyclic_orientation(matrix):
        # pre-randomized here. to prevent i->j always with i>j, return a lower triangle of the input
        return np.tril(matrix, k=-1)

    @staticmethod
    def _remove_isolating_node(matrix):
        non_iso_index = np.logical_or(matrix.any(axis=0), matrix.any(axis=1))
        return matrix[non_iso_index][:, non_iso_index]

    @staticmethod
    def _graph_to_adjmat(undirected_graph):
        return np.array(undirected_graph.get_adjacency().data)


def simulate_cards(adj_matrix, card_param=None):
    """ this function simulates cardinalities of nodes so that the simulation can be conducted quickly"""
    if card_param is None:
        card_param = {'lower': 2, 'upper': 6, 'mu': 2, 'basesigma': 1.5}  # truncated normal distribution

    def _max_peers():
        """
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
        """
        in_degrees = adj_matrix.sum(axis=0)
        peers_num = in_degrees[:, None] * adj_matrix.T
        return peers_num.max(axis=0)

    lower, upper, mu, basesigma = card_param['lower'], card_param['upper'], card_param['mu'], card_param['basesigma']
    sigma = basesigma / np.exp2(_max_peers())  ########## simply _max_peers() !
    cards = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma). \
        rvs(size=adj_matrix.shape[0]).round().astype(int)
    return cards


def simulate_discrete_bn(adj_matrix, n_sample=10000, card_param=None, uniform_alpha_param=None, return_bn=False):
    """Get a Bayesian network from an adj. matrix, if needed, return simulated data instead"""
    if uniform_alpha_param is None:
        uniform_alpha_param = {'lower': 0.1, 'upper': 1.0}  # 1.2? # larger alpha, smaller variation. is uniform good?
        nonuniform_alpha_param = {'lower': 0.05,
                                  'upper': 0.1}  # 1.2? # larger alpha, smaller variation. is uniform good?

    def _uniform_random_alpha():
        return np.random.uniform(uniform_alpha_param['lower'], uniform_alpha_param['upper'])

    def _non_uniform_random_alpha():
        return np.random.uniform(nonuniform_alpha_param['lower'], nonuniform_alpha_param['upper'])

    cards = simulate_cards(adj_matrix, card_param=card_param)
    diEdges = list(map(lambda x: (str(x[0]), str(x[1])), np.argwhere(adj_matrix == 1)))  # list of tuples
    bn = BayesianNetwork(diEdges)  # so isolating nodes will echo error
    for node in range(len(cards)):

        do_uniform = True

        parents = np.where(adj_matrix[:, node] == 1)[0].tolist()
        parents_card = [cards[prt] for prt in parents]

        if do_uniform:
            random_alpha = _uniform_random_alpha()
        else:
            random_alpha = _non_uniform_random_alpha()
        rand_ps = np.array(
            [np.clip(np.random.dirichlet(np.ones(cards[node]) * random_alpha), .0, 1.) for _ in
             range(int(np.prod(parents_card)))]).T.tolist()

        if min(rand_ps)[0] < 0:
            print(rand_ps)

        cpd = TabularCPD(str(node), cards[node], rand_ps,
                         evidence=list(map(str, parents)), evidence_card=parents_card)
        cpd.normalize()
        bn.add_cpds(cpd)
    if return_bn:
        return bn
    else:
        sample_data_from_bn(bn, n_sample, cards)


def sample_data_from_bn(bn: BayesianNetwork, n_sample, cards):
    inference = BayesianModelSampling(bn)
    df = inference.forward_sample(size=n_sample)
    if isinstance(cards, int):
        df = df[[str(i) for i in range(cards)]]
    elif isinstance(cards, list):
        df = df[[str(i) for i in range(len(cards))]]
    return df.to_numpy().astype(int)


class GraphSimulator:
    def __init__(self, cfg):
        GraphSpecific = namedtuple("GraphSpecific", "num_of_graphs_per_type nodes_range_per_type")
        self.save_dir = cfg["save_dir"]
        self.graph_spec_dict = {}
        self.n_simulation_samples = cfg["num_simulation_samples"]
        self.visualization = cfg["visualization"]
        self.avgInDegree = cfg["avgInDegree"]
        for key in cfg["graph_spec_dict"].keys():
            spec = cfg["graph_spec_dict"][key]
            self.graph_spec_dict[key] = GraphSpecific(spec["number_of_samples"], spec["node_number_range"])
        os.makedirs(os.path.join(self.save_dir), exist_ok=True)
        with open(os.path.join(self.save_dir, 'config.yml'), 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)

    def simulate_graphs(self):
        os.makedirs(os.path.join(self.save_dir, 'graph'), exist_ok=True)
        with open(os.path.join(self.save_dir, 'config.yml'), 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)
        os.makedirs(os.path.join(self.save_dir, 'graph', 'imgs'), exist_ok=True)
        dag_simulator = DagSimulator()
        for cname in self.graph_spec_dict.keys():
            lower, upper = self.graph_spec_dict[cname].nodes_range_per_type
            for gtype in ['ER', 'SF']:
                for id in range(self.graph_spec_dict[cname].num_of_graphs_per_type):
                    synname = f'{cname}_{gtype}_{id}'
                    graphtxtpath = os.path.join(self.save_dir, 'graph', f'{synname}_graph.txt')
                    if os.path.exists(graphtxtpath):
                        print(f"{graphtxtpath} exists, continue.")
                        continue

                    # sample # of nodes, edges
                    d = np.random.randint(lower, upper)
                    s0 = np.round(np.random.uniform(self.avgInDegree[0], self.avgInDegree[1]) * d).astype(int)
                    B = dag_simulator.simulate(d, s0, gtype)
                    np.savetxt(graphtxtpath, B, fmt='%i')

                    DirectedEdges = list(map(tuple, np.argwhere(B == 1)))
                    NodeIDs = list(range(len(B)))

                    # visualization
                    if self.visualization:
                        G = nx.DiGraph()
                        G.add_nodes_from(NodeIDs)
                        G.add_edges_from(DirectedEdges)
                        pos = nx.kamada_kawai_layout(G)
                        nx.draw_networkx(G, pos)

                        plt.title(f'{synname}, {len(NodeIDs)} nodes, {len(DirectedEdges)} edges')
                        plt.savefig(os.path.join(self.save_dir, 'graph', 'imgs', f'{synname}.png'))
                        plt.clf()

                    print(synname, f"expected: d={d}, s={s0}\treturned: d={len(B)}, s={len(DirectedEdges)}")

    def simulate_data_discrt(self):
        assert os.path.exists(os.path.join(self.save_dir, 'graph')), "Graphs for simulation not existed!"
        os.makedirs(os.path.join(self.save_dir, 'data'), exist_ok=True)
        for cname in self.graph_spec_dict.keys():
            for gtype in ['ER', 'SF', ]:
                for id in range(self.graph_spec_dict[cname].num_of_graphs_per_type):
                    synname = f'{cname}_{gtype}_{id}'
                    if os.path.exists(f'{self.save_dir}/data/{synname}_{self.n_simulation_samples}.npy'):
                        continue
                    print(f"start simulation on graph {synname}")
                    B = np.loadtxt(f'{self.save_dir}/graph/{synname}_graph.txt')
                    sim_data = simulate_discrete_bn(B, n_sample=self.n_simulation_samples)
                    file_name = f'{synname}_{self.n_simulation_samples}.npy'
                    np.save(os.path.join(self.save_dir, 'data', file_name), sim_data)


def resample_benchmarks(samplesize=10000):
    os.makedirs(f"../benchmarks/csv{samplesize}/", exist_ok=True)
    os.makedirs(f"../benchmarks/npy{samplesize}/", exist_ok=True)
    os.makedirs('../synthetics/Rscripts/', exist_ok=True)
    for bname in benchmark_names:
        print(bname)
        _ = Graph.DiGraph(f"../benchmarks/{bname}_graph.txt")  # used to generate VStrucs.txt, ...
        R_script_string = f'library(bnlearn)\nnet = read.bif("../benchmarks/bif/{bname}.bif")\n'
        R_script_string += f'sim = rbn(net, n={samplesize}, debug=FALSE)\n'
        R_script_string += f'write.csv(sim, "../benchmarks/csv{samplesize}/{bname}.csv", row.names = FALSE)'
        with open(f'../synthetics/Rscripts/{bname}.R', 'w') as fout:
            fout.write(R_script_string)
        os.system(f'Rscript ../synthetics/Rscripts/{bname}.R')

        my_data = np.genfromtxt(f"../benchmarks/csv{samplesize}/{bname}.csv", delimiter=',', dtype=None,
                                encoding="utf8")[1:]
        converted_data = np.zeros(my_data.shape, dtype=np.uint8)
        for colind in range(my_data.shape[1]):
            coldata = my_data[:, colind]
            _, converted_col = np.unique(coldata, return_inverse=True)
            converted_data[:, colind] = converted_col
        np.save(f"../benchmarks/npy{samplesize}/{bname}.npy", converted_data)

    os.system(f'rm -rf ../benchmarks/csv{samplesize}/')


def simulate_diff(controltag='nodenum'):  # control in 'nodenum', 'sparsity', 'graphtype', 'samplesize'
    for control_item in controls_range[controltag]:
        nodenum = control_item if controltag == 'nodenum' else NodeNum
        sparsity = control_item if controltag == 'sparsity' else Sparsity
        graphtype = control_item if controltag == 'graphtype' else GraphType
        samplesize = control_item if controltag == 'samplesize' else SampleSize
        dag_simulator = DagSimulator()

        for traintag in ['train', 'test']:
            for id in range(SYN_NUM[traintag]):
                edgenum = np.round(sparsity * nodenum).astype(int)
                B = dag_simulator.simulate(nodenum, edgenum, graphtype)
                graphdir = f'../synthetics/transferability/{controltag}/{control_item}/{traintag}/graph/'
                graph_txt_pth = os.path.join(graphdir, f'{id}_graph.txt')
                os.makedirs(graphdir, exist_ok=True)
                np.savetxt(graph_txt_pth, B, fmt='%i')
                _ = Graph.DiGraph(graph_txt_pth)  # used to generate VStrucs.txt, ...

                datadir = f'../synthetics/transferability/{controltag}/{control_item}/{traintag}/data/'
                data_npy_pth = os.path.join(datadir, f'{id}.npy')
                os.makedirs(datadir, exist_ok=True)
                np.save(data_npy_pth, simulate_discrete_bn(B, n_sample=samplesize))


if __name__ == '__main__':
    ########### for testing benchmarks data ###########
    # for samplesize in [1000, 5000, 10000, 15000, 20000]:
    # for samplesize in [10000]:
    #     resample_benchmarks(samplesize)
    ########### for testing benchmarks data ###########

    ########### for training synthetic data ###########
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="graph generation config file", default="../configs/generate_random_graphs.yml")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    graph_simulator = GraphSimulator(cfg)
    graph_simulator.simulate_graphs()
    graph_simulator.simulate_data_discrt()
    # ########### for training synthetic data ###########

    # ########### for transferability experiment ###########
    # for controltag in ['nodenum', 'sparsity', 'graphtype', 'samplesize']:
    #     simulate_diff(controltag)
    ########### for transferability experiment ###########
