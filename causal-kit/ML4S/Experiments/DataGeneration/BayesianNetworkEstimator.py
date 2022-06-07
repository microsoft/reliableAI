import random
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch, BDeuScore
from itertools import permutations
import numpy as np
from Tools import Dataset, Utility
from Tools import blip_helper
from p_tqdm import p_umap

class BayesianNetworkEstimator:
    def __init__(self, bname):
        self.bname = bname
        self.bn = None

    def estimate_bn_from_graph(self, graph_path: str):
        # TODO finish, this is useful in some cases
        data_path = f"../benchmarks/npy10000/{self.bname}.npy"
        dataset = Dataset.Dataset(data_path)
        data = dataset.to_pandas()
        dag = ""
        bn = BayesianNetwork(dag)
        bn.fit(data)
        self.bn = bn
        return bn

    def estimate_bn(self, proxy_model: str):
        """
        This function estimates a bayesian network with baseline method
        Args:
            bname: name of the benchmark
            proxy_model: name of baseline method

        Returns:
            Estimated bayesian network
        """
        assert proxy_model in ["hc", "blip", "notears", "daggnn", "real", "mmpc"], "proxy model not supported!"
        data_path = f"../benchmarks/npy10000/{self.bname}.npy"
        dataset = Dataset.Dataset(data_path)
        data = dataset.to_pandas()
        if proxy_model == "hc":
            dag = HillClimbSearch(data).estimate(scoring_method=BDeuScore(data))
        elif proxy_model == "blip":
            blip_res_path = f"../blip/data/{self.bname}.res"
            dag = blip_helper.get_blip(blip_res_path)
        elif proxy_model == "notears":
            notears_path = f"../notears/data/{self.bname}.txt"
            matrix = np.abs(np.loadtxt(notears_path))
            for i in range(matrix.shape[0]):
                weights = list(matrix[:, i])
                weights: list
                weights.sort(reverse=True)
                top5 = weights[5]
                for j in range(matrix.shape[0]):
                    if matrix[j, i] <= top5: matrix[j, i] = .0
            dag = DAG()
            dag.add_nodes_from([str(i) for i in range(matrix.shape[0])])
            for edge in permutations(range(matrix.shape[0]), 2):
                x, y = edge
                if matrix[x, y] != 0:
                    dag.add_edge(str(x), str(y))
        elif proxy_model == "daggnn":
            daggnn_path = f"../daggnn/data/{self.bname}.csv"
            matrix = np.loadtxt(daggnn_path, delimiter=",")
            dag = DAG()
            dag.add_nodes_from([str(i) for i in range(matrix.shape[0])])
            for edge in permutations(range(matrix.shape[0]), 2):
                x, y = edge
                if matrix[y, x] != 0:
                    dag.add_edge(str(x), str(y))
        elif proxy_model == "mmpc":
            mmpc_path = f"../REAL/data/{self.bname}_mmpc_bn.txt"
            with open(mmpc_path) as f:
                lines = f.readlines()
            dag = DAG()
            dag.add_nodes_from([str(i) for i in range(len(lines))])
            for x, line in enumerate(lines):
                line = line.strip()
                for y, s in enumerate(line):
                    if s == "2":
                        dag.add_edge(str(x), str(y))
        elif proxy_model == "real":
            real_path = f"../REAL/data/{self.bname}_bn.txt"
            with open(real_path) as f:
                lines = f.readlines()
            dag = DAG()
            dag.add_nodes_from([str(i) for i in range(len(lines))])
            for x, line in enumerate(lines):
                line = line.strip()
                for y, s in enumerate(line):
                    if s == "2":
                        dag.add_edge(str(x), str(y))
        bn = BayesianNetwork(dag)
        bn.fit(data)
        self.bn = bn
        return bn

    def generate_random_bn(self, edge_prob=0.5, fit_cpd=True):
        """
        This function generates a random Bayesian network with designated node name
        function in pgmpy doesn't support random bn with customized node names
        Args:
            bname: name of the benchmark
            edge_prob: probability of an edge exists
            fit_cpd: if we use data to fit cpd

        Returns:
            a BayesionNetwork
        """
        data_path = f"../benchmarks/npy10000/{self.bname}.npy"
        dataset = Dataset.Dataset(data_path)
        data = dataset.to_pandas()
        nodes = data.columns.values.tolist()

        # create BN with graph first
        dag = DAG()
        dag.add_nodes_from(nodes)
        bn = BayesianNetwork(dag)
        all_pairs = list(permutations(bn.nodes, 2))
        random.shuffle(all_pairs)
        for edge in all_pairs:
            if Utility.flip_coin(edge_prob):
                x, y = edge
                # try if an edge can be added from node x to y (not breaking acyclic)
                try:
                    bn.add_edge(x, y)
                except:
                    pass

        # fit a cpd with data
        bn.fit(data)
        if not fit_cpd:
            n_states_dict = bn.get_cardinality()

            # randomize cpd
            cpds = []
            for node in bn.nodes():
                parents = list(bn.predecessors(node))
                cpds.append(
                    TabularCPD.get_random(
                        variable=node, evidence=parents, cardinality=n_states_dict
                    )
                )
            bn.cpds = []
            bn.add_cpds(*cpds)
        return bn

    def estimate_all_alpha(self):
        def est_one(cpd: TabularCPD):
            variable = cpd.variable
            values = cpd.get_values().T
            if values.shape[0] > 500:
                return variable, None
                # trimmed_values = values[:500]
                # trimmed_alpha = [Utility.DirichletAlphaEstimation(np.array([value])) * .25 for value in trimmed_values]
                # alpha = copy(trimmed_alpha)
                # while len(alpha) < values.shape[0]:
                #     alpha.append(random.choice(trimmed_alpha))
                # return variable, alpha
            else:
                return variable, [Utility.DirichletAlphaEstimation(np.array([value])) for value in values]

        assert self.bn is not None, "No estimated Bayesian Network!"
        all_alpha = {i[0]: i[1] for i in p_umap(est_one, self.bn.get_cpds())}
        return all_alpha