from itertools import permutations
from typing import Dict
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

from p_tqdm import p_umap
from scipy.stats import truncnorm

import sys, os, random, pickle, pprint, shutil
import argparse
import yaml

from Tools import Graph, Utility
from Experiments.DataGeneration import GenerateData
from Experiments.DataGeneration.BayesianNetworkEstimator import BayesianNetworkEstimator

sys.path.append(os.path.dirname(os.getcwd()))


class SiblingGraphGenerator:
    def __init__(self, benchmark_name, config):
        config["benchmark_name"] = benchmark_name
        self._unpack_parameters(**config)
        self.name_list = ['hc', 'cpd', 'struc', 'synthetic']

        # create save directory
        self.sv_dir = f"{self.data_folder}/{self.dir_name}/{self.bname}"
        if os.path.exists(self.sv_dir) and os.path.isdir(self.sv_dir):
            assert self.remove_exist_data, f"Warning! {self.sv_dir} already exists\n" \
                                           f"set param remove_exist_data to true to remove data and proceed.\n" \
                                           f"or use another path to save data."
            shutil.rmtree(self.sv_dir)
        os.makedirs(self.sv_dir, exist_ok=True)

        # save config in the data folder
        with open(os.path.join(self.sv_dir, 'config.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

        self.est_bn, self.all_alpha = None, None
        self.bn_estimator = BayesianNetworkEstimator(bname)

    def _unpack_parameters(self, benchmark_name, data_folder, dir_name, 
                           case0_num=0, case1_num=0, case2_num=1000, case3_num=0, n_samples=10000, 
                           random_bn=True, remove_exist_data=True):
        self.bname = benchmark_name
        self.data_folder, self.dir_name = data_folder, dir_name
        self.case0_num, self.case1_num, self.case2_num, self.case3_num = case0_num, case1_num, case2_num, case3_num
        self.n_samples = n_samples
        self.random_bn = random_bn
        self.remove_exist_data = remove_exist_data

    @staticmethod
    def _export_bn(bn: BayesianNetwork):
        # assert bn.check_model()
        B = np.zeros((len(bn.nodes), len(bn.nodes)), dtype=int)
        for edge in bn.edges:
            x, y = edge
            x, y = int(x), int(y)
            B[x, y] = 1
        return B

    def cpd_replacement(self, bn: BayesianNetwork, all_alpha: Dict):
        new_bn = bn.copy()
        cpd: TabularCPD
        for cpd in bn.get_cpds():
            cpd = cpd.copy()
            variable = cpd.variable
            if all_alpha[variable] is None:
                # print("skip", variable)
                continue
            variable_card = cpd.cardinality[0]
            evidence = cpd.variables[1:] if len(cpd.variables) > 1 else None
            evidence_card = cpd.cardinality[1:] if len(cpd.variables) > 1 else None
            state_name = cpd.state_names
            values = []
            for alpha in all_alpha[variable]:
                values.append(np.random.dirichlet(alpha * 0.2))
            values = np.array(values).T
            values = values.tolist()
            new_bn.remove_cpds(cpd)
            new_cpd = TabularCPD(
                variable, variable_card, values, evidence, evidence_card, state_name
            )
            new_cpd.normalize()
            new_bn.add_cpds(new_cpd)
        return self._export_bn(new_bn), new_bn

    def edge_manipulation(self, bn: BayesianNetwork, all_alpha: Dict):
        new_bn = bn.copy()
        manipulation_num = 0
        modified = set()
        all_pairs = list(permutations(bn.nodes, 2))
        random.shuffle(all_pairs)
        for edge in all_pairs:
            x, y = edge
            if y in modified:
                continue
            has_edge = new_bn.has_edge(x, y)
            in_degree = new_bn.in_degree(x)
            out_degree = new_bn.out_degree(y)

            if not has_edge:
                pass
                if all_alpha[y] is None:
                    continue
                new_dag = new_bn.copy()

                # try if an edge can be added from node x to y (not breaking acyclic)
                try:
                    new_dag.add_edge(x, y)
                    allowed = True
                    new_dag.remove_edge(x, y)
                except:
                    allowed = False

                # an edge can be added
                if allowed:
                    if Utility.flip_coin(.1) and len(new_bn.minimal_dseparator(x, y)) <= 3:
                        cpd: TabularCPD
                        cpd = new_bn.get_cpds(y)
                        new_bn.remove_cpds(y)
                        variable = cpd.variable
                        variable_card = cpd.variable_card
                        evidence = cpd.variables[1:] if len(cpd.variables) > 1 else None
                        evidence_card = cpd.cardinality[1:] if len(cpd.variables) > 1 else None
                        if evidence is None:
                            evidence = [x]
                            evidence_card = np.array([new_bn.get_cardinality(x)])
                        else:
                            evidence = [x] + evidence
                            evidence_card = np.hstack([[new_bn.get_cardinality(x)], evidence_card])
                        state_name = cpd.state_names
                        state_name[x] = bn.get_cpds(x).state_names[x]
                        values = []
                        for i in range(new_bn.get_cardinality(x)):
                            for alpha in all_alpha[variable]:
                                values.append(np.random.dirichlet(alpha * .01))
                        values = np.array(values).T
                        # print(new_bn.get_cardinality(x), np.array(all_alpha[variable]).shape, values.shape, evidence, evidence_card)
                        values = values.tolist()
                        new_bn.add_edge(x, y)
                        new_bn.add_cpds(TabularCPD(
                            variable, variable_card, values, evidence, evidence_card, state_name
                        ).normalize(inplace=False))
                        modified.add(y)
                        manipulation_num += 1
            elif has_edge:
                # perform edge deletion with probability of 10%
                if Utility.flip_coin(.1):
                    cpd: TabularCPD
                    cpd = new_bn.get_cpds(y)
                    new_bn.remove_cpds(y)
                    cpd.marginalize([x])
                    cpd.normalize()
                    new_bn.remove_edge(x, y)
                    new_bn.add_cpds(cpd)
                    modified.add(y)
                    manipulation_num += 1
        # print(manipulation_num)
        return self._export_bn(new_bn), new_bn

    def pure_synthetic(self, bn: BayesianNetwork, *args):
        node_mu = len(bn.nodes)
        edge_mu = len(bn.edges)
        if node_mu < 20:
            node_sigma = 3
        elif node_mu < 50:
            node_sigma = 8
        else:
            node_sigma = 15
        if edge_mu < 20:
            edge_sigma = 3
        elif edge_mu < 50:
            edge_sigma = 8
        else:
            edge_sigma = 15

        node_num = truncnorm((0.7 * node_mu - node_mu) / node_sigma, (1.3 * node_mu - node_mu) / node_sigma,
                             loc=node_mu,
                             scale=node_sigma). \
            rvs().round().astype(int)
        edge_num = truncnorm((0.7 * edge_mu - edge_mu) / edge_sigma, (1.3 * edge_mu - edge_mu) / edge_sigma,
                             loc=edge_mu,
                             scale=edge_sigma). \
            rvs().round().astype(int)
        simulator = GenerateData.DagSimulator()
        B = simulator.simulate(node_num, edge_num, "SF")
        bn = GenerateData.simulate_discrete_bn(B, return_bn=True)
        return B, bn

    def no_modification(self, bn: BayesianNetwork, *args):
        return self._export_bn(bn), bn

    def generate_one_sibling_graph(self, parameter):
        case, idx = parameter
        function_list = [self.no_modification, self.cpd_replacement, self.edge_manipulation, self.pure_synthetic]
        graphtxtpath = os.path.join(self.sv_dir, f"{self.name_list[case]}_{idx}.txt")
        datapath = os.path.join(self.sv_dir, f"{self.name_list[case]}_{idx}.npy")
        B, bn = function_list[case](self.est_bn, self.all_alpha)
        try:
            bn.check_model()
            np.savetxt(graphtxtpath, B, fmt='%i')
            sim_data = GenerateData.sample_data_from_bn(bn, self.n_samples, B.shape[0])
            np.save(datapath, sim_data)
        except:
            print("error, skipped")

    def generate_sibling(self):
        # load graph
        graph_path = f"../benchmarks/{bname}_graph.txt"
        truth = Graph.DiGraph(graph_path)

        if not self.random_bn:
            # if the estimated bayesian network doesn't exist, run estimation process
            # otherwise load previous result
            if not os.path.exists(os.path.join(self.sv_dir, "est_bn.pkl")):
                self.est_bn = self.bn_estimator.estimate_bn("blip")
                self.all_alpha = self.bn_estimator.estimate_all_alpha()
                with open(os.path.join(self.sv_dir, "alpha.pkl"), "wb") as f:
                    pickle.dump(self.all_alpha, f)
                with open(os.path.join(self.sv_dir, "est_bn.pkl"), "wb") as f:
                    pickle.dump(self.est_bn, f)
            else:
                with open(os.path.join(self.sv_dir, "alpha.pkl"), "rb") as f:
                    self.all_alpha = pickle.load(f)
                with open(os.path.join(self.sv_dir, "est_bn.pkl"), "rb") as f:
                    self.est_bn = pickle.load(f)
        else:
            print("Use randomly generated Bayesian Network")
            self.est_bn = self.bn_estimator.generate_random_bn(0.02)
            self.all_alpha = self.bn_estimator.estimate_all_alpha()

        print("Proxy Model Performance", Utility.compute_metric(self.est_bn, truth))
        est_bn: BayesianNetwork
        print("Model Check", self.est_bn.check_model())

        # create a list of tasks with their case and index
        tasks = [(1, i) for i in range(self.case1_num)]
        tasks += [(2, i) for i in range(self.case2_num)]
        tasks += [(3, i) for i in range(self.case3_num)]
        tasks += [(0, i) for i in range(self.case0_num)]

        p_umap(self.generate_one_sibling_graph, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfile", help="benchmark name file", default="../configs/benchmark.yml")
    parser.add_argument("--config", help="graph generation config file", default="../configs/generate_sibling.yml")
    args = parser.parse_args()
    with open(args.bfile, 'r') as file:
        benchmark_names = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.config, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    # set random seed for reproducibility
    if cfg['seed']:
        np.random.seed(cfg['seed'])
    else:
        np.random.seed()
    cfg.pop('seed')

    for bname in benchmark_names["benchmarks"]:
        print(f"Working on benchmark {bname}")
        print(f"config detail:\n{pprint.pformat(cfg)}")
        generator = SiblingGraphGenerator(bname, cfg)
        generator.generate_sibling()
