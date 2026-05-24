import time
import os
import random
import copy
import numpy as np
from scipy.stats import truncnorm
from itertools import permutations
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from multiprocessing import Pool, cpu_count

from pgmpy.estimators import BDeuScore
from base.dataloader import Dataset

from proxy_generation.bn_estimator import BayesianNetworkEstimator
from proxy_generation.dag_simulator import DagSimulator
from base.dag import DiGraph
from tools.metric import get_compared_components, metric_skeleton_level, metric_target_level, metric_cpdag_level
from tools.utils import flip_coin, dirichlet_alpha_estimation
from tools.sampling import sample_data_from_bn
from tools.convert import bn_to_adj


class SiblingGraphGenerator:
    def __init__(self, args):
        self._unpack_parameters(args)
        self.est_bn = None
        self.all_alpha = None
        self.folder_dir = f"../datasets/experiment"
        os.makedirs(f'{self.folder_dir}/siblings/{self.exp_number}/datasets/{self.benchmark_name}', exist_ok=True)
        self.bn_estimator = BayesianNetworkEstimator(self.benchmark_name, self.intervention_size)
        self.accept, self.reject = 0, 0
    
    def _unpack_parameters(self, args):
        self.benchmark_name = args.benchmark_name
        self.exp_number = args.exp_number
        self.proxy_type = args.proxy_type
        self.proxy_model = args.proxy_model
        self.case_num = args.case_num
        self.proxy_degree = args.proxy_degree
        self.mixed_samples = args.mixed_samples
        self.intervention_size = args.intervention_size
        self.purely_random = args.purely_random
        if self.purely_random == True:
            self.proxy_type = 'random'
        self.use_int_bais = args.use_int_bais
        self.mcmc_step = args.mcmc_step
    
    @staticmethod
    def _export_bn_adj_matrix(bn: BayesianNetwork):
        """This function is used to obtain the adjacency matrix of a given Bayesian network

        Args:
            bn (BayesianNetwork): a given Bayesian network

        Returns:
            adj_matrix (np.array): adjacency matrix of given Bayesian network
        """
        adj_matrix = np.zeros((len(bn.nodes), len(bn.nodes)), dtype=int)
        for x, y in bn.edges:
            adj_matrix[int(x), int(y)] = 1
        return adj_matrix
    
    def edge_manipulation(self, bn, all_alpha):
        new_bn = bn.copy()
        modified = set()
        all_pairs = list(permutations(bn.nodes, 2))
        random.shuffle(all_pairs)

        for x, y in all_pairs:
            if y in modified:
                continue
            
            if new_bn.has_edge(x, y): # perform edge deletion with probability of self.proxy_degree%
                if len(list(new_bn.predecessors(y))) == 1:  # avoid lonely node
                    continue

                if flip_coin(self.proxy_degree):
                # if True:
                    cpd = new_bn.get_cpds(y)
                    new_bn.remove_cpds(y)
                    cpd.marginalize([x])
                    cpd.normalize()
                    new_bn.remove_edge(x, y)
                    new_bn.add_cpds(cpd)
                    modified.add(y)
                else:
                    continue
            else: # perform edge insertion with probability of self.proxy_degree%
                if self.use_int_bais:
                    if int(y) >= len(new_bn.nodes) - self.intervention_size:  # only for sys->sys / env->sys
                        continue
                
                if all_alpha[y] is None:
                    continue
                
                test_bn = new_bn.copy() # try if an edge can be added from node x to y (not breaking acyclic)
                try:
                    test_bn.add_edge(x, y)
                    if flip_coin(self.proxy_degree) and len(new_bn.minimal_dseparator(x, y)) <= 3:
                    # if len(new_bn.minimal_dseparator(x, y)) <= 3:
                        cpd = new_bn.get_cpds(y)
                        new_bn.remove_cpds(y)
                        variable = cpd.variable
                        variable_card = cpd.variable_card
                        values = np.array([np.random.dirichlet(alpha * .01) for alpha in all_alpha[variable] for _ in range(new_bn.get_cardinality(x))]).T.tolist()
                        evidence = [x] + cpd.variables[1:] if len(cpd.variables) > 1 else [x]
                        evidence_card = np.hstack([[new_bn.get_cardinality(x)], cpd.cardinality[1:]]) if len(cpd.variables) > 1 else np.array([new_bn.get_cardinality(x)])
                        state_name = cpd.state_names
                        state_name[x] = new_bn.get_cpds(x).state_names[x]
                        new_bn.add_edge(x, y)
                        new_bn.add_cpds(TabularCPD(variable, variable_card, values, evidence, evidence_card, state_name).normalize(inplace=False))
                        modified.add(y)
                except:
                    continue
            # break
        return self._export_bn_adj_matrix(new_bn), new_bn
    
    def modify_edge_manipulation(self, bn, all_alpha):
        new_bn = bn.copy()
        modified = set()
        all_pairs = list(permutations(bn.nodes, 2))
        exist_pairs = bn.edges
        edge_del_prob = len(exist_pairs) / len(all_pairs)
        edge_ins_prob = (len(all_pairs) - len(exist_pairs)) / len(all_pairs)

        for x, y in all_pairs:
            if y in modified:
                continue
            
            if new_bn.has_edge(x, y): # perform edge deletion with probability of self.proxy_degree%
                if flip_coin(edge_del_prob):
                    continue
                if len(list(new_bn.predecessors(y))) == 1:  # avoid lonely node
                    continue
                cpd = new_bn.get_cpds(y)
                new_bn.remove_cpds(y)
                cpd.marginalize([x])
                new_bn.remove_edge(x, y)
                new_bn.add_cpds(cpd)
                modified.add(y)
            else: # perform edge insertion with probability of self.proxy_degree%
                if flip_coin(edge_ins_prob):
                    continue
                if int(y) >= len(new_bn.nodes) - self.intervention_size:  # avoid sys->env / env->env
                    continue

                if all_alpha[y] is None:
                    continue
                
                test_bn = new_bn.copy() # try if an edge can be added from node x to y (not breaking acyclic)
                try:
                    test_bn.add_edge(x, y)
                    if len(new_bn.minimal_dseparator(x, y)) <= 3:
                        cpd = new_bn.get_cpds(y)
                        new_bn.remove_cpds(y)
                        variable = cpd.variable
                        variable_card = cpd.variable_card
                        values = np.array([np.random.dirichlet(alpha * .01) for alpha in all_alpha[variable] for _ in range(new_bn.get_cardinality(x))]).T.tolist()
                        evidence = [x] + cpd.variables[1:] if len(cpd.variables) > 1 else [x]
                        evidence_card = np.hstack([[new_bn.get_cardinality(x)], cpd.cardinality[1:]]) if len(cpd.variables) > 1 else np.array([new_bn.get_cardinality(x)])
                        state_name = cpd.state_names
                        state_name[x] = new_bn.get_cpds(x).state_names[x]
                        new_bn.add_edge(x, y)
                        new_bn.add_cpds(TabularCPD(variable, variable_card, values, evidence, evidence_card, state_name).normalize(inplace=False))
                        modified.add(y)
                    else:
                        continue
                except:
                    continue
        
        return self._export_bn_adj_matrix(new_bn), new_bn
    
    def pure_synthetic(self, bn: BayesianNetwork):
        np.random.seed(42)
        node_num = len(bn.nodes)
        edge_num = np.round(np.random.uniform(1.2, 1.7) * node_num).astype(int)
        dag_simulator = DagSimulator()
        adj_matrix = dag_simulator.simulate_graph(node_num, edge_num)
        # adj_matrix[:, -self.intervention_size:] = 0
        bn = dag_simulator.simulate_discrete_bn(adj_matrix)
        return adj_matrix, bn
    
    def generate_one_sibling_graph(self, index):
        """This function is used to generate vicnal graph through proxy algorithm type, further obtain the corresponding forward sampling dataset

        Args:
            parameter (tuple): (proxy algorithm type, graph index)
        """
        # First, prepare to proxy algorithm type, graph index, file name of proxy vicinal graph and forward sampling dataset
        graph_path = f"{self.folder_dir}/siblings/{self.exp_number}/datasets/{self.benchmark_name}/{self.proxy_type}_{index}.txt"
        data_path = f"{self.folder_dir}/siblings/{self.exp_number}/datasets/{self.benchmark_name}/{self.proxy_type}_{index}.npy"

        # Second, use different proxy algorithms for the base vicinal graph
        if self.proxy_type == 'vicinal':
            adj_matrix, bn = self.edge_manipulation(self.est_bn, self.all_alpha)
        elif self.proxy_type == 'random':
            # adj_matrix, bn = self.pure_synthetic(self.est_bn)
            adj_matrix, bn = self.edge_manipulation(self.est_bn, self.all_alpha)
        
        alpha = min(1, np.exp(self.bdeu.score(bn) / self.norm_size) / np.exp(self.base_val / self.norm_size))  # accept rate alpha
        
        # Third, MCMC generate training graph and dataset
        # if True:
        
        if random.uniform(0, 1) < alpha:
            try:
                bn.check_model()  # check whether the proxy vicinal graph is legal. If yes, save the graph and the forward sampling dataset. Otherwise, skip
                sim_data = sample_data_from_bn(bn, self.mixed_samples)
                
                with open(graph_path, 'wb') as f:
                    np.savetxt(f, adj_matrix, fmt='%i')
                with open(data_path, 'wb') as f:
                    np.save(f, sim_data)
            except:
                print('skip')
        else:
            sim_data = sample_data_from_bn(self.est_bn, self.mixed_samples)
            with open(graph_path, 'wb') as f:
                np.savetxt(f, self._export_bn_adj_matrix(self.est_bn), fmt='%i')
            with open(data_path, 'wb') as f:
                np.save(f, sim_data)
    
    def generate_one_sibling_graph_withmcmc(self, index):
        """This function is used to generate vicnal graph through proxy algorithm type, further obtain the corresponding forward sampling dataset

        Args:
            parameter (tuple): (proxy algorithm type, graph index)
        """
        # First, prepare to proxy algorithm type, graph index, file name of proxy vicinal graph and forward sampling dataset
        graph_path = f"{self.folder_dir}/siblings/{self.exp_number}/datasets/{self.benchmark_name}/{self.proxy_type}_{index}.txt"
        data_path = f"{self.folder_dir}/siblings/{self.exp_number}/datasets/{self.benchmark_name}/{self.proxy_type}_{index}.npy"

        # Second, use different proxy algorithms for the base vicinal graph, MCMC generate training graph and dataset
        for idx in range(self.mcmc_step):
            adj_matrix, bn = self.edge_manipulation(self.est_bn, self.all_alpha)
            alpha = min(1, np.exp(self.bdeu.score(bn) / self.norm_size) / np.exp(self.bdeu.score(self.est_bn) / self.norm_size))  # accept rate alpha
            if random.uniform(0, 1) < alpha:
                self.est_bn = bn.copy()
                self.all_alpha = {cpd.variable:[dirichlet_alpha_estimation(np.array([value])) for value in cpd.get_values().T] for cpd in self.est_bn.get_cpds()}
            else:
                # self.est_bn = bn.copy()
                break
        
        
        try:
            self.est_bn.check_model()  # check whether the proxy vicinal graph is legal. If yes, save the graph and the forward sampling dataset. Otherwise, skip
            sim_data = sample_data_from_bn(self.est_bn, self.mixed_samples)
            
            with open(graph_path, 'wb') as f:
                np.savetxt(f, adj_matrix, fmt='%i')
            with open(data_path, 'wb') as f:
                np.save(f, sim_data)
        except:
            print('skip')
    
    
    def generate_sibling(self, logger):

        aug_data_path = f"{self.folder_dir}/original/{self.exp_number}/aug_samples/{self.benchmark_name}_aug_dataset.npy"
        
        if self.proxy_type == "vicinal":
            self.est_bn = self.bn_estimator.generate_vicinal_bn(aug_data_path, self.proxy_model, self.exp_number)
        elif self.proxy_type == "random":
            self.est_bn = self.bn_estimator.generate_random_bn(aug_data_path, 0.2)
        self.all_alpha = self.bn_estimator.estimate_all_alpha()
        
        logger.info(f"Get Base BN from {self.proxy_type} algorithm. Model Check: {self.est_bn.check_model()}\n")
        
        data = Dataset(data_path=aug_data_path)
        self.bdeu = BDeuScore(data.to_pandas())
        self.base_val = self.bdeu.score(self.est_bn)
        self.norm_size = data.SampleSize * data.VarCount
        
        proxy_graph_path = f"{self.folder_dir}/siblings/{self.exp_number}/proxy_predict/{self.benchmark_name}_proxy_graph.txt"
        os.makedirs(f'{self.folder_dir}/siblings/{self.exp_number}/proxy_predict', exist_ok=True)
        
        with open(proxy_graph_path, 'wb') as f:
            np.savetxt(f, bn_to_adj(self.est_bn), fmt='%i')
        
        start_time = time.time()
        if self.purely_random:
            pass
        else:
            with Pool(cpu_count()) as p:
                p.map(self.generate_one_sibling_graph, range(self.case_num))
        end_time = time.time()
        logger.info(f'vicinal generate finished, total time is {round(end_time - start_time, 2)}\n')
        