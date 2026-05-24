import os
import random
import copy
import pickle
import networkx as nx
import numpy as np
from pgmpy.base import DAG
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BDeuScore
from itertools import permutations
from castle.algorithms import PC

import sys
sys.path.append('../')

from baselines.blip import run_blip
from base.dataloader import Dataset
from tools.utils import flip_coin, dirichlet_alpha_estimation
from pathos.multiprocessing import Pool, cpu_count
from tools.convert import adj_to_dag


class BayesianNetworkEstimator:
    def __init__(self, benchmark_name, intervention_size):
        self.bn = None
        self.benchmark_name = benchmark_name
        self.intervention_size = intervention_size
    
    def generate_random_bn(self, read_dir, edge_prob=0.5):
        """This function generates a random bayesian network with dataset and designated node name. (function in pgmpy doesn't support customized node names)

        Args:
            edge_prob (float, optional): probability of an edge exists. Defaults to 0.5. (The smaller the value, the sparser the graph)
            fit_cpd (bool, optional): if we use data to fit cpd. Defaults to True.

        Returns:
            bn (class): a random Bayesian Networks compatible with observational dataset
        """

        # First, get dataset and nodes
        dataset = Dataset(data_path=read_dir)
        data = dataset.to_pandas()
        nodes = data.columns.values.tolist()
        
        # Second, create BN with base nodes(DAG)
        dag = DAG()
        dag.add_nodes_from(nodes)
        for edge in list(permutations(dag.nodes, 2)):
            if flip_coin(edge_prob):
                x, y = edge
                if dag.has_edge(x, y) or dag.has_edge(y, x):
                    continue
                test_bn = BayesianNetwork(dag)
                try:  # try if an edge can be added from node x to y (not breaking acyclic)
                    test_bn.add_edge(x, y)
                    dag.add_edge(x, y)
                except:
                    pass
        
        # Third, Remove edges that violate assumptions (here we remove all env->env/sys->env in proxy dag)
        nodes = list(dag.nodes)[-self.intervention_size:]
        for node in nodes:
            parents = list(dag.predecessors(node))
            for parent in parents:
                dag.remove_edge(parent, node)
        
        # Fourth, estimates the CPD for each variable based on a dataset, otherwise, randomize CPD
        self.bn = BayesianNetwork(dag)
        self.bn.fit(data)
        return self.bn
    
    def generate_vicinal_bn(self, read_dir, proxy_model, exp_number):
        """This function estimates a bayesian network with proxy_model
        
        Args:
            proxy_model (str): name of proxy method

        Returns:
            bn (class): Estimated bayesian network
        """
        # First, get observation dataset and nodes
        dataset = Dataset(data_path=read_dir)
        data = dataset.to_pandas()

        # Second, create BN with proxy model(DAG)
        if proxy_model == "blip":
            possible_file = f'../baselines/blip/temp_{exp_number}/{self.benchmark_name}_graph.txt'
            if os.path.exists(possible_file):
                with open(possible_file, 'rb') as f:
                    dag = adj_to_dag(np.loadtxt(possible_file))
            else:
                dag = adj_to_dag(run_blip.dag_learning(data, self.benchmark_name, exp_number))
        elif proxy_model == "hc":
            dag = HillClimbSearch(data).estimate(scoring_method=BDeuScore(data), show_progress=False)
        elif proxy_model == "pc":
            pc = PC(alpha=1e-3, variant='original', ci_test='chi2')
            pc.learn(data)
            dag = adj_to_dag(pc.causal_matrix)
        
        # Third, Remove edges that violate assumptions 

        # method 1 (here we remove all sys->env / env->env in proxy dag)
        env_nodes = list(dag.nodes)[-self.intervention_size:]
        for y in env_nodes:
            parents = list(dag.predecessors(y))
            for x in parents:
                dag.remove_edge(x, y)
        
        # # method 2 (here we remove all env->env and inverse sys->env in proxy dag)
        # env_nodes = list(dag.nodes)[-self.intervention_size:]
        # sys_nodes = list(dag.nodes)[:-self.intervention_size]
        # for x, y in list(dag.edges()):
        #     if x in env_nodes and y in env_nodes:
        #         dag.remove_edge(x, y)
        #     if x in sys_nodes and y in env_nodes:
        #         dag.remove_edge(x, y)
        #         dag.add_edge(y, x)
        

        # Fourth, estimates the CPD for each variable based on a observation + interventional dataset
        self.bn = BayesianNetwork(dag)
        self.bn.fit(data)
        
        return self.bn
    
    
    def estimate_all_alpha(self):
        """This function is used to estimate all dirichlet distribution parameter alpha of each variable

        Returns:
            all_alpha (dict): all dirichlet distribution parameter alpha of each variable
        """
        def est_one(cpd):
            variable_name = cpd.variable
            variable_values = cpd.get_values().T
            if variable_values.shape[0] > 500:
                # trimmed_values = variable_values[:500]
                # trimmed_alpha = [dirichlet_alpha_estimation(np.array([value])) * .25 for value in trimmed_values]
                # alpha = trimmed_alpha.copy()
                # while len(alpha) < variable_values.shape[0]:
                #     alpha.append(random.choice(trimmed_alpha))
                # return variable_name, alpha
                return variable_name, None
            else:
                return variable_name, [dirichlet_alpha_estimation(np.array([value])) for value in variable_values]
        
        with Pool(cpu_count()) as p:
            all_alpha = {var: alpha for var, alpha in p.map(est_one, self.bn.get_cpds())}
        
        return all_alpha