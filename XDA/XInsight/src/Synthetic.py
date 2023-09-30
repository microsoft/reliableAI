# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env python
#-*- coding: utf-8 -*-
import itertools
import sys, os, random
from turtle import color
from typing import List

import pandas as pd
from tqdm import tqdm
from p_tqdm import p_umap
import pickle
sys.path.append(os.path.dirname(os.getcwd()))
import igraph as ig
import numpy as np
from scipy.stats import truncnorm
from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edges import Edges
from causallearn.graph.Endpoint import Endpoint

from src.DAG2PAG import dag2pag
from src.FCI import get_color_edges
from src.CausalSemanticModel import Edge
from src.logger import logging
import src.CausalSemanticModel as CSM

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.
       partly stolen from https://github.com/xunzheng/notears
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF
        restrict_indegree (bool): set True to restrict nodes' indegree, specifically,
            for ER: from skeleton to DAG, randomly acyclic orient each edge.
                    if a node's degree (in+out) is large, we expect more of its degree is allocated for out, less for in.
                    so permute: the larger degree, the righter in adjmat, and
                                after the lower triangle, the lower upper bound of in-degree
            for SF: w.r.t. SF natrue that in-degree may be exponentially large, but out-degree is always low,
                    explicitly set the MAXTOLIND. transpose in/out when exceeding. refer to _transpose_in_out(B)
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _acyclic_orientation(B_und):
        # pre-randomized here. to prevent i->j always with i>j
        return np.tril(B_und, k=-1)

    def _remove_isolating_node(B):
        non_iso_index = np.logical_or(B.any(axis=0), B.any(axis=1))
        return B[non_iso_index][:, non_iso_index]

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
    elif graph_type == 'SF':
        G_und = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False, outpref=True, power=-3)
    else:
        raise ValueError('unknown graph type')

    B_und = _graph_to_adjmat(G_und)
    B_und = _random_permutation(B_und)
    B = _acyclic_orientation(B_und)
    B = _remove_isolating_node(B)
    B_perm = _random_permutation(B).astype(int)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def simulate_cards(B, card_param=None):
    if card_param == None:
        card_param = {'lower': 2, 'upper': 6, 'mu': 3.5, 'basesigma': 1.5} # truncated normal distribution
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
    cards = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).\
        rvs(size=B.shape[0]).round().astype(int)
    return cards
    
def simulate_discrete(B, n, card_param=None, uniform_alpha_param=None, return_bn=False):
    # try:
    if uniform_alpha_param == None:
        uniform_alpha_param = {'lower': 0.1, 'upper': 1.0}
    def _random_alpha():
        return np.random.uniform(uniform_alpha_param['lower'], uniform_alpha_param['upper'])

    cards = simulate_cards(B, card_param=card_param)
    diEdges = list(map(lambda x: (str(x[0]), str(x[1])), np.argwhere(B == 1))) # list of tuples
    bn = BayesianModel(diEdges) # so isolating nodes will echo error
    fd_num = len(cards) * 0.15
    fd_edges = []
    for node in range(len(cards)):
        parents = np.where(B[:, node] == 1)[0].tolist()
        parents_card = [cards[prt] for prt in parents]

        if len(parents) == 1 and fd_num > 0:
            rand_ps = []
            for _ in range(int(np.prod(parents_card))):
                dist = np.zeros(cards[node])
                dist[np.random.randint(cards[node])] = 1.
                rand_ps.append(dist)
            rand_ps = np.array(rand_ps).T.tolist()
            fd_edges.append((parents[0], node))
            fd_num -= 1
        else:
            rand_ps = np.array([np.random.dirichlet(np.ones(cards[node]) * _random_alpha()) for _ in
                                range(int(np.prod(parents_card)))]).T.tolist()
        cpd = TabularCPD(str(node), cards[node], rand_ps,
                        evidence=list(map(str, parents)), evidence_card=parents_card)
        cpd.normalize()
        bn.add_cpds(cpd)
    inference = BayesianModelSampling(bn)
    df = inference.forward_sample(size=n, show_progress = False)
    df = df[[str(i) for i in range(len(cards))]]
    return df.to_numpy().astype(np.int), fd_edges

def simulate_graph(node_num:int, edge_num:int, graph_type:str):
    logging.info("GENERATE SYNTHETIC DAG")
    B = simulate_dag(node_num, edge_num, graph_type)
    logging.info("FORWARD SAMPLING")
    data, fd_edges = simulate_discrete(B, n=10000)
    nodes = []
    node_num = B.shape[0]
    for i in range(node_num):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        node.add_attribute("col_name", f"col_{i}")
        nodes.append(node)
    dag = GeneralGraph(nodes)

    for i in range(node_num):
        for j in range(node_num):
            if B[i,j] == 1: dag.add_edge(Edges().directed_edge(nodes[i], nodes[j]))
    logging.info("GENERATE PAG")
    pag = dag2pag(dag, random.choices(nodes, k=int(node_num * 0.05)))
    pag_fd_edge = []
    for fd_edge in fd_edges:
        x, y = fd_edge
        node_x, node_y = None, None
        for node in pag.get_nodes():
            node:GraphNode
            if node.get_attribute("id") == x: node_x = node
            if node.get_attribute("id") == y: node_y = node
        if node_x is not None and node_y is not None:
            pag_fd_edge.append((node_x.get_attribute("col_name"), node_y.get_attribute("col_name")))
    #         if pag.is_adjacent_to(node_x, node_y):
    #             edge = pag.get_edge(node_x, node_y)
    #             pag.remove_edge(edge)
    #         pag.add_edge(Edges().directed_edge(node_x, node_y))
    # pag = dag2pag(dag, [])
    observed_data = {}
    for node in pag.get_nodes():
        node: GraphNode
        node_id = node.get_attribute("id")
        node_name = node.get_attribute("col_name")
        observed_data[node_name] = data[:, node_id]
    df = pd.DataFrame(observed_data)
    return pag, df, pag_fd_edge

def get_pag_edges(pag: GeneralGraph, col_names:List[str]=None):
    if col_names is not None:
        for node in pag.get_nodes():
            node: GraphNode
            node.add_attribute("col_name", col_names[node.get_attribute("id")])
    edges = get_color_edges(pag)
    csm_edges = []
    for edge in edges:
        edge: Edge
        node1 = edge.node1
        node2 = edge.node2
        node1: GraphNode
        node2: GraphNode
        if edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.CIRCLE:
            csm_edge = CSM.Edge(node1.get_attribute("col_name"), node2.get_attribute("col_name"), CSM.Edge.EdgeType.NonDirect)
        elif edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW:
            csm_edge = CSM.Edge(node1.get_attribute("col_name"), node2.get_attribute("col_name"), CSM.Edge.EdgeType.Direct)
        elif edge.endpoint1 == Endpoint.CIRCLE and edge.endpoint2 == Endpoint.ARROW:
            csm_edge = CSM.Edge(node1.get_attribute("col_name"), node2.get_attribute("col_name"), CSM.Edge.EdgeType.SemiDirect)
        elif edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.ARROW:
            csm_edge = CSM.Edge(node1.get_attribute("col_name"), node2.get_attribute("col_name"), CSM.Edge.EdgeType.BiDirect)
        elif edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL:
            csm_edge = CSM.Edge(node2.get_attribute("col_name"), node1.get_attribute("col_name"), CSM.Edge.EdgeType.Direct)
        elif edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.CIRCLE:
            csm_edge = CSM.Edge(node2.get_attribute("col_name"), node1.get_attribute("col_name"), CSM.Edge.EdgeType.SemiDirect)
        csm_edges.append(csm_edge)
    # csm = CSM.CausalSemanticModel(col_names, csm_edges)
    return csm_edges

def compare_pag(est_edges: List[Edge], true_edges: List[Edge], partial: bool=True):
    
    true_positive = 0
    skl_true_positive = 0
    for true_edge in true_edges:
        for est_edge in est_edges:
            if Edge.edge_match(true_edge, est_edge):
                skl_true_positive += 1
            if Edge.full_match(true_edge, est_edge):
                true_positive += 1
                break
            elif partial and Edge.partial_match(true_edge, est_edge):
                true_positive += 1
                break
    precision = true_positive / len(est_edges)
    recall = true_positive / len(true_edges)
    f1 = 2 * (precision * recall) / (precision + recall)

    skl_precision = skl_true_positive / len(est_edges)
    skl_recall = skl_true_positive / len(true_edges)
    skl_f1 = 2 * (skl_precision * skl_recall) / (skl_precision + skl_recall)

    return {"f1": f1, "precision": precision, "recall": recall, "skl_f1": skl_f1, "skl_precision": skl_precision, "skl_recall": skl_recall}

def run_one(node_num, idx):
    dataset_path = f"data/synthetic/{node_num}-{idx}.csv"
    pag_path = f"data/synthetic/{node_num}-{idx}-pag.pkl"
    fd_path = f"data/synthetic/{node_num}-{idx}-fd.pkl"
    
    # with open(pag_path, "rb") as f:
    #     pag = pickle.load(f)
    pag, df, fd_edges = simulate_graph(node_num, int(node_num*1.2), "ER")
    with open(pag_path, "wb") as f:
        pickle.dump(pag, f)
    with open(fd_path, "wb") as f:
        pickle.dump(fd_edges, f)
    df.to_csv(dataset_path, index=False)
    return 0

def run_one2(node_num, idx):
    dataset_path = f"data/synthetic/{node_num}-{idx}.csv"
    pag_path = f"data/synthetic/{node_num}-{idx}-pag.pkl"
    fd_path = f"data/synthetic/{node_num}-{idx}-fd.pkl"
    xl_path = f"data/synthetic/{node_num}-{idx}-xl.pkl"
    with open(pag_path, "rb") as f:
        pag = pickle.load(f)
    with open(fd_path, "rb") as f:
        fd_edges = pickle.load(f)
    from src.XLearner import XLearner
    xl = XLearner(dataset_path, fd_edges=fd_edges)
    with open(xl_path, "wb") as f:
        pickle.dump(xl, f)
    true_edges = get_pag_edges(pag)
    est_edges = xl.csm_edges
    rlt = compare_pag(est_edges, true_edges)
    rlt.update({"node num": node_num, "idx": idx})
    print(rlt)
    return rlt

def run_fci(node_num, idx):
    dataset_path = f"data/synthetic/{node_num}-{idx}.csv"
    pag_path = f"data/synthetic/{node_num}-{idx}-pag.pkl"
    fci_result_path = f"data/synthetic/{node_num}-{idx}-fci-result.pkl"
    with open(pag_path, "rb") as f:
        pag = pickle.load(f)
    df = pd.read_csv(dataset_path)
    g, _ = fci(df.to_numpy(), independence_test_method=chisq, alpha=0.01, depth=5, max_path_length=5)
    with open(fci_result_path, "wb") as f:
        pickle.dump(g, f)
    true_edges = get_pag_edges(pag)
    est_edges = get_pag_edges(g, df.columns)
    rlt = compare_pag(est_edges, true_edges)
    rlt.update({"node num": node_num, "idx": idx})
    print(rlt)
    return rlt

def run_one_star(a_b):
    run_one(*a_b)
    return run_one2(*a_b)

def run_fci_star(a_b):
    return run_fci(*a_b)