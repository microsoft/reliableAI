# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env python
#-*- coding: utf-8 -*-
from queue import PriorityQueue
import numpy as np
import sklearn.preprocessing
from ML4C_Tools import Graph
PI = np.pi

def Erf(x):
    # error function. handle either positive or negative x.
    # because error function is negatively symmetric of x
    a = 0.140012
    b = x ** 2
    item = -b * (4 / PI + a * b) / (1 + a * b)
    return np.abs(np.sqrt(1 - np.exp(item)))

def GaussianSignificance(x, u, sigma):
    '''
    calculate the statistical significance for a gaussian distribution.
    :param x: the observed x value
    :param u: mean value
    :param sigma: the standard deviation
    :return: stat significance for x relative to the gaussian distribution
    '''
    x1 = np.abs(x - u)
    return Erf(x1 / sigma / 1.414213562373095)

def percentile(a, q):
    return np.percentile(a, q).tolist() if len(a) else [-1.] * len(q)

def meanstdmaxmin(a):
    return [np.mean(a), np.std(a), np.max(a), np.min(a)] if len(a) else [-1., -1., -1., -1.]

def deTuple(listOfTuples):
    return zip(*listOfTuples) if listOfTuples else ((), ())

def divide(arr1, arr2, inf_to_num):
    '''
    feature: 0/0=1,
            >0/0=given inf_to_num, if there is max left
    if you write like this,
      ratio_pval = np.array(XY_ST_pval) / np.array(XY_S_pval)
      ratio_svrt = np.array(XY_ST_svrt) / np.array(XY_S_svrt)
    there will be 0/0=nan, >0/0=inf, and further
    percentile([...nan, ...], [...])=[nan]_full
    percentile([...inf, ...], [...])=maybe nan or inf, e.g.
      percentile([inf], [10])=[nan], percentile([inf], [100])=[nan]
      percentile([1,2,inf], [10,50,90])=[1.2,nan,inf]
    '''
    res = arr1 / arr2
    res[np.isnan(res)] = 1.
    res[np.isinf(res)] = inf_to_num # very rare
    return res

class Kernel_Embedding(object):
    # kernel embedding from https://github.com/lopezpaz/causation_learning_theory
    def __init__(self, k=5, s=None, d=1): # in RCC k=100
        if not s: s = [0.15, 1.5, 15]
        self.k, self.s, self.d = k, s, d
        self.w = np.hstack(( # w in shape 2*15
            np.vstack([si * np.random.randn(k, d) for si in s]), # shape 15*1, first 5 rows~N(0, 0.15), then ~N(0, 1.5), ~N(0, 15)
            2 * PI * np.random.rand(k * len(s), 1) # shape 15*1, ~N(0, 2pi)
        )).T

    def get_empirical_embedding(self, a):
        # param: a (list) is the same as that in percentile(a, q): samples P_S to a distribution P
        if len(a) == 0: return [-1.] * self.k * len(self.s)      #np.ones((self.k * len(self.s))) * -1.
        arr = sklearn.preprocessing.scale(a)[:, None]            # arr = np.array(a)[:, None]
        return np.cos(np.dot(np.hstack((arr, np.ones((arr.shape[0], 1)))), self.w)).mean(axis=0).tolist()


class MaxHeap(PriorityQueue):
    # stores the k minimum seveirtys, with the heaptop being max
    def _init(self, maxsize: int):
        super()._init(maxsize=maxsize)
        self.validnum = 0

    def _put(self, item):
        # item is (bool isvalid, tuple S, double pValue, double severity)
        prior = self._get_priority(item)
        if not super().full():
            self.validnum += item[0]
            return super()._put((prior, item))
        head_prior, head_item = super()._get()
        if prior > head_prior: # i.e. severity less than the current k-th minimum
            self.validnum = self.validnum + item[0] - head_item[0]
            return super()._put((prior, item))
        return super()._put((head_prior, head_item))

    def _get(self):
        item = super()._get()[1]
        self.validnum -= item[0]
        return item

    def _get_priority(self, item):
        return -item[3] # use opposite for max heap

    def get_all_sepsets(self):
        sepsets = set()
        while not super().empty():
            sepsets.add(super()._get()[1][1])
        return sepsets

    def get_all_valid_sepsets(self):
        # if there're no valid sepset (every pvalue < 0.01),
        # then return only one sepset with the minimum severity
        sepsets = set()
        popleft = None
        while not super().empty():
            popleft = super()._get()[1]
            if popleft[0]: sepsets.add(popleft[1])
        # the last popped has the minimum severity
        return sepsets if len(sepsets) > 0 else {popleft[1]}

def cal_score(truth_G: Graph.DiGraph, result_G: Graph.MixedGraph):
    '''
    in general, ground_truth_G and result_G should be of the same skeleton.
        but in some algorithm, the output dosen't strictly follow the given skeleton (whitelist),
        and thus this scenario should also be considered.
    "identifiability" should always be considered, so we first get CPDAGs for truth_G and result_G (by extract Vs, meeks..),
        and then calculate metrics between these two CPDAGs.
    which metric scores should be used?:
        +-------------------+-----------------+---------------+-------------+
        | in result CPDAG → | iden (directed) |     uniden    |   missing   |
        |                   +--------+--------+  (undirected) | in skeleton |
        |  in truth CPDAG ↓ |  right |  wrong |               |             |
        +-------------------+--------+--------+---------------+-------------+
        |        iden       |   √ 1  |   × 2  |      × 3      |     × 4     |
        +-------------------+--------+--------+---------------+-------------+
        |       uniden      |       × 5       |      √ 6      |     × 7     |
        +-------------------+-----------------+---------------+-------------+
        |      nonexist     |       × 8       |      × 9      |     √ 10    |
        +-------------------+-----------------+---------------+-------------+
    SHD: "×" means ++SHD; "√" makes no change to SHD.
    i.e. equivalent code:
        edges_1 = truth_CPDAG.DirectedEdges.intersection(result_CPDAG.DirectedEdges)
        edges_2 = {(i, j) for (i, j) in result_CPDAG.DirectedEdges if (j, i) in truth_CPDAG.DirectedEdges}
        edges_3 = {(i, j) for (i, j) in result_CPDAG.UndirectedEdges if
                        (i, j) in truth_CPDAG.DirectedEdges or (j, i) in truth_CPDAG.DirectedEdges}
        edges_4 = {(i, j) for (i, j) in truth_CPDAG.DirectedEdges if not result_CPDAG.adjacent_in_mixed_graph(i, j)}
        edges_5 = {(i, j) for (i, j) in result_CPDAG.DirectedEdges if truth_CPDAG.has_undi_edge(i, j)}
        edges_6 = truth_CPDAG.UndirectedEdges.intersection(result_CPDAG.UndirectedEdges)
        edges_7 = {(i, j) for (i, j) in truth_CPDAG.UndirectedEdges if not result_CPDAG.adjacent_in_mixed_graph(i, j)}
        edges_8 = {(i, j) for (i, j) in result_CPDAG.DirectedEdges if not truth_CPDAG.adjacent_in_mixed_graph(i, j)}
        edges_9 = {(i, j) for (i, j) in result_CPDAG.UndirectedEdges if not truth_CPDAG.adjacent_in_mixed_graph(i, j)}
        SHD = len(edges_2) + len(edges_3) + len(edges_4) + len(edges_5) + len(edges_7) + len(edges_8) + len(edges_9)

    To calculate precision-recall, create Venn graph from the table above:
        ╔═══╦═══╦═══╗
        ║   ║ B ║   ║
        ║ A ╠═══╣ C ║
        ║   ║ D ║   ║
        ╚═══╩═══╩═══╝
        where A+B+D is the left circle (True), and C+B+D is the right circle (Positive), and
            A = 3∪4
            B = 1
            C = 5∪8
            D = 2
    i-TPR = B/(A+B+D) = precision : how many identifiable edges are oriented, and oriented in correct direction?
    i-FDR = (C+D)/(C+B+D) = 1-recall  : how many oriented edges are oriented wrong (either in wrong direction, or even should not)?
    v-TPR, v-FDR similar as i-above, but on each's respective v-edges, instead of on identifiable edges.
    i.e. equivalent code:
        i_TPR = len(edges_1) / max((len(edges_1) + len(edges_2) + len(edges_3) + len(edges_4)), 1)
        i_FDR = 1. - len(edges_1) / max((len(edges_1) + len(edges_2) + len(edges_5) + len(edges_8)), 1)

    :param truth_G: Graph.DiGraph
    :param result_G: Graph.MixedGrpah (some algorithm only returns CPDAG)
    :return: dictionary of metrics
    '''

    def _cal_SHD(targ_adj, pred_adj):
        diff = np.abs(targ_adj - pred_adj)
        diff = diff + diff.transpose()
        diff[diff > 1] = 1
        # Ignoring the double edges, only count them as one mistake.
        # eg1. truth i->j, pred j->i
        # eg2: truth i j, pred i--j
        return int(np.sum(diff) / 2) # it must be int itself

    truth_CPDAG = truth_G.getCPDAG()
    result_CPDAG = result_G.getCPDAG()

    truth_v_edges = set()
    for (j, i, k) in truth_G.vstrucs:
        truth_v_edges.add((i, j))
        truth_v_edges.add((k, j))
    result_v_edges = set()
    for (j, i, k) in result_G.vstrucs:
        result_v_edges.add((i, j))
        result_v_edges.add((k, j))

    SHD = _cal_SHD(truth_CPDAG.getAdjacencyMatrix(), result_CPDAG.getAdjacencyMatrix())
    i_TPR = len(result_CPDAG.DirectedEdges.intersection(truth_CPDAG.DirectedEdges)) / max(len(truth_CPDAG.DirectedEdges), 1)
    i_FDR = 1. - len(result_CPDAG.DirectedEdges.intersection(truth_CPDAG.DirectedEdges)) / max(len(result_CPDAG.DirectedEdges), 1)
    precision, recall = i_TPR, 1. - i_FDR
    i_F1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.

    v_TPR = len(result_v_edges.intersection(truth_v_edges)) / max(len(truth_v_edges), 1)
    v_FDR = 1. - len(result_v_edges.intersection(truth_v_edges)) / max(len(result_v_edges), 1)
    precision, recall = v_TPR, 1. - v_FDR
    v_F1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0.
    # or len(result_v_edges - truth_v_edges) / max(len(result_v_edges), 1), but corner case: 0/0, e.g. sachs
    # so for 0/0 case, set precision(or recall) as 0., so TPR=0, FDR=1

    scores = {'vstrucs_edges_TPR': v_TPR, 'vstrucs_edges_FDR': v_FDR, 'vstrucs_edges_F1': v_F1,
              'identfb_edges_TPR': i_TPR, 'identfb_edges_FDR': i_FDR, 'identfb_edges_F1': i_F1,
              'SHD': SHD}

    return scores