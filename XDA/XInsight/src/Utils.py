# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from typing import Any, Callable, Dict, List, Set, Tuple, TypeVar
from copy import copy
import time, graphlib

from src.CausalSemanticModel import Edge

class FixedPredicate:
    def __init__(self, cols: List, vals: List):
        self.cols = cols
        self.vals = vals
    
    def to_sql(self):
        predicate_strs = []
        if len(self.cols) > 0:
            for idx, col in enumerate(self.cols):
                val = self.vals[idx]
                if isinstance(val, int) or isinstance(val, float):
                    predicate_strs.append(f" {col}={val} ")
                else:
                    predicate_strs.append(f" {col}=\"{val}\" ")
            return " AND ".join(predicate_strs)
        else:
            return " 1=1 "

class InterventionalPredicate:
    def __init__(self, cols: List, vals: List):
        self.cols = cols
        self.vals = vals
    
    def to_sql(self):
        predicate_strs = []
        if len(self.cols) > 0:
            col_val_pairs = []
            for idx, col in enumerate(self.cols):
                val = self.vals[idx]
                col_val_pairs.append((col, val))
            col_val_pairs.sort()
            for pair in col_val_pairs:
                col, val = pair
                if isinstance(val, int) or isinstance(val, float):
                    predicate_strs.append(f" {col}!={val} ")
                else:
                    predicate_strs.append(f" {col}!=\"{val}\" ")
            return " AND ".join(predicate_strs)
        else:
            return " 1=1 "
    
    def to_contingency(self):
        predicate_strs = []
        if len(self.cols) > 0:
            col_val_pairs = []
            for idx, col in enumerate(self.cols):
                val = self.vals[idx]
                col_val_pairs.append((col, val))
            col_val_pairs.sort()
            for pair in col_val_pairs:
                col, val = pair
                if isinstance(val, int) or isinstance(val, float):
                    predicate_strs.append(f" {col}=={val} ")
                else:
                    predicate_strs.append(f" {col}==\"{val}\" ")
            return " OR ".join(predicate_strs)
        else:
            return " 1=1 "
    
    def to_counterfactual(self):
        assert len(set(self.cols)) == 1
        return CounterfactualPredicate(self.cols[0], self.vals)
    
    def copy(self):
        return InterventionalPredicate(copy(self.cols), copy(self.vals))
    
    def partion(self, k):
        return InterventionalPredicate(self.cols[:k], self.vals[:k]), InterventionalPredicate(self.cols[k:], self.vals[k:]), 
    
    def add(self, col, val):
        self.cols.append(col)
        self.vals.append(val)
    
    def contain_col(self, col):
        return col in self.cols
    
    def size(self):
        return len(self.cols)
    
    def __str__(self) -> str:
        return self.to_sql()

class CounterfactualPredicate:
    def __init__(self, col: str, vals: List[str]):
        assert len(vals) > 0
        self.col = col
        self.vals = vals
    
    def to_sql(self):
        predicate_strs = []
        col = self.col
        for idx, val in enumerate(self.vals):
            if isinstance(val, int) or isinstance(val, float):
                predicate_strs.append(f" {col}!={self.vals[idx]} ")
            else:
                predicate_strs.append(f" {col}!=\"{self.vals[idx]}\" ")
        return " AND ".join(predicate_strs)
    
    def size(self):
        return len(self.vals)
    
    def to_intervention(self):
        cols = [self.col] * len(self.vals)
        return InterventionalPredicate(cols, self.vals)
    
    def __str__(self) -> str:
        return self.to_sql()

class DiffPredicate:
    def __init__(self, col: str, vals: Tuple[str, str]):
        self.col = col
        self.vals = vals
    
    def to_sql(self):
        col, vals = self.col, self.vals
        if isinstance(vals[0], int) or isinstance(vals[0], float):
            pred_str1 = f" {col}={vals[0]} "
        else:
            pred_str1 = f" {col}=\"{vals[0]}\" "
        if isinstance(vals[1], int) or isinstance(vals[1], float):
            pred_str2 = f" {col}={vals[1]} "
        else:
            pred_str2 = f" {col}=\"{vals[1]}\" "
        return pred_str1, pred_str2

class Measure:
    def __init__(self, col: str, func: str) -> None:
        self.col = col
        assert func.lower() in ["avg", "sum", "udf"]
        self.func = func
    
    def to_sql(self):
        return f" {self.func}({self.col}) "

class Timer:
    def __init__(self, timeout=0) -> None:
        self.timeout = time.time() + timeout
    
    def is_timeout(self):
        return time.time() - self.timeout > 0
    
    def reset(self, timeout):
        self.timeout = time.time() + timeout

class FDGraph:

    class FDNode:
        def __init__(self, col_name: str):
            self.col_name = col_name

    def __init__(self, col_names: List[str], fd_list: Set[Tuple[str, str]], cardinalities: Dict[str, int]):
        self.nodes = col_names
        self.nonroot_node = {fd[1] for fd in fd_list}
        self.edges = {}
        self.cardinalities = cardinalities
        for col_name in col_names:
            self.edges[col_name] = set()
        for fd in fd_list:
            self.edges[fd[0]].add(fd[1])
        ts = graphlib.TopologicalSorter(self.edges)
        self.order_node = list(ts.static_order())
        self.retained_node = copy(set(self.order_node))
        self.resulting_edges: List[Edge] = list()
        for node in self.order_node:
            if node in self.nonroot_node:
                parent = self.get_parent(node)[0] # get parent with minimal cardinality
                self.retained_node.remove(node)
                self.resulting_edges.append(Edge(parent, node, Edge.EdgeType.Direct))
        self.retained_node = list(self.retained_node)
                
    def has_nonroot(self):
        return sum([len(self.edges[parent]) for parent in self.edges]) > 0
    
    def get_parent(self, child):
        parents = list()
        for parent in self.edges:
            if child in self.edges[parent]: parents.append(parent)
        parents.sort(key=lambda x: self.cardinalities[x])
        return parents

    def remove_all_edge(self, node):
        self.edges[node] = set()
        for parent in self.edges:
            self.edges[parent].discard(node)

class DataError(Exception):
    pass