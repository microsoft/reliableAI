# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations
from itertools import permutations
from typing import List, Tuple
import pandas as pd
import pickle

from src.FCI import xlearner
from src.CausalSemanticModel import CausalSemanticModel
from src.DiffQuery import DiffQuery
from src.BLIP import skeleton_learning as blip_sl
from src.REAL import skeleton_learning as real_sl
from src.Utils import *
from src.logger import *
from src.Search import SumSearch, AvgSearch

class XLearner:
    def __init__(self, csv_path: str, col_names: List[str]=None, sl_algo: str="default", sl_file: str=None, fd_edges: List[Tuple[str, str]]=None) -> None:
        if col_names is None: 
            self.df = pd.read_csv(csv_path)
        else: 
            self.df = pd.read_csv(csv_path, names=col_names)
        self.sl_algo = sl_algo
        self.sl_file = sl_file
        self.learn(fd_edges=fd_edges)
    
    @staticmethod
    def from_file(file_path: str) -> XLearner:
        with open(file_path, "rb") as f:
            xl = pickle.load(f)
        return xl
    
    @staticmethod
    def sl_from_file(file_path: str, df: pd.DataFrame, retained_nodes: List[str]):
        skeleton = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                if "-" in line:
                    x,y = line.split("-")
                    if df.columns[int(x)] in retained_nodes and df.columns[int(y)] in retained_nodes:
                        skeleton.append((df.columns[int(x)], df.columns[int(y)]))
        print(len(skeleton))
        return skeleton

    def to_file(self, file_path: str) -> XLearner:
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    def resolve_fd(self, fd_edges: List[Tuple[str, str]]=None):
        fd_list = set()
        cols = self.df.columns
        count = {col:len(pd.unique(self.df[col])) for col in cols}
        for col in count:
            if count[col] == 1: logging.warning(f"COL: {col} has only one value.")
        cardinalities = {col:count[col] for col in cols}
        if fd_edges is not None: fd_list = set(fd_edges)
        else:
            for cmb in permutations(cols, 2):
                col1 = cmb[0]
                col2 = cmb[1]
                if (col2, col1) in fd_list: continue
                mapper = {}
                has_dup = False
                if count[col2] == 1 or count[col2] > count[col1]:
                    continue
                for index, row in self.df.iterrows():
                    key = row[col1]
                    val = row[col2]
                    if key in mapper and mapper[key] != val:
                        has_dup = True
                        break
                    if key not in mapper:
                        mapper[key] = val
                if not has_dup:
                    fd_list.add(cmb)
        self.fd_graph = FDGraph(cols, fd_list, cardinalities)
    
    def learn(self, fd_edges: List[Tuple[str, str]]=None, resolve_fd: bool=True, mask:bool=False):
        if resolve_fd: 
            self.resolve_fd(fd_edges)
            if mask:
                df = self.df[self.fd_graph.retained_node]
            else:
                df = self.df
        else:
            df = self.df
        dataset = df.to_numpy(dtype=int, copy=True)
        if self.sl_algo.lower() == "blip":
            skeleton = blip_sl(df)
            csm_edges = xlearner(dataset, df.columns, skeleton=skeleton)
        elif self.sl_algo.lower() == "blip-fast":
            skeleton = blip_sl(df, True)
            csm_edges = xlearner(dataset, df.columns, skeleton=skeleton, depth=3)
        elif self.sl_algo.lower() == "real":
            skeleton = real_sl(df)
            csm_edges = xlearner(dataset, df.columns, skeleton=skeleton)
        elif self.sl_algo.lower() == "file":
            skeleton = XLearner.sl_from_file(self.sl_file, self.df, df.columns)
            csm_edges = xlearner(dataset, df.columns, skeleton=skeleton)
        elif self.sl_algo.lower() == "default":
            csm_edges = xlearner(dataset, df.columns)
        else:
            raise RuntimeError(f"{self.sl_algo} not found")
        if resolve_fd:
            final_csm_edges = copy(self.fd_graph.resulting_edges)
            for edge in csm_edges:
                has_match = False
                for fd_edge in self.fd_graph.resulting_edges:
                    if fd_edge.end == edge.end or fd_edge.end == edge.start: 
                        has_match = True
                        break
                if not has_match: 
                    final_csm_edges.append(edge)
            self.csm = CausalSemanticModel(self.df.columns, final_csm_edges)
            self.csm_edges = final_csm_edges
        else:
            self.csm = CausalSemanticModel(df.columns, csm_edges)
            self.csm_edges = csm_edges
    
    def get_explainable_cols(self, diff: DiffQuery):
        retained_col = diff.retained_cols
        retained_col.remove(diff.diff_pred.col)
        condition_set = [diff.diff_pred.col] + diff.fixed_pred.cols
        explainable_cols = []
        semantics = {}
        for col in retained_col:
            semantic = self.csm.check_semantic(diff.measure.col, col, condition_set)
            semantics[col] = semantic
            if semantic != CausalSemanticModel.SemanticType.Unexplainable:
                explainable_cols.append(col)
        return explainable_cols, semantics
