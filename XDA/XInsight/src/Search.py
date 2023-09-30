# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List
from duckdb import DuckDBPyConnection
import os, sys, itertools
from copy import copy, deepcopy
from p_tqdm import p_umap
from tqdm import tqdm
sys.path.append(os.path.dirname(os.getcwd()))
from src.logger import *
from src.Utils import *
from src.DiffQuery import *

class SumSearch:
    delta = .15 # ratio-based thoreshold
    sigma = .3 # penalty for large explanation
    min_row_num = 10

    def __init__(self, diff: DiffQuery, search_space: Dict[str, List], conn: DuckDBPyConnection) -> None:
        self.diff = diff
        self.conn = conn
        self.search_space = search_space
        self.Delta = self.diff.get_raw_diff(self.conn)
        self.epsilon = SumSearch.delta * self.Delta
        assert self.Delta > 0 and self.diff.measure.func.lower() == "sum"

    def run(self, prior_cols=None):
        if prior_cols is None: prior_cols = self.diff.retained_cols
        self.explanation: List[Tuple[CounterfactualPredicate, float]] = []
        for col in prior_cols:
            if col not in self.diff.retained_cols: continue
            diff_list = []
            val_set = []
            for val in self.search_space[col]:
                pred = InterventionalPredicate([col], [val])
                row_num, diff = self.diff.get_contingency_diff(pred, self.conn)
                if row_num > SumSearch.min_row_num:
                    if self.diff.measure.func.lower() == "sum": # and diff > 0:
                        val_set.append((val, diff))
                        diff_list.append(diff)
            diff_list.sort(reverse=True)
            val_set.sort(key=lambda x:-x[1])
            for idx in range(len(diff_list)-1):
                # print(self.Delta - sum(diff_list[:idx+1]), self.Delta - sum(diff_list[:idx]))
                if self.Delta - sum(diff_list[:idx+1]) <= self.epsilon and self.Delta - sum(diff_list[:idx]) > self.epsilon:
                    canonical_predicate = val_set[:idx+1]
                    tau = sum(diff_list[:idx+1])
                    cutoff = (SumSearch.sigma * self.Delta) / ((1+tau/self.Delta) ** 2)
                    
                    resp = sum([max(0, i-cutoff) for i in diff_list])
                    exp = CounterfactualPredicate(col, [i[0] for i in canonical_predicate if i[1]>cutoff])
                    # cont_vals = [i[0] for i in canonical_predicate if i[1]<=cutoff]
                    # cont = InterventionalPredicate([col] * len(cont_vals), cont_vals)
                    # print(cont_vals)
                    # _, diff = self.diff.get_contingency_diff(cont, self.conn)
                    # resp = 1/(1+diff/self.Delta)
                    self.explanation.append((exp, resp))
                    break
    
    def bruteforce(self, explanation: CounterfactualPredicate):
        start = time.time()
        remaining_vals = set(self.search_space[explanation.col]) - set(explanation.vals)
        resp = 0
        for cont_size in range(1, 6):
            print(cont_size)
            cols = [explanation.col] * cont_size
            for cont in tqdm(itertools.combinations(remaining_vals, cont_size)):
                _, diff = self.diff.get_contingency_diff(InterventionalPredicate(cols, list(cont)), self.conn)
                resp = max(resp, 1/(1+diff/self.Delta))
        print(time.time()-start)
        return resp - SumSearch.sigma * explanation.size()

class AvgSearch:
    delta = .3 # ratio-based thoreshold
    sigma = .1 # penalty for large explanation
    min_row_num = 10

    def __init__(self, diff: AvgDiffQuery, search_space: Dict[str, List], conn: DuckDBPyConnection) -> None:
        self.diff = diff
        self.conn = conn
        self.search_space = search_space
        self.Delta = self.diff.get_raw_diff(self.conn)
        self.epsilon = SumSearch.delta * self.Delta
        assert self.Delta > 0 and self.diff.measure.func.lower() == "avg"
    
    def run(self, prior_cols=None):
        if prior_cols is None: prior_cols = self.diff.retained_cols
        self.explanation: List[Tuple[CounterfactualPredicate, float]] = []
        for col in prior_cols:
            if col not in self.diff.retained_cols: continue
            # find canonical predicate
            current_PC = InterventionalPredicate([], [])
            for i in range(min(len(self.search_space[col]), int(1/AvgSearch.sigma))):
                _, diff = self.diff.get_interventional_diff(current_PC, self.conn)
                if diff <= self.epsilon: break
                candidate = set(self.search_space[col]) - set(set(current_PC.vals))
                next_PC_vals = None
                min_diff = 0xfffffffff
                new_PC_cols = copy(current_PC.cols) + [col]
                for val in candidate:
                    new_PC_vals = copy(current_PC.vals) + [val]
                    new_PC = InterventionalPredicate(new_PC_cols, new_PC_vals)
                    _, diff = self.diff.get_interventional_diff(new_PC, self.conn)
                    if diff < min_diff:
                        min_diff = diff
                        next_PC_vals = new_PC_vals
                current_PC = InterventionalPredicate(new_PC_cols, next_PC_vals)
            _, canonical_diff = self.diff.get_interventional_diff(current_PC, self.conn)
            if canonical_diff > self.epsilon: 
                break
            optimal_exp = None
            optimal_resp = 0
            for idx in range(current_PC.size()):
                exp, cont = current_PC.partion(idx+1)
                _, exp_diff = self.diff.get_interventional_diff(exp, self.conn)
                if cont.size() == 0: 
                    resp = 1 - AvgSearch.sigma * exp.size()
                else:
                    resp = 1/(1+max(0, (exp_diff-canonical_diff)/self.Delta)) - AvgSearch.sigma * exp.size()
                # print(exp, resp)
                if resp > optimal_resp:
                    optimal_resp = resp
                    optimal_exp = exp.to_counterfactual()
            self.explanation.append((optimal_exp, optimal_resp))
            
 
if __name__ == "__main__":
    from src.XplainerDataModel import XplainerDataModel
    import json
    agg = "sum"
    # for n in  [5, 10, 15, 30, 50, 100]:
    # for n in  [5]:
    for n in  [10000, 20000, 50000, 100000, 500000, 1000000]:
    # for n in [10, 15, 20, 30, 50, 100]:
        dm = XplainerDataModel(f"data/synthetic-B-{agg}/n/{n}.csv")
        fp = FixedPredicate([], [])
        dp = DiffPredicate("X", [1,0])
        measure = Measure("Z", agg)
        if measure.func == "avg":
            dq = AvgDiffQuery(fp, dp, measure, dm.col_names)
            dq.create_scope(dm.conn)
            raw_diff = dq.get_interventional_diff(FixedPredicate([],[]), dm.conn)
            search = AvgSearch(dq, dm.value_set, dm.conn)
        else:
            dq = DiffQuery(fp, dp, measure, dm.col_names)
            dq.create_scope(dm.conn)
            raw_diff = dq.get_interventional_diff(FixedPredicate([],[]), dm.conn)
            search = SumSearch(dq, dm.value_set, dm.conn)
        start = time.time()
        search.run(["Y"])
        with open(f"data/synthetic-B-{agg}/n/{n}.json") as f:
            config = json.load(f)
        groundtruth = set(config["exp_val"])
        explanation = set(search.explanation[0][0].vals)
        print(explanation,groundtruth)
        precision = len(explanation.intersection(groundtruth)) / len(explanation)
        recall = len(explanation.intersection(groundtruth)) / len(groundtruth)
        f1 = 2*precision*recall/(precision+recall) if precision+recall > 0 else 0
        print(n, f1, time.time()-start)
        # for cc, score in search.actual_cause:
        #     print(cc, score)
    # print(search.actual_cause)
    # print(search.counterfactual_cause)