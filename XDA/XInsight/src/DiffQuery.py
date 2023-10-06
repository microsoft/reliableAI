# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List
from duckdb import DuckDBPyConnection
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.getcwd()))
from src.logger import *
from src.Utils import *

class DiffQuery:
    def __init__(self, fixed_pred: FixedPredicate, diff_pred: DiffPredicate, measure: Measure, columns: List[str]) -> None:
        assert diff_pred.col not in fixed_pred.cols
        self.fixed_pred = fixed_pred
        self.diff_pred = diff_pred
        self.measure = measure
        self.columns = columns
        self.cache = {}
    
    def create_scope(self, conn: DuckDBPyConnection):
        retained_cols = []
        for col in self.columns:
            if col not in self.fixed_pred.cols: retained_cols.append(col)
        self.retained_cols = retained_cols
        retained_col_str = ", ".join(retained_cols)
        fix_pred_str = self.fixed_pred.to_sql()
        diff_pred_str1, diff_pred_str2 = self.diff_pred.to_sql()
        sql = f"CREATE TABLE diff_scope AS SELECT {retained_col_str} FROM raw_table WHERE {fix_pred_str} AND ({diff_pred_str1} OR {diff_pred_str2})"
        conn.execute("DROP TABLE IF EXISTS diff_scope")
        conn.execute(sql)
        conn.execute("SELECT COUNT(*) FROM diff_scope")
        row_num = conn.fetchone()[0]
        self.scope_row_num = row_num
        if row_num <= 10:
            logging.warn(f"CREATE DIFF_SCOPE WITH {row_num} ROWS")
        else:
            logging.info(f"CREATE DIFF_SCOPE WITH {row_num} ROWS")
        
    def get_interventional_diff(self, intervention: InterventionalPredicate, conn: DuckDBPyConnection):
        diff_col = self.diff_pred.col
        sql = f"SELECT {diff_col}, COUNT(*), {self.measure.to_sql()} FROM DIFF_SCOPE WHERE {intervention.to_sql()} GROUP BY {diff_col}" 
        if sql in self.cache: return self.cache[sql]
        conn.execute(sql)
        rlt = conn.fetchall()
        if len(rlt) != 2: 
            self.cache[sql] = (0, 0)
            return 0, 0
        row_num = min([row[1] for row in rlt])
        if rlt[0][0] == self.diff_pred.vals[0]: 
            self.cache[sql] = (row_num, rlt[0][2] - rlt[1][2])
            return row_num, rlt[0][2] - rlt[1][2]
        else: 
            self.cache[sql] = (row_num, rlt[1][2] - rlt[0][2])
            return row_num, rlt[1][2] - rlt[0][2]
    
    def get_counterfactual_diff(self, counterfactual: CounterfactualPredicate, intervention: InterventionalPredicate, conn: DuckDBPyConnection):
        diff_col = self.diff_pred.col
        sql = f"SELECT {diff_col}, COUNT(*), {self.measure.to_sql()} FROM DIFF_SCOPE WHERE {intervention.to_sql()} AND {counterfactual.to_sql()} GROUP BY {diff_col}" 
        if sql in self.cache: return self.cache[sql]
        conn.execute(sql)
        rlt = conn.fetchall()
        if len(rlt) != 2: 
            self.cache[sql] = (0, 0)
            return 0, 0
        row_num = min([row[1] for row in rlt])
        if rlt[0][0] == self.diff_pred.vals[0]: 
            self.cache[sql] = (row_num, rlt[0][2] - rlt[1][2])
            return row_num, rlt[0][2] - rlt[1][2]
        else: 
            self.cache[sql] = (row_num, rlt[1][2] - rlt[0][2])
            return row_num, rlt[1][2] - rlt[0][2]
    
    def get_contingency_diff(self, contingency: InterventionalPredicate, conn: DuckDBPyConnection):
        diff_col = self.diff_pred.col
        sql = f"SELECT {diff_col}, COUNT(*), {self.measure.to_sql()} FROM DIFF_SCOPE WHERE {contingency.to_contingency()} GROUP BY {diff_col}" 
        # print(sql)
        if sql in self.cache: return self.cache[sql]
        conn.execute(sql)
        rlt = conn.fetchall()
        # print(rlt)
        if len(rlt) != 2: 
            self.cache[sql] = (0, 0)
            return 0, 0
        row_num = sum([row[1] for row in rlt])
        if rlt[0][0] == self.diff_pred.vals[0]: 
            self.cache[sql] = (row_num, rlt[0][2] - rlt[1][2])
            return row_num, rlt[0][2] - rlt[1][2]
        else: 
            self.cache[sql] = (row_num, rlt[1][2] - rlt[0][2])
            return row_num, rlt[1][2] - rlt[0][2]
    
    def get_raw_diff(self, conn: DuckDBPyConnection):
        return self.get_interventional_diff(FixedPredicate([],[]), conn)[1]

    def get_row_count(self, intervention: InterventionalPredicate, conn: DuckDBPyConnection, counterfactual: CounterfactualPredicate=None):
        diff_col = self.diff_pred.col
        if counterfactual != None:
            sql = f"SELECT COUNT(*) FROM DIFF_SCOPE WHERE {intervention.to_sql()} AND {counterfactual.to_sql()} GROUP BY {diff_col}" 
        else:
            sql = f"SELECT COUNT(*) FROM DIFF_SCOPE WHERE {intervention.to_sql()} GROUP BY {diff_col}" 
        conn.execute(sql)
        all_rlt = conn.fetchall()
        if len(all_rlt) != 2: return 0
        row_num = min([row[1] for row in all_rlt])
        return row_num
    
    def plot(self, fig_path: str, conn: DuckDBPyConnection, counterfactual: CounterfactualPredicate=None):
        diff_col = self.diff_pred.col
        sql = f"SELECT {diff_col}, COUNT(*), {self.measure.to_sql()} FROM DIFF_SCOPE GROUP BY {diff_col}" 
        conn.execute(sql)
        rlt = conn.fetchall()
        if len(rlt) != 2: raise DataError
        if rlt[0][0] == self.diff_pred.vals[0]: m1, m2 = rlt[0][2], rlt[1][2]
        else: m1, m2 = rlt[1][2], rlt[0][2]
        x = self.diff_pred.vals
        y = [m1, m2]
        plt.bar(x, y, alpha=0.5)
        if counterfactual is not None:
            sql = f"SELECT {diff_col}, COUNT(*), {self.measure.to_sql()} FROM DIFF_SCOPE WHERE {counterfactual.to_sql()} GROUP BY {diff_col}" 
            conn.execute(sql)
            rlt = conn.fetchall()
            if len(rlt) != 2: raise DataError
            if rlt[0][0] == self.diff_pred.vals[0]: cm1, cm2 = rlt[0][2], rlt[1][2]
            else: cm1, cm2 = rlt[1][2], rlt[0][2]
            c = [cm1, cm2]
            print(c,y)
            plt.bar(x, c, alpha=0.5)
            plt.legend(["True", "Counterfactual"])
            plt.title(f"Explain Diff by {counterfactual.to_sql()}")
        else:
            plt.legend(["True"])
            plt.title(f"Diff")
        plt.xlabel(self.diff_pred.col)
        plt.ylabel(self.measure.to_sql())
        plt.savefig(fig_path)
        plt.clf()
    
    def plot_all(self, fig_path: str, counterfactuals: List[CounterfactualPredicate], conn: DuckDBPyConnection):
        diff_col = self.diff_pred.col
        sql = f"SELECT {diff_col}, COUNT(*), {self.measure.to_sql()} FROM DIFF_SCOPE GROUP BY {diff_col}" 
        conn.execute(sql)
        rlt = conn.fetchall()
        if len(rlt) != 2: raise DataError
        if rlt[0][0] == self.diff_pred.vals[0]: m1, m2 = rlt[0][2], rlt[1][2]
        else: m1, m2 = rlt[1][2], rlt[0][2]
        x = self.diff_pred.vals
        y = [m1, m2]

        line = int(np.ceil(len(counterfactuals)/4))
        fig, axs = plt.subplots(line, min(4, len(counterfactuals)), figsize=(4*min(4, len(counterfactuals)), 4*line), sharex=True, sharey=True)
        for idx, ax in enumerate(axs.flat):
            if idx < len(counterfactuals):
                counterfactual = counterfactuals[idx]
                ax.bar(x, y, alpha=0.5)
                sql = f"SELECT {diff_col}, COUNT(*), {self.measure.to_sql()} FROM DIFF_SCOPE WHERE {counterfactual.to_sql()} GROUP BY {diff_col}" 
                conn.execute(sql)
                rlt = conn.fetchall()
                if len(rlt) != 2: raise DataError
                if rlt[0][0] == self.diff_pred.vals[0]: cm1, cm2 = rlt[0][2], rlt[1][2]
                else: cm1, cm2 = rlt[1][2], rlt[0][2]
                c = [cm1, cm2]
                ax.bar(x, c, alpha=0.5)
                ax.legend(["True", "Counterfactual"])
                ax.set_title(f"{counterfactual.to_sql()}")
                ax.set_xlabel(self.diff_pred.col)
                ax.set_ylabel(self.measure.to_sql())
        fig.tight_layout()
        plt.savefig(fig_path)
        plt.clf()
    
    def reset_cache(self):
        self.cache = {}

class AvgDiffQuery(DiffQuery):
    def __init__(self, fixed_pred: FixedPredicate, diff_pred: DiffPredicate, measure: Measure, columns: List[str]) -> None:
        assert measure.func.lower() == "avg"
        super().__init__(fixed_pred, diff_pred, measure, columns)
    
    def get_predicate_wise_info(self, target_column: str, contingency: InterventionalPredicate, conn: DuckDBPyConnection) -> Dict[str, Dict[str, Tuple[int, int]]]:
        diff_col = self.diff_pred.col
        if contingency is None:
            sql = f"SELECT {diff_col}, {target_column}, COUNT(*), {self.measure.to_sql()} FROM DIFF_SCOPE GROUP BY {diff_col}, {target_column}" 
        else:
            sql = f"SELECT {diff_col}, {target_column}, COUNT(*), {self.measure.to_sql()} FROM DIFF_SCOPE WHERE {contingency.to_sql()} GROUP BY {diff_col}, {target_column}"
        print(sql)
        conn.execute(sql)
        all_rlt = conn.fetchall()

        # convert to dict
        result = {self.diff_pred.vals[0]: {}, self.diff_pred.vals[1]: {}}
        vals = set()
        for row in all_rlt:
            vals.add(row[1])
            result[row[0]][row[1]] = (row[2], row[3]) # Dict[diff, Dict[pred, Tuple[support, measure]]]
        
        # add placeholders
        for val in vals:
            if val not in result[self.diff_pred.vals[0]]: result[self.diff_pred.vals[0]][val] = (0,0)
            if val not in result[self.diff_pred.vals[1]]: result[self.diff_pred.vals[1]][val] = (0,0)
        return result

if __name__ == "__main__":
    from src.XplainerDataModel import XplainerDataModel
    dm = XplainerDataModel("../data/child.csv", [f"col_{i}" for i in range(20)])
    fp = FixedPredicate(["col_0", "col_1"], [0, 0])
    dp = DiffPredicate("col_3", [0,1])
    measure = Measure("col_7", "sum")
    dq = DiffQuery(fp, dp, measure, dm.col_names)
    dq.create_scope(dm.conn)
    raw_diff = dq.get_interventional_diff(FixedPredicate([],[]), dm.conn)
    print(raw_diff)
    cond = InterventionalPredicate(["col_5", "col_6"], [0, 1])
    print(dq.get_interventional_diff(cond, dm.conn))