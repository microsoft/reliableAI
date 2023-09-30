# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List
import duckdb
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
from src.logger import *
from src.XLearner import XLearner

class XplainerDataModel:
    def __init__(self, csv_path: str, col_names=None):
        self.csv_path = csv_path
        self.col_names = col_names
        self.value_set = {}
        self.conn = self.initialize_database()
    
    @staticmethod
    def preprocess_columns(csv_path, xl:XLearner=None):
        df = pd.read_csv(csv_path)
        print(df.columns)
        # print(df.columns, xl.df.columns)
        for col in df.columns:
            col:str
            if xl is not None and col not in xl.df.columns:
                df.drop(col, 1, inplace=True)
            else:
                new_col = col.replace(".","_").replace(" ","_").replace("-","_")
                if new_col != col:
                    df[new_col] = df[col]
                    df.drop(col, 1, inplace=True)            
        df.to_csv(f"{csv_path}.xplainer", index=False)
        print(df.columns)
        return f"{csv_path}.xplainer"
    
    @staticmethod
    def recover_target_column(csv_path, raw_csv_path, target_col):
        df = pd.read_csv(csv_path)
        raw_df = pd.read_csv(raw_csv_path)
        df[target_col] = raw_df[target_col]
        df.to_csv(f"{csv_path}.xplainer", index=False)
        return f"{csv_path}.xplainer"

    def initialize_database(self):
        csv_path = self.csv_path
        if self.col_names is None: 
            df = pd.read_csv(csv_path)
            self.col_names = df.columns
        else: df = pd.read_csv(csv_path, names=self.col_names)
        conn = duckdb.connect(database=':memory:')
        conn.register('raw_table', df)
        conn.execute("SELECT COUNT(*) FROM raw_table")
        row_num = conn.fetchone()[0]
        logging.info(f"LOAD TABLE WITH {row_num} ROWS")
        for col in self.col_names:
            conn.execute(f"SELECT DISTINCT {col} FROM raw_table")
            self.value_set[col] = [row[0] for row in conn.fetchall()]
        # print(self.value_set)
        return conn
    
    # def get_columns(self, table="raw_table"):
    #     sql = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}' ORDER BY ORDINAL_POSITION"
    #     self.conn.execute(sql)
    #     return [row[0] for row in self.conn.fetchall()]

if __name__ == "__main__":
    data_model = XplainerDataModel("../data/child.csv", [f"col_{i}" for i in range(20)])
    data_model.conn.execute("SELECT COUNT(*) FROM raw_table")
    print(data_model.conn.fetchone())