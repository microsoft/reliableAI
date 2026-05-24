import os
import inspect
import tempfile
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import datetime
from pgmpy.base import DAG
import argparse


def generate_dat(df: pd.DataFrame, tmp: tempfile.TemporaryDirectory, benchmark_name):
    ####################################################
    print('make data form')
    ####################################################
    dat_path = os.path.join(tmp, f"{benchmark_name}_data.dat")
    out_str = " ".join(map(str, df.columns)) + "\n"
    card = df.apply(pd.Series.nunique)
    out_str += " ".join([str(card[col]) for col in df.columns]) + "\n"
    for _, row in df.iterrows():
        out_str += " ".join([str(row[col]) for col in df.columns]) + "\n"
    with open(dat_path, "w") as f:
        f.write(out_str)


def parent_set_iden(tmp: tempfile.TemporaryDirectory, benchmark_name):
    ####################################################
    print('parent_set_iden')
    ####################################################

    # java -jar blip.jar scorer.is -d data/child-5000.dat -j data/child-5000.jkl -t 10 -b 0
    dat_path = os.path.join(tmp, f"{benchmark_name}_data.dat")
    jkl_path = os.path.join(tmp, f"{benchmark_name}_parent_set.jkl")

    os.system(f"java -Xmx200G -jar {tmp}/../blip.jar scorer.is -d {dat_path} -j {jkl_path} -t {TIMEOUT1} -b 0")

    if not os.path.exists(jkl_path):
        print("failed")


def general_struc_opt(tmp: tempfile.TemporaryDirectory, benchmark_name):
    ####################################################
    print('general_struc_opt')
    ####################################################

    # java -jar blip.jar solver.winasobs.adv -smp ent -d data/child-5000.dat -j data/child-5000.jkl -r data/child.wa.res -t 10 -b 0
    
    dat_path = os.path.join(tmp, f"{benchmark_name}_data.dat")
    jkl_path = os.path.join(tmp, f"{benchmark_name}_parent_set.jkl")
    res_path = os.path.join(tmp, f"{benchmark_name}_graph.res")

    os.system(f"java -Xmx200G -jar {tmp}/../blip.jar solver.winasobs.adv -smp ent  -d {dat_path} -j {jkl_path} -r {res_path} -t {TIMEOUT2} -b 0")

    if not os.path.exists(res_path):
        print("failed")


def parse_res_file(col_names: List[str], tmp: tempfile.TemporaryDirectory, benchmark_name) -> List[Tuple[str, str]]:
    ####################################################
    print('parse_res_file')
    ####################################################

    res_path = os.path.join(tmp, f"{benchmark_name}_graph.res")
    adj_path = os.path.join(tmp, f"{benchmark_name}_graph.txt")

    with open(res_path) as f:
        lines = [l.strip() for l in f.readlines() if not l.startswith("Score") and l.strip() != ""]
    
    adj = np.zeros([len(col_names), len(col_names)])
    for line in lines:
        if "(" not in line:
            continue
        child = int(col_names[int(line.split(":")[0].strip())])
        start_idx = line.index("(") + 1
        end_idx = line.index(")")
        parent_raw = line[start_idx: end_idx].split(",")
        for p in parent_raw:
            parent = int(col_names[int(p)])
            adj[parent, child] = 1
    
    with open(adj_path, 'wb') as f:
        np.savetxt(adj_path, adj, fmt='%d')
    return adj


def dag_learning(df: pd.DataFrame, benchmark_name: str, exp_number: str, is_small: bool = False) -> List[Tuple[str, str]]:
    global TIMEOUT1, TIMEOUT2
    if is_small:
        TIMEOUT1 = 300
        TIMEOUT2 = 60
    else:
        TIMEOUT1 = len(df.columns)
        TIMEOUT2 = len(df.columns)
    
    tmp=os.path.join(os.path.dirname(os.path.realpath(inspect.getfile(inspect.currentframe()))), 'temp_'+exp_number)
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    generate_dat(df, tmp, benchmark_name)
    parent_set_iden(tmp, benchmark_name)
    general_struc_opt(tmp, benchmark_name)
    dag = parse_res_file(df.columns, tmp, benchmark_name)
    return dag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BLIP')
    parser.add_argument("--exp_number", type=str, default="exp_1", help="")
    args = parser.parse_args()

    for benchmark_name in ['01earthquake', '02survey', '03asia', '04sachs',  '05child', '06insurance', '07water', '08mildew', '09alarm', '10barley', '11hailfinder', '12hepar2', '13win95pts', '14pathfinder']:
        start_time = datetime.now()

        input_file = f'../../datasets/experiment/original/{args.exp_number}/aug_samples/{benchmark_name}_aug_dataset.npy'
        df = pd.DataFrame(np.load(input_file))
        
        dag = dag_learning(df, benchmark_name, args.exp_number)
        
        end_time = datetime.now()
        print(f"Done! Duration: {end_time - start_time}\n")