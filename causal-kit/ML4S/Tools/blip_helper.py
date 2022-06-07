from typing import List, Tuple
import pandas as pd
import os, tempfile
from pgmpy.base import DAG

TIMEOUT1 = 600
TIMEOUT2 = 1000

def generate_dat(df: pd.DataFrame, tmp: tempfile.TemporaryDirectory):
    dat_path = os.path.join(tmp, "data.dat")
    out_str = " ".join(map(str, range(len(df.columns)))) + "\n"
    card = df.apply(pd.Series.nunique)
    out_str += " ".join([str(card[col]) for col in df.columns]) + "\n"
    for index, row in df.iterrows():
        out_str += " ".join([str(row[col]) for col in df.columns]) + "\n"
    with open(dat_path, "w") as f:
        # print(out_str)
        f.write(out_str)

def parent_set_iden(tmp: tempfile.TemporaryDirectory):
    # java -jar blip.jar scorer.is -d data/child-5000.dat -j data/child-5000.jkl -t 10 -b 0 
    dat_path = os.path.join(tmp, f"data.dat")
    jkl_path = os.path.join(tmp, f"parent_set.jkl")

    os.system(f"java -Xmx200G -jar ../blip/lib/blip/blip.jar scorer.is -d {dat_path} -j {jkl_path} -t {TIMEOUT1} -b 0")

    if not os.path.exists(jkl_path):
        print("failed")

def general_struc_opt(tmp: tempfile.TemporaryDirectory, res_path: str):
    # java -jar blip.jar solver.winasobs.adv -smp ent -d data/child-5000.dat -j data/child-5000.jkl -r data/child.wa.res -t 10 -b 0

    dat_path = os.path.join(tmp, f"data.dat")
    jkl_path = os.path.join(tmp, f"parent_set.jkl")

    os.system(f"java -Xmx200G -jar ../blip/lib/blip/blip.jar solver.winasobs.adv -smp ent  -d {dat_path} -j {jkl_path} -r {res_path} -t {TIMEOUT2} -b 0")

    if not os.path.exists(res_path):
        print("failed")

def run_blip(df: pd.DataFrame, res_path: str):
    tmp = tempfile.TemporaryDirectory()
    generate_dat(df, tmp)
    parent_set_iden(tmp)
    general_struc_opt(tmp, res_path)
    dag = get_blip(res_path)
    return dag
    
def get_blip(res_path: str):
    with open(res_path) as f:
        lines = [l.strip() for l in f.readlines() if not l.startswith("Score") and l.strip() != ""]
    dag = DAG()
    dag.add_nodes_from([str(i) for i in range(len(lines))])
    for line in lines:
        if "(" not in line: continue
        child = line.split(":")[0]
        start_idx = line.index("(") + 1
        end_idx = line.index(")")
        parent_raw = line[start_idx: end_idx].split(",")
        for p in parent_raw:
            dag.add_edge(p, child)
    return dag

if __name__ == "__main__":
    run_blip(pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), "../blip/data/child.res")