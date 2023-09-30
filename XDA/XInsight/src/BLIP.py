# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List, Tuple
import pandas as pd

import os, tempfile
from src.logger import *

if os.path.exists("./lib/jdk-11.0.13+8/bin/java"):
    raise RuntimeError("java not found. Please download it from https://www.oracle.com/java/technologies/javase-jdk11-downloads.html")

if os.path.exists("./lib/blip/blip.jar"):
    raise RuntimeError("blip.jar not found. Please download it from https://github.com/mauro-idsia/blip")


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

    os.system(f"./lib/jdk-11.0.13+8/bin/java -Xmx200G -jar ./lib/blip/blip.jar scorer.is -d {dat_path} -j {jkl_path} -t {TIMEOUT1} -b 0")

    if not os.path.exists(jkl_path):
        print("failed")

def general_struc_opt(tmp: tempfile.TemporaryDirectory):
    # java -jar blip.jar solver.winasobs.adv -smp ent -d data/child-5000.dat -j data/child-5000.jkl -r data/child.wa.res -t 10 -b 0

    dat_path = os.path.join(tmp, f"data.dat")
    jkl_path = os.path.join(tmp, f"parent_set.jkl")
    res_path = os.path.join(tmp, f"graph.res")

    os.system(f"./lib/jdk-11.0.13+8/bin/java -Xmx200G -jar ./lib/blip/blip.jar solver.winasobs.adv -smp ent  -d {dat_path} -j {jkl_path} -r {res_path} -t {TIMEOUT2} -b 0")

    if not os.path.exists(res_path):
        print("failed")

def parse_res_file(col_names: List[str], tmp: tempfile.TemporaryDirectory) -> List[Tuple[str, str]]:
    res_path = os.path.join(tmp, f"graph.res")
    with open(res_path) as f:
        lines = [l.strip() for l in f.readlines() if not l.startswith("Score") and l.strip() != ""]
    skeleton = []
    for line in lines:
        if "(" not in line: continue
        child = col_names[int(line.split(":")[0].strip())]
        start_idx = line.index("(") + 1
        end_idx = line.index(")")
        parent_raw = line[start_idx: end_idx].split(",")
        for p in parent_raw:
            parent = col_names[int(p)]
            skeleton.append((child, parent))
    return skeleton

def skeleton_learning(df: pd.DataFrame, is_small: bool=False) -> List[Tuple[str, str]]:
    global TIMEOUT1, TIMEOUT2
    if is_small:
        TIMEOUT1 = 300
        TIMEOUT2 = 120
    else:
        TIMEOUT1 = 600
        TIMEOUT2 = 1000
    with tempfile.TemporaryDirectory() as tmp:
        logging.info("LEARNING SKELETON [BLIP]")
        generate_dat(df, tmp)
        parent_set_iden(tmp)
        general_struc_opt(tmp)
        skl = parse_res_file(df.columns ,tmp)
    return skl
