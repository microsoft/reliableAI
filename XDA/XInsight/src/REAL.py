# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os, tempfile
from typing import List, Tuple
import pandas as pd

from src.logger import *

EXE_PATH = "./lib/real/PGM.Experiments.exe"

if not os.path.exists(EXE_PATH):
    raise RuntimeError("REAL not found. This is currently not open source.")

def generate_tsv(df: pd.DataFrame, tmp: tempfile.TemporaryDirectory):
    tsv_path = os.path.join(tmp, "data.tsv")
    df.to_csv(tsv_path, sep="\t", index=False)

def run_real(tmp: tempfile.TemporaryDirectory):
    tsv_path = os.path.join(tmp, "data.tsv")
    graph_path = os.path.join(tmp, "graph.txt")
    temp_path = os.path.join(tmp, "temp.txt")

    os.system(f"mono {EXE_PATH} REAL2 {tsv_path} {graph_path} {temp_path}")

def parse_graph(col_names: List[str], tmp: tempfile.TemporaryDirectory) -> List[Tuple[str, str]]:
    graph_path = os.path.join(tmp, "graph.txt")
    with open(graph_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip() != ""]
    skeleton = []
    for i, line in enumerate(lines):
        for j, has_edge in enumerate(line.split()):
            if has_edge =="1": skeleton.append((col_names[int(i)], col_names[int(j)]))
    return skeleton

def skeleton_learning(df: pd.DataFrame, is_small: bool=True) -> List[Tuple[str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        logging.info("LEARNING SKELETON [REAL]")
        generate_tsv(df, tmp)
        run_real(tmp)
        skl = parse_graph(df.columns, tmp)
    return skl