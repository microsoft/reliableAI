# Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning
[![Conference](http://img.shields.io/badge/AISTATS-2025-4b44ce.svg)](https://aistats.org/aistats2025/) 

This repository contains the open-sourced official implementation of the SiCL approach proposed in paper: "Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning" (AISTATS 2025). 

## Run

1. skeleton learning

    > python main_skeleton_graph_learning.py [--config /path/to/config]

2. v-structure learning

    > python main_orientation.py [--config /path/to/config]

3. CPDAG prediction && test

    > python main_prediction_test.py [--config /path/to/config]

The default configure files are in folder `configs/`. It can be modified accordingly.


## Environment

Use conda to create the environment:
> conda env create -f sicl.yml

## Cite
TBD