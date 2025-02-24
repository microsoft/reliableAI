# Introduction

This repository contains the open-sourced official implementation of the paper:
"ML4S: Learning Causal Skeleton from Vicinal Graphs" (KDD '22). If you find this
repository helpful, please cite the following paper:

```
@inproceedings{ml4s,
author = {Ma, Pingchuan and Ding, Rui and Dai, Haoyue and Jiang, Yuanyuan and Wang, Shuai and Han, Shi and Zhang, Dongmei},
title = {ML4S: Learning Causal Skeleton from Vicinal Graphs},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539447},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {1213–1223},
numpages = {11},
keywords = {bayesian network, structure learning, causal discovery},
location = {Washington DC, USA},
series = {KDD '22}
}
```

For any questions/comments, please feel free to open GitHub issues.

# Config

1. Use __configs/benchmark.yml__ to specify the datasets;
2. Use __configs/generate_random_graphs.yml__ and
   __configs/generate_sibling.yml__ to specify the parameters for generating
   vicinal graphs;
3. Use  __configs/ml4s.yml__ to specify the parameters in ML4S (e.g., data path).

# Prepare Pseudo Graph

`ML4S` is dependent on existing DAG learning algorithms as a proxy to first
learn a pseudo graph and then generate vicinal graphs (i.e., training data) on
the top of the pseudo graph. Currently, we provide supports to several
algorithms. Change line 291 in
__Experiments/DataGeneration/BayesianNetworkEstimator.py__ to specify proxy
algorithm and also check `estimate_bn` method at line 28 to ensure that the path
is set correctly.

## BLIP as proxy algorithm

1. Clone `https://github.com/mauro-idsia/blip` to the folder;
2. Following its instruction to build BLIP project with Java;
3. Modify __Tools/blip_helper.py__ to run BLIP and store `res` file.

## HC (Hill Climbing) as proxy algorithm

1. We use the `pgmpy` implementation to run HC. simply change `blip` to `hc` in
line 291 in __Experiments/DataGeneration/BayesianNetworkEstimator.py__ would
work. 

Note that HC does not scale well to large datasets.

## NOTEARS as proxy algorithm

1. Clone `https://github.com/xunzheng/notears`;
2. Follow its instruction to run `NOTEARS` and save the results accordingly
   (i.e., __notears/data/{benchmark name}.txt__).

## DAGGNN as proxy algorithm

1. Clone `https://github.com/ronikobrosly/DAG_from_GNN`;
2. Follow its instruction to run `DAGGNN` and save the results accordingly
   (i.e., __daggnn/data/{benchmark name}.txt__).

## MMPC, REAL and others

We are still working on making our implementation on MMPC and REAL publicly
available. Please stay tuned. 

You may also modify the `estimate_bn` method to support other algorithms as long
as it returns a valid instance of `pgmpy.models.BayesianNetwork`.

# Run `ML4S`

1. Use env.yml to initialize the conda environment
2. Core code is in __Experiments__
3. Use __GenerateSibling.py__ to create vicinal graph
4. __ML4S.py__ is for skeleton learning

# Use `ML4S` on your own data

1. Prepare your data in numpy array format as `npy10000/{your data}.npy` and corresponding
   ground-truth adjacency matrix as `{your data}.txt` in __benchmarks__ folder.
2. Modify  __configs/benchmark.yml__ to include `{your data}` and comment other benchmarks.
3. Follow the instruction in **Prepare Pseudo Graph** and **Run `ML4S`**.

## known issues

Due to the limitation of forward sampling in `pgmpy`, when the dataset is
complex (e.g., with dense graph structure or high cardinality), it may fail (or
very slow) to produce vicinal graphs and associated datasets. We confirm that
`ML4S` successfully process all datasets used in our paper.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
