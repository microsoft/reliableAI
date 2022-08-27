# Introduction

This repository contains the open-sourced official implementation of the paper:
"pureGAM: Learning an Inherently Pure Additive Model" (KDD '22). If you find this
repository helpful, please cite the following paper:

```
@inproceedings{ml4s,
author = {Ma, Pingchuan and Ding, Rui and Dai, Haoyue and Jiang, Yuanyuan and Wang, Shuai and Han, Shi and Zhang, Dongmei},
title = {pureGAM: Learning an Inherently Pure Additive Model},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539256},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {1728–1738},
numpages = {11},
keywords = {identifiability, pureness, interpretable machine learning, generalized additive model, pure coding},
location = {Washington DC, USA},
series = {KDD '22}
}
```

For any questions/comments, please feel free to open GitHub issues.

# pureGAM Model overview
1. pureGAM, an inherently pure additive model of both main effects and higher-order interactions, is implemented
in __pureGAM_model/pureGAM.py__ ;
2. Imitate __experiments/run_experiment_realdata.py__ or __experiments/run_experiment_realdata.py__ 
(which is examples of applying pureGAM in real datasets and synthetic datasets, respectively) to specify 
the datasets and hyper-parameters of pureGAM.
3. To use your own data
   1. For real data, you may place the datasets in __realdata/__ folder
   2. For synthetic data, you can write your own scripts in __synthetic_datagenerator/__
4. pureGAM and other XAI model can use metrics in __metrics/__ to evaluate the pureness of their results.

# Run `pureGAM` experiments

1. Use requirements.txt to initialize the environment and install packages.
2. Core code is in __experiments__
   1. Run __experiments/run_experiment_realdata.py__ for real data experiment
   2. Run __synthetic_datagenerator/categorical_gen.py__ and __synthetic_datagenerator/numerical_gen.py__ 
   to generate synthetic data.
   3. Run __experiments/run_synthetic_cate.py__ and __experiments/run_synthetic_num.py__ for synthetic data experiment


<!--
## MMPC, REAL and others

We are still working on making our implementation on MMPC and REAL publicly
available. Please stay tuned. 

You may also modify the `estimate_bn` method to support other algorithms as long
as it returns a valid instance of `pgmpy.models.BayesianNetwork`.
-->


<!--
## known issues
1. Due to the limitation of this implementation, for categorical data with arbitrary interactions that is 
   not grid-close, `pureGAM` can throw an exception and crush.
-->

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
