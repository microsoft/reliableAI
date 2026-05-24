# TICL

Implementation for ICML'26 paper: Test-Time Learning of Causal Structure from Interventional Data.

## Dependencies

- ```conda env create -f environment.yml```

- Note:
  - You also need to configure the Java environment to run JCI-BLIP: apache-maven-3.8.4 / jdk-11.0.13+8
  - If you need SID indicators, follow the corresponding R package: https://cran.r-project.org/web/packages/SID/index.html

## Datasets

#### Raw data：
- bnlearn:  <https://www.bnlearn.com/bnrepository/>

#### Prepared parameter network：

- The prepocessed parameter graph should be upload in the folder [bif](./datasets/raw/bif/).
- Note: bif forms of each dataset for forward-sampling.

## Usage
- Preprocess original data. (If you want to use other dataset, please prepare the bif data format under the folder data.)
- ```cd src```
- ```python main.py```
- Adjust the hyperparameters and strategies according to the needs
  - e.g. ```python main.py --exp_number exp_1 --intervention_type soft --support_type hard --observation_sample 20000 --intervention_sample 10000```

## Comparison of Baselines
  + I-CPDAG Discovery:  
    + [GIES](./baselines/exps_of_baselines/gies_exps.ipynb)
    + [IGSP](./baselines/exps_of_baselines/igsp_exps.ipynb)
    + [UT-IGSP](./baselines/exps_of_baselines/ut-igsp_exps.ipynb)
    + [ENCO](./baselines/exps_of_baselines/enco_exps.ipynb)
    + [AVICI](./baselines/exps_of_baselines/avici_exps.ipynb)
    + [JCI-BLIP](./baselines/exps_of_baselines/jci-blip_exps.ipynb)
    + [JCI-HC](./baselines/exps_of_baselines/jci-hc_exps.ipynb)
    + [JCI-PC](./baselines/exps_of_baselines/jci-pc_exps.ipynb)
    + [JCI-GOLEM](./baselines/exps_of_baselines/jci-golem_exps.ipynb)
    + others ...
  
  + Intervention Target Detection
    + [CITE](./baselines/exps_of_baselines/cite_exps.ipynb)
    + [PreDITEr](./baselines/exps_of_baselines/pre_exps.ipynb)
    + others ...

## Cite
```
@inproceedings{chen2026test,
  title={Test-Time Learning of Causal Structure from Interventional Data},
  author={Chen, Wei and Ding, Rui and Huang, Bojun and Zhang, Yang and Fu, Qiang and Liang, Yuxuan and Shi, Han and Zhang, Dongmei},
  booktitle={Forty-three International Conference on Machine Learning},
}