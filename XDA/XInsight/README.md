# Introduction

This repository contains the open-sourced official implementation of the paper:
"XInsight: eXplainable Data Analysis Through The Lens of Causality" (SIGMOD '2). If you find this
repository helpful, please cite the following paper:

```
@article{ma2023xinsight,
  title={XInsight: eXplainable Data Analysis Through The Lens of Causality},
  author={Ma, Pingchuan and Ding, Rui and Wang, Shuai and Han, Shi and Zhang, Dongmei},
  journal={Proceedings of the ACM on Management of Data},
  volume={1},
  number={2},
  pages={1--27},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```

For any questions/comments, please feel free to open GitHub issues.

# Installation

```
pip install -r requirements.txt
```

1. Download BLIP from https://github.com/mauro-idsia/blip and JDK 11;
2. Place these files in `lib` folder according to the specification in `src/BLIP.py`;

# Run a query

## Query Spec

example query JSON file:

```json
{
   "fp": {
      "cols": [],
      "vals": []
   },
   "dp": {
      "col": "X",
      "vals": [1, 0]
   },
   "measure": {
      "col": "Z",
      "func": "sum"
   },
   "prior_cols": ["Y"]
}
```

fields:
- `fp`: The `fp` field in the JSON file stands for Fixed Predicate. This field is composed of two subfields:
  - `cols`: the list of column names in the predicate;
  - `vals`: the list of values in the predicate.
- `dp`: The `dp` field in the JSON file stands for Diff Predicate.
  - `col`: the column name of the Diff Predicate;
  - `vals`: two values to be compared.
- `measure`: The `measure` field in the JSON file stands for the measure of the Diff Query
  - `col`: the (numerical) column name of the measure
  - `func`: the aggregation function of the column (currently support `avg` and `sum`).
- `prior_cols` (optional): The list of columns that will be searched over by XPlainer. If not provided, use XLearner to identify columns with causal semantics.

## Run XInsight

```
python run.py --data /path/to/datafile --query /path/to/queryfile --use_xlearner
```


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
