# VICNTM
a self-superivised neural topic model with variance-invariance-covariance regularization

<img src=2in1vic_v2.jpg width=500px>

data_{dataset name}.py are python files that download and preprocess the datasets used in experiments.

For 20NG and IMDb datasets, simply run data_20ng.py and data_IMDb.py.
For wiki dataset, you should first download the raw wikitext-103 dataset from https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip, and then run data_wiki103.py to preprocess it.

