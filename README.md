# Supervised Hierarchical Clustering with Exponential Linkage
This repository contains code used in experiments for our ICML 2019 paper titled  "[Supervised Hierarchical Clustering with Exponential Linkage](http://proceedings.mlr.press/v97/yadav19a.html)". 

## Setup ##

Clone<sup>*</sup>  and setup **xcluster** repository from <https://github.com/iesl/xcluster>. 
Make sure **xcluster** repo is cloned in the same folder as this repo i.e. you should have **xcluster** and **expLinkage** folder in the same parent folder.  

Set environment variables:

```
cd expLinkage
source bin/setup.sh
```

## Data Setup ##


#### Data in *n*-dim vector space ####

`clusterFile` parameter in config files should point to the tsv file which contains data with each line in following format:

`<point_id> <cluster_id> <dim-1> <dim-2> .... <dim-n>`

#### Data with features defined on every pair of points ####

`dataDir` parameter in config files should point to data folder which should be present in the following format:
```bash
├── NP_Coref
|   ├── doc1
|       ├── gtClusters.tsv
|       ├── pairFearues.tsv
|   ├── doc2
|   ├── ...
|   ├── docn
    
```

All data should be in a single folder with a separate sub-folder for each canopy or set of points. Each sub-folder contains files: `gtClusters.tsv` and `pairFeatures.tsv`. 
 
`gtClusters.tsv` contains information about ground-truth clusters for each point in following format:
`<pointId> <clusterId>`

`pairFeatures.tsv` contains feature vector for each pair of points in following format:  
`<pointId_1> <pointId_2> <feature_1>  <feature_2> ... <feature_n>`

Set of points in each subfolder will be clustered separately.

## Run Code ##

#### For data in *n*-dim vector space ####

```bash
cd expLinkage
python src/trainer/train_vect_data.py --config=<rel_path_to_config_file> --seed=<random_seed>
```

#### For data with features on every pair of points ####

```bash
cd expLinkage
python src/trainer/train_pair_feat.py --config=<rel_path_to_config_file> --seed=<random_seed>
```

Config files for all experiments in the paper are present in [config](config) folder.


## Notes ##
- *Code from **xcluster** repository is only used for evaluating dendrogram purity and is not crucial for training as such (if evaluation does not involve computing dendrogram purity or no evaluation on dev set is peformed during training).
