# Selecting Subgraph Matching Algorithm via Machine Learning

## Introduction
Subgraph matching is one of the fundamental problems in graph analysis. Numerous subgraph matching algorithms have been developed for efficient processing. However, it is known that the best algorithm differs across query and data graphs. A previous study has proposed manually designed rule-based models for selecting algorithms depending on the characteristics of query and data graphs based on their experimental results, but it may not be effective if we use different data graphs. In this paper, we propose a framework with machine learning to select a subgraph matching algorithm for improving efficiency depending on datasets and queries. Our framework learns from pairs of subgraph matching algorithms and their performance. It consists of two key components: labeling strategy and featurizer. The labeling strategy adds labels to experimental results, and the featurizer extracts well-designed features from datasets and queries. Our experiments show that our framework can select high-performing subgraph matching algorithms and improve EPS by up to 76% over baselines.

## Code for Subgraph Matching
We generate training data for machine learning using SIGMOD'2020 paper "In-Memory Subgraph Matching: an In-depth Study" by Dr. Shixuan Sun and Prof. Qiong Luo and ICDE'2025 paper "PILOS: Scalable Large-Subgraph Matching by Online Spectral Filtering" by Konstantinos Skitsas, Associate Prof. Davide Mottin, and Prof. Panagiotis Karras. 
The training data contains the results (e.g., EPS, time, the number of embeddings) of 60 algorithms for all queries.

## Compile
Under the root directory of the project, execute the following commands to compile the source code.
```
mkdir
cd build
cmake ..
make
```

## Extract training data
Under the test directory, execute the following commands to get training data including EPS, execution time, and the number of embeddings of each algorithm.
```
python3 calc_EPS.py ../build/matching/SubgraphMatching.out
```

## Featurizer
Under the query graph directory (e.g., ```query_graph_DBLP```), execute the following commands to get features for each query and data graph.
```
python3 featurizer.py ../../build_cand/matching/SubgraphMatching.out
```
When compiling in the root directory, please make sure that build_cand uses the file ```StudyPerformance_candidate.cpp``` instead of ```StudyPerformance.cpp``` for calculating the candidate size.
A training data including not only EPS e.c.t., but also features is in ```test/MLData/All_data.csv``` and you can use this file for evaluate our methods with this data.

## Subgraph Matching with ML
Under the test directory, execute the following commands to evaluate our methods.
```
python3 MLSM.py
```
The type of our methods can be selected by the variable ```task```.
|task|Ours|
|----|----|
|topx|Top-X|
|incy|Inc-Y|
|wei|Weight|

When using Top-X and Inc-Y, ```attribute_num``` can be selected from the following values. These values can be determined through hyper-parameter tuning.
|Ours|attribute_num|
|----|----|
|Top-X|1, 2, 3, 4, 5, 6, 7|
|Inc-Y|1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5|

Other hyper-parameters are below:
|Hyper-param|search space|
|----|----|
|learning rate|5e-4, 1e-3, 5e-3, 1e-2, 5e-2|
|drop out rate|0.0, 0.1, 0.2, 0.3, 0.4, 0.5|
|weight decay|0.0, 1e-5, 1e-4, 1e-3, 1e-2|
|hidden layer size|16, 32, 64, 128|

We conducted experiments using five different seed values: 1, 2, 3, 4, and 5, which were used for tasks such as splitting the training data.\
If you test our methods with unseen dataset (unseen query size), you can execute ```Unseen_dataset.py (Unseen_size.py)```.\
All results of our methods are in ```Results``` directory.

## RecAlgo
The rule of RecAlgo is below:
|stage|algorithms|conditions|
|----|----|----|
|Filtering|DPiso<br>LDF/NLF<br>VEQ|default<br>high selectivity<br>low selectivity|
|Ordering|RI/RM<br>GQL|default<br>sparse dataset & small candidate size|
|Enumeration|KSS<br>VEQ<br>KSS|small tree width<br>default<br>large candidate size|

The threshold for each feature, such as high or low values, is defined as the top 10% or bottom 10% of the entire dataset.
Also, in the table, the conditions listed higher have higher priority.

## Note
