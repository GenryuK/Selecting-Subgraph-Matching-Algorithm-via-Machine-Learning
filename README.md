# Selecting-Subgraph-Matching-Algorithm-via-Machine-Learning

## Introduction
Subgraph matching is one of the fundamental problems in graph
analysis. Numerous subgraph matching algorithms have been developed
for efficient processing. However, it is known that the best
algorithm differs across query and data graphs. A previous study
has proposed manually designed rule-based models for selecting algorithms
depending on the characteristics of query and data graphs
based on their experimental results, but it may not be effective if
we use different data graphs. In this paper, we propose a framework
with machine learning to select a subgraph matching algorithm
for improving efficiency depending on datasets and queries. Our
framework learns from pairs of subgraph matching algorithms
and their performance. It consists of two key components: labeling
strategy and featurizer. The labeling strategy adds labels to experimental
results, and the featurizer extracts well-designed features
from datasets and queries. Our experiments show that our framework
can select high-performing subgraph matching algorithms
and improve EPS by up to 76% over baselines.

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
