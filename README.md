# Web data classification
An attempt to improve existing solution of web pages classification.

## How to
Check the solution by doing. 

```bash

# Copy your dataset to ./data forlder
# and then run
# 
# It may take a while, on Intel Xeon E5-2680v2 8 cores/2.8GHz 
# it takes ~75-100 min to run everything.
 
make
```

## Workflow
1. Understand the existing solution
2. Look closer to the data
3. Try different base algorithm/parameters/feature transformators
4. Use OneVsRest approach to solve multilabel-classification problem
5. Clean the sample (reduce number of unique objects) to be able to train different models
6. Search parameters for OvR approach and compare the methods
7. Clustering(for unlabelled/ feature propagation), dimensionality reduction, voting schemes were not considered yet.


## Results
Here is the output that I get on my machine

```
Running original approach on clean dataset
Loading data ...
Loaded 168621 objects
Thres 0.01 avg f1 0.17400940461333647
Thres 0.02 avg f1 0.2749392341140847
Thres 0.03 avg f1 0.3091445013484581
Thres 0.04 avg f1 0.32699119759324713
Thres 0.05 avg f1 0.3364683864648282
Thres 0.06 avg f1 0.34126627107697416
Thres 0.07 avg f1 0.34583145923880343
Thres 0.08 avg f1 0.3473901838796516
Thres 0.09 avg f1 0.3482033012953878
Thres 0.1 avg f1 0.3480773056634144
Best threshold found 0.09
Best F1 0.3482033012953878

Running hybrid solution
Loading data ...
Loaded 133851 objects
Thres 0.01 avg f1 0.287958705818203
Thres 0.02 avg f1 0.3703813464325074
Thres 0.03 avg f1 0.38547569004192733
Thres 0.04 avg f1 0.3913487486060178
Thres 0.05 avg f1 0.39327046361083956
Thres 0.06 avg f1 0.3946631887671244
Thres 0.07 avg f1 0.3954701923802124
Thres 0.08 avg f1 0.39496643770020334
Thres 0.09 avg f1 0.39424552775417915
Thres 0.1 avg f1 0.3934446431951139
Best threshold found 0.07
Best F1 0.3954701923802124

Running OvR solution
Loading data ...
On test sample F1: 0.415669205658
Using custom F1 measure 0.326537719335
```