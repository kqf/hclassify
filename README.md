# Web data classification
An attempt to improve existing solution of web pages classification.

## How to
Check the solution by doing

```bash

# Copy your dataset to ./data forlder
# and then run

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