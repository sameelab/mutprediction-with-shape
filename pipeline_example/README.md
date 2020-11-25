# README
### Zian Liu
#### 11/25/2020

## Description

This is an example run of our Lasso-based selection model accompanying the publication.

As a reminder, the goal of our study is to build a shape + sequence based model that predicts mutation rate variations with better performance than the current state-of-the-art model.

To run, simply call:
```
python main.py ./input/ben_data_7mer_bayesian_test_training_AFR_10 ./input/ref_7mers_structure_cpg.xlsx
```
In the command line.


This script takes the structured mutation rate data from *Aggarwala and Voight*, as well as the DNAshapeR library, and returns the built model and predictions made on both training and testing data.

Note that since our Lasso-based feature selection can take a very long time on complex feature inputs, we used a relatively simple input of 1st order shape + 2nd order nucleotide features. As a result, the performance of the resulting model won't be optimal. 

The script includes a function for transforming 1st order DNA shape features into 2nd order shape features with only neighboring interactions. We have a block that is currently commented out which does so; uncomment the block to conduct 2nd order feature transformation. 
