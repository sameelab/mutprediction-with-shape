# README for example pipeline
### Zian Liu
#### Last updated: 12/16/2020

## Description

This is an example run of our Lasso-based selection model accompanying the publication.

As a reminder, the goal of our study is to build a shape + sequence based model that predicts mutation rate variations with better performance than the current state-of-the-art model.

To run, simply call:
```
python main.py ./input/example_mutrates.txt ./input/ref_7mers_structure_cpg.xlsx
```
In the command line.


This script takes a structured mutation rate data, as well as the *DNAshapeR* library, and returns the built model and predictions made on both training and testing data.

Note: the input data to this pipeline should be a file named "ben_data_7mer_bayesian_test_training_AFR_10" that is available on Dr. Benjamin Voight's GitHub. However, we strongly encourage you to reach out to Dr. Voight before using their data. We have included a similarly structured mutation rate file that is generated from calculating noncoding mutation rates of chromosomes 1 and 2. We **DO NOT** encourage you to re-use this piece of data for other purposes, as it is designed to be a toy dataset and we don't have the documentation for reproducing it. If you want to have a readily usable set of mutation rate data, please contact Dr. Hassan Samee and we are willing to help. 

Note that since our Lasso-based feature selection can take a very long time on complex feature inputs, we used a relatively simple input of 1st order shape + 2nd order nucleotide features. As a result, the performance of the resulting model won't be optimal. 

The script includes a function for transforming 1st order DNA shape features into 2nd order shape features with only neighboring interactions. We have a block that is currently commented out which does so; uncomment the block to conduct 2nd order feature transformation. 
