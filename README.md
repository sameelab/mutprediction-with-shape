# Predicting mutation rate variations with DNA shape
### Zian Liu
##### Last updated: 11/25/2020

## Introduction

This is the Github repository for the Liu and Samee 2020 publication, predicting single nucleotide mutation rate variations with a combination of DNA shape features and sequence context features.

This repo contains the main notebook for our publication, as well as other codes and snippets used for generating our final results. 

## What are the input data?

For the mutation rates data, please see Dr. Benjamin Voight's Github (https://github.com/bvoight/mutvar_fwdRegr). This repo contains the input file for our study. 

For the *DNAshape* reference table, we have included a 7-mer reference table in the "data_input" directory. However, we strongly encourage you to also check out our other repo (https://github.com/ZnL-BCM/DNAshapeR_reference) which contains scripts for extracting the reference table from the DNAshapeR package. Please make sure to cite the DNAshapeR papers when using this excel spreadsheet. 

For the *DNAshapeR* package, please visit Tsu-Pei Chiu's awesome Github page (http://tsupeichiu.github.io/DNAshapeR/) for more information.

## How can I run this for myself?

We have included our main Jupyter notebook in this repo. However, since it is very messy, we encourage using it mostly as a "reference" document. Instead, we have separately prepared an example pipeline in the "pipeline_example" directory. 

To run our model, simply call "main.py" from the example directory. The included README file will share more regarding what to do, and the script file is well annotated for you to follow.

## Citations

If you are using the input data from *Aggarwala and Voight*, please make sure to cite:

* Aggarwala, V. & Voight, B. F. An expanded sequence context model broadly explains variability in polymorphism levels across the human genome. *Nature Genetics 48*, 349–355 (2016).


If you are using any data pertinent to the *DNAshape* method and/or the *DNAshapeR* package, please make sure to cite (yes, all four):

* Chiu, T.-P. et al. DNAshapeR: an R/Bioconductor package for DNA shape prediction and feature encoding. *Bioinformatics 32*, 1211–1213 (2016).
* Chiu, T.-P., Rao, S., Mann, R. S., Honig, B. & Rohs, R. Genome-wide prediction of minor-groove electrostatic potential enables biophysical modeling of protein–DNA binding. *Nucleic Acids Res 45*, 12565–12576 (2017).
* Li, J. et al. Expanding the repertoire of DNA shape features for genome-scale studies of transcription factor binding. *Nucleic Acids Res 45*, 12877–12887 (2017).
* Rao, S. et al. Systematic prediction of DNA shape changes due to CpG methylation explains epigenetic effects on protein–DNA binding. *Epigenetics & Chromatin 11*, 6 (2018).


For all other usages, please cite our publication:

* Liu, Z. * Samee, M. A. H. Title. ......


## Contact 

Please contact Md. Abul Hassan Samee, Ph.D. (md.abulhassan.samee@bcm.edu) for questions related to our publication. 

Please contact Zian Liu (zian.liu@bcm.edu) or Dr. Samee for questions related to specifically our research. Note that if you are accessing this page on/after Spring 2023, Zian Liu might be overly ecstatic on completing his Ph.D. and will not be available to answer your questions; if you don't hear back from Zian for 2 days, please email Dr. Samee. 

# Thank you!
