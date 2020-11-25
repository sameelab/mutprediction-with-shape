# Predicting mutation rate variations with DNA shape
### Zian Liu
### Last updated: 11/25/2020

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


If you are using any data pertinent to the *DNAshape* method and/or the *DNAshapeR* package, please make sure to cite:


For all other usages, please cite our publication:


## Contact 

Please contact Md. Abul Hassan Samee, Ph.D. (md.abulhassan.samee@bcm.edu) for questions related to our publication. If you have questions regarding our research, please contact Dr. Samee or Zian Liu (zian.liu@bcm.edu). 

# Thank you!
