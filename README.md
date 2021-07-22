# Predicting mutation rate variations with DNA shape
### Zian Liu
### Last updated: 7/22/2021

## Introduction

This is the GitHub repository for the Liu Z and Samee MAH 2021 publication, Mutation rate variations in the human genome are encoded in DNA shape. The manuscript is currently under review and is submitted to bioRxiv (https://doi.org/10.1101/2021.01.15.426837); see the bottom of the page for citation information.

This repo contains the main notebook for our publication, as well as other codes and snippets used for generating our final results. 

## What are the input data?

For the mutation rates data, please request Dr. Benjamin F. Voight; their data is also available from Dr. Voight's GitHub (https://github.com/bvoight/mutvar_fwdRegr). We would **strongly encourage** you to first communicate with and request access from Dr. Voight and cite their study in the case you want to use their data. 

As you might have noticed, we included an input mutation rate data file in our example script directory. We would **strongly discourage** you to directly use this data for other purposes. This input data is generated by one of our in-production pipelines, and then re-formatted to match the format of the Aggarwala and Voight data. It is intended to be a toy dataset and we do not currently have documentation for how to generate it. If you are interested, please stay tuned as we do have plans to release our pipeline to the Samee Lab GitHub, or contact us and we are more than happy to pass the data (as well as the steps to generate it) to you.

For the *DNAshape* reference table, we have included a 7-mer reference table in the "data_input" directory. We also encourage checking out Zian Liu's repo (https://github.com/ZnL-BCM/DNAshapeR_reference) which contains scripts for extracting the reference table from the *DNAshapeR* package. Please make sure to cite the four *DNAshapeR* papers when using this excel spreadsheet.

For the *DNAshapeR* package, please visit Tsu-Pei Chiu's GitHub page (http://tsupeichiu.github.io/DNAshapeR/) for more information.

## How can I run this for myself?

We have included our main Jupyter notebook ("Publication_note.ipynb") as a reference document. We have separately prepared an example pipeline in the "pipeline_example" directory. 

To run our model, call:

``python main.py input_mutation_file reference_dnashape_file.xlsx``

from the example directory. The included README file will share more regarding what to do, and the script file is well annotated for you to follow.

## Citations

If you are using the input data from *Aggarwala and Voight*, please make sure to cite:

* Aggarwala, V. & Voight, B. F. An expanded sequence context model broadly explains variability in polymorphism levels across the human genome. *Nature Genetics 48*, 349–355 (2016).

If you are using any data pertinent to the *DNAshape* method, the *DNAshapeR* package, or our curated DNA shape tables, please make sure to cite all four of the following:

* Chiu, T.-P. et al. DNAshapeR: an R/Bioconductor package for DNA shape prediction and feature encoding. *Bioinformatics 32*, 1211–1213 (2016).
* Chiu, T.-P., Rao, S., Mann, R. S., Honig, B. & Rohs, R. Genome-wide prediction of minor-groove electrostatic potential enables biophysical modeling of protein–DNA binding. *Nucleic Acids Res 45*, 12565–12576 (2017).
* Li, J. et al. Expanding the repertoire of DNA shape features for genome-scale studies of transcription factor binding. *Nucleic Acids Res 45*, 12877–12887 (2017).
* Rao, S. et al. Systematic prediction of DNA shape changes due to CpG methylation explains epigenetic effects on protein–DNA binding. *Epigenetics & Chromatin 11*, 6 (2018).

For all other usages pertinent to our work, our manuscript is currently under review. In the meantime, please cite the following submission in bioRxiv:

* Liu, Z. & Samee, M. A. H. Mutation rate variations in the human genome are encoded in DNA shape. *bioRxiv 2021.01.15.426837*. doi: https://doi.org/10.1101/2021.01.15.426837


## Contact 

Please contact Md. Abul Hassan Samee, Ph.D. (samee@bcm.edu) for questions related to our publication or other logistics-related questions. 

Please contact Zian Liu (zian.liu@bcm.edu) for questions specifically related to our research. Note that if you are accessing this page on or after **Spring 2023** and you don't hear back from Zian for 2 days, please email Dr. Samee directly. 

# Thank you!
