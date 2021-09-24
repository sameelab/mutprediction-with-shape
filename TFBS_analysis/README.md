### Motif processing

Due to size restrictions on GitHub, nearly all of our generated files have been removed. The following describes (to the best of my knowledge) how to reproduce our analyses from scratch.

For our mutations, we have created a file named "1kg_afr_muts_all.bed", which is a BED file documenting all occurring SNVs in the 1000 Genomes Project in the African population. 
Since the pipeline used for generating this file is not yet released, you can substitute this with any other reference BED file as long as it 1) follows BED format, and 2) has the fourth column as a description of the SNV mutation type separated by an underline (e.g. "A>T_0.00075643").

We downloaded motifs from compbio.mit.edu/encode-motifs/, where the main file was referred to as "motifs_tf.bed"; we also performed some region-based filtering where we excluded certain genomic regions. We also downloaded motif logo files from the same source and placed them in the **logos/** directory.

As you can see, the PBS job files are named after the order they should be run. 

* Step 1: Pre-processing 1000 Genomes mutation files (NOTE: this is dependent on an internally developed package which we plan to release soon, skip if unnecessary). 
* Step 2: Preprocess the downloaded BED file into individual BED files by motif.
* Step 3: Retrieve the fasta sequence and DNA shapes using the generated BED files.
* Step 4: Retrieve the positions on each TFBS at which mutations occur.
* Step 5: Trim the above results to specific genomic regions, note that we also count how many times a TF occurs in a genomic region, which is used to exclude certain TFBSs from our VISTA enhancers analysis.
* Step 6: Main data analysis file.
* Step 7: Analyzes mutation rate fold changes within each TFBS.

For the rest of the files in this directory: 
* binding_count.txt: This file shows how many times each TFBS occurred in the entire genome.
* mutrate_ratio_tf.txt: This is the output of Step 7.
* script_\*: **IMPORTANT!** These are the scripts that process our data, whereas the numbered PBS scripts are more like job submission commands. Make sure to not remove them or change their names.
* shape_acc.xlsx: This is the final output of Step 6 and could be considered one of the main results files of this experiment.
* uniq_*: Lists of motif names. Note that three of the files refer to a filtered list where all included TFBSs satisfy certain conditions; the file with "vista_gt200_mt50" shows that these TFBSs each occurred at least 200 times with at least 50 mutations in the VISTA enhancer regions (and was used in our paper).
* README.md: You are reading this.
