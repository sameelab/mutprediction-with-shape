# python3

# Libraries
import os
import sys
import re
import numpy as np
import pandas as pd
from collections import Counter
from Bio import SeqIO, motifs
from Bio.Seq import Seq
from scipy.stats import pearsonr, spearmanr, kstest, entropy


# Import filenames list
file_shape, file_muts, file_logo, filename_out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


# Sequence logo and conservation score
TF_logo = pd.read_csv(file_logo, sep=" ", header=None, skiprows=[0])
TF_logo.pop(0)
#TF_conserve = (2 * np.max(TF_logo, axis=1) - np.sum(TF_logo, axis=1)).values
TF_conserve = entropy(TF_logo, qk=np.full(np.shape(TF_logo), fill_value=0.25), axis=1)

# Define TF length
len_tf = len(TF_conserve)

# TFBS shape distribution
DF_pos_shape = pd.read_csv(file_shape)
# TFBS mutation ref and alt distribution
DF_pos_muts = pd.read_csv(file_muts, sep="\t", index_col=None, header=None)
DF_pos_muts.columns = ["chr", "start", "end", "mut", "MAF", "pos", "kmer_xtend", "kmer"]
# 5-mer reference DF
DF_strucval_5mersheet = pd.read_csv("ref_5mers_structure.csv", index_col=0)

temp_altks = [0] * len(DF_pos_muts)
temp_alt7 = [0] * len(DF_pos_muts)
for i in range(len(temp_altks)):
    temp_kmer, temp_7mer = DF_pos_muts['kmer'][i].upper(), DF_pos_muts['kmer_xtend'][i].upper()
    temp_alt = DF_pos_muts['mut'][i].split(">")[1]
    temp_altks[i] = temp_kmer[0:2] + temp_alt + temp_kmer[3:5]
    temp_alt7[i] = temp_7mer[0:3] + temp_alt + temp_7mer[4:7]
DF_pos_muts['kmer_alt'] = temp_altks
DF_pos_muts['kmer_alt_xtend'] = temp_alt7


DF_pos_muts.index = [item.upper() for item in DF_pos_muts['kmer'].values]
DF_pos_muts_ref = DF_pos_muts.join(DF_strucval_5mersheet, how="left")
DF_pos_muts_ref.sort_values(by=["pos", "kmer"], inplace=True)
DF_pos_muts.index = DF_pos_muts['kmer_alt']
DF_pos_muts_alt = DF_pos_muts.join(DF_strucval_5mersheet, how="left")
DF_pos_muts_alt.sort_values(by=["pos", "kmer"], inplace=True)


temp_counter = Counter(DF_pos_muts_ref['pos'])
for i in range(len_tf):
    if i not in temp_counter.keys():
        temp_counter[i] = 0
DF_observed_mut = pd.DataFrame([temp_counter]).transpose()
DF_observed_mut.sort_index(inplace=True)
DF_observed_mut = DF_observed_mut / len(DF_pos_shape)

temp_arr = DF_observed_mut.values.flatten()
temp_stat = np.max(temp_arr) / np.min(temp_arr[np.nonzero(temp_arr)])
print(file_shape, temp_stat)

with open(filename_out, "a+") as f:
    f.write(file_shape + "\t" + str(temp_stat))
