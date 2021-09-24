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


shape_picks = np.arange(np.shape(DF_strucval_5mersheet)[1])
DF_KSstat = np.zeros(shape=(len(shape_picks), len_tf))
Shape_types = []
j = 0
for i in shape_picks:
    Shape_types.append(DF_pos_muts_ref.columns[10+i])
    for pos_select in range(max(DF_pos_muts_ref['pos'])+1):
        # Temporary calculations
        temp_refval = DF_pos_muts_ref[DF_pos_muts_ref['pos'] == pos_select].iloc[:, 10+i].values
        temp_altval = DF_pos_muts_alt[DF_pos_muts_alt['pos'] == pos_select].iloc[:, 10+i].values
        # Skips iteration if there is no observed mutations in location
        if len(temp_refval) == 0:
            continue
        colname_shape = DF_pos_muts_ref.columns[10+i]
        colname_shape = colname_shape.split("_")[0] + "_" + str(int(colname_shape.split("_")[1]) + pos_select)
        # In the rare case that the column name isn't found:
        if colname_shape not in DF_pos_shape.columns:
            print("Current column is " + colname_shape + ", not found in shape DF; TF is " + file_logo)
            continue
        temp_bgval = DF_pos_shape[colname_shape]
        # Add to arrays
        DF_KSstat[j, pos_select] = kstest(temp_bgval, temp_altval)[0]
    j += 1

temp_counter = Counter(DF_pos_muts_ref['pos'])
for i in range(len_tf):
    if i not in temp_counter.keys():
        temp_counter[i] = 0
    print(temp_counter)
DF_observed_mut = pd.DataFrame([temp_counter]).transpose()
DF_observed_mut.sort_index(inplace=True)
DF_observed_mut = DF_observed_mut / len(DF_pos_shape)

DF_corr = np.zeros(shape=(len(shape_picks), 3))
for i in range(len(shape_picks)):
    DF_corr[i, 0] = spearmanr(DF_KSstat[i], TF_conserve)[0]
    DF_corr[i, 1] = spearmanr(DF_KSstat[i], DF_observed_mut.values.flatten())[0]
    DF_corr[i, 2] = spearmanr(TF_conserve, DF_observed_mut.values.flatten())[0]
DF_corr = pd.DataFrame(DF_corr, columns=["KS_conserve", "KS_mutrate", "conserve_mutrate"], 
                       index=Shape_types)

# Save
DF_corr.to_csv(filename_out)
