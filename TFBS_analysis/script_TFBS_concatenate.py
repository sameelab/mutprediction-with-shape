# python3

# Libraries
import os
import sys
import re
import numpy as np
import pandas as pd

# Import files
filedir_in, filename_out = sys.argv[1], sys.argv[2]

# Setup, import files
List_allTFcors, TF_names = [], []
for File in os.listdir(filedir_in):
    List_allTFcors.append( pd.read_csv(filedir_in+File, index_col=0) )
    TF_names.append(File.strip(".csv"))

Arr_conserve_mutrate = np.zeros(shape=(len(List_allTFcors), ), dtype=float)
DF_KS_conserve, DF_KS_mutrate = [], []

# Reorganize inputs
for i in range(len(List_allTFcors)):
    Arr_conserve_mutrate[i] = List_allTFcors[i]["conserve_mutrate"][0]
    DF_KS_conserve.append(List_allTFcors[i]["KS_conserve"].transpose().values)
    DF_KS_mutrate.append(List_allTFcors[i]["KS_mutrate"].transpose().values)

# Transform into dataframes
DF_KS_conserve = pd.DataFrame(DF_KS_conserve, columns=List_allTFcors[0].index, index=TF_names)
DF_KS_mutrate = pd.DataFrame(DF_KS_mutrate, columns=List_allTFcors[0].index, index=TF_names)
DF_conserve_mutrate = pd.DataFrame(Arr_conserve_mutrate, columns=["conserve_mutrate"], index=TF_names)

# Write output
with pd.ExcelWriter(filename_out) as writer:
    DF_KS_mutrate.to_excel(writer, sheet_name="KS_mutrate")
    DF_KS_conserve.to_excel(writer, sheet_name="KS_conserve")
    DF_conserve_mutrate.to_excel(writer, sheet_name="conserve_mutrate")
