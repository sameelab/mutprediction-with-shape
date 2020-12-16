# python 3
# predictor_generation.py   Builds the machine learning model
# Zian Liu
# Last update: 12/16/2020


"""
Call this script by:
python main.py ./input/example_mutrates.txt ./input/ref_7mers_structure_cpg.xlsx

In this script, we will give a demonstration of running our model and predicting mutation rates.
We will use a relatively simple architecture of 1st order shape + 2nd order nucleotide features.
First, import the functions:
"""

# Import from libraries
import numpy as np
import pandas as pd
import sys
from joblib import dump, load, Parallel, delayed
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
# Import from functions.py
from functions import *


"""
Next, prepare the files for input. The input files are:
1. Mutation rate file as annotated by Aggarwala and Voight. Please see the original file for formatting.
2. DNAshapeR reference file. See our Github repo for how to extract the data from the package.
Note that these should be location 1 and location 2 of the command line input.
"""

ratefile_in = sys.argv[1]   # This is the mutation rate file
shaperef = sys.argv[2]   # This is the DNAshapeR reference for 7-mers

"""
If you prefer running these in an interpreter instead of calling from the command line,
uncomment the following and use these to import the files instead.
"""
# ratefile_in = "../data_input/example_mutrates.txt"
# shaperef = "../data_input/ref_7mers_structure_cpg.xlsx"   # This is the location of the file


"""
In the next section, we will load the rate files using our custom functions.
First, we need to load the DNA shape reference table.
Then, we will run the custom import function to import shape and nucleotide features and transform them, 
as well as loading effectors, and mutation class indices. 
We will also set up a k-fold object; this will be useful for cross validation tests.
"""

# Load the DNA shape reference table:
Shapes_name = ['HelT', 'Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Buckle', 'Opening', 'ProT', 'Shear',
               'Stagger', 'Stretch', 'MGW', 'EP']
with pd.ExcelFile(shaperef) as F_strucref:
    DF_strucval = pd.DataFrame()
    for item in Shapes_name:
        temp_df = F_strucref.parse(sheet_name=item, index_col=0)
        if np.shape(temp_df)[1] == 3:
            temp_df.rename({'V1': item+'_L', 'V2': item+'_C', 'V3': item+'_R'}, axis='columns', inplace=True)
        else:
            temp_df.rename({'V1': item+'_L', 'V2': item+'_CL', 'V3': item+'_CR', 'V4': item+'_R'}, axis='columns', inplace=True)
        DF_strucval = pd.concat([DF_strucval, temp_df], axis=1)

# Run the function to obtain shape predictors, effector, and mutation class index
Pred_shape_raw, Eff, Eff_test, Index_class = load_file_preprocessed(
    ratefile_in, mode='shape', structural_ref_df=DF_strucval)
Index_class_name = ['A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T', 'CpG_C>A', 'CpG_C>G', 'CpG_C>T']
# Make the min-max normalized shapes
Pred_1d = MinMaxScaler(feature_range=(0, 1)).fit_transform(Pred_shape_raw)

# Also import nucleotide predictors
Pred_sc_raw, _, _, Index_class_sc = load_file_preprocessed(
    ratefile_in, mode='sequence', structural_ref_df=DF_strucval)
# Transforms the raw predictor into one-hot encoding
Pred_sc_1d = OneHotEncoder(
    categories=[['C','G','T'], ['C','G','T'],['C','G','T'],['C','G','T'],['C','G','T'],['C','G','T']],
    sparse=True, handle_unknown='ignore').fit_transform(Pred_sc_raw).toarray()

# Set up a k-fold object, this will be useful for later cross validations
Kfoldobj = KFold(n_splits=8, shuffle=True, random_state=42)


"""
This generates the basic 1st degree predictors. 
Next, use them to conduct polynomial transforms on predictors. 
For this example, we will perform 2nd order transforms on nucleotide features.
We will then bin the 1st order shape and 2nd order nucleotide features for our final predictor.
Due to the amount of computation required, the code for 2nd order transformation of shape features is currently
commented out by the following function; remove the extra function to run it.
"""

run_shape2 = False
# run_shape2 = True
if run_shape2:
    # Make 2nd order shape features, neighboring interactions only
    Pred_2dneibr = make_2dshape_neighbor(Pred_shape_raw, Pred_shape_raw.columns)
    # Minmax scaling
    Labels_2dneibr = Pred_2dneibr.columns
    Pred_2dneibr = MinMaxScaler(feature_range=(0, 1)).fit_transform(Pred_2dneibr)
    # Reduce variance, note that it is advised to retain the variance reduction object for future use
    Var_red_neibr = VarianceThreshold(threshold=0.01).fit(Pred_2dneibr)
    Pred_2dneibr = Var_red_neibr.transform(Pred_2dneibr)
    Labels_2dneibr = Var_red_neibr.transform(np.array(Labels_2dneibr).reshape(1, len(Labels_2dneibr)))[0]

# Make 2d nucleotide features
Pred_sc_2d, Labels_sc_2d = make_4dshape(Pred_sc_1d, Pred_sc_raw, degree=2)

# Concatenate predictors
Pred_binned = np.concatenate((Pred_1d, Pred_sc_2d), axis=1)


"""
Next, we want to conduct L1-based feature selection after defining a list of alpha values for selection.
"""

# Make a list of alpha values in log-scale descending order
alpha_val_list = list(np.arange(9, 0, -1)*1e-06) + list(np.arange(9, 0, -1)*1e-07) + \
    list(np.arange(9, 0, -1)*1e-08) + list(np.arange(9, 0, -1)*1e-09)

# Run L1 selection on each mutation class, note we are doing it in parallel to speed things up
_ = Parallel(n_jobs=5)(delayed(L1select_byalpha_byclass)(Pred_binned, Eff, alpha_val_list, Index_class, Kfoldobj,
    n_jobs=1, select=select, verbose=1, savefile=True,
    save_name_model="intermediate/Models_L1_sh1_sc2_subclass_"+str(select)+".joblib",
    save_name_results="intermediate/Results_L1_sh1_sc2_subclass_"+str(select)+".joblib")
    for select in range(9)
)


"""
The previous script should generate several model and result joblib files in the "intermediate" directory. 
Reload these results to get the optimal alpha value for each mutation class, 
defined by minimizing validation MSE:
"""

alpha_list, Indices_rmL1 = [], []
results_dict_in = [load("intermediate/Results_L1_sh1_sc2_subclass_"+str(select)+".joblib") for select in range(9)]
model_dict_in = [load("intermediate/Models_L1_sh1_sc2_subclass_"+str(select)+".joblib") for select in range(9)]
for select in range(9):
    # Find the location of the alpha value that minimizes validation ('testing') MSE
    min_loc = np.argmin([np.mean(results_dict_in[select][key]['test_mse']) for key in results_dict_in[select].keys()])
    # Add the particular alpha value to alpha list
    alpha_list.append( list(results_dict_in[select].keys())[min_loc] )
    model_select = model_dict_in[select][alpha_list[select]]
    Indices_rmL1.append(model_select.coef_ != 0)
    print("Class " + str(select) + ", Number of predictors: " + str(np.sum(Indices_rmL1[select])))


"""
Build the model using the optimal number of features as defined above, show predictions, 
and show results by showing 8-fold cross validation R^2 in training/validation, as well as 
R^2 and adjusted R^2 values in training and independing testing data.
"""

Model_L1, CV_L1, Result_L1 = standard_l1modelfitting(
    Pred_binned, Eff, Eff_test, Index_class, Indices_rmL1, Kfoldobj, n_jobs=4,
    plot_out="R2_comparison.jpg")
print(Result_L1)


"""
Finally, make a scatterplot. 
The left column show the scatterplot of predicted vs measured mutation rates in Euclidean and log scale, 
the right column show residual plots in Euclidean and log scale.
"""

plot_scatter_residual(
    Model_L1.predict(Pred_binned, Index_class), Eff_test, alpha_value=0.5, use_index=True,
    index=Index_class, index_name=Index_class_name, filename="scatterplot.jpg")


"""
This is the end of the demo.
"""
