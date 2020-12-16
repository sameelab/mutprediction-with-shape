# python 3
# functions.py
# Zian Liu
# Last update: 12/16/2020

"""
This is a script accompanying main.py for building machine learning models.
Note that this script only contains functions and should not be run independently.
"""


def load_file_preprocessed(filename, mode=['sequence', 'shape'], structural_ref_df=None):
    """This function imports the file downloaded from Dr. Voight's github."""
    import numpy as np
    import pandas as pd

    df_import = pd.read_csv(filename, sep='\t', header=2)
    # Make the predictor, note that this is different for shape/nucleotide:
    if mode == 'sequence':
        temp = [list(item)[0:3]+list(item)[4:7] for item in list(df_import['trans'])]
        predictor = pd.DataFrame(temp)
        predictor.columns = ['3L', '2L', '1L', '1R', '2R', '3R']
    elif mode == 'shape':
        temp1 = [item.split(',')[0] for item in list(df_import['trans'])]
        temp2 = [item.split(',')[1] for item in list(df_import['trans'])]
        df_val_major = structural_ref_df.loc[temp1, :]
        df_val_alter = structural_ref_df.loc[temp2, :]
        predictor = pd.concat([df_val_major.reset_index(drop=True), df_val_alter.reset_index(drop=True)], axis=1, ignore_index=True)
        colnames = list(df_val_major.columns) + [str(item) + '_r' for item in list(df_val_alter.columns)]
        predictor.columns = colnames
    # Make the effector
    effector_train, effector_test = np.array(df_import['train_rate']), np.array(df_import['test_rate'])
    # Make the mutation class (A>C, A>G, CpG C>A, ...) index
    index_class = np.zeros(shape=(len(df_import), ), dtype=int)
    g_label = np.array([item[4] for item in list(df_import['trans'])])
    for select in range(1, 6):
        sclass = ['A,G', 'A,T', 'C,A', 'C,G', 'C,T'][select-1]
        index_class[df_import['class'] == sclass] = select
    for select in range(6, 9):
        sclass = ['C,A', 'C,G', 'C,T'][select-6]
        index_class[np.all([df_import['class'] == sclass, g_label == 'G'], axis=0)] = select
    return predictor, effector_train, effector_test, index_class


def make_2dshape_neighbor(shape_df, labels):
    """A custom polynomial transform function that transforms 1D shape features into 2D (neighboring
    interactions only)"""
    import numpy as np
    import pandas as pd
    from copy import deepcopy

    temp_posit = [item.split('_')[1] for item in labels]
    temp_df = np.array(deepcopy(shape_df))
    temp_labels = deepcopy(labels)
    for i in range(np.shape(shape_df)[1]):
        for j in range(i, np.shape(shape_df)[1]):
            posit1 = temp_posit[i]
            posit2 = temp_posit[j]
            if posit1 == 'L':
                if posit2 in ['L', 'CL', 'C']:
                    append_arr = np.array(shape_df.iloc[:,i] * shape_df.iloc[:,j]).reshape(-1, 1)
                    temp_df = np.concatenate((temp_df, append_arr), axis=1)
                    temp_labels = np.append(temp_labels, temp_labels[i]+'*'+temp_labels[j])
            elif posit1 == 'CL':
                if posit2 in ['CL', 'C', 'CR']:
                    append_arr = np.array(shape_df.iloc[:,i] * shape_df.iloc[:,j]).reshape(-1, 1)
                    temp_df = np.concatenate((temp_df, append_arr), axis=1)
                    temp_labels = np.append(temp_labels, temp_labels[i]+'*'+temp_labels[j])
            elif posit1 == 'C':
                if posit2 in ['C', 'CR', 'R']:
                    append_arr = np.array(shape_df.iloc[:,i] * shape_df.iloc[:,j]).reshape(-1, 1)
                    temp_df = np.concatenate((temp_df, append_arr), axis=1)
                    temp_labels = np.append(temp_labels, temp_labels[i]+'*'+temp_labels[j])
            elif posit1 == 'CR':
                if posit2 in ['CR', 'R']:
                    append_arr = np.array(shape_df.iloc[:,i] * shape_df.iloc[:,j]).reshape(-1, 1)
                    temp_df = np.concatenate((temp_df, append_arr), axis=1)
                    temp_labels = np.append(temp_labels, temp_labels[i]+'*'+temp_labels[j])
            elif posit1 == 'R':
                if posit2 in ['R']:
                    append_arr = np.array(shape_df.iloc[:,i] * shape_df.iloc[:,j]).reshape(-1, 1)
                    temp_df = np.concatenate((temp_df, append_arr), axis=1)
                    temp_labels = np.append(temp_labels, temp_labels[i]+'*'+temp_labels[j])
    df_strucval_neighbor = pd.DataFrame(temp_df)
    df_strucval_neighbor.columns = temp_labels
    return df_strucval_neighbor


def make_4dshape(predictor_1d, predictor_raw, degree=4, neighbors_only=False):
    """This function makes a 2~4D polynoimal transformed version of the nucleotide predictor."""
    import numpy as np
    from copy import deepcopy

    pred_wide = np.shape(predictor_1d)[1]
    predictor_out = deepcopy(predictor_1d)
    Labels_ref = np.array(['C', 'G', 'T', 'C', 'G', 'T', 'C', 'G', 'T', 'C', 'G', 'T', 'C', 'G', 'T', 'C', 'G', 'T'], dtype=object)
    labels_out = np.array([])
    Labels_raw = np.array(predictor_raw)
    # Add first degree, labels only
    for i in range(pred_wide):
        posit = int(i//3)
        new_label = ['_'] * 6
        new_label[posit] = Labels_ref[i]
        labels_out = np.append(labels_out, ''.join(new_label))
    # Add second degree label
    if degree >= 2:
        for i in range(pred_wide):
            posit1 = int(i//3)
            for j in range(int((posit1+1)*3), pred_wide):
                posit2 = int(j//3)
                if neighbors_only:
                    if (posit2 - posit1) > 1:
                        continue
                new_arr = predictor_1d[:, i] * predictor_1d[:, j]
                new_arr.shape = (len(new_arr), 1)
                new_label = ['_'] * 6
                new_label[posit1] = Labels_ref[i]
                new_label[posit2] = Labels_ref[j]
                predictor_out = np.concatenate((predictor_out, new_arr), axis=1)
                labels_out = np.append(labels_out, ''.join(new_label))
    # Add third degree label
    if degree >= 3:
        for i in range(pred_wide):
            posit1 = int(i//3)
            for j in range(int((posit1+1)*3), pred_wide):
                posit2 = int(j//3)
                if neighbors_only:
                    if (posit2 - posit1) > 1:
                        continue
                for k in range(int((posit2+1)*3), pred_wide):
                    posit3 = int(k//3)
                    if neighbors_only:
                        if (posit3 - posit2) > 1:
                            continue
                    new_arr = predictor_1d[:, i] * predictor_1d[:, j] * predictor_1d[:, k]
                    new_arr.shape = (len(new_arr), 1)
                    new_label = ['_'] * 6
                    new_label[posit1] = Labels_ref[i]
                    new_label[posit2] = Labels_ref[j]
                    new_label[posit3] = Labels_ref[k]
                    predictor_out = np.concatenate((predictor_out, new_arr), axis=1)
                    labels_out = np.append(labels_out, ''.join(new_label))
    # Add fourth degree label
    if degree >= 4:
        for i in range(pred_wide):
            posit1 = int(i//3)
            for j in range(int((posit1+1)*3), pred_wide):
                posit2 = int(j//3)
                if neighbors_only:
                    if (posit2 - posit1) > 1:
                        continue
                for k in range(int((posit2+1)*3), pred_wide):
                    posit3 = int(k//3)
                    if neighbors_only:
                        if (posit3 - posit2) > 1:
                            continue
                    for l in range(int((posit3+1)*3), pred_wide):
                        posit4 = int(l//3)
                        if neighbors_only:
                            if (posit4 - posit3) > 1:
                                continue
                        new_arr = predictor_1d[:, i] * predictor_1d[:, j] * predictor_1d[:, k] * predictor_1d[:, l]
                        new_arr.shape = (len(new_arr), 1)
                        new_label = ['_'] * 6
                        new_label[posit1] = Labels_ref[i]
                        new_label[posit2] = Labels_ref[j]
                        new_label[posit3] = Labels_ref[k]
                        new_label[posit4] = Labels_ref[l]
                        predictor_out = np.concatenate((predictor_out, new_arr), axis=1)
                        labels_out = np.append(labels_out, ''.join(new_label))
    return predictor_out, labels_out


def standard_cv(model_used, kfold_used, X, y, subclass_index=None, log=False, use_index=False, record_index=False):
    """A custom CV function that records k-fold CV R^2 and MSE for train/test. Can change cv number, record by-subclass,
    and accommodate log-transformed or class-specific models."""
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score

    cv = kfold_used.n_splits
    output_dict = dict()
    output_dict['train_r2'], output_dict['test_r2'] = np.zeros(shape=(cv, )), np.zeros(shape=(cv, ))
    output_dict['train_mse'], output_dict['test_mse'] = np.ones(shape=(cv, )), np.ones(shape=(cv, ))
    if record_index:
        output_dict['train_r2_subclass'], output_dict['test_r2_subclass'] = dict(), dict()
        output_dict['train_mse_subclass'], output_dict['test_mse_subclass'] = dict(), dict()
        for _iter_c in set(subclass_index):
            output_dict['train_r2_subclass'][_iter_c] = np.zeros(shape=(cv, ))
            output_dict['test_r2_subclass'][_iter_c] = np.zeros(shape=(cv, ))
            output_dict['train_mse_subclass'][_iter_c] = np.ones(shape=(cv, ))
            output_dict['test_mse_subclass'][_iter_c] = np.ones(shape=(cv, ))
    _iter = 0
    for train_idx, test_idx in kfold_used.split(X, y):
        # Make train/test splits
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        # Also do this for the subclasses, will be useful for calculating R^2 by class
        if use_index or record_index:
            subclass_train = subclass_index[train_idx]
            subclass_test = subclass_index[test_idx]
        if use_index:
            model_used.fit(X_train, y_train, subclass_index[train_idx])
            temp_tr = model_used.predict(X_train, subclass_index[train_idx])
            temp_tt = model_used.predict(X_test, subclass_index[test_idx])
        else:
            model_used.fit(X_train, y_train)
            temp_tr = model_used.predict(X_train)
            temp_tt = model_used.predict(X_test)
        if log:   # Transform things back with exp
            temp_tr, temp_tt = np.exp(temp_tr), np.exp(temp_tt)
            y_train, y_test = np.exp(y_train), np.exp(y_test)
        # Add predictions
        output_dict['train_r2'][_iter], output_dict['test_r2'][_iter] = r2_score(y_train, temp_tr), r2_score(y_test, temp_tt)
        output_dict['train_mse'][_iter], output_dict['test_mse'][_iter] = mean_squared_error(y_train, temp_tr), mean_squared_error(y_test, temp_tt)
        if record_index:
            for _iter_c in set(subclass_index):
                output_dict['train_r2_subclass'][_iter_c][_iter] = r2_score(y_train[subclass_train==_iter_c], temp_tr[subclass_train==_iter_c])
                output_dict['test_r2_subclass'][_iter_c][_iter] = r2_score(y_test[subclass_test==_iter_c], temp_tt[subclass_test==_iter_c])
                output_dict['train_mse_subclass'][_iter_c][_iter] = mean_squared_error(y_train[subclass_train==_iter_c], temp_tr[subclass_train==_iter_c])
                output_dict['test_mse_subclass'][_iter_c][_iter] = mean_squared_error(y_test[subclass_test==_iter_c], temp_tt[subclass_test==_iter_c])
        _iter += 1
    return output_dict


def L1select_byalpha_byclass(X, y, alpha_val_list, index_class, kfold_obj, n_jobs, select, verbose=1, savefile=False,
                             save_name_model=None, save_name_results=None):
    """Runs L1(Lasso) selection on different alpha values, save each subclass separately as a
    joblib file."""
    import numpy as np
    from joblib import dump
    from sklearn.linear_model import LinearRegression, Lasso

    dict_out_model, dict_out_results = dict(), dict()
    alpha_val_select = alpha_val_list
    for _iter in range(len(alpha_val_select)):
        alpha_val = alpha_val_select[_iter]
        temp_x = X[index_class == select]
        temp_y = y[index_class == select]
        model_L1 = Lasso(alpha=alpha_val, precompute=True, warm_start=True, selection='random', random_state=42,
                         max_iter=int(1e08)).fit(temp_x, temp_y)
        if np.sum(model_L1.coef_) == 0:
            continue
        elif np.sum(model_L1.coef_) == np.shape(X)[1]:
            break
        else:
            temp_coefs = (model_L1.coef_ != 0)
        dict_out_model[alpha_val] = model_L1
        model = LinearRegression(n_jobs=n_jobs)
        temp_x_crop = temp_x[:, temp_coefs]
        dict_out_results[alpha_val] = standard_cv(model, kfold_obj, temp_x_crop, temp_y, index_class, record_index=False)
        if verbose >= 1:
            print("Alpha value "+str(alpha_val)+" done.")
        if _iter > 1:
            if np.mean(dict_out_results[alpha_val]['test_mse']) > np.mean(dict_out_results[alpha_val_select[_iter-1]]['test_mse']):
                if np.mean(dict_out_results[alpha_val]['test_mse']) > np.mean(dict_out_results[alpha_val_select[_iter-2]]['test_mse']):
                    if verbose >= 1:
                        print("Detected rise in test MSE, break loop for class "+str(select)+" at alpha = "+str(alpha_val))
                    break
    if verbose >= 1:
        print("Class "+str(select)+" done.")
    if savefile:
        dump(dict_out_model, save_name_model)
        dump(dict_out_results, save_name_results)
    return dict_out_model, dict_out_results


class linreg_bysubclassall_alin2_rmcor_ind():
    """This is the class for our regression model. It makes a class of binned regression items with custom model indices
    and linear/log-linear selection."""
    def __init__(self, model_indices=None, lin_or_log=None, n_jobs=1):
        self.paral = int(n_jobs)
        self.indices = model_indices
        self.model_type = lin_or_log

    def fit(self, X=None, y=None, Subclass_list=None):
        import numpy as np
        from sklearn.linear_model import LinearRegression
        subclass = np.array(Subclass_list)
        self.model = dict()
        for _iter in range(9):
            if self.model_type[_iter] == 0:
                y_i = np.array(y)[subclass == _iter]
            elif self.model_type[_iter] == 1:
                y_i = np.log(y)[subclass == _iter]
            X_i = X[subclass == _iter][:, self.indices[_iter]]
            self.model[_iter] = LinearRegression(n_jobs=self.paral).fit(X_i, y_i)
        return self

    def predict(self, X=None, Subclass_list=None):
        import numpy as np
        from sklearn.linear_model import LinearRegression
        subclass = np.array(Subclass_list)
        if len(X) != len(subclass):
            raise ValueError("Predictor matrix and substitution class identity list are not of same length!")
        tmpout = dict()
        iter_count = dict()
        for _iter in range(9):
            X_i = X[subclass == _iter][:, self.indices[_iter]]
            if self.model_type[_iter] == 0:
                tmpout[_iter] = self.model[_iter].predict(X_i)
            elif self.model_type[_iter] == 1:
                tmpout[_iter] = np.exp(self.model[_iter].predict(X_i))
            iter_count[_iter] = 0
        # Make output
        out = np.zeros(shape=(len(X), ), dtype=float)
        for iter_i in range(len(subclass)):
            model_ind = subclass[iter_i]
            out[iter_i] = tmpout[model_ind][iter_count[model_ind]]
            iter_count[model_ind] += 1
        return out


def r2_adjust(y_true, y_pred, p):
    from sklearn.metrics import r2_score

    """Adjusted R^2. Need to define the number of predictors p. """
    _R2 = r2_score(y_true, y_pred)
    n = len(y_true)
    _R2adj = 1 - ((1-_R2) * (n - 1) / ( n - p - 1))
    return _R2adj


def plot_CV_subclass_r2(cv_object, filename=None):
    """Makes a simple visualization of the R^2 and MSE of different subclasses and the overall R^2,
    use in conjunction with standard_cv function."""
    import matplotlib.pyplot as plt

    temp_trainr, temp_testr = [cv_object['train_r2']], [cv_object['test_r2']]
    for _iter in range(9):
        temp1, temp2 = cv_object['train_r2_subclass'][_iter], cv_object['test_r2_subclass'][_iter]
        temp1[temp1 < 0] = 0
        temp2[temp2 < 0] = 0
        temp_trainr.append(temp1)
        temp_testr.append(temp2)
    temp_fig = plt.figure(figsize=[15, 5])
    plt.subplot(121)
    plt.boxplot(temp_trainr)
    plt.xticks(list(range(1, 11)), ['overall', 'A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T', 'CpG_C>A', 'CpG_C>G', 'CpG_C>T'])
    plt.ylabel("train")
    plt.ylim((-0.05, 1.05))
    plt.subplot(122)
    plt.boxplot(temp_testr)
    plt.xticks(list(range(1, 11)), ['overall', 'A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T', 'CpG_C>A', 'CpG_C>G', 'CpG_C>T'])
    plt.ylabel('test')
    plt.ylim((-0.05, 1.05))
    plt.show()
    if filename:
        temp_fig.savefig(filename, dpi=300)


def standard_traintestresult_df(pred_output, effector_train, effector_test, index_class, p, display='df'):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score

    """Makes a df containing R2 and R2-adjust given an input model prediction and number of features."""
    train_r2, test_r2, train_r2adj, test_r2adj = np.zeros(shape=(10,)), np.zeros(shape=(10,)), np.zeros(shape=(10,)), np.zeros(shape=(10,))
    train_r2[9], test_r2[9] = r2_score(effector_train, pred_output), r2_score(effector_test, pred_output)
    if type(p) == int:
        train_r2adj[9], test_r2adj[9] = r2_adjust(effector_train, pred_output, p), r2_adjust(effector_test, pred_output, p)
    else:
        train_r2adj[9], test_r2adj[9] = r2_adjust(effector_train, pred_output, sum(p)), r2_adjust(effector_test, pred_output, sum(p))
    for select in range(9):
        train_r2[select] = r2_score(effector_train[index_class==select], pred_output[index_class==select])
        test_r2[select] = r2_score(effector_test[index_class==select], pred_output[index_class==select])
        if type(p) == int:
            train_r2adj[select] = r2_adjust(effector_train[index_class==select], pred_output[index_class==select], p)
            test_r2adj[select] = r2_adjust(effector_test[index_class==select], pred_output[index_class==select], p)
        else:
            train_r2adj[select] = r2_adjust(effector_train[index_class==select], pred_output[index_class==select], p[select])
            test_r2adj[select] = r2_adjust(effector_test[index_class==select], pred_output[index_class==select], p[select])
    output_df = pd.DataFrame({'train_R2': train_r2, 'train_R2_adj': train_r2adj,
                             'test_R2': test_r2, 'test_R2_adj': test_r2adj})
    output_df.index = ['A>C', 'A>G', 'A>T', 'C>A', 'C>G', 'C>T', 'CpG_C>A', 'CpG_C>G', 'CpG_C>T', 'overall']
    return output_df


def standard_l1modelfitting(predictor, effector, effector_test, index_class, l1_index_in, kfold_obj, n_jobs=4, plot_out=None):
    import numpy as np
    from sklearn.metrics import r2_score

    """Fit a model using a given predictor, selection index, and show the CV results as well as the independent testing
    result. """
    lin_or_log = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    model_out = linreg_bysubclassall_alin2_rmcor_ind(model_indices=l1_index_in, lin_or_log=lin_or_log, n_jobs=n_jobs)
    model_out.fit(predictor, effector, index_class)
    tmp_out = model_out.predict(predictor, index_class)
    print(r2_score(effector, tmp_out))   # This is the R^2 value
    print(r2_score(effector[index_class!=8], tmp_out[index_class!=8]))   # This is the R^2 value for non CpG C>T
    # Cross validate
    model = linreg_bysubclassall_alin2_rmcor_ind(model_indices=l1_index_in, lin_or_log=lin_or_log, n_jobs=n_jobs)
    cv_out = standard_cv(model, kfold_obj, predictor, effector, index_class, use_index=True, record_index=True)
    plot_CV_subclass_r2(cv_out, filename=plot_out)
    # Results file
    p_length = [np.sum(l1_index_in[i]) for i in range(9)]
    results_df_out = standard_traintestresult_df(
        tmp_out, effector, effector_test, index_class, p=p_length, display='df')
    return model_out, cv_out, results_df_out


def plot_scatter_residual(predicted_Y, actual_Y, alpha_value=0.5, use_index=False, index=None, index_name=None, filename=False):
    """Plot scatterplot of actual vs predicted results and residual plot, both in normal and log axes."""
    import matplotlib.pyplot as plt

    temp_fig = plt.figure(figsize=[15, 15])
    for _iter in range(1, 5):
        plt.subplot(int(220+_iter))
        plotting_x = actual_Y
        if _iter in [2, 4]:
            plotting_y = actual_Y - predicted_Y
        elif _iter in [1, 3]:
            plotting_y = predicted_Y
        if use_index:
            for _class in set(index):
                plt.scatter(plotting_x[index==_class], plotting_y[index==_class], alpha=alpha_value, s=1)
            plt.legend(index_name)
        else:
            plt.scatter(plotting_x, plotting_y, alpha=alpha_value, s=1)   # X is predicted value, Y is actual
        plt.xlabel("Actual frequencies")
        if _iter in [3, 4]:
            plt.xscale('log')
            #plt.xlim(min(plotting_x)*1.05, max(plotting_x)*0.95)
        else:
            plt.xlim(-0.01, max(plotting_x)+0.01)
        if _iter == 1:
            plt.ylim(-0.01, max(plotting_x)+0.01)
        elif _iter == 2:
            plt.ylim(min(plotting_y)*1.05, max(plotting_y)*1.05)
        elif _iter == 3:
            plt.yscale('log')
            #plt.ylim(min(plotting_x)*1.05, max(plotting_x)*0.95)
        elif _iter == 4:
            plt.ylim(min(plotting_y)*1.05, max(plotting_y)*1.05)
    plt.show()
    if filename:
        temp_fig.savefig(filename, dpi=300)
