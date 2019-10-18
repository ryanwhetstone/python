from scipy.stats import pearsonr
from sklearn.model_selection import (StratifiedKFold, KFold)
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from pprint import pprint as pp
import seaborn as sns
import pandas as pd
import numpy as np
import os, shutil
import lib.functions as f
import config


def display_histogram(data, title, axes):
    axes.set_title(title)
    sns.distplot(data, bins=30, ax=axes, axlabel=False)

def show_missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (round(df.isnull().sum()*100 / df.isnull().count(), 2)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])
    print(missing_data.head(10))

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['Percent'])
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)

def print_df(df):
    print()
    print("Columns are:")
    print(df.columns)
    print(df.head(30))
    print(df.info())
    print(df.describe())


# Handle column and row transformations
def column_transformation(df, values, df_all, global_show_print=False):
    # col = values[0]
    # transformation = values[1]
    # if len(values) >=3:
    #     print("Before" + str(transformation))
    #     print(df[col])
    df = transformation_function(df, values, df_all, global_show_print)
    # if len(values) >=3:
    #     print(col + " after transformation: " + str(transformation))
    #     print(df[col].head(50))

    return df


def transformation_function(df, values, df_all, global_show_print):
    col = values[0]
    transformation = values[1]
    show_print = False
    if len(values) >=3 and global_show_print:
        show_print = True

    if transformation[0] == 'drop_column':
        if col in df.columns:
            df = df.drop(col, axis=1)
            
    if transformation[0] == 'copy_column':
        df_print = print_add_base_column(show_print, df, transformation[1])
        
        df[col] = df[transformation[1]]
        
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'add_columns':
        df_print = print_add_base_column(show_print, df, transformation[1])        
        
        df[col] = df[transformation[1]] + df[transformation[2]]
        
        df_print = print_add_column(show_print, df_print, df, transformation[2])
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'nan_static':
        df[col].fillna(transformation[1], inplace=True)

    if transformation[0] == 'nan_drop_row':
        df.dropna(subset=[col], inplace=True)

    if transformation[0] == 'nan_mean':
        df[col].fillna(df_all[col].mean(), inplace=True)

    if transformation[0] == 'nan_mean_from_columns':
        columns = transformation[1].copy()
        # grouped_mean_columns = pd.DataFrame()
        grouped_mean_columns = df_all.groupby(columns).mean()
        columns.append(col)
        grouped_mean_columns = grouped_mean_columns.reset_index()[columns]
        for index, row in df.iterrows():
            if(np.isnan(row[col])):
                mean = grouped_mean_columns.copy()
                for column in transformation[1]:
                    if len(mean[mean[column] == row[column]].index) > 0:
                        mean = mean[mean[column] == row[column]]
                df.at[index,col] = mean[col].values[0]

    if transformation[0] == 'nan_boolean':
        df[col] = df[col].notnull().astype('int')

    if transformation[0] == 'string_length':
        df_print = print_add_base_column(show_print, df, col)
        
        df[col] = df[col].apply(len)
        
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'continuous_string_length':
        df_print = print_add_base_column(show_print, df, transformation[1])
        
        df[col] = df[transformation[1]]
        df[col] = df[col].apply(len)
        
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'normalized_string_length':
        df_print = print_add_base_column(show_print, df, transformation[1])
        
        df[col] = df[transformation[1]]
        df[col] = df[col].apply(len)
        df[col] = column_normalize(df, col)
        
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'count_fequencies':
        df_print = print_add_base_column(show_print, df, col)
        
        df[col] = df_all.groupby(transformation[1])[transformation[1]].transform('count')
        
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'extract_dummy':
        df_print = print_add_base_column(show_print, df, transformation[1])
        
        df[col] = 0
        if(transformation[2] == '>'):
            df.loc[df[transformation[1]] > transformation[3], col] = 1
        if(transformation[2] == '>='):
            df.loc[df[transformation[1]] >= transformation[3], col] = 1
        if(transformation[2] == '<'):
            df.loc[df[transformation[1]] < transformation[3], col] = 1
        if(transformation[2] == '<='):
            df.loc[df[transformation[1]] <= transformation[3], col] = 1
        if(transformation[2] == '=' or transformation[2] == '=='):
            df.loc[df[transformation[1]] == transformation[3], col] = 1
        if(transformation[2] == '><'):
            df.loc[(df[transformation[1]] >= transformation[3]) & (df[transformation[1]] <= transformation[4]), col] = 1
        
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'duplicate_dummies':
        df_all['duplicates'] = df_all.duplicated([col], keep=False)
        for index, row in df.iterrows():
            if(df_all.iloc[index]['duplicates'] == False):
                df.at[index,col] = np.NaN
        df[col] = df[col].astype('category').cat.codes
        df = column_one_hot(show_print, df_print, transformation, df, col)        
    
    if transformation[0] == 'first_character':
        df_print = print_add_base_column(show_print, df, col)

        df[col] = df[col].astype(str).str[0]
        
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'continuous_first_character':
        df_print = print_add_base_column(show_print, df, transformation[1])
   
        df[col] = df[transformation[1]]
        df[col] = df[col].astype(str).str[0]
        df[col] = df[col].astype('category').cat.codes

        print_transformation(show_print, df, col, df_print, transformation)
    
    if transformation[0] == 'normalized_first_character':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = df[col].astype(str).str[0]
        df[col] = df[col].astype('category').cat.codes
        df[col] = column_normalize(df, col)

        print_transformation(show_print, df, col, df_print, transformation)
    
    if transformation[0] == 'split_by_qcut':
        df_print = print_add_base_column(show_print, df, col)

        df[col] = pd.qcut(df[col], transformation[1], duplicates='drop', labels=False)

        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'split_by_cut':
        df_print = print_add_base_column(show_print, df, col)

        df[col] = pd.cut(df[col], transformation[1], duplicates='drop', labels=False)

        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'split_by_defined':
        df_print = print_add_base_column(show_print, df, col)

        df[col] = pd.cut(df[col], transformation[1], duplicates='drop')
    
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'regex_extract':
        df_print = print_add_base_column(show_print, df, col)

        df[col] = df[col].str.extract(transformation[1], expand=False)
    
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'str_replace':
        df_print = print_add_base_column(show_print, df, col)

        df[col] = df[col].replace(transformation[1], transformation[2])
    
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'numeric_categories':
        df_print = print_add_base_column(show_print, df, col)

        df[col] = df[col].astype('category').cat.codes
    
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'continuous_numeric_categories':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = df[col].astype('category').cat.codes
    
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'normalized_numeric_categories':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = df[col].astype('category').cat.codes
        df[col] = column_normalize(df, col)
    
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'one_hot':
        df_print = print_add_base_column(show_print, df, col)

        df = column_one_hot(show_print, df_print, transformation, df, col)        
    
    if transformation[0] == 'normalize':
        df_print = print_add_base_column(show_print, df, col)

        df[col] = column_normalize(df, col)
    
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'normalized_column':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = column_normalize(df, col)
    
        print_transformation(show_print, df, col, df_print, transformation)

    if transformation[0] == 'binary_string_length':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = df[col].apply(len)
        df = column_one_hot(show_print, df_print, transformation, df, col)
    
    if transformation[0] == 'binary_one_hot':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df = column_one_hot(show_print, df_print, transformation, df, col)
    
    if transformation[0] == 'binary_numeric_categories':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = df[col].astype('category').cat.codes
        df = column_one_hot(show_print, df_print, transformation, df, col)
    
    if transformation[0] == 'binary_split_by_defined':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = pd.cut(df[col], transformation[2], duplicates='drop', labels=False)
        df = column_one_hot(show_print, df_print, transformation, df, col)
    
    if transformation[0] == 'binary_split_by_cut':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = pd.cut(df[col], transformation[2], duplicates='drop', labels=False)
        df = column_one_hot(show_print, df_print, transformation, df, col)
    
    if transformation[0] == 'binary_split_by_qcut':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = pd.qcut(df[col], transformation[2], duplicates='drop', labels=False)
        df = column_one_hot(show_print, df_print, transformation, df, col)
    
    if transformation[0] == 'binary_first_character':
        df_print = print_add_base_column(show_print, df, transformation[1])

        df[col] = df[transformation[1]]
        df[col] = df[col].astype(str).str[0]
        df[col] = df[col].astype('category').cat.codes
        df = column_one_hot(show_print, df_print, transformation, df, col)

    return df

def column_one_hot(show_print, df_print, transformation, df, column):
    if not config.values["debug_without_one_hot"]:
        df_orig = df.copy()
        df = pd.get_dummies(df, columns=[column], prefix=[column])
        new_cols = set( df.columns ) - set( df_orig.columns )
        if show_print:
            for c in new_cols:
                df_print = print_add_column(show_print, df_print, df, c)
            df_print = df_print.reindex(sorted(df_print.columns), axis=1)
            print_transformation(show_print, df, False, df_print, transformation)


    return df

def column_normalize(df, col):
    x = df[[col]].values.astype(float)
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[col] = pd.DataFrame(x_scaled)
    return df[col]

def print_transformation(show_print, df, col, df_print, transformation):
    if show_print:
        if col != False:
            df_print = pd.concat([df_print, pd.DataFrame(df[col])], axis=1)
        if col != False:
            print(col + " after transformation: " + str(transformation))
        print(df_print.head(50))

def print_add_base_column(show_print, df, column):
    if show_print: 
        df_print = df[column]
        return df_print
    else:
        return False

def print_add_column(show_print, df_print, df, column):
    if show_print:
        df_print = pd.concat([df_print, pd.DataFrame(df[column])], axis=1)
        return df_print
    else:
        return False
    
 

# split [column, operator, value, drop_columns]
def split_data(df, split):
    col = split[0]
    included_df = pd.DataFrame()
    excluded_df = pd.DataFrame()
    if(split[1] == ">"):
        included_df = df[df[col] > split[2]]
        excluded_df = df[df[col] <= split[2]]
    if(split[1] == ">="):
        included_df = df[df[col] >= split[2]]
        excluded_df = df[df[col] < split[2]]
    if(split[1] == '<'):
        included_df = df[df[col] < split[2]]
        excluded_df = df[df[col] >= split[2]]
    if(split[1] == '<='):
        included_df = df[df[col] <= split[2]]
        excluded_df = df[df[col] > split[2]]
    if(split[1] == '=' or split[1] == '=='):
        included_df = df[df[col] == split[2]]
        excluded_df = df[df[col] != split[2]]

    for column in split[3]:
        included_df = included_df.drop(column, axis=1)
        excluded_df = excluded_df.drop(column, axis=1)

    return included_df, excluded_df

def column_correlations(df, target_col):
    correlations = {}
    # Get rid of columns that are just constant values so we don't get the warning
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]
    for i, col in enumerate(df.columns.tolist()):
        if(df[col].dtype == np.float64 or df[col].dtype == np.int64 or df[col].dtype == np.uint8 or df[col].dtype == np.int8):
            df_copy = df.copy()
            df_copy.dropna(subset=[col], inplace=True)
            corr, _ = pearsonr(df_copy[col], df_copy[target_col])
            correlations.update({col: abs(corr)})
            
    print()
    print("Correlations with the column '" + target_col + "' (only showing columns with numeric data)")
    for key, value in sorted(correlations.items(), key=lambda item: item[1]):
        print("%s:  %s" % (round(value, 3), key))
    return correlations

def column_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df.corr(), vmax=.8, square=True)
    plt.show()

def remove_correlated_columns(df, limit=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > limit)]
    df = df.drop(df[to_drop], axis=1)
    return df

def get_main_transformations():
    transformations = config.values['main_transformations']
    if config.values['use_transformation'] != 'all':
        use_transformation = transformations[config.values['use_transformation']]
        transformations.clear() 
        transformations[config.values['use_transformation']] = use_transformation
    return transformations

def fit_train_model(model, train, y, test, fold_type=None, n_fold=None):
    if (fold_type is not None and fold_type is not 'None'):
        if (fold_type == 'stratified'):
            folds = StratifiedKFold(n_splits=n_fold, random_state=1)
        else:
            folds = KFold(n_splits=n_fold, random_state=1)
        test_pred = np.empty((0, 1), float)
        train_pred = np.empty((0, 1), float)
        for train_indices, val_indices in folds.split(train, y.values):
            x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
            y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

            model.fit(X=x_train, y=y_train)
            train_pred = np.append(train_pred, model.predict(x_val))
    else:
        model.fit(X=train, y=y)
        train_pred = model.predict(train)

    test_pred = model.predict(test)
    train_acc = accuracy_score(y, model.predict(train)) * 100
    return test_pred.reshape(-1, 1),train_pred,train_acc

def train_models(model, train, y, fold_type=None, n_fold=None):
    if (fold_type is not None and fold_type is not 'None'):
        if (fold_type == 'stratified'):
            folds = StratifiedKFold(n_splits=n_fold, random_state=1)
        else:
            folds = KFold(n_splits=n_fold, random_state=1)
        train_pred = np.empty((0, 1), float)
        for train_indices, val_indices in folds.split(train, y.values):
            x_train, x_val = train.iloc[train_indices], train.iloc[val_indices]
            y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

            model.fit(X=x_train, y=y_train)
            train_pred = np.append(train_pred, model.predict(x_val))
    else:
        model.fit(X=train, y=y)
        train_pred = model.predict(train)

    train_acc = accuracy_score(y, model.predict(train)) * 100
    return model, train_pred,train_acc

def remove_files(directory):
    for the_file in os.listdir(directory):
        file_path = os.path.join(directory, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def create_model_image(model, column_names, model_name, transformation_type):
    tree = ""
    if model_name == "RandomForest" or model_name == "DecisionTree" or model_name == "ExtraTrees" or model_name == "GradientBoosting":
        if model_name == "RandomForest" or model_name == "ExtraTrees" or model_name == "GradientBoosting":
            tree = model.estimators_[5]
        if model_name == "DecisionTree":
            tree = model
        if model_name == "GradientBoosting":
            tree = model.estimators_[5,0]
        file_name = config.values["graphs_data_path"] + transformation_type + '_' + model_name.lower()
        export_graphviz(tree, out_file=file_name + '.dot', 
                        feature_names = column_names,
                        # class_names = model.target_names,
                        rounded = True, proportion = False, 
                        precision = 2, filled = True)

        # Convert to png using system command (requires Graphviz)
        from subprocess import call
        call(['dot', '-Tpng', file_name + '.dot', '-o', file_name + '.png', '-Gdpi=600'])

        # Display in jupyter notebook
        from IPython.display import Image
        Image(filename = file_name + '.png')

        os.unlink(file_name + '.dot')

def get_exploring_transformations(column, column_type, model_type, limited_uniques=False):
    transformations = []
    # Number transformations
    if column_type == 'int64' or column_type == 'float64':
        if limited_uniques == True:
            if model_type == 'binary':
                transformations = [
                    [
                        [column+'_one_hot',    ["binary_one_hot", column]],
                    ]
                ]
            if model_type == 'continuous':
                transformations = []
            if model_type == 'normalized':
                transformations = [
                    [
                        [column+'_normalized',  ["normalized_column", column]],
                    ]
                ]
        else:
            if model_type == 'binary':
                transformations = [
                    [
                        [column+'_split_qcut_4',    ["binary_split_by_qcut", column, 4]],
                    ],
                    [
                        [column+'_split_cut_4',     ["binary_split_by_cut", column, 4]],
                    ],
                    [
                        [column+'_split_qcut_8',    ["binary_split_by_qcut", column, 8]],
                    ],
                    [
                        [column+'_split_cut_8',     ["binary_split_by_cut", column, 8]],
                    ],
                ]
            if model_type == 'continuous':
                transformations = []
            if model_type == 'normalized':
                transformations = [
                    [
                        [column+'_normalized',  ["normalized_column", column]],
                    ]
                ]

    
    # String transformations
    if column_type == 'object':  
        if limited_uniques == True:
            if model_type == 'binary':
                transformations = [
                    [
                        [column+'_numeric_categories',    ["binary_numeric_categories", column]],
                    ]
                ]
            if model_type == 'continuous':
                transformations = [
                    [
                        [column+'_numeric_categories',    ["continuous_numeric_categories", column]],
                    ]
                ]
            if model_type == 'normalized':
                transformations = [
                    [
                        [column+'_numeric_categories',    ["normalized_numeric_categories", column]],
                    ]
                ]
        else:
            if model_type == 'binary':
                transformations = [
                    [
                        [column+'_string_len',    ["binary_string_length", column]],
                    ],
                    [
                        [column+'_first_character',    ["binary_first_character", column]],
                    ]
                ]
            if model_type == 'continuous':
                transformations = [
                    [
                        [column+'_string_len',    ["continuous_string_length", column]],
                    ],
                    [
                        [column+'_first_character',    ["continuous_first_character", column]],
                    ]
                ]
            if model_type == 'normalized':
                transformations = [
                    [
                        [column+'_string_len',    ["normalized_string_length", column]],
                    ],
                    [
                        [column+'_first_character',    ["normalized_first_character", column]],
                    ]
                ]
    return transformations

def print_drop_column_transformations(df):
    print()
    print('# Drop statements for all columns, uncomment any you want to drop')
    for column in df.columns:
        drop_column_transformations = [
            [column, ["drop_column"]],
        ]
        if column != config.values['target_column']:
            for t in drop_column_transformations:
                print('#'+str(t)+",")

def print_nancolumn_transformations(df):
    missing_data = df.columns[df.isnull().any()]
    print()
    print('# Options for columns that are missing data either in the training or test data sets')
    for column in missing_data:
        nan_transformations = [
            [column, ["nan_boolean"]],
            [column, ["nan_static", 'string']],
            [column, ["nan_mean"]],
            [column, ["nan_mean_from_columns", ['column_name', 'column_name', 'column_name']]],
        ]
        if column != config.values['target_column']:
            for t in nan_transformations:
                print(str(t)+",")

def print_column_transformations(df, column_transformation_groups):
    print()
    print('# Transformations on feature data')
    for key, value in list(column_transformation_groups.items()):
        show_column = False
        for column in df.columns:
            if key in column:
                show_column = True
                # del column_transformation_groups[key]
        if show_column == True:
            for t in value:
                print(str(t)+",")

def model_optimization(df, transformation_type, model_type, X, y ):
    if config.values["show_model_optimizations"]:
        from sklearn.model_selection import GridSearchCV, cross_val_score
        from sklearn.svm import SVC
        from sklearn.ensemble import (RandomForestClassifier)

        if transformation_type == "normalized" and model_type=="RandomForest":
            param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 200, 300, 400, 700, 1000, 1500]}
            rf = RandomForestClassifier()
            clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, cv=5)
            clf.fit(X, y)
            print()
            print("Best Params for normalized date and random forest:")
            print(clf.best_params_)
            # print("oob score:", round(model.oob_score_, 4)*100, "%")
                
        if transformation_type == "binary" and model_type=="SVM1":
            param_grid = [
            {'C': [1, 10, 100, 200, 500, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 200, 500, 1000], 'gamma': [0.001, 0.0001, 'scale'], 'kernel': ['rbf']},
            ]
            estimator = SVC()
            clf = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=5)
            clf.fit(X, y)
            print()
            print("Best Params for binary date and SVC:")
            print(clf.best_params_)
            # print("oob score:", round(model.oob_score_, 4)*100, "%")