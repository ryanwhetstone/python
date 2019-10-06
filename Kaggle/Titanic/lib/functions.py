from scipy.stats import pearsonr
from sklearn.model_selection import (StratifiedKFold, KFold)
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from pprint import pprint as pp
import seaborn as sns
import pandas as pd
import numpy as np
import os, shutil
import lib.functions as f


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
    print(df.head(30))
    print(df.info())
    print(df.describe())


# Handle column and row transformations
def column_transformation(df, values, df_all):
    col = values[0]
    transformation = values[1]

    if transformation[0] == 'drop_column':
        if col in df.columns:
            df = df.drop(col, axis=1)
    if transformation[0] == 'copy_column':
        df[col] = df[transformation[1]]
    if transformation[0] == 'add_columns':
        df[col] = df[transformation[1]] + df[transformation[2]]
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
        df[col] = df[col].apply(len)
    if transformation[0] == 'count_fequencies':
        df[col] = df_all.groupby(transformation[1])[transformation[1]].transform('count')
    if transformation[0] == 'extract_dummy':
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
        # df.loc[df[transformation[1]] == transformation[2], col] = 1
    if transformation[0] == 'duplicate_dummies':
        df_all['duplicates'] = df_all.duplicated([col], keep=False)
        for index, row in df.iterrows():
            if(df_all.iloc[index]['duplicates'] == False):
                df.at[index,col] = np.NaN
        df[col] = df[col].astype('category').cat.codes
        df = pd.get_dummies(df, columns=[col], prefix=[col])
    if transformation[0] == 'first_character':
        df[col] = df[col].astype(str).str[0]
    if transformation[0] == 'split_by_qcut':
        df[col] = pd.qcut(df[col], transformation[1], duplicates='drop')
    if transformation[0] == 'split_by_cut':
        df[col] = pd.cut(df[col], transformation[1], duplicates='drop')
    if transformation[0] == 'split_by_defined':
        df[col] = pd.cut(df[col], transformation[1], duplicates='drop')
    if transformation[0] == 'regex_extract':
        df[col] = df[col].str.extract(transformation[1], expand=False)
    if transformation[0] == 'str_replace':
        df[col] = df[col].replace(transformation[1], transformation[2])
    if transformation[0] == 'numeric_categories':
        df[col] = df[col].astype('category').cat.codes
    if transformation[0] == 'one_hot':
        df = pd.get_dummies(df, columns=[col], prefix=[col])

    return df

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
    for i, col in enumerate(df.columns.tolist()):
        if(df[col].dtype == np.float64 or df[col].dtype == np.int64 or df[col].dtype == np.uint8):
            corr, _ = pearsonr(df[col], df[target_col])
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