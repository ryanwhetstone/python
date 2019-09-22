import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
import pprint


def display_histogram(data, title, axes):
    axes.set_title(title)
    sns.distplot(data, bins=30, ax=axes, axlabel=False)


def print_df(df):
    print()
    print(df.head(30))
    print(df.info())
    print(df.describe())


# Handle column and row transformations
def column_transformation(df, values):
    col = values[0]
    transformation = values[1]

    if transformation[0] == 'drop_column':
        df = df.drop(col, axis=1)
    if transformation[0] == 'copy_column':
        df[transformation[1]] = df[col]
    if transformation[0] == 'nan_static':
        df[col].fillna(transformation[1], inplace=True)
    if transformation[0] == 'nan_drop_row':
        df.dropna(subset=[col], inplace=True)
    if transformation[0] == 'nan_mean':
        df[col].fillna(df[col].mean(), inplace=True)
    if transformation[0] == 'nan_boolean':
        df[col] = df[col].notnull().astype('int')
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


def column_correlations(df, target_col):
    correlations = {}
    for i, col in enumerate(df.columns.tolist()):
        corr, _ = pearsonr(df[col], df[target_col])
        correlations.update({col: abs(corr)})
        print(col)
        # print('Pearsons correlation for column' + str(col) + ' is ' + str(round(corr, 3)))
    for key, value in sorted(correlations.items(), key=lambda item: item[1]):
        print("%s:  %s" % (round(value, 3), key))
    return correlations
