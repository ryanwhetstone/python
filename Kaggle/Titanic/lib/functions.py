import seaborn as sns
import pandas as pd

def display_histogram(data, title, axes):
    axes.set_title(title)
    sns.distplot(data, bins=30, ax=axes, axlabel=False)

def print_df(df):
    print()
    print(df.head(30))
    print(df.info())
    print(df.describe())

# Handle column and row transformations
def column_transformations(df, transformation):
    for col, value in transformation.items():
        if value[0] == 'drop_column':
            df = df.drop(col, axis=1)
        if value[0] == 'nan_static':
            df[col].fillna(value[1], inplace=True)
        if value[0] == 'nan_drop_row':
            df.dropna(subset=[col], inplace=True)
        if value[0] == 'nan_mean':
            df[col].fillna(df[col].mean(), inplace=True)
        if value[0] == 'first_character':
            df[col] = df[col].astype(str).str[0]
        if value[0] == 'split':
            # df[col] = pd.cut(df[col], 4)
            df[col] = pd.qcut(df[col], [.25, .5, .75, 1], duplicates='drop')
        if value[0] == 'regex_extract':
            df[col] = df[col].str.extract(value[1], expand=False)
        if value[0] == 'str_replace':
            df[col] = df[col].replace(value[1], value[2])
        if value[0] == 'numeric_categories':
            df[col] = df[col].astype('category').cat.codes
        if value[0] == 'one_hot':
            df = pd.get_dummies(df, columns=[col], prefix = [col])
    return df

