import seaborn as sns
import pandas as pd

def display_histogram(data, title, axes):
    axes.set_title(title)
    sns.distplot(data, bins=30, ax=axes, axlabel=False)

def print_df(df):
    print()
    print(df.head(10))
    print(df.info())
    print(df.describe())

# Drop Columns first
def drop_columns(df, columns):
    for col in columns:
        df = df.drop(col, axis=1)
    return df

# Handle NaN values next
def nan_columns(df, columns):
    for col, value in columns.items():
        if value[0] == 'static':
            df[col].fillna(value[1], inplace=True)
        if value[0] == 'drop_row':
            df.dropna(subset=[col], inplace=True)
        if value[0] == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
    return df

# Handle intermediate transformations
def column_transformations(df, transformation):
    for col, value in transformation.items():
        if value[0] == 'drop_column':
            df = df.drop(col, axis=1)
        if value[0] == 'first_character':
            df[col] = df[col].astype(str).str[0]
        if value[0] == 'split':
            # df[col] = pd.cut(df[col], 4)
            df[col] = pd.qcut(df[col], [.25, .5, .75, 1], duplicates='drop')
        if value[0] == 'one_hot':
            df = pd.get_dummies(df, columns=[col], prefix = [col])
    return df

# Convert to Numeric Categories
def convert_to_numerical_categories(df, columns):
    for col in columns:
        df[col] = df[col].astype('category').cat.codes
    return df
