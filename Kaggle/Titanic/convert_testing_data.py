from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import lib.functions as f


# Simply drop these columns from the data
drop_columns = ["Name", "Ticket"]

# How to handle Nan values (Current options are 'static', 'drop_row', 'mean')
NaN_handling = {
    "Cabin": ["static", "_"],
    "Age": ["mean"],
    "Fare": ["mean"],
    # "Age": ["drop_row"]
}

# We can do intermediate transformations here. ie. just take the first character of each row
first_transformations = {
    "Cabin": ["first_character"],
    "Age": ["split"],
    "Fare": ["split"],
    "SibSp": ["split"],
    "Parch": ["split"],
}

# Convert to from unique strings/character to a numeric code
convert_to_numeric_categories = ["Sex", "Fare", "Age", "SibSp", "Parch"]

# Final transformations
last_transformations = {
    "Embarked": ["one_hot"],
    "Cabin": ["one_hot"],
    "Pclass": ["one_hot"],
    "SibSp": ["one_hot"],
    "Parch": ["one_hot"],
    "Age": ["one_hot"],
    "Fare": ["one_hot"],
}


data_path = 'data/original/'
data_file = 'test.csv'
df = pd.read_csv(data_path + data_file)
f.print_df(df)

df = f.drop_columns(df, drop_columns)

df = f.nan_columns(df, NaN_handling)

df = f.column_transformations(df, first_transformations)

df = f.convert_to_numerical_categories(df, convert_to_numeric_categories)

df = f.column_transformations(df, last_transformations)

f.print_df(df)

df.to_csv('data/converted/' + data_file, index=False)
