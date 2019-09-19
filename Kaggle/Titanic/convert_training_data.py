from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import lib.functions as f

transformations = []
# Simply drop these columns from the data
# drop_columns = ["Ticket", "PassengerId"]
transformations.append({
    "Ticket": ["drop_column"],
    "PassengerId": ["drop_column"],
})

# How to handle Nan values (Current options are 'static', 'drop_row', 'mean')
NaN_handling = {
    "Cabin": ["static", "_"],
    # "Age": ["mean"],
    "Age": ["drop_row"],
}

# We can do intermediate transformations here. ie. just take the first character of each row
transformations.append({
    "Cabin": ["first_character"],
    "Age": ["split" ],
    "Fare": ["split"],
    "SibSp": ["split"],
    "Parch": ["split"],
})

# Convert to from unique strings/character to a numeric code
convert_to_numeric_categories = ["Sex", "Fare", "Age", "SibSp", "Parch"]

# Final transformations
transformations.append({
    "Embarked": ["one_hot"],
    "Cabin": ["one_hot"],
    "Pclass": ["one_hot"],
    "SibSp": ["one_hot"],
    "Parch": ["one_hot"],
    "Age": ["one_hot"],
    "Fare": ["one_hot"],
})


data_path = 'data/original/'
data_file = 'train.csv'
df = pd.read_csv(data_path + data_file)
# f.print_df(df)

# df = f.drop_columns(df, drop_columns)

# df = f.nan_columns(df, NaN_handling)

for transformation in transformations:
    df = f.column_transformations(df, transformation)


# df = f.convert_to_numerical_categories(df, convert_to_numeric_categories)

# df = f.column_transformations(df, last_transformations)

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

# extract titles
df['Title'] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
# replace titles with a more common title or as Rare
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                        'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
# convert titles into numbers
df['Title'] = df['Title'].map(titles)
# filling NaN with 0, to get safe
df['Title'] = df['Title'].fillna(0)
df = df.drop(['Name'], axis=1)

f.print_df(df)



df.to_csv('data/converted/' + data_file, index=False)
