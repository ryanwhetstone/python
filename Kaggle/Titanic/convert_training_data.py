from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import lib.functions as f

transformations = []
# First drop these columns from the data
transformations.append({
    "Ticket": ["drop_column"],
    "PassengerId": ["drop_column"],
})

# How to handle Nan values (Current options are 'nan_static', 'nan_drop_row', 'nan_mean')
transformations.append({
    "Cabin": ["nan_static", "_"],
    "Age": ["nan_drop_row"],
})

transformations.append({
    "Name": ["regex_extract", " ([A-Za-z]+)\."]
})

# We can do intermediate transformations here. ie. just take the first character of each row, split into 4 bins, etc
transformations.append({
    "Cabin": ["first_character"],
    "Age": ["split" ],
    "Fare": ["split"],
    "SibSp": ["split"],
    "Parch": ["split"],
    "Name": ["str_replace", ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'],
})

transformations.append({
    "Name": ["str_replace", ['Mlle', 'Ms'], 'Miss'],
})

transformations.append({
    "Name": ["str_replace", 'Mme', 'Mrs'],
})

# Convert to from unique strings/character to a numeric code
transformations.append({
    "Sex": ["numeric_categories"],
    "Fare": ["numeric_categories"],
    "Age": ["numeric_categories"],
    "SibSp": ["numeric_categories"],
    "Parch": ["numeric_categories"],
    # "Name": ["numeric_categories"],
})

# Final transformations
transformations.append({
    "Embarked": ["one_hot"],
    "Cabin": ["one_hot"],
    "Pclass": ["one_hot"],
    "SibSp": ["one_hot"],
    "Parch": ["one_hot"],
    "Age": ["one_hot"],
    "Fare": ["one_hot"],
    "Name": ["one_hot"],
})


data_path = 'data/original/'
data_file = 'train.csv'
df = pd.read_csv(data_path + data_file)

f.print_df(df)


for transformation in transformations:
    df = f.column_transformations(df, transformation)


f.print_df(df)



df.to_csv('data/converted/' + data_file, index=False)
