from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import lib.functions as f


transformations = [
    ["PassengerId", ["drop_column"]],
    ["Ticket",      ["drop_column"]],

    ["Cabin",       ["copy_column", "CabinBool"]],
    ["Cabin",      	["drop_column"]],
	["Name",       	["copy_column", "Title"]],
    ["Name",       	["drop_column"]],

    ["Age",         ["nan_drop_row"]],
    ["CabinBool",   ["nan_boolean"]],
    # ["Fare",        ["nan_mean"]],  # mean

    # ["Cabin",       ["first_character"]],

    ["Age",         ["split_by_defined", [0, 15, 25, 40, 60, 90]]],
    ["Fare",        ["split_by_qcut", 4]],
    ["Parch",       ["split_by_qcut", 4]],
    ["SibSp",       ["split_by_qcut", 4]],

    ["Title",        ["regex_extract", " ([A-Za-z]+)\."]],
    ["Title",        ["str_replace", ['Lady', 'Countess', 'Dona'], 'Royalty']],
    ["Title",        ["str_replace", ['Capt', 'Col', 'Major', 'Rev'], 'Officer']],
    ["Title",        ["str_replace", ['Jonkheer', 'Don', 'Sir'], 'Royalty']],
    ["Title",        ["str_replace", ['Mlle', 'Ms'], 'Miss']],
    ["Title",        ["str_replace", 'Mme', 'Mrs']],

    # [ "Age",         ["numeric_categories"] ],
    # [ "Fare",        ["numeric_categories"] ],
    # [ "Parch",       ["numeric_categories"] ],
    # [ "Sex",         ["numeric_categories"] ],
    # [ "SibSp",       ["numeric_categories"] ],

    ["Age",         ["one_hot"]],
    # ["Cabin",       ["one_hot"]],
    ["Embarked",    ["one_hot"]],
    ["Fare",        ["one_hot"]],
    ["Title",        ["one_hot"]],
    ["Parch",       ["one_hot"]],
    ["Pclass",      ["one_hot"]],
    ["Sex",         ["one_hot"]],
    ["SibSp",       ["one_hot"]],
]


data_path = 'data/original/'
data_file = 'train.csv'
df = pd.read_csv(data_path + data_file)

f.print_df(df)

for values in transformations:
    df = f.column_transformation(df, values)

correlations = f.column_correlations(df, "Survived")

ignore_columns = ["Survived", "PassengerId"]
for col, value in correlations.items():
    if col not in ignore_columns:
        if(value < 0.06):
            df = df.drop(col, axis=1)

f.print_df(df)


df.to_csv('data/converted/' + data_file, index=False)
