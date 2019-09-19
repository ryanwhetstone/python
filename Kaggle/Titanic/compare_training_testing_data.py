from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import lib.functions as f


ignore_columns = ["Survived", "PassengerId"]

df_train = pd.read_csv('data/converted/train.csv')
df_test = pd.read_csv('data/converted/test.csv')

# print("test columns")
# print(df_test.columns )
print("train columns")
f.print_df(df_train)
# Compare columns in test and training data 
missing_cols = set( df_test.columns ) - set( df_train.columns )
for c in missing_cols:
    if c not in ignore_columns:
        # df[c] = 0
        print(str(c) + " is in test data but not in the training data")


missing_cols = set( df_train.columns ) - set( df_test.columns )
for c in missing_cols:
    if c not in ignore_columns:
        df_train = df_train.drop(c, axis=1)
        print(str(c) + " is in training data but not in the test data")

df_train.to_csv('data/converted/train.csv', index=False)
