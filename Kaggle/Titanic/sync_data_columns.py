from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import lib.functions as f
import glob, os
import config


ignore_columns = [config.values['target_column'], config.values['index_column']]

test_csv_files = {}
for file in glob.glob(config.values['transformed_data_path'] + "test_*"):
    test_csv_files[file.replace('.csv', '').replace(config.values['transformed_data_path'], '')] = pd.read_csv(file)

train_csv_files = {}
for file in glob.glob(config.values['transformed_data_path'] + "train_*"):
    train_csv_files[file.replace('.csv', '').replace(config.values['transformed_data_path'], '')] = pd.read_csv(file)

df_train = pd.read_csv(config.values['transformed_data_path'] + config.values['train_file'])
df_test = pd.read_csv(config.values['transformed_data_path'] + config.values['test_file'])


# Compare columns in test but not in the training data, then update the csv files so the columns match accordingly
missing_cols = set( df_test.columns ) - set( df_train.columns )
for c in missing_cols:
    if c not in ignore_columns:
        df_test = df_test.drop(c, axis=1)
        # print(str(c) + " is in test data but not in the training data")

if test_csv_files:
    for file, df in test_csv_files.items():
        for c in missing_cols:
            if c not in ignore_columns:
                df = df.drop(c, axis=1)
        df.to_csv(config.values['transformed_data_path'] + file + '.csv', index=False)


# Now compare columns in training but not in the test data
missing_cols = set( df_train.columns ) - set( df_test.columns )
for c in missing_cols:
    if c not in ignore_columns:
        # df_train = df_train.drop(c, axis=1)
        df_test[c] = 0

        # print(str(c) + " is in training data but not in the test data")

if train_csv_files:
    for c in missing_cols:
        if c not in ignore_columns:
            for file, df in train_csv_files.items():
                df = df.drop(c, axis=1)
        df.to_csv(config.values['transformed_data_path'] + file + '.csv', index=False)
df_test.to_csv(config.values['transformed_data_path'] + config.values['test_file'], index=False)
df_train.to_csv(config.values['transformed_data_path'] + config.values['train_file'], index=False)
