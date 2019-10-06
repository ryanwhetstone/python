import numpy as np
import pandas as pd
import lib.functions as f
import config
import os

transformations = config.values['main_transformations']

transformations.insert( 0, [config.values['index_column'], ["drop_column"]] )


df = pd.read_csv(config.values['original_data_path'] + config.values['train_file'])
df_test = pd.read_csv(config.values['original_data_path'] + config.values['test_file'])

df_all = df.append(df_test, ignore_index=True, sort=False)

for values in transformations:
    df = f.column_transformation(df, values, df_all)
    df_all = f.column_transformation(df_all, values, df_all)

# df = f.remove_correlated_columns(df, 0.9)
# f.column_heatmap(df)

correlations = f.column_correlations(df, config.values['target_column'])

ignore_columns = [config.values['target_column'], config.values['index_column']]
for col, value in correlations.items():
    if col not in ignore_columns:
        if(value < 0.05):
            df = df.drop(col, axis=1)

# split_data = [
#     ["Sex_0", "==", 1, ['Sex_0', 'Sex_1']],
# ]
# data_splits = {}

# for split in split_data:
#     included_df, excluded_df = f.split_data(df, split)
#     included_df.to_csv(config.values['transformed_data_path'] + 'train_selected_' + split[0].lower() + '.csv', index=False)
#     excluded_df.to_csv(config.values['transformed_data_path'] + 'train_excluded_' + split[0].lower() + '.csv', index=False)



df.to_csv(config.values['transformed_data_path'] + config.values['train_file'], index=False)
