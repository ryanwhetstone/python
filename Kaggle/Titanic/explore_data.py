from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr
import lib.functions as f
import config

df = pd.read_csv(config.values['original_data_path'] + config.values['train_file'])
# df = pd.read_csv(config.values['transformed_data_path'] + config.values['train_file'])

f.print_df(df)
print()

f.show_missing_data(df)

print()
columns = df.columns.tolist()
columns_to_plot = columns.copy()
# Pare down the columns to only ones that are numeric or will be converted to categorical numbers
for i, col in enumerate(columns):
    if(df[col].dtype == np.float64 or df[col].dtype == np.int64):
        # Data Type is numeric
        print(col + ' is numeric data.')
    else:
        unique_values = df[col].unique()
        if(len(unique_values) < 10):
            print(col + ' is string data. There are ' +
                  str(len(unique_values)) + ' unique values. Since there is a low number of unique values it makes sense to categorize this column')
        else:
            print(col + ' is string data. There are ' +
                  str(len(unique_values)) + ' unique values.')
            columns_to_plot.remove(col)

# Setting up the Histograms for the numeric data
n_cols = 2
n_rows = 2
n_plots_per_chart = n_cols * n_rows
n_charts = math.ceil(len(columns_to_plot)/(n_plots_per_chart))

column_chunks = [columns_to_plot[i * n_plots_per_chart:(i + 1) * n_plots_per_chart] for i in range((len(columns_to_plot) + n_plots_per_chart - 1) // n_plots_per_chart )]

for n in range(n_charts):
    columns = column_chunks[n]
    fig, subplots = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    for i, ax in enumerate(subplots.flatten()):
        if(i < len(columns)):
            col = columns[i]
            df.dropna(subset=[col], inplace=True)

            if(df[col].dtype == np.float64 or df[col].dtype == np.int64):
                data = df[col].fillna('-100').astype(float)
                f.display_histogram(data, col, ax)
            else:
                unique_values = df[col].unique()
                if(len(unique_values) < 10):
                    data = df[col].fillna('_').astype('category').cat.codes
                    f.display_histogram(data, col, ax)

correlations = f.column_correlations(df, config.values['target_column'])


plt.show()

print()
print("Starting point for transformations in the config file.")
for col in df.columns.tolist():
    if col != config.values["index_column"] and col != config.values["target_column"]:
        print('["' + col + '",       	["drop_column"]],')