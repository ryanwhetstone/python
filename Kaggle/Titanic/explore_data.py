from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
from scipy.stats import pearsonr
import lib.functions as f


df = pd.read_csv('data/original/train.csv')
print('First ten rows')
print()
print(df.head(10))
print()
print("- Consider removing any column with uncorrelated data (ie. df.drop('Name', axis=1))")
print(
    "- Code text categorical data (ie. df['Gender'] = df['Gender'].astype('category').cat.codes)")
print("- Consider converting NaN values to either a mean or something within a standard deviation of the median (this will lessen the effect of this column")
print("- Consider converting NaN values to a number like -99 or something similar that will allow categorizing the column")
print("- df['ColumnName'].fillna('_',inplace=True) # This replaces all NaN's with a static value")
print("- df['ColumnName'] = df['ColumnName'].astype(str).str[0] # This pulls off the first character of a string")
print("- df['ColumnName'] = df['ColumnName'].astype('category').cat.codes # This turns a column into a category type, then replaces text categories with integers representing the text categories")

print()
print('Table showing columns of the features and number of rows filled with data for each feature')
print()
df.info()
print()
print()
print(df.describe())
print()

total = df.isnull().sum().sort_values(ascending=False)
percent = (round(df.isnull().sum()*100 /
                 df.isnull().count(), 2)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=[
                         'Total Missing', 'Percent'])
print(missing_data.head(10))

f, ax = plt.subplots(figsize=(15, 8))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

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
                corr, _ = pearsonr(data, df['Survived'])
            else:
                unique_values = df[col].unique()
                if(len(unique_values) < 10):
                    data = df[col].fillna('_').astype('category').cat.codes
                    f.display_histogram(data, col, ax)
                    corr, _ = pearsonr(data, df['Survived'])
            print(corr)
            print('Pearsons correlation: %.3f' % corr)


plt.show()
