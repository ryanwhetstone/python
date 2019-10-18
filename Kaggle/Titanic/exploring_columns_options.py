import numpy as np
import pandas as pd
import lib.functions as f
import config
import os


df = pd.read_csv(config.values['original_data_path'] + config.values['train_file'])
f.print_df(df)
df_test = pd.read_csv(config.values['original_data_path'] + config.values['test_file'])

df_all = df.append(df_test, ignore_index=True, sort=False)
column_transformation_groups = {}
model_type = 'normalized'
for column in df.columns:
    if column != config.values['target_column'] and column != config.values['index_column']:
        column_type = df[column].dtype
        df = f.column_transformation(df, [column, ["nan_drop_row"]], df_all)
        if column_type == 'int64' or column_type == 'float64':
            unique_values = df[column].unique()
            if(len(unique_values) < 20):
                # Few Unique Numerical Values
                transformation_groups = f.get_exploring_transformations(column, column_type, model_type, True)
            else:
                transformation_groups = f.get_exploring_transformations(column, column_type, model_type)

        if column_type == 'object':
            unique_values = df[column].unique()
            if(len(unique_values) < 50):
                # Few Unique String Values
                transformation_groups = f.get_exploring_transformations(column, column_type, model_type, True)
            else:
                transformation_groups = f.get_exploring_transformations(column, column_type, model_type)

        
        for transformation_group in transformation_groups:
            column_name = transformation_group[0][0]
            column_transformation_groups.update({column_name :  transformation_group})
            for t in transformation_group:
                df = f.column_transformation(df, t, df_all)


f.print_df(df)
# f.show_missing_data(df_all)


# Give Nan columns options
f.print_nancolumn_transformations(df_all)


df = f.remove_correlated_columns(df, 0.85)
f.column_heatmap(df)


# Rest of columns
f.print_column_transformations(df, column_transformation_groups)

# Show all columns to consider dropping
f.print_drop_column_transformations(df_all)


correlations = f.column_correlations(df, config.values['target_column'])

