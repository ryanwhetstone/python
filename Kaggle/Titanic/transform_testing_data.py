import numpy as np
import pandas as pd
import lib.functions as f
import config


transformations = config.values['main_transformations']

data_file = config.values['test_file']
df = pd.read_csv(config.values['original_data_path'] + config.values['test_file'])
df_train = pd.read_csv(config.values['original_data_path'] + config.values['train_file'])

df_all = df.append(df_train, ignore_index=True, sort=True)
# f.print_df(df)

for values in transformations:
    df = f.column_transformation(df, values, df_all)
    df_all = f.column_transformation(df_all, values, df_all)

# f.print_df(df)


# split_data = [
#     ["Sex_0", "==", 1, ['Sex_0', 'Sex_1']],
# ]
# data_splits = {}

# for split in split_data:
#     included_df, excluded_df = f.split_data(df, split)
#     included_df.to_csv(config.values['transformed_data_path'] + 'test_selected_' + split[0].lower() + '.csv', index=False)
#     excluded_df.to_csv(config.values['transformed_data_path'] + 'test_excluded_' + split[0].lower() + '.csv', index=False)



df.to_csv(config.values['transformed_data_path'] + config.values['test_file'], index=False)
