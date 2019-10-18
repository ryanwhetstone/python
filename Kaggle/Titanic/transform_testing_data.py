import numpy as np
import pandas as pd
import lib.functions as f
import config

transformations = f.get_main_transformations()

for key, transformation in transformations.items():


    df = pd.read_csv(config.values['original_data_path'] + config.values['test_file'])
    df_train = pd.read_csv(config.values['original_data_path'] + config.values['train_file'])

    df_all = df.append(df_train, ignore_index=True, sort=True)
    # f.print_df(df)

    for values in transformation:
        df = f.column_transformation(df, values, df_all)
        df_all = f.column_transformation(df_all, values, df_all)


    f.print_df(df)
    f.show_missing_data(df)


    df = df.reindex(sorted(df.columns), axis=1)

    df.to_csv(config.values['transformed_data_path'] + key + '_' + config.values['test_file'], index=False)
