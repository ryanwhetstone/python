import numpy as np
import pandas as pd
import lib.functions as f
import config

transformations = f.get_main_transformations()

for key, transformation in transformations.items():

    transformation.insert( 0, [config.values['index_column'], ["drop_column"]] )
    df = pd.read_csv(config.values['original_data_path'] + config.values['train_file'])
    df_test = pd.read_csv(config.values['original_data_path'] + config.values['test_file'])

    df_all = df.append(df_test, ignore_index=True, sort=False)

    for values in transformation:
        df = f.column_transformation(df, values, df_all, True)
        df_all = f.column_transformation(df_all, values, df_all)

    if config.values["debug_without_one_hot"] or config.values["debug"]:
        f.print_df(df)
    # f.show_missing_data(df)


    # df = f.remove_correlated_columns(df, 0.9)
    # f.column_heatmap(df)

    correlations = f.column_correlations(df, config.values['target_column'])

    # ignore_columns = [config.values['target_column'], config.values['index_column']]
    # for col, value in correlations.items():
    #     if col not in ignore_columns:
    #         if(value < 0.05):
    #             df = df.drop(col, axis=1)

    df = df.reindex(sorted(df.columns), axis=1)

    df.to_csv(config.values['transformed_data_path'] + key + '_' + config.values['train_file'], index=False)
