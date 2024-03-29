import glob
import os
import config
import joblib
import pandas as pd
import xgboost as xgb
import lib.functions as f


transformations = f.get_main_transformations()

x_stacked = pd.DataFrame()
for transformation_type, transformation in transformations.items():

    df_test = pd.read_csv(config.values['transformed_data_path'] + transformation_type + '_' + config.values['test_file'])
    p_test = pd.DataFrame(df_test[config.values['index_column']])
    x_test = df_test.drop(config.values['index_column'], axis=1)
    # x_test = df_test.copy()
    results = {}
    train_accuracies = {}

    for model_file in sorted(glob.glob(config.values['models_data_path'] + transformation_type + '_' + "*.pkl")):
        model_name = model_file.replace('.pkl', '').replace(config.values['models_data_path'], '')

        if model_name != transformation_type + '_' + 'stacked':

            # Load the model
            model = joblib.load(model_file)

            test_pred = model.predict(x_test)
            x_stacked = pd.concat([x_stacked, pd.DataFrame(test_pred, columns=[model_name])], axis=1)
            p_test[config.values['target_column']] = test_pred.copy()
            p_test.to_csv(config.values['output_data_path'] + 'test/results_' + model_name.lower() + '.csv', index=False)

# Load stacked model
stacked_model = joblib.load(config.values['models_data_path'] + 'stacked.pkl')
p_test[config.values['target_column']] = stacked_model.predict(x_stacked)
p_test.to_csv(config.values['output_data_path'] + 'test/results_' + 'stacked.csv', index=False)
