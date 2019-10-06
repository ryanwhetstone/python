import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import numpy as np
import config

df = pd.read_csv(config.values['transformed_data_path'] + config.values['train_file']) 
y = df[config.values['target_column']]

# Get all files that we want to blend for the final prediction
path = config.values['models_data_path'] + '/train/'
files = os.listdir(path)
predictions = pd.DataFrame()
for file in files:
    df = pd.read_csv(path + file)
    predictions[file.replace('.csv', '')] = df[config.values['target_column']]


gbm = xgb.XGBClassifier()

gbm.fit(predictions, y)
pred = gbm.predict(predictions)
acc = accuracy_score(y, pred) * 100
print(acc)

path = config.values['output_data_path'] + 'test/'
files = os.listdir(path)
models_x_test = pd.DataFrame()
for file in files:
    df = pd.read_csv(path + file)
    models_x_test[file.replace('.csv', '')] = df[config.values['target_column']]

print(models_x_test)

df_test = pd.read_csv(config.values['transformed_data_path'] + config.values['test_file']) 
p_test = pd.DataFrame(df_test[config.values['index_column']])
# x_test = df_test.drop(config.values['index_column'], axis=1)

p_test[config.values['target_column']] = gbm.predict(models_x_test)
p_test.to_csv('data/output/results_stacked.csv', index=False)


print()
print("Listing of Importances of columns for the Classifier")
importances = pd.DataFrame({'feature':models_x_test.columns,'importance':np.round(gbm.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances.head(30))
