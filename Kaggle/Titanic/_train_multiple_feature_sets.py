from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import (LogisticRegression, Perceptron)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (cross_val_score, cross_val_predict, KFold)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from pprint import pprint as pp
import xgboost as xgb
import seaborn as sns
import numpy as np
import pandas as pd
import lib.functions as f
import glob, os
import config


results = {}
train_accuracies = {}

for file in glob.glob(config.values['transformed_data_path'] + "train_*"):
    file_name = file.replace('.csv', '').replace(config.values['transformed_data_path'], '').replace('train_','')
    df = pd.read_csv(file)
    # df = pd.read_csv(config.values['transformed_data_path'] + config.values['train_file'])

    y = df[config.values['target_column']].copy()
    # p_train = pd.DataFrame(df[config.values['index_column']])
    p_train = pd.DataFrame()
    X = df.drop(config.values['target_column'], axis=1)


    # Classifiers
    clf_svm1 = SVC(gamma='scale')
    clf_svm2 = SVC(gamma='auto')
    clf_svm3 = SVC(gamma='scale', shrinking=False)
    clf_rf = RandomForestClassifier(n_estimators=100)
    clf_ada = AdaBoostClassifier()
    clf_gradient = GradientBoostingClassifier()
    clf_et = ExtraTreesClassifier(n_estimators=100)
    clf_tree = tree.DecisionTreeClassifier()
    clf_perceptron = Perceptron()
    clf_knn = KNeighborsClassifier()
    clf_mlp1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 8), random_state=1)
    clf_mlp2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 6), random_state=1)
    clf_mlp3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 4), random_state=1)
    clf_mlp4 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(24, 2), random_state=1)
    clf_mlp5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 8), random_state=1)

    models = {
        "SVM1": clf_svm1,
        "SVM2": clf_svm2,
        "SVM3": clf_svm3,
        "RandomForest": clf_rf,
        "AdaBoost": clf_ada,
        "GradientBoosting": clf_gradient,
        "ExtraTrees": clf_et,
        "DecisionTree": clf_tree,
        "Perceptron": clf_perceptron,
        "KNN": clf_knn,
        "MultiLayerPerceptron1": clf_mlp1,
        "MultiLayerPerceptron2": clf_mlp2,
        "MultiLayerPerceptron3": clf_mlp3,
        "MultiLayerPerceptron4": clf_mlp4,
        "MultiLayerPerceptron5": clf_mlp5,
    }

    df_test = pd.read_csv(file.replace('train', 'test'))
    p_test = pd.DataFrame(df_test[config.values['index_column']])
    x_test = df_test.drop(config.values['index_column'], axis=1)

    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for key, model in models.items():

        test, train, train_acc = f.fit_train_model(model, X, y, x_test, 'stratified', 5)

        df_train = pd.concat([df_train, pd.DataFrame(train, columns=[key])], axis=1)
        df_test = pd.concat([df_test, pd.DataFrame(test, columns=[key])], axis=1)
        train_accuracies.update({(key + '_' + file_name).lower() : round(train_acc,1)})

        p_test[config.values['target_column']] = test.copy()
        if key in results.keys():
            results.update({key : pd.concat([results[key], p_test], axis=0, sort=True).sort_values(by=[config.values['index_column']])})
        else:
            results.update({key : p_test})
        p_test.to_csv(config.values['output_data_path'] + 'test/results_' + file_name + '_' + key.lower() + '.csv', index=False)

pp(train_accuracies)
for key, result in results.items():
    result.to_csv(config.values['output_data_path'] + 'test/results_' + key.lower() + '.csv', index=False)
