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
import joblib


transformations = f.get_main_transformations()
x_stacked = pd.DataFrame()

for transformation_type, transformation in transformations.items():

    df = pd.read_csv(config.values['transformed_data_path'] + transformation_type + '_' + config.values['train_file'])

    y = df[config.values['target_column']].copy()
    X = df.drop(config.values['target_column'], axis=1)

    # Classifiers
    clf_rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, min_samples_split=5, max_features="auto", n_jobs=-1, oob_score = True)
    clf_et = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=3)
    clf_tree = tree.DecisionTreeClassifier()
    clf_gradient = GradientBoostingClassifier(n_estimators=200, min_samples_leaf=3, max_features=None)
    clf_ada = AdaBoostClassifier()
    clf_svm1 = SVC(gamma='scale')
    clf_svm2 = SVC(C=1, gamma='scale', kernel='rbf')
    clf_svm3 = SVC(gamma='scale', shrinking=False)
    clf_knn = KNeighborsClassifier()
    clf_perceptron = Perceptron()
    clf_mlp1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 8), random_state=1)
    clf_mlp2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 6), random_state=1)
    clf_mlp3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 4), random_state=1)
    clf_mlp4 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(24, 2), random_state=1)
    clf_mlp5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(115, 30, 8), random_state=1)

    models = { # Keep in alphabetical order
        "RandomForest": clf_rf,
        # "ExtraTrees": clf_et,
        # "DecisionTree": clf_tree,
        # "GradientBoosting": clf_gradient,
        # "AdaBoost": clf_ada,
        "SVM1": clf_svm1,
        "SVM2": clf_svm2,
        # "SVM3": clf_svm3,
        # "KNN": clf_knn,
        # "Perceptron": clf_perceptron,
        # "MultiLayerPerceptron1": clf_mlp1,
        # "MultiLayerPerceptron2": clf_mlp2,
        # "MultiLayerPerceptron3": clf_mlp3,
        # "MultiLayerPerceptron4": clf_mlp4,
        # "MultiLayerPerceptron5": clf_mlp5,
    }

    train_accuracies = {}
    for key, model in sorted(models.items()):

        model, train, train_acc = f.train_models(model, X, y, config.values['fold_type'], config.values['n_fold'])

        x_stacked = pd.concat([x_stacked, pd.DataFrame(train, columns=[transformation_type + '_' + key.lower()])], axis=1)
        train_accuracies.update({key : round(train_acc,1)})

        # Save Model
        joblib.dump(model, config.values['models_data_path']  + transformation_type + '_' + key.lower() + '.pkl')

        f.create_model_image(model, X.columns, key, transformation_type)

        f.model_optimization(df, transformation_type, key, X, y )


    print()
    print("Train Accuracies for '" + transformation_type + "'")
    pp(train_accuracies)


model = xgb.XGBClassifier()
pred = model.fit(x_stacked,y)
acc = accuracy_score(y, model.predict(x_stacked)) * 100
print()
print("Train accuracy for 'stacked': " + str(round(acc,1)))
# train_accuracies.update({'Stacked' : round(acc,1)})
joblib.dump(model, config.values['models_data_path'] + 'stacked.pkl')

# print()
# importances = pd.DataFrame({'feature':X.columns,'importance':np.round(clf_rf.feature_importances_,3)})
# importances = importances[importances['importance']<0.01].sort_values('importance',ascending=False).set_index('feature')
# if len(importances)>0:
#     print("Consider removing the following features because they have a very low importance to the algorithm.")
#     for index, row in importances.iterrows():
#         print('["' + index + '",       	["drop_column"]],')
