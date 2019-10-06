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


df = pd.read_csv(config.values['transformed_data_path'] + config.values['train_file'])

y = df[config.values['target_column']].copy()
X = df.drop(config.values['target_column'], axis=1)

# Classifiers
clf_rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, max_features=0.75, n_jobs=-1)
clf_svm1 = SVC(gamma='scale')
clf_svm2 = SVC(gamma='auto')
clf_svm3 = SVC(gamma='scale', shrinking=False)
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
clf_mlp5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(115, 30, 8), random_state=1)

models = {
    "RandomForest": clf_rf,
    "SVM1": clf_svm1,
    # "SVM2": clf_svm2,
    # "SVM3": clf_svm3,
    # "AdaBoost": clf_ada,
    # "GradientBoosting": clf_gradient,
    # "ExtraTrees": clf_et,
    # "DecisionTree": clf_tree,
    # "Perceptron": clf_perceptron,
    # "KNN": clf_knn,
    # "MultiLayerPerceptron1": clf_mlp1,
    # "MultiLayerPerceptron2": clf_mlp2,
    "MultiLayerPerceptron3": clf_mlp3,
    # "MultiLayerPerceptron4": clf_mlp4,
    "MultiLayerPerceptron5": clf_mlp5,
}

train_accuracies = {}
df_train = pd.DataFrame()
for key, model in sorted(models.items()):

    model, train, train_acc = f.train_models(model, X, y, config.values['fold_type'], config.values['n_fold'])

    df_train = pd.concat([df_train, pd.DataFrame(train, columns=[key.lower()])], axis=1)
    train_accuracies.update({key : round(train_acc,1)})

    # Save Model
    joblib.dump(model, config.values['models_data_path'] + key.lower() + '.pkl')


model = xgb.XGBClassifier()
pred = model.fit(df_train,y)
acc = accuracy_score(y, model.predict(df_train)) * 100
train_accuracies.update({'Stacked' : round(acc,1)})
joblib.dump(model, config.values['models_data_path'] + 'stacked.pkl')

print()
importances = pd.DataFrame({'feature':X.columns,'importance':np.round(clf_rf.feature_importances_,3)})
importances = importances[importances['importance']<0.01].sort_values('importance',ascending=False).set_index('feature')
if len(importances)>0:
    print("Consider removing the following features because they have a very low importance to the algorithm.")
    for index, row in importances.iterrows():
        print('["' + index + '",       	["drop_column"]],')

pp(train_accuracies)
