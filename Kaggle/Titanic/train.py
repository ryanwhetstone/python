from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.read_csv("data/converted/train.csv") 
df.info()

Y = df['Survived']
X = df.drop('Survived', axis=1)

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC(gamma='scale')
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()
clf_RF = RandomForestClassifier(n_estimators=100)

# # Training the models
clf_tree.fit(X, Y)
clf_svm.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_KNN.fit(X, Y)
clf_RF.fit(X, Y)

#Testing using the same data
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = clf_svm.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = clf_perceptron.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = clf_KNN.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

pred_RF = clf_RF.predict(X)
acc_RF = accuracy_score(Y, pred_RF) * 100
print('Accuracy for RF: {}'.format(acc_RF))

# The best classifier from svm, per, KNN
index = np.argmax([acc_svm, acc_per, acc_KNN, acc_tree, acc_RF])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN', 3: 'Decision Tree', 4: 'Random Forest'}
print('Best classifier is {}'.format(classifiers[index]))


importances = pd.DataFrame({'feature':X.columns,'importance':np.round(clf_RF.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances.head(15))

df = pd.read_csv("data/converted/test.csv") 

P = pd.DataFrame(df['PassengerId'])
X = df.drop('PassengerId', axis=1)

pred_tree = clf_KNN.predict(X)

P['Survived'] = pred_tree

P.to_csv('results.csv', index=False)