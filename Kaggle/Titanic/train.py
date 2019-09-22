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
import pprint

df = pd.read_csv("data/converted/train.csv") 

Y = df['Survived']
X = df.drop('Survived', axis=1)

# Classifiers
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC(gamma='scale')
clf_perceptron = Perceptron()
clf_knn = KNeighborsClassifier()
clf_rf = RandomForestClassifier(n_estimators=100)

models = {
    "DecisionTree" : clf_tree,
    "SVM" : clf_svm,
    "Perceptron" : clf_perceptron,
    "KNN" : clf_knn,
    "RandomForest" : clf_rf,
}

df_test = pd.read_csv("data/converted/test.csv") 
p_test = pd.DataFrame(df_test['PassengerId'])
x_test = df_test.drop('PassengerId', axis=1)

accuracy = {}
for key, value in models.items():

    #Training the models
    value.fit(X, Y)
    
    #Testing and determining the accuracy using the training data
    pred = value.predict(X)
    acc = accuracy_score(Y, pred) * 100
    accuracy.update({key : round(acc,1)})
    
    #Creating predictions and saving results file
    p_test['Survived'] = value.predict(x_test)
    p_test.to_csv('data/output/results_' + key.lower() + '.csv', index=False)

print()
print("Listing of Importances of columns for the Random Forest Classifier")
importances = pd.DataFrame({'feature':X.columns,'importance':np.round(clf_rf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances.head(30))


print()
print("Accuracy of models on training data")
for key, value in sorted(accuracy.items(), key=lambda item: item[1]):
        print("%s:  %s" % (value, key))
