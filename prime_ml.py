from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# Read the data from the CSV file into a DataFrame
data = pd.read_csv('data.csv', nrows=1000000)

# Extract the features and target variable from the DataFrame
X = data.drop(['Sum of digits', 'Prime'], axis=1)
y = data['Prime']

train_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                    test_size=1 - train_size, shuffle=False)

#clf = LinearSVC(max_iter=1000000000).fit(X_train, y_train) # Does not predict everything to be non-prime! That is a great start
clf = DecisionTreeClassifier().fit(X_train, y_train)
#clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(25, 10, 5, 2)).fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", clf.score(X_test, y_test))

# Find the indices of the data that the model got wrong
wrong_indices = [i for i in range(len(y_test)) if y_test.iloc[i] != y_pred[i]]

print("Number of numbers being falsely labeled as false:", sum([y_pred[i] for i in wrong_indices]))
print("Number of wrong predictions:", len(wrong_indices))
print("Number of primes:", sum(data.Prime))
print('AUC-ROC:', roc_auc_score(y_test, y_pred))

#nrows=1000 gives Accuracy: 0.86625, Number of numbers being falsely labeled as false: 9, and AUC-ROC: 0.591723487596112