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
import numpy as np
import pandas as pd

# Read the data from the CSV file into a DataFrame
data = pd.read_csv('data.csv', nrows=1000)

# Extract the features and target variable from the DataFrame
#X = data.drop(['Modulo 2', 'Modulo 3', 'Modulo 5', 'Modulo 7', 'Modulo 11', 'Modulo 13', 'Sum of digits', 'Prime'], axis=1)
X = data.drop(['Prime'], axis=1)
y = data['Prime']

train_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                    test_size=1 - train_size, shuffle=False)

#clf = LogisticRegression().fit(X_train, y_train) # Accuracy of 0.921845
#clf = LinearRegression().fit(X_train, y_train) # Accuracy of 0.0005095
#clf = KernelRidge().fit(X_train, y_train) # Requires 4.66 TiB RAM
#clf = BayesianGaussianMixture().fit(X_train, y_train)
clf = LinearSVC(max_iter=1000000000).fit(X_train, y_train) # Does not predict everything to be non-prime! That is a great start
#clf = LinearSVR(max_iter=1000000000).fit(X_train, y_train)
#clf = MLPClassifier(solver='adam', activation='relu', alpha=1e-5, hidden_layer_sizes=(5, 2)).fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", clf.score(X_test, y_test))


# Find the indices of the data that the model got wrong
wrong_indices = [i for i in range(len(y_test)) if y_test.iloc[i] != y_pred[i]]

# Print the true labels and predicted labels for the data that the model got wrong
#print("Wrong numbers:", [X_test.iloc[i][0] for i in wrong_indices])
#print("True labels:", [y_test[i] for i in wrong_indices])
print("Number of numbers being falsely labeled as false:", sum([y_pred[i] for i in wrong_indices]))
print("This means the model always predicts a number as a non-prime")
print("Number of wrong predictions:", len(wrong_indices))
print("Number of primes:", sum(data.Prime))
print("Wrong predictions divided by total number of primes:", len(wrong_indices)/sum(data.Prime))
print('AUC-ROC:', roc_auc_score(y_test, y_pred))

#nrows=1000 gives Accuracy: 0.86625, Number of numbers being falsely labeled as false: 9, and AUC-ROC: 0.591723487596112