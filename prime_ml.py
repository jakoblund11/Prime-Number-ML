from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier
import numpy as np

def prime_range(start, end):
    assert start > 1, "start must be larger than 1"
    result = []
    for tal in range(start, end + 1):
        prime = True
        for tal2 in range(2, int(tal ** 0.5) + 1):
            if (tal % tal2) == 0:
                prime = False
                break

        if prime:
            result.append(tal)
    return result


first_num = 2
last_num = 1000000

number_line = np.array([x for x in range(first_num, last_num + 1)])

prime_number_line = np.array(prime_range(first_num, last_num + 1))

# following is true/false list of whether a number is prime or not starting from 2 to end
if not last_num == 1000000:
    true_false = [item in prime_number_line for item in tqdm(number_line)]

with open('Primes_true_false.txt', 'r') as f:
    if last_num == 1000000:
        true_false = f.read()
        lst = true_false[1:-1].split(', ')
        lst = [True if x == 'True' else False for x in lst]
        true_false = lst
    pass


train_size = 0.5
X_train, X_test, y_train, y_test = train_test_split(number_line.reshape(-1, 1), true_false, train_size=train_size,
                                                    test_size=1-train_size, shuffle=False)

#clf = LogisticRegression().fit(X_train, y_train) # Accuracy of 0.921845
#clf = LinearRegression().fit(X_train, y_train) # Accuracy of 0.0005095
#clf = KernelRidge().fit(X_train, y_train) # Requires 4.66 TiB RAM
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 5, 2)).fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", clf.score(X_test, y_test))

# Find the indices of the data that the model got wrong
wrong_indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]

# Print the true labels and predicted labels for the data that the model got wrong
print("Wrong numbers:", [X_test[i][0] for i in wrong_indices])
#print("True labels:", [y_test[i] for i in wrong_indices])
print("Number of numbers being falsely labeled as false:", sum([y_pred[i] for i in wrong_indices]))
print("This means the model always predicts a number as a non-prime")
print("Number of wrong predictions:", len(wrong_indices))
print("Number of primes:", len(prime_number_line))
