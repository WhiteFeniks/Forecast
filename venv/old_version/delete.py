import numpy as np
from reader import Reader
from sklearn.model_selection import TimeSeriesSplit


def get_price_data():
    """Метод получения данных по ценам за весь период."""
    x = Reader('USD000000TOD_1M_131001_131231.txt')
    data = x.read().splitlines()
    close_column = [x.split(',')[7] for x in data][1:]
    result = [float(item) for item in close_column]
    return result


def x_preparation(dataset, window):
    x = []
    for i in range(len(dataset)):
        if i + window < len(dataset):
            x.append(dataset[i: i + window])
    return x


def y_preparation(dataset, window):
    y = []
    for i in dataset[window:]:
        y += [i]
    return y


window = 10
dataset = get_price_data()
X = x_preparation(dataset, window)
y = y_preparation(dataset, window)
print("X =", X)
print("y =", y)
X = np.array(X).T

print("X[0]", len(X[0]))
print("X[1]", X[1])
print("X[2]", X[2])
print("X[3]", X[3])
y = np.array(y)

tscv = TimeSeriesSplit(n_splits=9)
print("-------------------------------------------------------------------------------------------------------------------------")

for train_index, test_index in tscv.split(X):
    print("\nTrain_index:", train_index, "Test_index:", test_index, "\n")
    X_train, X_test = X[train_index], X[test_index]
    print("X_train = ", X_train, "\nX_test = ", X_test)
    y_train, y_test = y[train_index], y[test_index]
    print("len(y_train) = ", len(y_train), "len(y_test) = ", len(y_test))
    print("-------------------------------------------------------------------------------------------------------------------------")

