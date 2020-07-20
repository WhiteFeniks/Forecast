import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reader import Reader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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

def z_preparation(dataset, window):
    z = []
    for i in dataset[window - 1:]:
        z += [i]
    return z

dataset = get_price_data()

window = 10
X = x_preparation(dataset, window)

y = y_preparation(dataset, window)

z = z_preparation(dataset, window)


slice = int(len(X) * 0.8)

X_train = X[:slice]
X_test = X[slice:]

y_train = y[:slice]
y_test = y[slice:]

z_train = z[:slice]
z_test = z[slice:]

# print("X_train =", X_train)
# print("y_train =", y_train)
# print("z_train =", z_train)
#
# print("X_test =", X_test)
# print("y_test =", y_test)
# print("z_test =", z_test)


"""Линейная регрессия"""
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)

y_pred = reg_all.predict(X_test)

# print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("Root Mean Squared Error: {}".format(rmse))


# """Вывод на графики полученных данных"""
# fig, ax = plt.subplots()
# print("исходные данные: ", y_test)
# print("предсказанные данные : ", list(y_pred))
# ax.plot(y_test, label='Исходные данные')
# ax.plot(y_pred, label='Линейная регрессия')
# ax.set_xlabel('Время (мин)')
# ax.set_ylabel('Цена (руб)')
# ax.legend()
# plt.show()

print("z_test", len(z_test))
print("y_test", len(y_test))
print("y_pred", len(list(y_pred)))

pred_true = 0
for ten, pred, real in zip(z_test, y_pred, y_test):
    if ten < pred and ten < real:
        pred_true += 1
        print(" True: before real = {0:10.6f}, predicted = {1:10.6f} and after "
              "real = {2:10.6f} ".format(ten, pred, real))
    elif ten > pred and ten > real:
        pred_true += 1
        print(" True: before real = {0:10.6f}, predicted = {1:10.6f} and after "
              "real = {2:10.6f} ".format(ten, pred, real))
    else:
        print("False: before real = {0:10.6f}, predicted = {1:10.6f} and after "
              "real = {2:10.6f} ".format(ten, pred, real))
accuracy = pred_true/len(y_test)
print("Accuracy =", accuracy)

# print("dataset =", dataset)
# print("X =", X)
# print("X[0] =", X[0])
# print("X[1] =", X[1])
# print("X[7614] =", X[7614])
# print("len(X) =", len(X))
# print("y =", y)
# print("len(y) =", len(y))
# print("Slice =", slice)
# print("X_train =", X_train)
# print("X_test =", X_test)
print("y_train =", len(y_train))
print("y_test =", len(y_test))




