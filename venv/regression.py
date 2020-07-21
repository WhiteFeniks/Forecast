import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reader import Reader
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

class Regression:
    def __init__(self, window, P0, Ptest, i):
        self.window = window
        self.P0 = P0
        self.Ptest = Ptest
        self.i = i
        self.data_train = self.get_data_train()
        self.x_train = self.get_x_train()
        self.y_train = self.get_y_train()
        self.data_test = self.get_data_test()
        self.x_test = self.get_x_test()
        self.y_test = self.get_y_test()
        # print("data_test", self.data_test)

    def get_data_train(self):
        """Метод получения данных по ценам за весь период."""
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        relative_price = []
        i = 0
        while i < len(result) - 1:
            relative_price += [result[i + 1] - result[i]]
            i += 1
        relative_price = relative_price[:int(len(relative_price)*P0)]
        return relative_price

    def get_data_test(self):
        """Метод получения данных по ценам за весь период."""
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        relative_price = []
        i = 0
        while i < len(result) - 1:
            relative_price += [result[i + 1] - result[i]]
            i += 1
        len_x_test = int(len(relative_price) * Ptest)
        len_x_train = int(len(relative_price) * P0)
        relative_price = relative_price[(len_x_train + (len_x_test * self.i)): len_x_train + len_x_test * (self.i + 1)]
        return relative_price

    def get_x_train(self):
        x = []
        for i in range(len(self.data_train)):
            if i + self.window < len(self.data_train):
                x.append(self.data_train[i: i + self.window])
        return x

    def get_y_train(self):
        y = []
        for i in self.data_train[self.window:]:
            y += [i]
        return y

    def get_x_test(self):
        x = []
        for i in range(len(self.data_test)):
            if i + self.window < len(self.data_test):
                x.append(self.data_test[i: i + self.window])
        return x

    def get_y_test(self):
        y = []
        for i in self.data_test[self.window:]:
            y += [i]
        return y


# class Simple_moving_average(Regression):
#     def callculate(self):
#         weights = np.repeat(1.0, self.window) / self.window
#         y_pred = np.convolve(self.data_train, weights)
#         return y_pred

# class Linear_regression(Regression):
#     def callculate(self):
#         reg_linear = LinearRegression().fit(self.x_train, self.y_train)
#         y_pred = reg_linear.predict(self.x_test)
#         return y_pred


class Nearest_neighbor(Regression):
    def callculate(self):
        reg_neighbor = KNeighborsRegressor(n_neighbors=window)
        reg_neighbor.fit(self.x_train, self.y_train)
        y_pred = reg_neighbor.predict(self.x_test)
        return y_pred

window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01
k = (Pk - P0)/Ptest
i = 0
y_real = []
y_pred = []
while i < k:
    # next_price = list(Linear_regression(window, P0, Ptest, i).callculate())
    next_price = Nearest_neighbor(window, P0, Ptest, i).callculate()
    y_pred.extend(next_price)
    # y_real.extend(list(Linear_regression(window, P0, Ptest, i).y_test))
    y_real.extend(Nearest_neighbor(window, P0, Ptest, i).y_test)
    # print("test", i + 1, list(next_price))
    # print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    i += 1


# def all_test():
#     """Метод получения данных по ценам за весь период."""
#     x = Reader('USD000000TOD_1M_131001_131231.txt')
#     data = x.read().splitlines()
#     close_column = [x.split(',')[7] for x in data][1:]
#     result = [float(item) for item in close_column]
#     result = result[int(len(result)*P0) + 10:]
#     return result


# dataset = all_test()[:5940]
dataset = y_real
print(len(dataset), dataset)
print(len(y_pred), y_pred)

# for t in zip(dataset, y_pred):
#     print("{0:10.6f} {1:10.6}".format(t[0], t[1]))


# dataset = Simple_moving_average(window, P0, Ptest, i).y_train
# y_pred = list(Simple_moving_average(window, P0, Ptest, i).callculate())
# y_pred = y_pred[window - 1: len(y_pred) + 1 - window]


"""Вывод на графики полученных данных"""
fig, ax = plt.subplots()
ax.plot(dataset, label='Исходные данные')
# ax.plot(y_pred, label='Данные с линейной регрессии')
ax.plot(y_pred, label='Данные с метода ближайших соседей')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена (руб)')
ax.legend()
plt.show()
