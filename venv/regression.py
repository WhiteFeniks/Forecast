import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reader import Reader
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


class Regression:
    def __init__(self, window, P0, Ptest, count):
        self.window = window
        self.P0 = P0
        self.Ptest = Ptest
        self.count = count
        self.data_train = self.get_data_train()
        self.x_train = self.get_x_train()
        self.y_train = self.get_y_train()
        self.data_test = self.get_data_test()
        self.x_test = self.get_x_test()
        self.y_test = self.get_y_test()

    def get_data_train(self):
        """Метод получения train выборки по ценам за весь период и перевод из абсолютного значения в относительные"""
        # x = Reader('USD000000TOD_1M_131001_131231.txt')
        x = Reader('1.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        relative_price = result[:int(len(result)*P0)]
        otnosit_price = []
        for i in range(len(relative_price)):
            if i + 1 < len(relative_price):
                otnosit_price.append(relative_price[i + 1] - relative_price[i])
        print(otnosit_price)
        return otnosit_price

    def get_data_test(self):
        """Метод получения test выборки по ценам за весь период и перевод из абсолютного значения в относительные"""
        # x = Reader('USD000000TOD_1M_131001_131231.txt')
        x = Reader('1.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        len_x_test = int(len(result) * Ptest)
        len_x_train = int(len(result) * P0)
        relative_price = result[(len_x_train + (len_x_test * self.count)): len_x_train + len_x_test * (self.count + 1) + 1]
        otnosit_price = []
        for i in range(len(relative_price)):
            if i + 1 < len(relative_price):
                otnosit_price.append(relative_price[i + 1] - relative_price[i])
        print(otnosit_price)
        return otnosit_price

    def get_x_train(self):
        """Метод разбивки train выборки на пачки"""
        x = []
        for i in range(len(self.data_train)):
            if i + self.window < len(self.data_train):
                x.append(self.data_train[i: i + self.window])
        return x

    def get_y_train(self):
        """Метод получения реальных train значений"""
        y = []
        for i in self.data_train[self.window:]:
            y += [i]
        return y

    def get_x_test(self):
        """Метод разбивки test выборки на пачки"""
        x = []
        for i in range(len(self.data_test)):
            if i + self.window < len(self.data_test):
                x.append(self.data_test[i: i + self.window])
        return x

    def get_y_test(self):
        """Метод получения реальных test значений"""
        y = []
        for i in self.data_test[self.window:]:
            y += [i]
        return y


class Linear_regression(Regression):
    """Метод использующий линейную регрессию для предсказания данных"""
    def callculate(self):
        reg_linear = LinearRegression().fit(self.x_train, self.y_train)
        y_pred = reg_linear.predict(self.x_test)
        return y_pred


class Nearest_neighbor(Regression):
    """Метод использующий ближайших соседей для предсказания данных"""
    def callculate(self):
        reg_neighbor = KNeighborsRegressor(n_neighbors=3)
        reg_neighbor.fit(self.x_train, self.y_train)
        y_pred = reg_neighbor.predict(self.x_test)
        print("Предсказанное значение:", *list(y_pred))
        return y_pred


def main(window, Pk, P0, Ptest):
    count = 0
    k = (Pk - P0) / Ptest
    data_real = []
    data_neigh = []
    data_lin = []
    while count < k:
        lin_price = list(Linear_regression(window, P0, Ptest, count).callculate())
        neighbor = list(Nearest_neighbor(window, P0, Ptest, count).callculate())
        data_neigh.extend(neighbor)
        data_lin.extend(lin_price)
        data_real.extend(list(Linear_regression(window, P0, Ptest, count).y_test))
        count += 1
    return data_real, data_lin, data_neigh


window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01

data_real, data_lin, data_neigh = main(window, Pk, P0, Ptest)


# window = 3
# P0 = 7 / 11
# Ptest = 4/11
# neighbor = Nearest_neighbor(window, P0, Ptest, 0).callculate()

""" Вывод на графики полученных данных """

fig, ax = plt.subplots()
ax.plot(data_real, label='Исходные данные')
ax.plot(data_neigh, label='Данные с метода ближайших соседей')
ax.plot(data_lin, label='Данные с метода линейной регрессии')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена, (руб)')
ax.legend()
plt.show()
