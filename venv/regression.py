import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
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
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        # x = Reader('1.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        relative_price = result[:int(len(result)*P0)]
        otnosit_price = []
        for i in range(len(relative_price)):
            if i + 1 < len(relative_price):
                otnosit_price.append(relative_price[i + 1] - relative_price[i])
        # print(otnosit_price)
        return otnosit_price

    def get_data_test(self):
        """Метод получения test выборки по ценам за весь период и перевод из абсолютного значения в относительные"""
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        # x = Reader('1.txt')
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
        # print(otnosit_price)
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

    @staticmethod
    def get_price_data():
        """Метод получения данных по ценам за весь период."""
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        return result


class Simple_moving_average(Regression):
    def callculate(self):
        """Метод подсчёта Simple moving average с помощью свертки и окна"""
        values = self.get_price_data()
        weights = np.repeat(1.0, self.window) / self.window
        smas = np.convolve(values, weights)
        smas = list(smas[self.window - 1:])
        smas = smas[:len(smas) - self.window]
        return smas


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
        # print("Предсказанное значение:", *list(y_pred))
        return y_pred


class Schedule:
    def __init__(self, window, Pk, P0, Ptest, experiment_num):
        self.window = window
        self.Pk = Pk
        self.P0 = P0
        self.Ptest = Ptest
        self.experiment_num = experiment_num
        self.k = (self.Pk - self.P0) / self.Ptest
        self.time = []
        self.price = []

    def get_datatime(self):
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_time = [x.split(',')[2] + x.split(',')[3] for x in data][1:]
        for i in close_time:
            self.time += [datetime.datetime(int(i[:4]), int(i[4:6]), int(i[6:8]), int(i[8:10]), int(i[10:12]))]
        self.time +=[self.time]
        return self.time

    def get_data_test(self):
        """Метод получения test выборки по ценам за весь период и перевод из абсолютного значения в относительные"""
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        # x = Reader('1.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        len_x_test = int(len(result) * Ptest)
        len_x_train = int(len(result) * P0)
        first_slise = len_x_train + (len_x_test * self.experiment_num)
        second_slise = len_x_train + len_x_test * (self.experiment_num + 1) + 1
        relative_price = result[first_slise: second_slise]
        return first_slise, second_slise, relative_price

    def get_price(self):
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        first_slise, second_slise, self.price = self.get_data_test()
        return first_slise, second_slise, self.price

    def calculate(self):
        datatime = self.get_datatime()
        first_slise, second_slise, price = self.get_price()
        length = len(datatime[first_slise:second_slise])
        begin = datatime[first_slise:second_slise][0]
        end = datatime[first_slise:second_slise][length - 1]
        # print(len(price), price)
        # print(len(datatime[first_slise:second_slise]), datatime[first_slise:second_slise])
        return begin, end


def main(window, Pk, P0, Ptest):
    count = 0
    k = (Pk - P0) / Ptest
    data_real = []
    data_neigh = []
    data_lin = []
    data_sma = list(Simple_moving_average(window, P0, Ptest, count).callculate())
    while count < k:
        lin_price = list(Linear_regression(window, P0, Ptest, count).callculate())
        neighbor = list(Nearest_neighbor(window, P0, Ptest, count).callculate())
        data_neigh.extend(neighbor)
        data_lin.extend(lin_price)
        data_real.extend(list(Linear_regression(window, P0, Ptest, count).y_test))
        count += 1
    return data_real, data_lin, data_neigh, data_sma


window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01
experiment_num = 5

data_real, data_lin, data_neigh, data_sma= main(window, Pk, P0, Ptest)
begin, end = Schedule(window, Pk, P0, Ptest, experiment_num - 1).calculate()


# print("Experiment number = ", experiment_num, "\n")
# print("Begin period:", begin)
# print("  End period:", end)

# window = 3
# P0 = 7 / 11
# Ptest = 4/11
# neighbor = Nearest_neighbor(window, P0, Ptest, 0).callculate()

#
# """ Вывод на графики полученных данных """
#
# fig, ax = plt.subplots()
# ax.plot(data_real, label='Исходные данные')
# ax.plot(data_neigh, label='Данные с метода ближайших соседей')
# ax.plot(data_lin, label='Данные с метода линейной регрессии')
# ax.plot(data_sma, label='Данные с SMA метода ')
# ax.set_xlabel('Время (мин)')
# ax.set_ylabel('Цена, (руб)')
# ax.legend()
# plt.show()
