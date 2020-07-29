import numpy as np
import matplotlib.pyplot as plt
from reader import Reader


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

    def get_real(self):
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        return result

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


window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01

