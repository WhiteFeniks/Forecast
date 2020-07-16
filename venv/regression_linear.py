import numpy as np
import matplotlib.pyplot as plt
import datetime
from reader import Reader
from sklearn.linear_model import LinearRegression


class Linear_regression:
    def __init__(self, dataset, window=1):
        self.dataset = dataset
        self.window = window
        self.x = []
        self.y = []
        self.r_sq = 0.0
        self.result = []

    def x_preparation(self):
        """Для получения x в методе для обучения с учителем """
        for i in range(len(self.dataset)):
            if i + self.window < len(self.dataset):
                self.x += [self.dataset[i: i + self.window]]
        return self.x

    def y_preparation(self):
        """Для получения y в методе для обучения с учителем """
        for i in dataset[self.window:]:
            self.y += [i]
        return self.y

    def calculate(self):
        """Метод подсчета данных для линейной регрессии."""
        x = self.x_preparation()
        y = self.y_preparation()
        model = LinearRegression().fit(x, y)
        self.r_sq = model.score(x, y)
        self.result = model.predict(x)
        return self.r_sq, self.result

    @staticmethod
    def get_price_data():
        """Метод получения данных по ценам за весь период."""
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        return result

    @staticmethod
    def get_price_time():
        """Метод получения данных времени за весь период."""
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_time = [x.split(',')[2] + x.split(',')[3] for x in data][1:]
        time = []
        for i in close_time:
            time += [datetime.datetime(int(i[:4]), int(i[4:6]), int(i[6:8]), int(i[8:10]), int(i[10:12]))]
        time +=[time]
        return time


"""Выделение данных и получения данных линейной регрессии"""
window = 10
dataset = Linear_regression.get_price_data()
time = Linear_regression.get_price_time()
new = Linear_regression(dataset, window)
r_sq, result = new.calculate()
# print("Coefficient of determination = ", r_sq)

"""Вывод на графики полученных данных"""
# ax.plot(time[:len(dataset)], dataset, label='Исходные данные')
# ax.plot(time[:len(result)], result, label='Линейная регрессия')

# fig, ax = plt.subplots()
# ax.plot(dataset, label='Исходные данные')
# ax.plot(result, label='Линейная регрессия')
# ax.set_xlabel('Время (мин)')
# ax.set_ylabel('Цена (руб)')
# ax.legend()
# plt.show()
