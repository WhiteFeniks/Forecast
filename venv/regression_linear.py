import numpy as np
import matplotlib.pyplot as plt
from reader import Reader
from sklearn.linear_model import LinearRegression


class Linear_regression:

    def __init__(self, dataset):
        self.dataset = dataset
        self.x = []
        self.y = []
        self.r_sq = 0.0
        self.result = []

    def x_preparation(self):
        """Для получения x в методе для обучения с учителем """
        for i in range(len(self.dataset)):
            if i + 10 < len(self.dataset):
                self.x += [self.dataset[i: i + 10]]
        return self.x

    def y_preparation(self):
        """Для получения y в методе для обучения с учителем """
        for i in dataset[10:]:
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
        open_price = [data[1].split(',')[4]]
        close_column = [x.split(',')[7] for x in data][1:]
        combined = open_price + close_column
        result = [float(item) for item in combined]
        return result


"""Выделение данных и получения данных линейной регрессии"""

dataset = Linear_regression.get_price_data()
new = Linear_regression(dataset)
r_sq, result = new.calculate()
print("Coefficient of determination = ", r_sq)

"""Вывод на графики полученных данных"""

fig, ax = plt.subplots()
ax.plot(dataset, label='Исходные данные')
ax.plot(result, label='Линейная регрессия')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена (руб)')
ax.legend()
plt.show()
