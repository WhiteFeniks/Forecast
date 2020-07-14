import numpy as np
import matplotlib.pyplot as plt
from reader import Reader


class Simple_moving_average:

    def __call__(self, values, window=1):
        """Метод подсчёта Simple moving average с помощью свертки и окна"."""
        weights = np.repeat(1.0, window) / window
        smas = np.convolve(values, weights, 'valid')
        return smas

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


"""Исходные данные"""
dataset = Simple_moving_average.get_price_data()

"""SMA данные"""
window = 3
sma = Simple_moving_average()
result = sma(dataset, window)

"""Вывод на графики полученных данных"""
fig, ax = plt.subplots()
ax.plot(dataset, label='Исходные данные')
ax.plot(result, label='SMA данные')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена (руб)')
ax.legend()
plt.show()
