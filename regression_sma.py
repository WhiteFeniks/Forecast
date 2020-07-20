import numpy as np
import matplotlib.pyplot as plt
import datetime
from reader import Reader


class Simple_moving_average:
    def __call__(self, values, window=1):
        """Метод подсчёта Simple moving average с помощью свертки и окна."""
        weights = np.repeat(1.0, window) / window
        smas = np.convolve(values, weights)
        smas = list(smas[9:])
        smas = smas[:7615]
        return smas

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
        return time


"""Исходные данные"""
dataset = Simple_moving_average.get_price_data()
time = Simple_moving_average.get_price_time()

"""SMA данные"""
window = 10
sma = Simple_moving_average()
result = sma(dataset, window)


"""Вывод на графики полученных данных"""
# fig, ax = plt.subplots()
# ax.plot(dataset, label='Исходные данные')
# ax.plot(result, label='SMA данные')
# ax.set_xlabel('Время (мин)')
# ax.set_ylabel('Цена (руб)')
# ax.legend()
# plt.show()