import numpy as np
import matplotlib.pyplot as plt
from reader import Reader
from sklearn.linear_model import LinearRegression


def get_price_data():
    """Метод получения данных по ценам за весь период."""
    x = Reader('USD000000TOD_1M_131001_131231.txt')
    data = x.read().splitlines()
    open_price = [data[1].split(',')[4]]
    close_column = [x.split(',')[7] for x in data][1:]
    combined = open_price + close_column
    result = [float(item) for item in combined]
    return result


dataset = get_price_data()

x = list()
for i in range(len(dataset)):
    if i + 10 < len(dataset):
        x += [dataset[i: i + 10]]
y = list()
for i in dataset[10:]:
    y += [i]

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)

result = model.predict(x)

print('predicted response:', result, sep='\n')
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# print(x)
# print(y)
# print(len(price))
# print(len(time))
# print(price)
# print(time)

"""Вывод на графики полученных данных"""

fig, ax = plt.subplots()
ax.text(10000, 10000, 'Coefficient of determination = 0.998696594488011')
ax.plot(dataset, label='Исходные данные')
ax.plot(result, label='Линейная регрессия')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена (руб)')
ax.legend()
plt.show()

# class Linear_regression:
#
#     def __call__(self, values, window=1):
#         """Метод подсчёта Simple moving average с помощью свертки и окна"."""
#         weights = np.repeat(1.0, window) / window
#         smas = np.convolve(values, weights, 'valid')
#         return smas
#
#     @staticmethod
#     def get_price_data():
#         """Метод получения данных по ценам за весь период."""
#         x = Reader('USD000000TOD_1M_131001_131231.txt')
#         data = x.read().splitlines()
#         open_price = [data[1].split(',')[4]]
#         close_column = [x.split(',')[7] for x in data][1:]
#         combined = open_price + close_column
#         result = [float(item) for item in combined]
#         return result
