import numpy as np
import matplotlib.pyplot as plt
import datetime
from reader import Reader
from sklearn.linear_model import LinearRegression

window = 10


def get_price_data():
    """Метод получения данных по ценам за весь период."""
    x = Reader('USD000000TOD_1M_131001_131231.txt')
    data = x.read().splitlines()
    close_column = [x.split(',')[7] for x in data][1:]
    result = [float(item) for item in close_column]
    return result


def x_preparation():
    """Для получения x в методе для обучения с учителем """
    x = []
    for i in range(len(dataset)):
        if i + window < len(dataset):
            x += [dataset[i: i + window]]
    return x


def y_preparation():
    """Для получения y в методе для обучения с учителем """
    y = []
    for i in dataset[window:]:
        y += [i]
    return y


dataset = get_price_data()
print(dataset)
x = x_preparation()
y = y_preparation()
print("x[0] = ", x[0])
print("x[1] = ", x[1])
print("x[2] = ", x[2])
print("x[3] = ", x[3])
print("x[4] = ", x[4])
print("y = ", y)


class Schedule:
    def __init__(self, dataset, window):
        dataset = dataset
        window = window

