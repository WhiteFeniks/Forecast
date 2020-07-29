import numpy as np
import matplotlib.pyplot as plt
from reader import Reader
from regression import Regression
from sklearn.linear_model import LinearRegression


class Linear_regression(Regression):
    """Метод использующий линейную регрессию для предсказания данных"""
    def callculate(self):
        reg_linear = LinearRegression().fit(self.x_train, self.y_train)
        y_pred = reg_linear.predict(self.x_test)
        return y_pred


def main(window, Pk, P0, Ptest):
    count = 0
    k = (Pk - P0) / Ptest
    data_real = []
    data_lin = []
    while count < k:
        lin_price = list(Linear_regression(window, P0, Ptest, count).callculate())
        data_lin.extend(lin_price)
        data_real.extend(list(Linear_regression(window, P0, Ptest, count).y_test))
        count += 1
    return data_real, data_lin

window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01

data_real, data_lin = main(window, Pk, P0, Ptest)

fig, ax = plt.subplots()
ax.plot(data_real, label='Исходные данные')
ax.plot(data_lin, label='Данные с метода линейной регрессии')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена, (руб)')
ax.legend()
plt.show()