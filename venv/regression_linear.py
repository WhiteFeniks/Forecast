import numpy as np
import matplotlib.pyplot as plt
from reader import Reader
from regression import Regression
from sklearn.linear_model import LinearRegression


class Linear_regression(Regression):
    """Метод использующий линейную регрессию для предсказания данных"""
    def callculate(self):
        reg_linear = LinearRegression().fit(self.X_train, self.y_train)
        y_pred = []
        for i in range(len(X_test)):
            y_pred.extend(reg_linear.predict(self.X_test[i]))
        real = []
        for j in range(len(self.y_test)):
            real.extend(self.y_test[j])
        return y_pred, real


window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01
filename = 'USD000000TOD_1M_131001_131231.txt'

X_train, X_test, y_train, y_test = Reader(filename, Pk, P0, Ptest, window).train_test()
y_pred, real = Linear_regression(X_train, X_test, y_train, y_test).callculate()

print("data_real", len(real), real)
print("data_lin", len(y_pred), y_pred)



fig, ax = plt.subplots()
ax.plot(real, label='Исходные данные')
ax.plot(y_pred, label='Данные с метода ближайших соседей')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена, (руб)')
ax.legend()
plt.show()