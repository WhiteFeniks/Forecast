import numpy as np
import matplotlib.pyplot as plt
from reader import Reader
from regression import Regression


class Simple_moving_average(Regression):
    """Метод подсчёта Simple moving average с помощью свертки и окна"""
    def callculate(self, values, window):
        weights = np.repeat(1.0, window) / window
        smas = np.convolve(values, weights)
        smas = list(smas[window - 1:])
        smas = smas[:len(smas) - window]
        return smas


window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01
filename = 'USD000000TOD_1M_131001_131231.txt'

data_real, time = Reader(filename, Pk, P0, Ptest, window).get_data()
X_train, X_test, y_train, y_test = Reader(filename, Pk, P0, Ptest, window).train_test()
y_pred = Simple_moving_average(X_train, X_test, y_train, y_test).callculate(data_real, window)
data_real = data_real[window:]

print("data_real", len(data_real), data_real)
print("data_sma", len(y_pred), y_pred)

fig, ax = plt.subplots()
ax.plot(data_real, label='Исходные данные')
ax.plot(y_pred, label='Данные с SMA метода')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена, (руб)')
ax.legend()
plt.show()