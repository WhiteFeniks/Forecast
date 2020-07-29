import numpy as np
import matplotlib.pyplot as plt
from reader import Reader
from regression import Regression


class Simple_moving_average(Regression):
    def callculate(self):
        """Метод подсчёта Simple moving average с помощью свертки и окна"""
        values = self.get_price_data()
        weights = np.repeat(1.0, self.window) / self.window
        smas = np.convolve(values, weights)
        smas = list(smas[self.window - 1:])
        smas = smas[:len(smas) - self.window]
        return smas


window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01

data_sma = list(Simple_moving_average(window, P0, Ptest, 0).callculate())
data_real = list(Regression(window, P0, Ptest, 0).get_real())

fig, ax = plt.subplots()
ax.plot(data_real, label='Исходные данные')
ax.plot(data_sma, label='Данные с SMA метода')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена, (руб)')
ax.legend()
plt.show()