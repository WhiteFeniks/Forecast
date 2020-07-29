import numpy as np
import matplotlib.pyplot as plt
from reader import Reader
from regression import Regression
from sklearn.neighbors import KNeighborsRegressor


class Nearest_neighbor(Regression):
    """Метод использующий ближайших соседей для предсказания данных"""
    def callculate(self):
        reg_neighbor = KNeighborsRegressor(n_neighbors=3)
        reg_neighbor.fit(self.x_train, self.y_train)
        y_pred = reg_neighbor.predict(self.x_test)
        # print("Предсказанное значение:", *list(y_pred))
        return y_pred


def main(window, Pk, P0, Ptest):
    count = 0
    k = (Pk - P0) / Ptest
    data_real = []
    data_neigh = []
    while count < k:
        neighbor = list(Nearest_neighbor(window, P0, Ptest, count).callculate())
        data_neigh.extend(neighbor)
        data_real.extend(list(Nearest_neighbor(window, P0, Ptest, count).y_test))
        count += 1
    return data_real, data_neigh

window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01

data_real, data_neigh, = main(window, Pk, P0, Ptest)

# window = 3
# P0 = 7 / 11
# Ptest = 4/11
# neighbor = Nearest_neighbor(window, P0, Ptest, 0).callculate()

fig, ax = plt.subplots()
ax.plot(data_real, label='Исходные данные')
ax.plot(data_neigh, label='Данные с метода ближайших соседей')
ax.set_xlabel('Время (мин)')
ax.set_ylabel('Цена, (руб)')
ax.legend()
plt.show()