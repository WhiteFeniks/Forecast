from reader import Reader
from regression_neighbor import Nearest_neighbor
from regression_linear import Linear_regression
from regression_sma import Simple_moving_average


class UpDown_classifier:
    def __init__(self, data_real, data_reg):
        self.data_real = data_real
        self.data_reg = data_reg

    def calculate(self):
        count_true = 0
        for x, y in zip(self.data_real, self.data_reg):
            if x > 0 and y > 0:
                print("  True: real = {0:10.6f}, predicted = {1:10.6f}".format(x, y))
                count_true += 1
            elif x < 0 and y < 0:
                print("  True: real = {0:10.6f}, predicted = {1:10.6f}".format(x, y))
                count_true += 1
            else:
                print(" False: real = {0:10.6f}, predicted = {1:10.6f}".format(x, y))
        print("-------------------------------------------------")
        accuracy = count_true / (len(self.data_real) - 1)
        print("Accuracy =", accuracy)

    def calculate_sma(self):
        count = 0
        for x, y, z in zip(self.data_real[window:], self.data_reg, self.data_real[window - 1:]):
            if x > z and y > z:
                print("  True: before real = {0:10.6f}, predicted = {1:10.6f} and after "
                      "real = {2:10.6f} ".format(z, y, x))
                count += 1
            if x < z and y < z:
                print("  True: before real = {0:10.6f}, predicted = {1:10.6f} and after "
                      "real = {2:10.6f} ".format(z, y, x))
                count += 1
            else:
                print(" False: before real = {0:10.6f}, predicted = {1:10.6f} and after "
                      "real = {2:10.6f} ".format(z, y, x))
        print("------------------------------------------------------------------------------------")
        accuracy = count / (len(self.data_reg) - 1)
        print("SMA accuracy =", accuracy)

window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01
filename = 'USD000000TOD_1M_131001_131231.txt'

X_train, X_test, y_train, y_test = Reader(filename, Pk, P0, Ptest, window).train_test()
data_neigh, neigh_real = Nearest_neighbor(X_train, X_test, y_train, y_test).callculate()
data_lin, lin_real = Linear_regression(X_train, X_test, y_train, y_test).callculate()
# UpDown_classifier(neigh_real, data_neigh).calculate()
# UpDown_classifier(lin_real, data_lin).calculate()

data_real, time = Reader(filename, Pk, P0, Ptest, window).get_data()
data_sma = Simple_moving_average(X_train, X_test, y_train, y_test).callculate(data_real, window)
# UpDown_classifier(data_real, data_sma).calculate_sma()
