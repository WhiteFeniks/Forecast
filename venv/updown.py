from regression_neighbor import Nearest_neighbor
from regression_linear import Linear_regression
from regression_sma import Simple_moving_average
from pprint import pprint
from reader import Reader


class UpDown_classifier:
    def __init__(self, real, reg_result, window=1):
        self.real = real
        self.reg_result = reg_result
        self.window = window - 1
        self.count_true = 0

    def calculate(self):
        """Метод сравнения реальной цены с предсказанной """
        for x, y, z in zip(self.real[self.window:], self.reg_result, self.real[self.window + 1:]):
            if x < y and x < z:
                print("  True: before real = {0:10.6f}, predicted = {1:10.6f} and after "
                      "real = {2:10.6f} ".format(x, y, z))
                self.count_true += 1
            else:
                print(" False: before real = {0:10.6f}, predicted = {1:10.6f} and after "
                      "real = {2:10.6f} ".format(x, y, z))
        print("------------------------------------------------------------------------------------")
        accuracy = self.count_true / (len(self.reg_result) - 1)
        print("Accuracy =", accuracy)


class Regression:
    def __init__(self, method):
        self.method = method

    def make(self):
        if self.method == 1:
            self.method_sma()
        elif self.method == 2:
            self.method_reg_lin()
        elif self.method == 3:
            self.method_neighbor()
        else:
            print("There is no such method number!")

    def method_sma(self):
        """ Для SMA """
        window = 10
        data_real = Simple_moving_average.get_price_data()
        sma = Simple_moving_average()
        data_sma = list(sma(data_real, window))
        result = UpDown_classifier(data_real, data_sma, window)
        print("{:^84}".format("SMA method"))
        print("------------------------------------------------------------------------------------")
        result.calculate()

    def method_reg_lin(self):
        """ Для Linear regression """
        window = 10
        data_real = Linear_regression.get_price_data()
        new = Linear_regression(data_real, window)
        r_sq, data_linear = new.calculate()
        result = UpDown_classifier(data_real, data_linear, window)
        print("{:^84}".format("Linear regression method"))
        print("------------------------------------------------------------------------------------")
        result.calculate()

    def method_neighbor(self):
        """ Для Nearest neighbor """
        window = 10
        data_real = Nearest_neighbor.get_price_data()
        new = Nearest_neighbor(data_real, window)
        data_neighbor = new.calculate()
        result = UpDown_classifier(data_real, data_neighbor, window)
        print("{:^84}".format("Nearest neighbor method"))
        print("------------------------------------------------------------------------------------")
        result.calculate()



print("Choose method number: \n1. Simple Moving Average\n2. "
      "Linear Regression\n3. Nearest classifier\n\nEnter the number(from 1 to 3): ", )
method = int(input())
x = Regression(method)
x.make()

