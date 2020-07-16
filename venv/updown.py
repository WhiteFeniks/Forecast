from regression_neighbor import Nearest_neighbor
from regression_linear import Linear_regression
from regression_sma import Simple_moving_average
from pprint import pprint
from reader import Reader


class UpDown_classifier:
    def __init__(self, real, reg_result, window=1):
        self.real = real
        self.reg_result = reg_result
        self.window = window

    def calculate(self):
        """Метод сравнения реальной цены с предсказанной """
        for x, y in zip(self.real[self.window:], self.reg_result):
            if x <= y:
                print(" True: real {0:6.4f} <= predicted {1:6.4f} ".format(x, y))
            else:
                print("False: real {0:6.4f} <= predicted {1:6.4f} ".format(x, y))


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
        print("{:^41}".format("SMA method"))
        print("-----------------------------------------")
        result.calculate()
        print("-----------------------------------------\n\n\n\n\n")

    def method_reg_lin(self):
        """ Для Linear regression """
        window = 10
        data_real = Linear_regression.get_price_data()
        new = Linear_regression(data_real, window)
        r_sq, data_linear = new.calculate()
        result = UpDown_classifier(data_real, data_linear, window)
        print("{:^41}".format("Linear regression method"))
        print("-----------------------------------------")
        result.calculate()
        print("-----------------------------------------\n\n\n\n\n")

    def method_neighbor(self):
        """ Для Nearest neighbor """
        window = 10
        data_real = Nearest_neighbor.get_price_data()
        new = Nearest_neighbor(data_real, window)
        data_neighbor = new.calculate()
        result = UpDown_classifier(data_real, data_neighbor, window)
        print("{:^41}".format("Nearest neighbor method"))
        print("-----------------------------------------")
        result.calculate()
        print("-----------------------------------------\n\n\n\n\n")


print("Choose method number: \n1. Simple Moving Average\n2. "
      "Linear Regression\n3. Nearest classifier\n\nEnter the number(from 1 to 3): ", )
method = int(input())
x = Regression(method)
x.make()
