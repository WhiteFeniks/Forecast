import datetime as datetime
from reader import Reader

class Schedule:
    def __init__(self, window, Pk, P0, Ptest, experiment_num):
        self.window = window
        self.Pk = Pk
        self.P0 = P0
        self.Ptest = Ptest
        self.experiment_num = experiment_num
        self.k = (self.Pk - self.P0) / self.Ptest
        self.time = []
        self.price = []

    def get_datatime(self):
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_time = [x.split(',')[2] + x.split(',')[3] for x in data][1:]
        for i in close_time:
            self.time += [datetime.datetime(int(i[:4]), int(i[4:6]), int(i[6:8]), int(i[8:10]), int(i[10:12]))]
        self.time +=[self.time]
        return self.time

    def get_data_test(self):
        """Метод получения test выборки по ценам за весь период и перевод из абсолютного значения в относительные"""
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        # x = Reader('1.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        len_x_test = int(len(result) * Ptest)
        len_x_train = int(len(result) * P0)
        first_slise = len_x_train + (len_x_test * self.experiment_num)
        second_slise = len_x_train + len_x_test * (self.experiment_num + 1) + 1
        relative_price = result[first_slise: second_slise]
        return first_slise, second_slise, relative_price

    def get_price(self):
        x = Reader('USD000000TOD_1M_131001_131231.txt')
        data = x.read().splitlines()
        close_column = [x.split(',')[7] for x in data][1:]
        result = [float(item) for item in close_column]
        first_slise, second_slise, self.price = self.get_data_test()
        return first_slise, second_slise, self.price

    def calculate(self):
        datatime = self.get_datatime()
        first_slise, second_slise, price = self.get_price()
        length = len(datatime[first_slise:second_slise])
        begin = datatime[first_slise:second_slise][0]
        end = datatime[first_slise:second_slise][length - 1]
        # print(len(price), price)
        # print(len(datatime[first_slise:second_slise]), datatime[first_slise:second_slise])
        return begin, end


window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01
experiment_num = 5

begin, end = Schedule(window, Pk, P0, Ptest, experiment_num - 1).calculate()

print("Experiment number = ", experiment_num, "\n")
print("Begin period:", begin)
print("  End period:", end)