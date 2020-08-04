import datetime as datetime
from reader import Reader

class Schedule:
    def __init__(self, data_real, time, experiment_num):
        self.data_real = data_real
        self.time = time
        self.experiment_num = experiment_num

    def calculate(self):
        first_slise = int(len(data_real) * P0) + ((experiment_num - 1) * int(len(data_real) * Ptest))
        second_slise = int(len(data_real) * P0) + (experiment_num * int(len(data_real) * Ptest))
        length = len(time[first_slise:second_slise])
        begin = time[first_slise:second_slise][0]
        end = time[first_slise:second_slise][length - 1]
        return begin, end


window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01
experiment_num = 2
filename = 'USD000000TOD_1M_131001_131231.txt'

data_real, time = Reader(filename, Pk, P0, Ptest, window).get_data()
begin, end = Schedule( data_real, time, experiment_num - 1).calculate()

print("Experiment number = ", experiment_num, "\n")
print("Begin period:", begin)
print("  End period:", end)
