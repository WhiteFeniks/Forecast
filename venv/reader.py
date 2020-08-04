import datetime as dt


class Reader:

    def __init__(self, filename, Pk, P0, Ptest, window):
        self.filename = filename
        self.Pk = Pk
        self.P0 = P0
        self.Ptest = Ptest
        self.window = window

    def read(self):
        try:
            file = open(self.filename)
            return file.read()
        except IOError:
            return "File not found"

    def get_data(self):
        file = self.read().splitlines()[1:]
        price_close = [float(x.split(',')[7]) for x in file]
        time = []
        for i in [x.split(',')[2] + x.split(',')[3] for x in file]:
            time += [dt.datetime(int(i[:4]), int(i[4:6]), int(i[6:8]), int(i[8:10]), int(i[10:12]))]

        # print('Price', len(price_close), price_close)
        # print('DataTime', len(time), time)

        return price_close, time

    @staticmethod
    def make_relative(lst):
        relative_price = []
        for i in range(len(lst)):
            if i + 1 < len(lst):
                relative_price.append(lst[i + 1] - lst[i])
        return relative_price

    def get_pack(self, lst):
        res = []
        for i in range(len(lst)):
            if i + self.window < len(lst):
                res.append(lst[i: i + self.window])
        return res

    def make_pack_test(self, pack_test):
        k = (self.Pk - self.P0) / self.Ptest
        test_lst = []
        for i in range(int(k)):
            test_lst += [pack_test[(int(len(pack_test)/k)) * i: int(len(pack_test)/k) * (i + 1) - self.window + 1]]
        return test_lst

    def get_y_test(self, X_test, relative_test):
        relative_test = relative_test[:len(X_test) * (len(X_test[0]) + self.window - 1) + 1]
        y_test = []
        for i in range(len(X_test)):
            y_test.append(relative_test[(len(X_test[0]) - 1 + self.window) * i + self.window:
                                        (len(X_test[0]) - 1 + self.window) * (i + 1) + 1])
        return y_test

    def get_y_train(self, X_train, relative_train):
        y_train = relative_train[self.window:]
        return y_train

    def for_sma(self):
        data, time = self.get_data()
        pack_data = self.get_pack(data)
        return pack_data


    def train_test(self):
        data, time = self.get_data()
        absolut_train = data[:int(len(data) * self.P0)]
        absolut_test = data[int(len(data) * self.P0):]
        relative_train = Reader.make_relative(absolut_train)
        relative_test = Reader.make_relative(absolut_test)
        pack_train = self.get_pack(relative_train)
        pack_test = self.get_pack(relative_test)
        X_train = pack_train
        X_test = self.make_pack_test(pack_test)
        y_test = self.get_y_test(X_test, relative_test)
        y_train = self.get_y_train(X_train, relative_train)


        # print('Price', len(data), data)
        # print('DataTime', len(time), time)
        # print('absolut_train', len(absolut_train), absolut_train)
        # print('absolut_test', len(absolut_test), absolut_test)
        # print('relative_train', len(relative_train), relative_train)
        # print('relative_test', len(relative_test), relative_test)
        # for i in pack_train:
        #     print(i)
        # print('\n\n\n')
        # for j in pack_test:
        #     print(j)
        # print('pack_train', len(pack_train), pack_train)
        # print('pack_test', len(pack_test), pack_test)

        return X_train, X_test, y_train, y_test

window = 10
Pk = 1
P0 = 0.1
Ptest = 0.01
filename = 'USD000000TOD_1M_131001_131231.txt'

x = Reader(filename, Pk, P0, Ptest, window)