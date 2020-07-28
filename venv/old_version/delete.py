import math
from reader import Reader


def get_dataset(P0, Ptest, i):
    file = Reader('USD000000TOD_1M_131001_131231.txt')
    data = file.read().splitlines()
    get_column = [x.split(',')[7] for x in data][1:]
    dataset = [float(item) for item in get_column]
    len_test = int(len(dataset) * Ptest)
    len_train = int(len(dataset) * P0)
    train = dataset[:len_train]
    test = dataset[(len_train + (len_test * i)): len_train + len_test * (i + 1)]
    return train, test


def get_table(dataset, window):
    buf = []
    for i in range(len(dataset)):
        if i + window < len(dataset):
            buf += [dataset[i: i + window]]
    table = []
    for i in range(window + 1):
        table.append([])
    for i in range(len(buf)):
        if i != 0:
            table[window].append(buf[i][window - 1])
        for j in range(len(buf[i])):
            table[j].append(buf[i][j])
    for i in range(len(table)):
        table[i] = table[i][:len(dataset) - window - 1]
    return table


def standartization(lst):
    maximum = max(lst)
    minimum = min(lst)
    lst_standart = []
    for i in lst:
        lst_standart.append((i - minimum)/(maximum - minimum))
    return lst_standart, maximum, minimum


def table_standart(table):
    extrem_list = []
    for i in range(len(table) - 1):
        table[i], maximum, minimum = standartization(table[i])
        extrem_list.append([maximum, minimum])
    return table, extrem_list


def table_back(table):
    buf = []
    len_pack = len(table[0])
    for i in range(len(table)):
        buf.extend(table[i])
    new_table = []
    for i in range(len_pack):
        new_table.append([])
    j = 0
    for i in range(len(buf)):
        new_table[j].append(buf[i])
        j += 1
        if j == len_pack:
            j = 0
    return (new_table)


def standartization_test(lst, maximum, minimum):
    lst_standart = []
    for i in lst:
        lst_standart.append((i - minimum) / (maximum - minimum))
    return lst_standart

def test_standart(test, extrem_list):
    for i in range(len(test) -1):
        test[i] = standartization_test(test[i], extrem_list[i][0], extrem_list[i][1])
    return test

def distance_point(point_test, point_train):
    return (point_test-point_train)**2


def distance(test, train, neighbor):
    i = 0
    while i < len(test):
        j = 0
        buf_distance = []
        index_list = []
        while j < len(train):
            k = 0
            buf = []
            while k < int(len(test[i]) - 1):
                    buf.append(distance_point(test[i][k], train[j][k]))
                    k += 1
            index_list += [train[j][window]]
            buf_distance += [math.sqrt(sum(buf))]
            j += 1
            my_dict = dict(zip(buf_distance, index_list))
        i += 1
        # print("\nmy_dict", i, my_dict)
        list_keys = list(my_dict.keys())
        list_keys.sort()
        count = 0
        predict = 0
        for n in list_keys:
            if count == neighbor:
                break
            predict = predict + my_dict[n]
            count += 1
        predict = predict / neighbor
        return (predict)



# k = (Pk - P0)/Ptest
# k = 1
# i = 0


def main(P0, Ptest, window, i):
    train, test = get_dataset(P0, Ptest, i)  # разбиваем на тест и трейн

    train = get_table(train, window)  # разделение трейн на колонны
    test = get_table(test, window)  # разделение тест на колонны

    train, extrem_list = table_standart(train)  # стандартизация колонн трейн
    test = test_standart(test, extrem_list)  # стандартизация колонн тест

    train = table_back(train)  # сборка обратно в таблицу трейн
    test = table_back(test)  # сборка обратно в таблицу тест

    return train, test

Pk = 1
P0 = 0.1
Ptest = 0.01
window = 10
neighbor = 3
k = (Pk - P0) / Ptest
# k = 1
m = 0
while m < k:
    train, test = main(P0, Ptest, window, m)
    print("train", train)
    print("test", test)
    i = 0
    y = []
    for i in range(len(train)):
        y += [train[i][window]]
        train[i] = train[i][:-1]
    i = 0
    for i in range(len(test)):
        test[i] = test[i][:-1]
    print("y", y)
    print("train", train)
    print("test", test)
    from sklearn.neighbors import KNeighborsRegressor

    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(train, y)
    m += 1
    print("sklearn", list(model.predict(test)))
    print("--------------------------------------------------------------------------------------------------------------------------------")

