from sklearn.neighbors import KNeighborsRegressor

X = [[65.75],
     [71.52],
     [69.40],
     [68.22],
     [67.79],
     [68.70],
     [69.80],
     [70.01],
     [67.90],
     [66.49],
    ]
y = [112.99, 136.49, 153.03, 142.34, 144.30, 123.30, 141.49, 136.46, 112.37, 127.45]
model = KNeighborsRegressor(n_neighbors=3, )
model.fit(X, y)

print(*model.predict([[60]]))