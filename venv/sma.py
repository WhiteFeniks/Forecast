# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# product = {'month': [1,2,3,4,5,6,7,8,9,10,11,12], 'demand': [290,260,288,300,310,303,329,340,316,330,308,310]}
# df = pd.DataFrame(product)
# print(df.head())
#
# for i in range(0,df.shape[0]-2):
#     df.loc[df.index[i+2],'SMA_3'] = np.round(((df.iloc[i,1]+ df.iloc[i+1,1] +df.iloc[i+2,1])/3),1)
#
# print(df.head())
#
# df['pandas_SMA_3'] = df.iloc[:,1].rolling(window=3).mean()
# print(df.head())
#
# for i in range(0,df.shape[0]-3):
#     df.loc[df.index[i+3],'SMA_4'] = np.round(((df.iloc[i,1]+ df.iloc[i+1,1] +df.iloc[i+2,1]+df.iloc[i+3,1])/4),1)
# print(df.head())
#
# df['pandas_SMA_4'] = df.iloc[:,1].rolling(window=4).mean()
# print(df.head())
#
#
#
# dataset = [290,260,288,300,310,303,329,340,316,330,308,310]
# window = 3
# weights = np.repeat(1.0, window) / window
# smas = np.convolve(dataset, weights, 'valid')
# print(smas)
#
# plt.grid(True)
# plt.plot(df['demand'],label='data')
# plt.plot(smas, label='My realization')
# plt.plot(df['SMA_3'],label='SMA 3 Months')
# plt.plot(df['SMA_4'],label='SMA 4 Months')
# plt.legend(loc=2)
# plt.show()
#
#
