import seaborn as sns
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


class Statistics:
    def calculate(self, lst):
        arr = np.array(lst)
        moda = st.mode(arr)
        mediana = np.median(arr)
        average = np.mean(arr)
        sigma = np.std(arr)
        se = st.sem(arr)
        interval = st.norm.interval(0.95, loc=average, scale=se)
        return moda, mediana, average, sigma, se, interval


# test = [33.05, 33.17, 33.44, 32.97, 32.91, 32.99, 33.03, 33.38, 33.41, 33.27, 33.31, 33.18, 33.03, 32.95, 33.02]
test = [0.12, 0.27, -0.47, -0.06, 0.08, 0.04, 0.35, 0.03, -0.14, 0.04, -0.13, -0.15, -0.08, 0.07]

moda, mediana, average, sigma, se, interval = Statistics().calculate(test)

print("Moda = ", moda)
print("Mediana =", mediana)
print("Average = ", average)
print("Standard deviation =", sigma)
print("Standard error =", se)
print("Confidence_interval", interval)

sns_plot = sns.distplot(test)
plt.title('Гистограмма')
plt.xlabel('Время (мин)')
plt.ylabel('Цена, (руб)')
plt.show()
