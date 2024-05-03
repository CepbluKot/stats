import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import math

with open('Москва_2021.txt', 'r') as file:
    data = list(map(int, file.read().split('\n')))

sigma = np.std(data)
z = 1.96
delta = 3
n = (z * sigma / delta)**2
n = int(np.ceil(n))
sample_means = []
last_choice_select = []
for i in range(36):
    sample = np.random.choice(data, size=n, replace=True)
    last_choice_select = np.copy(sample)
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)
print("Размер выборки:", n)
print("Кол-во средних:", len(sample_means))
print("Выборочные средние:", sample_means)
# минимальное и максимальное значения выборочных средних
min_value = int(np.floor(min(sample_means)))
max_value = int(np.ceil(max(sample_means)))

# количество интервалов
num_intervals = max_value - min_value + 1
intervals = np.zeros(num_intervals)
for value in sample_means:
    index = int(value - min_value)
    intervals[index] += 1

# вычисляем относительные частоты
rel_freqs = intervals / len(sample_means)
print("Интервальный ряд распределения:")
for i in range(num_intervals):
    left = min_value + i
    right = min_value + i + 1
    print(f"[{left}; {right}): {intervals[i]} ({rel_freqs[i]:.2f})")


# вычисляем выборочные моменты
lastValueRelFreqs = rel_freqs[2]
sample_mean = np.mean(sample_means)
sample_var = np.var(sample_means)

# находим точечные оценки параметров нормального распределения методом моментов
mu = sample_mean
sigma = np.sqrt(sample_var)

# создаем массив значений для построения кривой Гаусса
x = np.linspace(min_value-1, max_value+1, 100)

result = []
for oneValue in x:
  exponentialStep = -((oneValue - mu) ** 2) / (2 * (sigma ** 2))
  distributionValue = (math.exp(exponentialStep)) / (sigma * ((2 * math.pi) ** 0.5))
  result.append(distributionValue)
  #  result.append(1/(sigma * (2*3.14)**0.5) * ((2.71) ** ((-1)*(oneValue - lastValueRelFreqs)/(2*(sigma ** 2)))))

# вычисляем значения плотности нормального распределения
y = norm.pdf(x, mu, sigma)

# нормируем значения плотности, чтобы они соответствовали относительным частотам
y *= len(sample_means) * (max_value - min_value + 1)
plt.bar(np.linspace(min_value-1, max_value+1, len(rel_freqs)),rel_freqs, label='Частоты')
plt.plot(x, result, 'r-', label='Гаусс')
plt.xlabel("Значения выборочной средней")
plt.ylabel("Относительная частота")
plt.title("Аппроксимация гистограммы кривой Гаусса")
plt.legend()
plt.show()
print("Точечные оценки параметров нормального распределения:")
print(f"mu = {mu:.2f}")
print(f"sigma = {sigma:.2f}")
t_value = t.ppf(1 - ((1 - 0.95) / 2), n - 1 )

c_sqrt = 0
for x in last_choice_select:
    c_sqrt += ((x - np.mean(last_choice_select)) ** 2) / (n - 1)
c_sqrt = c_sqrt ** 0.5
print(t_value)

margin_of_error = t_value * c_sqrt / (n ** 0.5)
c_interval = (np.mean(last_choice_select) - margin_of_error, np.mean(last_choice_select) + margin_of_error)
print(f'Исправленное СКО: {c_sqrt}')
print("Доверительный интервал для математического ожидания:", c_interval[0], c_interval[-1])
print("Cреднее выборки", np.mean(last_choice_select))
print("Точность: ", margin_of_error)
print(n)
