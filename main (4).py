import numpy as np
from scipy.stats import chi2
import math
from scipy import stats

s = 7
alpha = 0.05
k = s - 1
critical_value = chi2.ppf(1 - alpha, k)
print(f"Критическое значение для α = {alpha} и {k} степеней свободы: {critical_value}")
# Загрузка данных из файла
with open('/home/oleg/Documents/stats_lecs/wrong/Москва_2021.txt', 'r') as f:
    lines = f.readlines()
arr = np.array(lines, dtype=int)

# Разбиение данных на 7 групп с равными интервалами
s = 7
min_val = np.min(arr)
max_val = np.max(arr)
raz = -(max_val - min_val)
print(raz)
if raz % s != 0:
    raz = -raz
    raz = -raz%s + raz

interval_size = int(raz / s)


intervals = [min_val + i * interval_size for i in range(s+1)]

print(intervals)

observed_freq, _ = np.histogram(arr, bins=intervals)

sr = np.zeros(s, dtype=int)
for i, _ in enumerate(sr):
    if i < len(intervals) - 1:
        sr[i] = (intervals[i] + interval_size / 2 - 1)
    else:
        sr[i] = (intervals[i] + interval_size / 2)

print(sr)
expected_freq = int(len(arr) / s)

print(observed_freq)

sr_ = 0
for i in range(s):
    sr_ += observed_freq[i] + sr[i]
sr_ /= s
print(sr_)

sko = 0
for i in range(s):
    sko += observed_freq[i] + (sr[i] - sr_)**2
sko = math.sqrt(sko)
sko /= (s - 1)
print(sko)

low_x_up = np.zeros((s, 2), dtype=int)
for i, x in enumerate(intervals):
    if i > 0 and i < len(low_x_up):
        low_x_up[i][0] = low_x_up[i-1][1]
        low_x_up[i][1] = intervals[i]


print(expected_freq)
expected_freq = expected_freq - sko
print(expected_freq)
chi_squared_statistic = np.sum(((observed_freq - expected_freq)**2) / expected_freq)


# Задаем уровень значимости и количество степеней свободы
# alpha = 0.05
# k = s - 1

# Вычисляем критическое значение
# critical_value = chi2.ppf(1 - alpha, k)
print('Возраст')
# print(f"Критическое значение для α = {alpha} и {k} степеней свободы: {critical_value}")
print(f"Критерий Пирсона: {chi_squared_statistic}")

# Проверка нулевой гипотезы
if chi_squared_statistic < critical_value:
    print("Нулевая гипотеза (о нормальном распределении) принимается")
else:
    print("Нулевая гипотеза (о нормальном распределении) отвергается")


#Б
# Заданные значения
gamma = 0.95
delta = 3
Z = 1.96  # для gamma = 0.95


# Расчет размера выборки
n = math.ceil((Z * np.std(arr) / delta) ** 2)

# Генерация выборок и расчет выборочных средних
means = []
for _ in range(36):
    sample = np.random.choice(arr, size=n, replace=True)
    means.append(np.mean(sample))

# Разбиение выборочных средних на интервалы
min_val = np.min(means)
max_val = np.max(means)
interval_size = (max_val - min_val) / s
intervals = [min_val + i * interval_size for i in range(s)]
intervals.append(max_val)

# Рассчитываем частоту вхождения каждого интервала выборочных средних
observed_freq, _ = np.histogram(means, bins=intervals)

# Ожидаемое количество значений в каждом интервале при равномерном распределении
expected_freq = len(means) / s

# Вычисляем критерий Пирсона
chi_squared_statistic = np.sum((observed_freq - expected_freq)**2 / expected_freq)

# Рассчитываем критическое значение для уровня значимости 0.05 и (s - 1) степеней свободы
df = s - 1
critical_value = chi2.ppf(1 - alpha, df)
print('\n \nСредний возраст')
print(f"Критическое значение для α = {alpha} и {k} степеней свободы: {critical_value}")
print(f"Критерий Пирсона: {chi_squared_statistic}")

if chi_squared_statistic < critical_value:
    print("Нулевая гипотеза (о нормальном распределении) принимается")
else:
    print("Нулевая гипотеза (о нормальном распределении) отвергается")




n1 = 50
n2 = 50

# Генерация двух выборок
sample1 = np.random.choice(arr, size=n1, replace=True)
sample2 = np.random.choice(arr, size=n2, replace=True)

# Вычисление выборочных дисперсий
variance1 = np.var(sample1, ddof=1)
variance2 = np.var(sample2, ddof=1)

# Вычисление статистики Фишера
F_statistic = variance1 / variance2 if variance1 > variance2 else variance2 / variance1

# Рассчитываем критическое значение для уровня значимости 0.05 и соответствующих степеней свободы
df1 = n1 - 1
df2 = n2 - 1
critical_value = stats.f.ppf(1 - alpha / 2, df1, df2)

print("\n\n")
# Проверка альтернативных гипотез
if F_statistic > critical_value:
    print("Альтернативная гипотеза (H1: D1 > D2) принимается")
else:
    print("Альтернативная гипотеза (H1: D1 > D2) отвергается")

if F_statistic > critical_value or F_statistic < 1 / critical_value:
    print("Альтернативная гипотеза (H1: D1 != D2) принимается")
else:
    print("Альтернативная гипотеза (H1: D1 != D2) отвергается")


