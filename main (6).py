import numpy as np
import math
from scipy.stats import norm, chi2, f

with open('Москва_2021.txt') as file:
    data = [int(line.strip()) for line in file.readlines()]

arr_age = np.array(data, dtype=int)


#Расчет интервала возрастов и середина частичных интервалов
def calc_interval(max_data, min_data, num_group):
    size_data = np.ceil((max_data - min_data) / num_group)
    range_data = np.zeros((num_group, 2), dtype=int)
    middle_data_x = np.zeros(num_group, dtype=float)
    for i in range(s):
        range_data[i][0] = min_data + size_data * i
        range_data[i][1] = range_data[i][0] + size_data
        middle_data_x[i] = (range_data[i][0] + range_data[i][1]) / 2
    return range_data, middle_data_x


#Расчет наблюдаемых частот
def observed_frequencies(data, range_data, num_group):
    observed_freq_data = np.zeros(num_group, dtype=int)
    for i in range(num_group):
        for age in data:
            if range_data[i][0] <= age < range_data[i][1]:
                observed_freq_data[i] += 1
    return observed_freq_data


#нахождение среднего
def find_avg(middle_data_x, observed_freq_data, num_group):
    mean_data_x = 0
    for i in range(num_group):
        mean_data_x += middle_data_x[i] * observed_freq_data[i]
    mean_data_x /= sum(observed_freq_data)
    return mean_data_x


#СКО и Дисперсия
def find_dev_variance(middle_data_x, avr_data_x, observed_freq_data, num_group):
    dev_data = 0
    var_data = 0
    for i in range(num_group):
        var_data += (middle_data_x[i] - avr_data_x)**2 * observed_freq_data[i]
    var_data /= sum(observed_freq_data)
    dev_data = math.sqrt(var_data)
    return dev_data, var_data

#Нормирование случайной велечины и вычисление функции лапласа
def normalization_and_laplace(range_data, avr_data_x, dev_data, num_group):
    z_data = np.zeros((num_group, 2), dtype=float)
    F_data = np.zeros((num_group, 2), dtype=float)
    for i in range(num_group):
        if i == 0:
            z_data[i][0] = float('-inf')
            z_data[i][1] = (range_data[i][1] - avr_data_x) / dev_data
            F_data[i][0] = norm.cdf(z_data[i][0]) - 0.5
            F_data[i][1] = norm.cdf(z_data[i][1]) - 0.5
        elif i == s-1:
            z_data[i][0] = (range_data[i][0] - avr_data_x) / dev_data
            z_data[i][1] = float('inf')
            F_data[i][0] = norm.cdf(z_data[i][0]) - 0.5
            F_data[i][1] = norm.cdf(z_data[i][1]) - 0.5
        else:
            z_data[i][0] = (range_data[i][0] - avr_data_x) / dev_data
            z_data[i][1] = (range_data[i][1] - avr_data_x) / dev_data
            F_data[i][0] = norm.cdf(z_data[i][0]) - 0.5
            F_data[i][1] = norm.cdf(z_data[i][1]) - 0.5
    return z_data, F_data


#Вычисление вероятностей и теоретических частот
def prob_expected_freq(F_data, observed_freq_data, num_group):
    p_data = np.zeros(num_group, dtype=float)
    expected_freq_data = np.zeros(num_group, dtype=float)
    for i in range(num_group):
        p_data[i] = F_data[i][1] - F_data[i][0]
        expected_freq_data[i] = p_data[i] * sum(observed_freq_data)
    return p_data, expected_freq_data

if __name__ == '__main__':
    '''Задание 1 а'''
    #Минимальный возраст и максимальный
    min_age = min(arr_age)
    max_age = max(arr_age)

    #Количсество групп
    s = 7

    #Интервал возрастов и середины этих интервалов
    range_age, middle_age = calc_interval(max_age, min_age, s)

    #Наблюдаемые частоты
    observed_freq = observed_frequencies(arr_age, range_age, s)

    #Среднее
    mean_age = find_avg(middle_age, observed_freq, s)

    #СКО и Дисперсия
    dev_age, var_age = find_dev_variance(middle_age, mean_age, observed_freq, s)

    #Нормирование случайной величины (Интервал Z) и функция лапласа
    z, F = normalization_and_laplace(range_age, mean_age, dev_age, s)

    #Определение вероятностей и теоретических частот
    p, expected_freq = prob_expected_freq(F, observed_freq, s)


    #Критери Пирсона
    chi_squared_statistic = sum((observed_freq - expected_freq)**2 / expected_freq)
    alpha = 0.05
    k = s - 1 - 2 # -2 т.к. у нас два параметра
    critical_value = chi2.ppf(1 - alpha, k)
    print('\tВозраст')
    print(f"Критическое значение: {critical_value}")
    print(f"Критерий Пирсона: {chi_squared_statistic}")

    # Проверка нулевой гипотезы
    if chi_squared_statistic < critical_value:
        print("Нулевая гипотеза (о нормальном распределении) принимается")
    else:
        print("Нулевая гипотеза (о нормальном распределении) отвергается")


    '''Задание 1 Б'''
    gamma = 0.95
    delta = 3
    Z = 1.96  # для gamma = 0.95

    #Объем выборки
    n = math.ceil((Z * np.std(arr_age) / delta) ** 2)
    # Генерация выборок и расчет выборочных средних
    means = []
    for _ in range(36):
        sample = np.random.choice(arr_age, size=n, replace=True)
        means.append(np.mean(sample))

    min_mean = int(np.floor(np.min(means)))
    max_mean = int(np.ceil(np.max(means)))
    s = max_mean - min_mean
    # Интервал возрастов и середины этих интервалов
    range_age_mean, middle_age_mean = calc_interval(max_mean, min_mean, s)

    # Наблюдаемые частоты
    observed_freq_mean = observed_frequencies(means, range_age_mean, s)

    # Среднее
    mean_age_mean = find_avg(middle_age_mean, observed_freq_mean, s)

    # Дисперсия и СКО
    dev_age_mean, var_age_mean = find_dev_variance(middle_age_mean, mean_age_mean, observed_freq_mean, s)

    # Нормирование случайной величины (Интервал Z) и функция лапласа
    z_mean, F_mean = normalization_and_laplace(range_age_mean, mean_age_mean, dev_age_mean, s)

    # Определение вероятностей и теоретических частот
    p_mean, expected_freq_mean = prob_expected_freq(F_mean, observed_freq_mean, s)

    # Критери Пирсона
    chi_squared_statistic = sum((observed_freq_mean - expected_freq_mean) ** 2 / expected_freq_mean)
    alpha = 0.05
    k = s - 1 - 2
    critical_value = chi2.ppf(1 - alpha, k)
    print('\n\tСредный возраст')
    print(f"Критическое значение: {critical_value}")
    print(f"Критерий Пирсона: {chi_squared_statistic}")

    # Проверка нулевой гипотезы
    if chi_squared_statistic < critical_value:
        print("Нулевая гипотеза о нормальном распределении принимается")
    else:
        print("Нулевая гипотеза о нормальном распределении отвергается")


    '''Задание 2'''
    #Формируем две выборки объемом n
    sel1 = np.random.choice(arr_age, size=n, replace=True)
    sel2 = np.random.choice(arr_age, size=n, replace=True)

    dic_sel1 = {}
    dic_sel2 = {}
    for age in sel1:
        dic_sel1[age] = dic_sel1.get(age, 0) + 1

    for age in sel2:
        dic_sel2[age] = dic_sel2.get(age, 0) + 1

    #Нахождение средних
    avr_sel1 = sum((age * freq) for age, freq in dic_sel1.items()) / sum(dic_sel1.values())
    avr_sel2 = sum((age * freq) for age, freq in dic_sel2.items()) / sum(dic_sel2.values())

    #Нахождение дисперсии
    var_sel1 = (sum(((age - avr_sel1)**2 * freq) for age, freq in dic_sel1.items()) / (sum(dic_sel1.values()) - 1))
    var_sel2 = (sum(((age - avr_sel2)**2 * freq) for age, freq in dic_sel2.items()) / (sum(dic_sel2.values()) - 1))

    #Проверка гипотез
    F_observed = max(var_sel1, var_sel2) / min(var_sel1, var_sel2)

    #Степень свободы
    k1 = n - 1
    k2 = n - 1

    #конкурирующая гипотеза H1: D1 > D2
    alpha = 0.05
    F_critical = f.ppf(1 - alpha, k1, k2)
    print("\n\nПроверка нулевая гипотеза H0: D1 = D2 при при конкурирующей гипотезе H1: D1 > D2")
    print(f'Fнабл = {F_observed} \nFкр = {F_critical}')
    if F_observed < F_critical:
        print('Нет оснований отвергать нулевую гипотезу')
    else:
        print('Нулевая гипотеза отвергается')

    #конкурирующая гипотеза H1: D1 != D2
    F_critical = f.ppf(1-alpha, k1, k2)
    print("\n\nПроверка нулевая гипотеза H0: D1 = D2 при конкурирующей гипотезе H1: D1 != D2")
    print(f'Fнабл = {F_observed} \nFкр = {F_critical}')
    if F_observed < F_critical:
        print('Нет оснований отвергать нулевую гипотезу')
    else:
        print('Нулевая гипотеза отвергается')