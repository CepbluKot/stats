import numpy as np
import math
from scipy import stats

# Задание данных
years = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])
x1_arr = np.array([4922.4, 4130.7, 4137.4, 3889.4, 4263.9, 4243.5, 3966.5, 3657, 3461.2, 4316, 3624.6, 2986.1])
x2_arr = np.array([23369, 26629, 29792, 32495, 34030, 36709, 39167, 43724, 47867, 51344, 57244, 65338])
x3_arr = np.array([1865.9, 1807.9, 1749.5, 1690, 1577, 1444.5, 1304.6, 1208.6, 1126.7, 1102.8, 1077.7, 1051.4])
x4_arr = np.array([24951.2, 14648.1, 39558.7, 32365, 46568.8, 27929.6, 37218.5, 51418.1, 116166.5, 126304.8, 159875.4, 197047])
x5_arr = np.array([319.8, 294.9, 295.9, 270, 245.4, 261.9, 211.9, 124.9, 285.1, 106.5, 429.9, 61.9])
y_arr = np.array([2404.8, 2302.2, 2206.2, 2190.6, 2388.5, 2160.1, 2058.5, 1991.5, 2024.3, 2044.2, 2004.4, 1966.8])

# Формирование матрицы C
C = np.zeros((len(x1_arr), 6), dtype=float)

for i, _ in enumerate(x1_arr):
    for j in range(6):
        if j == 0:
            C[i][j] = 1
        if j == 1:
            C[i][j] = x1_arr[i]
        if j == 2:
            C[i][j] = x2_arr[i]
        if j == 3:
            C[i][j] = x3_arr[i]
        if j == 4:
            C[i][j] = x4_arr[i]
        if j == 5:
            C[i][j] = x5_arr[i]


x1 = x1_arr.transpose()
x2 = x2_arr.transpose()
x3 = x3_arr.transpose()
x4 = x4_arr.transpose()
x5 = x5_arr.transpose()
y = y_arr.transpose()


# Транспонирование матрицы C
transposed_C = C.transpose()


# Вычисление C^T * C
CTC = np.dot(transposed_C, C)

# Вычисление (C^T * C)^(-1)
CTC_inv = np.linalg.inv(CTC)

# Вычисление Q
Q_1 = np.dot(CTC_inv, transposed_C)
Q = np.dot(Q_1, y)
Q_round = np.round(Q, 3)
print(f'Задание 1\nМнк-оценка: {Q_round[0]} {Q_round[1]} {Q_round[2]} '
      f'{Q_round[3]} {Q_round[4]} {Q_round[5]}')


#Задание 2
#Уравнение модели
equation_model = 'y = '
for i, q in enumerate(Q):
    if i == 0:
        equation_model += f'{q}'
    else:
        equation_model += f' + {q} * x{i}'

#Вектор прогноза y^ = Q0 + Q1x1 + Q2x2 + .. + Qmxm
n = len(x1_arr)
y_predict = np.zeros(n)

for i in range(n):
    y_predict[i] = Q[0] + Q[1] * x1_arr[i] + Q[2] * x2_arr[i] + Q[3] * x3_arr[i] + Q[4] * x4_arr[i] + Q[5] * x5_arr[i]

#Вектор остатков Xi = y - y^
Xi = np.zeros(n)
for i in range(n):
    Xi[i] = y[i] - y_predict[i]

#Вектор отклонений Delta = y - y_mean
Delta = np.zeros(n)
y_mean = sum(y_predict) / len(y_predict)
for i in range(n):
    Delta[i] = y[i] - y_mean

#Коэффициент детерминации R (2 способа)
#Способ 1: R = 1 - (Xi^T * Xi) / (Delta^T * Delta)
R = 1 - (np.dot(Xi.transpose(), Xi) / np.dot(Delta.transpose(), Delta))

#Способ 2: R = 1 - S_e / S_y
#Число степеней свободы: факторная, остаточная, общая
k_fact = 5 #Т.к. 5 переменные x
k_e = len(years) - 5 - 1 #5 - переменных x
k = len(years) - 1

#Сумма квадратов отклонения
#Сумма квадратов отклонений, объясненная регрессией
S_fact = sum((y_pre - y_mean)**2 for y_pre in y_predict)
#Сумма квадратов отклонений (необъясненная)
S_e = sum((y_i - y_pre)**2 for y_i, y_pre in zip(y_arr, y_predict))
#Общая сумма
S_y = sum((y_i - y_mean)**2 for y_i in y_arr)

#Средний квадрат отклонений на одну степень свободы
S_fact_2 = S_fact / k_fact
S_e_2 = S_e / k_e
S_y_2 = S_y / k

#Коэффициент детерминации R
R_2 = 1 - S_e / S_y

#Нормированный коэффициент детерминации R_adj = 1 - S_e_2 / S_y_2
R_adj = 1 - S_e_2 / S_y_2



print(f'\n\nЗадание 2\nУравнение модели: {equation_model}')
print(f'\n\nВектор прогноза:\n{y_predict}')
print(f'\n\nВектор остатков:\n{Xi}')
print(f'\n\nВектор отклонений:\n{Delta}')
print(f'\n\nКоэффициент детерминации: {R}')
print(f'Нормированный коэффициент детерминации: {R_adj}')


#Задание 3
#Корреляционный анализ
#коэф корреляции
def correlation_coe(data_1, data_2):
    # Нахождение корреляционного момента cov_xy = ср(xy) - ср(x) * ср(y))
    avr_1_2 = (sum((i_1 * i_2) for i_1, i_2 in zip(data_1, data_2)) / n)
    avr_1 = sum(data_1) / n
    avr_2 = sum(data_2) / n
    cov_1_2 = avr_1_2 - avr_1 * avr_2
    # Нахождение СКО(x) СКО(y)
    std_1 = math.sqrt(sum((i_2 - avr_2) ** 2 for i_2 in data_2) / n)
    std_2 = math.sqrt(sum((i_1 - avr_1) ** 2 for i_1 in data_1) / n)
    r = cov_1_2 / (std_1 * std_2)
    return r


all_variables = [x1_arr, x2_arr, x3_arr, x4_arr, x5_arr, y_arr]
correlation_matrix = np.zeros((len(all_variables), len(all_variables)))

for i in range(len(all_variables)):
    for j in range(len(all_variables)):
        r = correlation_coe(all_variables[i], all_variables[j])
        correlation_matrix[i][j] = r

correlation_matrix_1 = np.zeros((len(all_variables)-1, len(all_variables)-1))
for i in range(len(all_variables) - 1):
    for j in range(len(all_variables) - 1):
        correlation_matrix_1[i][j] = correlation_matrix[i][j]


#Проверка на мультиколлинеарность (находим определитель матрицы)
determinant = np.linalg.det(correlation_matrix_1)
print('\n\nЗадание 3. Исследовать модель на мультиколлинеарность'
      '\nМатрица парных корреляций')
print(correlation_matrix)
print(f'\ndet Rx = {determinant}')

#Задание 4
alpha = 0.05

# Проверка статистической значимости уравнения регрессии (F-тест)
F = S_fact_2 / S_e_2
F_critical = stats.f.ppf(1 - alpha, k_fact, k_e)

print('\n\nЗадание 4')
print('Статистическая значимость уравнения регрессииv\n'
      'Нулевая гипотеза H0: Dфакт = De')
print(f'Наблюдаемое значение критерия: {F}')
print(f'Критическая точка: {F_critical}')
if F > F_critical:
    print('\nУравнение регрессии является статистически значимым')
else:
    print('\nУравнение регрессии не является статистически значимым')


# Проверка статистической значимости коэффициентов регрессии (t-тест)
t_critical = stats.t.ppf(1 - alpha / 2, k_e)

print('\n\nИсследование значимости коэффициентов регрессии')
print(t_critical)
for i in range(len(Q)):
    T = Q[i] / (math.sqrt(S_e_2 * CTC_inv[i][i]))
    if abs(T) > t_critical:
        print(f"Коэффициент Q[{i}] является статистически значимым")
    else:
        print(f"Коэффициент Q[{i}] не является статистически значимым")

#Задание 5
#шаговый регрессионный анализ для улучшения
# модели путем удаления факторов.
#Введем две перемнные significance_equ и significance_coe
# и удаление будте продолжаться пока они не будут True
print('\n\nЗадание 5')
all_variables_x_dic = {1: x1_arr, 2: x2_arr, 3: x3_arr,
                     4: x4_arr, 5: x5_arr}
count = 0
while True:

    C = np.zeros((len(years), len(all_variables_x_dic) + 1),
                 dtype=float)
    C[:, 0] = 1
    for i in range(len(years)):
        for j, data in enumerate(all_variables_x_dic.values()):
            C[i][j + 1] = data[i]

    # Транспонирование матрицы C
    transposed_C = C.transpose()

    # Вычисление C^T * C
    CTC = np.dot(transposed_C, C)

    # Вычисление (C^T * C)^(-1)
    CTC_inv = np.linalg.inv(CTC)

    # Вычисление Q
    Q = np.dot(np.dot(CTC_inv, transposed_C), y_arr)
    Q_round = np.round(Q, 4)

    # Проверка статистической значимости уравнения регрессии (F-тест)
    y_predict = np.dot(C, Q)
    y_mean = sum(y_predict) / len(y_predict)

    # Степени свободы
    k_fact = len(all_variables_x_dic)
    k_e = len(years) - k_fact - 1
    k = len(years) - 1

    # Сумма квадратов отклонения
    # Сумма квадратов отклонений, объясненная регрессией
    S_fact = sum((y_pre - y_mean) ** 2 for y_pre in y_predict)
    # Сумма квадратов отклонений (необъясненная)
    S_e = sum((y_i - y_pre) ** 2 for y_i, y_pre in zip(y_arr, y_predict))
    # Общая сумма
    S_y = sum((y_i - y_mean) ** 2 for y_i in y_arr)

    # Средний квадрат отклонений на одну степень свободы
    S_fact_2 = S_fact / k_fact
    S_e_2 = S_e / k_e
    S_y_2 = S_y / k

    F = S_fact_2 / S_e_2
    F_critical = stats.f.ppf(1 - alpha, S_fact, S_e)



    # Проверка статистической значимости коэффициентов регрессии (t-тест)
    t_critical = stats.t.ppf(1 - alpha / 2, k_e)
    T = Q / np.sqrt(S_e_2 * np.diag(CTC_inv))

    significant_coe = np.abs(T) > t_critical

    if F > F_critical and all(significant_coe):
        break

    p_values = 2 * (1 - stats.t.cdf(np.abs(T), k_e))
    p_values_dic = {}
    for p_val_i, x_kay in zip(range(1, len(p_values)), all_variables_x_dic.keys()):
        p_values_dic[x_kay] = p_values[p_val_i]

    print(f'\nШаг {count}')
    text = f'{Q_round[0]}'
    for key, Q_i in zip(all_variables_x_dic.keys(), Q_round[1:]):
        text += f' + {Q_i} * x{key}'
    print(text)

    max_x = max(p_values_dic, key=lambda x: p_values_dic[x])

    all_variables_x_dic.pop(max_x)
    print(f'Удаляем переменную x{max_x} с p-значение: {p_values_dic[max_x]}')

    count += 1


text = f'{Q_round[0]}'
for key, Q_i in zip(all_variables_x_dic.keys(), Q_round[1:]):
    text += f' + {Q_i} * x{key}'
print(f'\nУлучшенная модель: {text}')




#Задание 6
def detect_inter(avr_1, data1, data2):
    # Определяем x/y i и x/y i+1
    avr_1 = round(avr_1, 0)
    for i in range(len(data1) - 1):
        if data1[i] >= avr_1 >= data1[i + 1] or data1[i] <= avr_1 <= data1[i + 1]:
            x_inter = [data1[i], data1[i + 1]]
            y_inter = [data2[i], data2[i+1]]
            break
    return x_inter, y_inter

structures = {'y = ax + b': [],
              'y = ax^b': [],
              'y = ab^x': np.zeros(4),
              'y = a + b/x': np.zeros(4),
              'y = 1/(ax+b)': np.zeros(4),
              'y = x/(ax+b)': np.zeros(4),
              'y = a ln x + b': np.zeros(4)}

print('\n\nЗадание 6')
for key, val in all_variables_x_dic.items():
    print(f'Экспериментальные точки:\n'
          f'x{key}: \n {val}')
    print(f'\ny:\n{y_arr}')
    # Выбираем три точки: начальную, конечную, промежуточную
    start_xy = [val[0], y_arr[0]]
    end_xy = [val[-1], y_arr[-1]]
    for i, (struc, data) in enumerate(structures.items()):
        print(f'\n{i+1}: {struc}')
        if i == 0:
            avr_x = (start_xy[0] + end_xy[0]) / 2
            avr_y = (start_xy[1] + end_xy[1]) / 2
            x_s = avr_x
        if i == 1:
            avr_x = math.sqrt(start_xy[0] * end_xy[0])
            avr_y = math.sqrt(start_xy[1] * end_xy[1])
            x_s = avr_x
        if i == 2:
            avr_x = (start_xy[0] + end_xy[0]) / 2
            avr_y = math.sqrt(start_xy[1] * end_xy[1])
            x_s = avr_x
        if i == 3:
            avr_x = (2 * start_xy[0] * end_xy[0]) / (start_xy[0] + end_xy[0])
            avr_y = (start_xy[1] + end_xy[1]) / 2
            x_s = avr_x
        if i == 4:
            avr_x = (start_xy[0] + end_xy[0]) / 2
            avr_y = (2 * start_xy[1] * end_xy[1]) / (start_xy[1] + end_xy[1])
            x_s = avr_x
        if i == 5:
            avr_x = (2 * start_xy[0] * end_xy[0]) / (start_xy[0] + end_xy[0])
            avr_y = (2 * start_xy[1] * end_xy[1]) / (start_xy[1] + end_xy[1])
            x_s = avr_x
        if i == 6:
            avr_x = math.sqrt(start_xy[0] * end_xy[0])
            avr_y = (start_xy[1] + end_xy[1]) / 2
            x_s = avr_x

        x_inter, y_inter = detect_inter(avr_x, val, y_arr)
        if round(avr_x, 0) in val:
            index_i = np.where(val == round(avr_x, 0))[0][0]
            y_s = y_arr[index_i]
        else:
            y_s = y_inter[0] + (y_inter[1] - y_inter[0]) / (x_inter[1] - x_inter[0]) * (x_s - x_inter[0])
        structures[struc] = [round(num, 3) for num in [avr_x, avr_y, y_s, abs(avr_y - y_s)]]

        print(structures[struc])


print('\n\nРезультат структурной идентификации')
for i in range(1):
    min_key = min(structures, key=lambda k: structures[k][-1])
    print(f'{min_key}:  {structures[min_key]}')

coe_a_b = np.zeros(2)
avr_ay = sum((y_i * x_i) for y_i, x_i in zip(y_arr, x3_arr))/ len(years)
avr_y = sum(y_arr)/ len(years)
avr_x = sum(x3_arr) / len(years)
coe_a_b[-1] = (avr_ay - avr_y * avr_x) / np.var(x3_arr)

coe_a_b[0] = avr_y - coe_a_b[-1]*avr_x
print(f'\nКэффициенты из исходного уравнения: {text}')
print([round(num, 4) for num in Q_round])
print(f'\nКоэффициенты найденные для выбранного уравнения при помощи структурной идентификации:')
min_key = min(structures, key=lambda k: structures[k][-1])
print(f'{min_key}')
print([round(num, 4) for num in coe_a_b])
