import random
import numpy as np
import math

def activation(z):  # Сигмоидальная функция
    return float(1)/(1 + math.exp(-z))

def create(Weight_count):
    w = np.array([])
    for _ in range(0, Weight_count):
        w = np.append(w, random.uniform(-1, 1))
    return w

def trainmass(x, y, w, Inputs_number, n, iter):
    error = np.zeros(iter)
    for i in range(iter):
        for j in range(len(x)):
            w, err = train(x[j], y[j], w, Inputs_number, n)
            error[i] += err ** 2
        error[i] *= 0.5
        print(f"Итерация {i}, погрешность = {error[i]}")
    return w, error

def train(x, y, w, Inputs_number, n):
    a = 0  # Сумма произведения весов и входных нейронов
    for i in range(Inputs_number):
        a += + w[i] * x[i]
    y_predict = activation(a)  # Значение функции активации

    error = y - y_predict  # Погрешность с реальным значением y
    errorreturn = abs(error)

    global_delta = (y_predict - y) * y_predict * (1 - y_predict)

    for i in range(Inputs_number):
        w[i] += -n * global_delta * x[i]
    return w, errorreturn


def single_perceptron(n, x, y, epochs):
    Inputs_number = 2
    w = create(Inputs_number)
    w, error = trainmass(x, y, w, Inputs_number, n, epochs)
    return w, error

#print(f"Weights array:")
#print(w)

#for i in range(4):
#    for j in range(Inputs_number):
#        a = a + w[j] * x[i][j]
#    y_predict = activation(a)  # Значение функции активации
#    print(f"Predict {i} = {y_predict}")