import numpy as np
import random
import math


# многослойный персептрон. кол-во слоев = 1. N - кол-во нейронов на этом слое

def sigmoid(z):  # Сигмоидальная функция
    return 1 / (1 + math.exp(-z))


def train(x, y, epochs, n, N, train_count):  # Функция, управляющая процессом
    w = create(N)
    errors = np.zeros(epochs, float)

    for epoch in range(epochs):
        y_predict = np.zeros(train_count)
        for sample in range(0, train_count):
            y_predict[sample], w = trainmass(N, x[sample], y[sample], w, n)
            errors[epoch] += (y[sample] - y_predict[sample]) ** 2
        errors[epoch] *= 0.5
        print(f"Итерация №{epoch+1}, погрешность сети = {errors[epoch]}")
    return w, errors


def trainmass(N, x, y, w, n):  # Функция, выполняющая основные счисления
    preActivation = np.zeros(N)
    postActivation = np.zeros(N)

    y_predict = predict(preActivation, postActivation, x, w, N)

    delta = np.zeros(N)
    global_delta = (y_predict - y) * y_predict * (1 - y_predict)

    for i in range(N):
        delta[i] = (global_delta * w[1][i]) * postActivation[i] * \
                   (1 - postActivation[i])

    for i in range(N):  # Веса для последнего слоя
        w[1][i] += -n * global_delta * postActivation[i]

    for i in range(N):
        w[0][i] += -n * delta[i] * x[0]
        w[0][i + N] += -n * delta[i] * x[1]

    return y_predict, w


def predict(preActivation, postActivation, x, w, N):  # Высчитывание результата y_predict на основе исходных данный
    prePredict = 0
    for i in range(N):
        preActivation[i] = x[0] * w[0][i] + x[1] * w[0][i + N]
        postActivation[i] = sigmoid(preActivation[i])

    for i in range(N):
        prePredict += postActivation[i] * w[1][i]

    return sigmoid(prePredict)


def create(N):  # Заполение массива весов
    w = np.zeros((2, N * 2))
    for j in range(N * 2):
        w[0][j] = random.uniform(-0.5, 0.5)
    for j in range(N):
        w[1][j] = random.uniform(-0.5, 0.5)
    for j in range(N, N * 2):
        w[1][j] = np.nan
    return w


def multi_perceptron(x, y, n, N, epochs):
    #x = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    #y = np.array([0, 1, 1, 0])

    training_count = len(x)
    weights, error = train(x, y, epochs, n, N, training_count)
    return weights, error


def multi_perceptron_package(x, y, n, N, epochs):
    x = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    y = np.array([0, 1, 1, 0])

    training_count = len(x)
    weights, error = train_package(x, y, epochs, n, N, training_count)
    return weights, error


def train_package(x, y, epochs, n, N, train_count):  # Функция, управляющая процессом
    w = create(N)
    errors = np.zeros(epochs, float)

    for epoch in range(epochs):
        preActivation = np.zeros((train_count, N))
        postActivation = np.zeros((train_count, N))
        y_predict = np.zeros(train_count)

        for sample in range(0, train_count):
            y_predict[sample], preActivation[sample], postActivation[sample] = \
                trainmass_package(N, x[sample], y[sample], w, n)
            errors[epoch] += (y[sample] - y_predict[sample]) ** 2

        errors[epoch] *= 0.5
        print(f"Итерация №{epoch+1}, погрешность сети = {errors[epoch]}")

        # Пакетное корректирование весов
        for sample in range(0, train_count):
            delta = np.zeros(N)
            global_delta = (y_predict[sample] - y[sample]) * y_predict[sample] * (1 - y_predict[sample])

            for i in range(N):
                delta[i] = (global_delta * w[1][i]) * postActivation[sample][i] * \
                           (1 - postActivation[sample][i])

            for i in range(N):
                w[1][i] += -n * global_delta * postActivation[sample][i]

            for i in range(N):
                w[0][i] += -n * delta[i] * x[sample][0]
                w[0][i + N] += -n * delta[i] * x[sample][1]
    return w, errors


def trainmass_package(N, x, y, w, n):  # Функция, выполняющая основные счисления
    preActivation = np.zeros(N)
    postActivation = np.zeros(N)
    y_predict, preActivation, postActivation = predict_package(preActivation, postActivation, x, w, N)

    return y_predict, preActivation, postActivation


def predict_package(preActivation, postActivation, x, w, N):  # Высчитывание результата y_predict на основе исходных данный
    prePredict = 0
    for i in range(N):
        preActivation[i] = x[0] * w[0][i] + x[1] * w[0][i + N]
        postActivation[i] = sigmoid(preActivation[i])

    for i in range(N):
        prePredict += postActivation[i] * w[1][i]

    return sigmoid(prePredict), preActivation, postActivation
