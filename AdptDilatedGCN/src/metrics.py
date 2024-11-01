import numpy as np
import sklearn.metrics as m
from scipy.stats import pearsonr
from numba import njit


@njit
# ci
def c_index(y_true, y_pred):
    sum = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                sum += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                sum += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1
    if pair != 0:
        return sum / pair
    else:
        return 0


# RMSE
def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))


# MAE
def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)


# R
def CORR(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


# SD
def SD(y_true, y_pred):
    from sklearn.linear_model import LinearRegression
    y_pred = y_pred.reshape((-1, 1))
    lr = LinearRegression().fit(y_pred, y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))
