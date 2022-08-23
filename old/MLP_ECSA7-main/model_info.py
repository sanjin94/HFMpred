import numpy as np
from tensorflow.keras.models import load_model

import data


def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted


def mse(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(mse(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_error(actual, predicted)))


def metrics(actual: np.ndarray, predicted: np.ndarray):
    print("RMSE: ", rmse(actual, predicted))
    print("MSE: ", mse(actual, predicted))
    print("MAE: ", mae(actual, predicted))


if __name__ == "__main__":
    train_validation = 1/2

    ti_train = data.e2_ti[0:int(np.round(train_validation * len(data.e2_ti)))]
    te_train = data.e2_te[0:int(np.round(train_validation * len(data.e2_ti)))]
    q_train = data.e2_q[0:int(np.round(train_validation * len(data.e2_ti)))]

    ti_validation = data.e2_ti[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]
    te_validation = data.e2_te[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]
    q_validation = data.e2_q[int(np.round(train_validation * len(data.e2_ti))):len(data.e2_ti)]

    merged_array = np.stack([ti_train, te_train], axis=1)
    merged_array1 = np.stack([ti_validation, te_validation], axis=1)

    filepath = './saved_model_e2_1_2_1l3'  # ANN model 1_4, 1_2 or 2_3
    model = load_model(filepath, compile=True)

    predictions = model.predict(merged_array)
    predictions1 = model.predict(merged_array1)
    result = np.append(predictions, predictions1)

    print("Whole sequence")
    metrics(data.e2_q, result)
