import numpy as np

def calculate_errors(y_true, y_pred):
    raw_error = y_true - y_pred
    absolute_error = np.abs(raw_error)
    return raw_error, absolute_error
