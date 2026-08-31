import numpy as np

def calculate_rolling_efficiency(y_true, y_pred):
    efficiencies = []
    for i in range(10, len(y_true)):
        y_t = y_true[:i]
        y_p = y_pred[:i]
        sq_err_ai = np.sum((y_t - y_p) ** 2)
        sq_err_avg = np.sum((y_t - np.mean(y_t)) ** 2) + 1e-8
        eff = (1 - (sq_err_ai / sq_err_avg)) * 100
        efficiencies.append(eff)
    
    # Pad the beginning to match array lengths
    pad = [efficiencies[0]] * 10
    return np.array(pad + efficiencies)

def calculate_efficiency_rate(efficiencies):
    rate = np.zeros_like(efficiencies)
    rate[1:] = np.diff(efficiencies)
    return rate
