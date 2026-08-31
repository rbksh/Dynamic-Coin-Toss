import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def train_and_predict(X_features, Y_target):
    lookback = 10
    X, Y = [], []
    X_norm = (X_features - np.mean(X_features, axis=0)) / np.std(X_features, axis=0)
    
    for i in range(len(X_norm) - lookback):
        X.append(X_norm[i : i + lookback])
        Y.append(Y_target[i + lookback])

    X = np.array(X)
    Y = np.array(Y)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    model = Sequential([
        LSTM(64, input_shape=(lookback, 5), activation='tanh', return_sequences=True),
        Dropout(0.1),
        LSTM(32, activation='tanh'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
    
    predictions = model.predict(X_test).flatten()
    return y_test, predictions
