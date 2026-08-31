import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def train_physics_decoder(z_vector, sensor_noise_variance=0.02):
    """Builds and trains a temporal sequence model using imperfect sensor data."""
    # Simulating microscopic measurement error in initial conditions
    noisy_z_vector = z_vector + np.random.normal(0, sensor_noise_variance, len(z_vector))
    
    lookback = 20
    X, Y = [], []
    for i in range(len(noisy_z_vector) - lookback):
        # The AI observes flawed noisy data...
        X.append(noisy_z_vector[i : i + lookback])
        # ...but is graded against the absolute ground-truth physics
        Y.append(z_vector[i + lookback])

    X = np.reshape(np.array(X), (len(X), lookback, 1))
    Y = np.array(Y)
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    model = Sequential([
        LSTM(128, input_shape=(lookback, 1), activation='tanh', return_sequences=True),
        Dropout(0.15),
        LSTM(64, activation='tanh'),
        Dropout(0.1),
        Dense(1)
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mse')
    
    print(f"\nTraining Model on Imperfect Data (Noise Variance: {sensor_noise_variance})...")
    model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=1)
    
    predictions = model.predict(X_test)
    return y_test, predictions.flatten()
