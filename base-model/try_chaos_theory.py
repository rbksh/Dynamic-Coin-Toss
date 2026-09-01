import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def run_chaos_simulation():
    """Generates chaotic data, trains a high-efficiency LSTM model, and returns the results."""
    def generate_logistic_map(x0, r, steps):
        x = np.zeros(steps)
        x[0] = x0
        for n in range(1, steps):
            x[n] = r * x[n-1] * (1 - x[n-1])
        return x

    data = generate_logistic_map(x0=0.4, r=3.9, steps=3000)

    X, Y = [], []
    lookback = 10 
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        Y.append(data[i + lookback])

    X = np.reshape(np.array(X), (len(X), lookback, 1))
    Y = np.array(Y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    model = Sequential([
        LSTM(64, input_shape=(lookback, 1), activation='tanh', return_sequences=True),
        Dropout(0.1),
        LSTM(32, activation='tanh'),
        Dropout(0.1),
        Dense(1)
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error')
    
    print("Training the High-Efficiency ML model on chaotic sequences...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    print("\nTesting the AI on unseen data...")
    predictions = model.predict(X_test)
    
    return y_test, predictions.flatten()

if __name__ == "__main__":
    y_test, predictions = run_chaos_simulation()
    
    squared_errors_ai = np.sum((y_test - predictions) ** 2)
    squared_errors_average = np.sum((y_test - np.mean(y_test)) ** 2)
    efficiency_percentage = (1 - (squared_errors_ai / squared_errors_average)) * 100
    
    raw_error = y_test - predictions
    absolute_error = np.abs(raw_error)
    time_steps = np.arange(len(raw_error))
    
    error_pct_change = np.zeros_like(absolute_error)
    error_pct_change[1:] = ((absolute_error[1:] - absolute_error[:-1]) / (absolute_error[:-1] + 1e-8)) * 100

    # =====================================================================
    # WINDOW 1: ACTUAL VS PREDICTED & EQUATION (STATIC)
    # =====================================================================
    plt.figure("Window 1: AI vs Chaos", figsize=(10, 5))
    plt.plot(y_test[:50], label='Actual Chaotic Value', color='black', marker='o')
    plt.plot(predictions[:50], label='ML Prediction', color='red', linestyle='--', marker='x')
    
    score_text = f"Model Efficiency: {efficiency_percentage:.2f}%"
    plt.text(0.02, 0.05, score_text, transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='darkgreen',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    equation_text = r"Governing Equation: $x_{n+1} = r \cdot x_n (1 - x_n)$"
    plt.text(0.02, 0.17, equation_text, transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='darkblue',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.title('Machine Learning Decoding Chaos')
    plt.xlabel('Time Steps')
    plt.ylabel('State Value')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # =====================================================================
    # WINDOW 3: ERROR PERCENTAGE CHANGE (STATIC)
    # =====================================================================
    plt.figure("Window 3: Error % Change", figsize=(10, 4))
    bar_colors = np.where(error_pct_change[1:100] > 0, 'crimson', 'forestgreen')
    plt.bar(time_steps[1:100], error_pct_change[1:100], color=bar_colors, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Interval-to-Interval Error Change (%)')
    plt.xlabel('Time Steps')
    plt.ylabel('% Change in Error')
    plt.tight_layout()

    # =====================================================================
    # WINDOW 2: ERROR MAGNITUDE AND DIRECTION (ANIMATED)
    # =====================================================================
    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(10, 6), num="Window 2: Animated Error Analysis")
    
    ax2_1.set_xlim(0, 100)
    ax2_1.set_ylim(np.min(raw_error[:200]) * 1.5, np.max(raw_error[:200]) * 1.5)
    ax2_1.axhline(0, color='black', linestyle='--')
    ax2_1.set_title('Live Tracking: Prediction Errors Over Time')
    line_raw, = ax2_1.plot([], [], color='purple', label='Raw Error')
    ax2_1.legend(loc='upper right')

    ax2_2.set_xlim(0, 100)
    ax2_2.set_ylim(0, np.max(absolute_error[:200]) * 1.5)
    ax2_2.axhline(np.mean(absolute_error), color='darkred', linestyle='-.', label='Average Error')
    ax2_2.set_title('Live Tracking: Pure Magnitude of Mistakes')
    line_abs, = ax2_2.plot([], [], color='crimson', label='Absolute Error')
    ax2_2.legend(loc='upper right')
    fig2.tight_layout()

    x_data2, y_raw, y_abs = [], [], []

    def update_w2(frame):
        x_data2.append(frame)
        y_raw.append(raw_error[frame])
        y_abs.append(absolute_error[frame])
        
        line_raw.set_data(x_data2, y_raw)
        line_abs.set_data(x_data2, y_abs)
        
        if frame >= 100:
            ax2_1.set_xlim(frame - 100, frame + 5)
            ax2_2.set_xlim(frame - 100, frame + 5)
            
        return line_raw, line_abs

    ani2 = animation.FuncAnimation(fig2, update_w2, frames=len(y_test[:300]), interval=100, blit=False, repeat=False)

    # =====================================================================
    # WINDOW 4: LIVE ANIMATED SIMULATION (ANIMATED)
    # =====================================================================
    fig4, (ax4_1, ax4_2) = plt.subplots(2, 1, figsize=(10, 8), num="Window 4: Live Simulation")

    ax4_1.set_xlim(0, 100)
    ax4_1.set_ylim(0, 1.1)
    ax4_1.set_title("Live Tracking: Actual vs Predicted")
    ax4_1.grid(True, linestyle=':', alpha=0.6)
    line_actual, = ax4_1.plot([], [], color='black', marker='o', label='Actual Value', markersize=4)
    line_pred, = ax4_1.plot([], [], color='red', marker='x', linestyle='--', label='AI Prediction', markersize=4)
    ax4_1.legend(loc="upper right")

    ax4_2.set_xlim(0, 100)
    ax4_2.set_ylim(0, np.max(absolute_error[:200]) * 1.5)
    ax4_2.set_title("Live Tracking: Absolute Error Magnitude")
    ax4_2.grid(True, linestyle=':', alpha=0.6)
    line_error, = ax4_2.plot([], [], color='crimson', label='Error Magnitude', linewidth=2)
    ax4_2.legend(loc="upper right")

    x_data4, y_actual, y_pred, y_err = [], [], [], []

    def update_w4(frame):
        x_data4.append(frame)
        y_actual.append(y_test[frame])
        y_pred.append(predictions[frame])
        y_err.append(absolute_error[frame])
        
        line_actual.set_data(x_data4, y_actual)
        line_pred.set_data(x_data4, y_pred)
        line_error.set_data(x_data4, y_err)
        
        if frame >= 100:
            ax4_1.set_xlim(frame - 100, frame + 5)
            ax4_2.set_xlim(frame - 100, frame + 5)
            
        return line_actual, line_pred, line_error

    ani4 = animation.FuncAnimation(fig4, update_w4, frames=len(y_test[:300]), interval=100, blit=False, repeat=False)
    fig4.tight_layout()

    plt.show()
