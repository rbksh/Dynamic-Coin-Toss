import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def generate_dhm_kinematics():
    """Simulates coin toss kinematics based on DHM principles (rigid body mechanics)."""
    # Classical mechanics parameters
    mass = 0.005  # kg
    force_thumb = 1.5  # N (impulses velocity)
    time_contact = 0.01  # s
    
    v0_z = (force_thumb * time_contact) / mass 
    g = 9.81
    t_flight = (2 * v0_z) / g
    
    # Rotational vectors (DHM precession model)
    omega_spin = 38.0  # Angular velocity (rad/s)
    precession_angle = np.radians(15)  # Wobble off the normal axis
    air_damping = 0.02  # Aerodynamic friction altering angular momentum
    
    steps = 3000
    t = np.linspace(0, t_flight, steps)
    
    # Damping applied to angular momentum over time
    damped_omega = omega_spin * np.exp(-air_damping * t)
    phase = np.cumsum(damped_omega) * (t_flight / steps)
    
    # Z-component of the normal vector (1.0 = Heads, -1.0 = Tails)
    coin_state = np.cos(phase) * np.cos(precession_angle) + np.sin(phase) * np.sin(precession_angle) * np.sin(t * 10)
    
    # Extract streaks to demonstrate Law of Small Numbers
    binary_outcomes = np.where(coin_state > 0, 1, 0)
    sample_flips = binary_outcomes[::60][:50]
    streaks = np.diff(np.where(np.diff(np.append([0], sample_flips)) != 0)[0])
    
    print("\n=========================================")
    print("      LAW OF SMALL NUMBERS ANALYSIS      ")
    print("=========================================")
    print(f"Sample of 50 intervals: {sample_flips}")
    print(f"Max consecutive same-side streak: {np.max(streaks)}")
    print("Conclusion: Physics generates clusters that human psychology misinterprets as non-random.\n")
    
    return coin_state

def build_and_train_model(coin_state):
    """Trains an LSTM to decode the physical continuous state."""
    X, Y = [], []
    lookback = 15
    for i in range(len(coin_state) - lookback):
        X.append(coin_state[i : i + lookback])
        Y.append(coin_state[i + lookback])

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
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='mean_squared_error')
    
    print("Training Physics Decoder Model...")
    model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)
    
    predictions = model.predict(X_test)
    return y_test, predictions.flatten()

if __name__ == "__main__":
    coin_state = generate_dhm_kinematics()
    y_test, predictions = build_and_train_model(coin_state)
    
    # Pre-calculate errors
    squared_errors_ai = np.sum((y_test - predictions) ** 2)
    squared_errors_average = np.sum((y_test - np.mean(y_test)) ** 2)
    efficiency_percentage = (1 - (squared_errors_ai / squared_errors_average)) * 100
    
    raw_error = y_test - predictions
    absolute_error = np.abs(raw_error)
    time_steps = np.arange(len(raw_error))
    error_pct_change = np.zeros_like(absolute_error)
    error_pct_change[1:] = ((absolute_error[1:] - absolute_error[:-1]) / (absolute_error[:-1] + 1e-8)) * 100

    # =====================================================================
    # WINDOW 1: ACTUAL VS PREDICTED (STATIC)
    # =====================================================================
    plt.figure("Window 1: DHM Physics vs ML", figsize=(10, 5))
    plt.plot(y_test[:75], label='Actual Coin Normal Vector', color='black', marker='o', markersize=3)
    plt.plot(predictions[:75], label='AI Trajectory Prediction', color='red', linestyle='--', marker='x', markersize=3)
    plt.axhline(0, color='gray', linestyle=':', label='Heads/Tails Boundary')
    
    score_text = f"DHM Model Efficiency: {efficiency_percentage:.2f}%"
    plt.text(0.02, 0.05, score_text, transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='darkgreen',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    equation_text = r"Normal Z-Component: $\cos(\phi)\cos(\theta) + \sin(\phi)\sin(\theta)\sin(\omega t)$"
    plt.text(0.02, 0.17, equation_text, transform=plt.gca().transAxes, fontsize=11, fontweight='bold', color='darkblue',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
             
    plt.title('Decoding Coin Toss Kinematics (Continuous State)')
    plt.xlabel('Microsecond Steps')
    plt.ylabel('Normal Vector Z-Component')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # =====================================================================
    # WINDOW 3: ERROR PERCENTAGE CHANGE (STATIC)
    # =====================================================================
    plt.figure("Window 3: Physics Friction Error %", figsize=(10, 4))
    bar_colors = np.where(error_pct_change[1:100] > 0, 'crimson', 'forestgreen')
    plt.bar(time_steps[1:100], error_pct_change[1:100], color=bar_colors, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Vector Prediction Error Change Per Interval (%)')
    plt.xlabel('Time Steps')
    plt.ylabel('% Change in Error')
    plt.tight_layout()

    # =====================================================================
    # WINDOW 2: ERROR MAGNITUDE AND DIRECTION (ANIMATED)
    # =====================================================================
    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(10, 6), num="Window 2: Animated Aerodynamic Error")
    
    ax2_1.set_xlim(0, 100)
    ax2_1.set_ylim(np.min(raw_error[:200]) * 1.5, np.max(raw_error[:200]) * 1.5)
    ax2_1.axhline(0, color='black', linestyle='--')
    ax2_1.set_title('Live Tracking: Vector Trajectory Errors')
    line_raw, = ax2_1.plot([], [], color='purple', label='Raw Error (Phase Drift)')
    ax2_1.legend(loc='upper right')

    ax2_2.set_xlim(0, 100)
    ax2_2.set_ylim(0, np.max(absolute_error[:200]) * 1.5)
    ax2_2.axhline(np.mean(absolute_error), color='darkred', linestyle='-.')
    ax2_2.set_title('Live Tracking: Pure Magnitude of Physics Mistakes')
    line_abs, = ax2_2.plot([], [], color='crimson', label='Absolute Error Magnitude')
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

    ani2 = animation.FuncAnimation(fig2, update_w2, frames=len(y_test[:300]), interval=50, blit=False, repeat=False)

    # =====================================================================
    # WINDOW 4: LIVE ANIMATED SIMULATION (ANIMATED)
    # =====================================================================
    fig4, (ax4_1, ax4_2) = plt.subplots(2, 1, figsize=(10, 8), num="Window 4: High-Speed Camera Tracking")

    ax4_1.set_xlim(0, 100)
    ax4_1.set_ylim(-1.2, 1.2)
    ax4_1.axhline(0, color='gray', linestyle=':')
    ax4_1.set_title("Live Rotation: Actual Vector vs Predicted Vector")
    line_actual, = ax4_1.plot([], [], color='black', marker='o', label='Actual Coin State', markersize=3)
    line_pred, = ax4_1.plot([], [], color='red', marker='x', linestyle='--', label='AI Prediction', markersize=3)
    ax4_1.legend(loc="upper right")

    ax4_2.set_xlim(0, 100)
    ax4_2.set_ylim(0, np.max(absolute_error[:200]) * 1.5)
    ax4_2.set_title("Live Tracking: Error Cascade")
    line_error, = ax4_2.plot([], [], color='crimson', label='Error Margin', linewidth=2)
    ax4_2.legend(loc="upper right")

    x_data4, y_actual_sim, y_pred_sim, y_err_sim = [], [], [], []

    def update_w4(frame):
        x_data4.append(frame)
        y_actual_sim.append(y_test[frame])
        y_pred_sim.append(predictions[frame])
        y_err_sim.append(absolute_error[frame])
        
        line_actual.set_data(x_data4, y_actual_sim)
        line_pred.set_data(x_data4, y_pred_sim)
        line_error.set_data(x_data4, y_err_sim)
        
        if frame >= 100:
            ax4_1.set_xlim(frame - 100, frame + 5)
            ax4_2.set_xlim(frame - 100, frame + 5)
            
        return line_actual, line_pred, line_error

    ani4 = animation.FuncAnimation(fig4, update_w4, frames=len(y_test[:300]), interval=50, blit=False, repeat=False)
    fig4.tight_layout()

    plt.show()
