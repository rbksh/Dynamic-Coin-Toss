import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from physics_engine import calculate_rigid_body_kinematics, analyze_law_of_small_numbers
from pinn_model import train_physics_decoder

if __name__ == "__main__":
    z_vector, t_flight = calculate_rigid_body_kinematics()
    analyze_law_of_small_numbers(z_vector)
    
    y_test, predictions = train_physics_decoder(z_vector, sensor_noise_variance=0.03)
    
    squared_errors_ai = np.sum((y_test - predictions) ** 2)
    squared_errors_average = np.sum((y_test - np.mean(y_test)) ** 2)
    efficiency_percentage = (1 - (squared_errors_ai / squared_errors_average)) * 100
    
    raw_error = y_test - predictions
    absolute_error = np.abs(raw_error)
    time_steps = np.arange(len(raw_error))
    
    # Calculate the Horizon of Predictability
    # Threshold defined as error > 0.5 (predicting Heads when physics dictates Tails)
    critical_failure_index = np.argmax(absolute_error > 0.5) if np.any(absolute_error > 0.5) else len(absolute_error)

    # WINDOW 1: ACTUAL VS PREDICTED (STATIC)
    plt.figure("Window 1: Rigid Body Determinism", figsize=(10, 5))
    plt.plot(y_test[:100], label='Actual Kinematic Vector', color='black', marker='o', markersize=3)
    plt.plot(predictions[:100], label='Model Prediction', color='red', linestyle='--', marker='x', markersize=3)
    plt.axhline(0, color='gray', linestyle=':', label='Heads/Tails Boundary')
    
    score_text = f"Decoding Efficiency: {efficiency_percentage:.2f}%"
    plt.text(0.02, 0.05, score_text, transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='darkgreen',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.title('Euler Rigid Body Equations vs Machine Learning (with Sensor Noise)')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # WINDOW 3: HORIZON OF PREDICTABILITY (STATIC)
    plt.figure("Window 3: Lyapunov Horizon Mapping", figsize=(10, 4))
    plt.plot(time_steps[:150], absolute_error[:150], color='crimson', label='Exponential Error Growth')
    plt.axhline(0.5, color='black', linestyle='--', label='Critical Failure Threshold')
    if critical_failure_index < 150:
        plt.axvline(critical_failure_index, color='darkred', linestyle=':', label=f'Horizon Reached at Step {critical_failure_index}')
        plt.fill_between(time_steps[:150], 0, absolute_error[:150], where=(time_steps[:150] > critical_failure_index), color='red', alpha=0.2)
    
    plt.title('Horizon of Predictability: Mapping Chaotic Divergence')
    plt.xlabel('Sequential Time Steps')
    plt.ylabel('Absolute Error Magnitude')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # WINDOW 2: ERROR MAGNITUDE (ANIMATED)
    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(10, 6), num="Window 2: Animated Phase Drift")
    ax2_1.set_xlim(0, 100)
    ax2_1.set_ylim(np.min(raw_error[:200]) * 1.5, np.max(raw_error[:200]) * 1.5)
    ax2_1.axhline(0, color='black', linestyle='--')
    ax2_1.set_title('Live Tracking: Predictive Failure Over Time')
    line_raw, = ax2_1.plot([], [], color='purple', label='Phase Drift Error')
    ax2_1.legend(loc='upper right')

    ax2_2.set_xlim(0, 100)
    ax2_2.set_ylim(0, np.max(absolute_error[:200]) * 1.5)
    ax2_2.axhline(np.mean(absolute_error), color='darkred', linestyle='-.')
    ax2_2.set_title('Live Tracking: Absolute Error Magnitude')
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
    ani2 = animation.FuncAnimation(fig2, update_w2, frames=len(y_test[:300]), interval=50, blit=False, repeat=False)

    # WINDOW 4: LIVE ANIMATED SIMULATION
    fig4, (ax4_1, ax4_2) = plt.subplots(2, 1, figsize=(10, 8), num="Window 4: Target Acquisition")
    ax4_1.set_xlim(0, 100)
    ax4_1.set_ylim(-1.2, 1.2)
    ax4_1.axhline(0, color='gray', linestyle=':')
    ax4_1.set_title("Live Rotation: Physical Vector vs Calculated Vector")
    line_actual, = ax4_1.plot([], [], color='black', marker='o', markersize=3)
    line_pred, = ax4_1.plot([], [], color='red', marker='x', linestyle='--', markersize=3)

    ax4_2.set_xlim(0, 100)
    ax4_2.set_ylim(0, np.max(absolute_error[:200]) * 1.5)
    ax4_2.set_title("Live Tracking: Error Cascade")
    line_error, = ax4_2.plot([], [], color='crimson', linewidth=2)

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
