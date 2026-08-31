import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from physics_engine import generate_5dof_kinematics
from model_predictor import train_and_predict
from metrics_error import calculate_errors
from metrics_efficiency import calculate_rolling_efficiency, calculate_efficiency_rate
from metrics_progress import log_progress, compile_final_stats

def run_animated_suite():
    log_progress("Generating 5-DoF Kinematics")
    X_features, Y_target = generate_5dof_kinematics()
    
    log_progress("Training LSTM Predictor")
    y_true, y_pred = train_and_predict(X_features, Y_target)
    
    log_progress("Calculating Error Matrices")
    raw_err, abs_err = calculate_errors(y_true, y_pred)
    
    log_progress("Calculating Efficiency Derivatives")
    efficiencies = calculate_rolling_efficiency(y_true, y_pred)
    eff_rate = calculate_efficiency_rate(efficiencies)
    
    compile_final_stats(efficiencies, np.max(abs_err))

    # Setup the 5-Panel Animation Figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2)
    
    ax_pred = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_eff = fig.add_subplot(gs[1, 1])
    ax_eff_err = fig.add_subplot(gs[2, 0])
    ax_rate = fig.add_subplot(gs[2, 1])
    
    # 1. Predicted vs Actual
    ax_pred.set_xlim(0, 100)
    ax_pred.set_ylim(-1.2, 1.2)
    ax_pred.set_title("Target Acquisition: Actual vs Predicted (5-DoF)")
    line_true, = ax_pred.plot([], [], 'k-', label='Actual Z-Vector')
    line_pred, = ax_pred.plot([], [], 'r--', label='LSTM Prediction')
    ax_pred.legend(loc='upper right')

    # 2. Error vs Time
    ax_err.set_xlim(0, 100)
    ax_err.set_ylim(0, np.max(abs_err[:200]) * 1.2)
    ax_err.set_title("Absolute Error Magnitude vs Time")
    line_err, = ax_err.plot([], [], 'm-')

    # 3. Efficiency Calculation
    ax_eff.set_xlim(0, 100)
    ax_eff.set_ylim(-100, 100)
    ax_eff.set_title("Rolling Model Efficiency (%)")
    ax_eff.axhline(0, color='k', linestyle=':')
    line_eff, = ax_eff.plot([], [], 'g-')

    # 4. Efficiency vs Error (Phase Space)
    ax_eff_err.set_xlim(0, np.max(abs_err[:200]) * 1.2)
    ax_eff_err.set_ylim(-100, 100)
    ax_eff_err.set_title("Phase Space: Efficiency vs Error")
    scatter_eff_err = ax_eff_err.scatter([], [], c='b', s=10)

    # 5. Efficiency Rate of Change
    ax_rate.set_xlim(0, 100)
    ax_rate.set_ylim(np.min(eff_rate[:200]) * 1.2, np.max(eff_rate[:200]) * 1.2)
    ax_rate.set_title("Rate of Efficiency Change (Derivative)")
    ax_rate.axhline(0, color='k', linestyle=':')
    line_rate, = ax_rate.plot([], [], 'c-')

    plt.tight_layout()

    # Animation Arrays
    frames = min(200, len(y_true))
    x_data = []

    def update(frame):
        x_data.append(frame)
        
        # Shift windows for scrolling effect
        if frame > 100:
            for ax in [ax_pred, ax_err, ax_eff, ax_rate]:
                ax.set_xlim(frame - 100, frame)

        line_true.set_data(x_data, y_true[:frame+1])
        line_pred.set_data(x_data, y_pred[:frame+1])
        line_err.set_data(x_data, abs_err[:frame+1])
        line_eff.set_data(x_data, efficiencies[:frame+1])
        line_rate.set_data(x_data, eff_rate[:frame+1])
        
        scatter_eff_err.set_offsets(np.c_[abs_err[:frame+1], efficiencies[:frame+1]])
        
        return line_true, line_pred, line_err, line_eff, scatter_eff_err, line_rate

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False, repeat=False)
    plt.show()

if __name__ == "__main__":
    run_animated_suite()
