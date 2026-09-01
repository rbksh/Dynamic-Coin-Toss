import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Link to the main engine to generate data
from try_chaos_theory import run_chaos_simulation

print("Initializing Chaos Simulation Engine for Live Animation...")
y_test, predictions = run_chaos_simulation()

# Calculate absolute error for the animation
error = np.abs(y_test - predictions)

# Set up the animation canvas
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), num="Live AI Chaos Simulation")

# Top Panel: Actual vs Predicted
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 1.1)
ax1.set_title("Live Tracking: Actual vs Predicted")
ax1.set_ylabel("State Value")
ax1.grid(True, linestyle=':', alpha=0.6)
line_actual, = ax1.plot([], [], color='black', marker='o', label='Actual Value', markersize=4)
line_pred, = ax1.plot([], [], color='red', marker='x', linestyle='--', label='AI Prediction', markersize=4)
ax1.legend(loc="upper right")

# Bottom Panel: Error vs Time
ax2.set_xlim(0, 100)
ax2.set_ylim(0, np.max(error[:200]) * 1.5)
ax2.set_title("Live Tracking: Absolute Error Magnitude")
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Error Size")
ax2.grid(True, linestyle=':', alpha=0.6)
line_error, = ax2.plot([], [], color='crimson', label='Error Magnitude', linewidth=2)
ax2.legend(loc="upper right")

# Storage arrays for the animation frames
x_data, y_actual, y_pred, y_err = [], [], [], []

def update_frame(frame):
    x_data.append(frame)
    y_actual.append(y_test[frame])
    y_pred.append(predictions[frame])
    y_err.append(error[frame])
    
    line_actual.set_data(x_data, y_actual)
    line_pred.set_data(x_data, y_pred)
    line_error.set_data(x_data, y_err)
    
    # Auto-scroll the X-axis once the animation reaches 100 steps
    if frame >= 100:
        ax1.set_xlim(frame - 100, frame + 5)
        ax2.set_xlim(frame - 100, frame + 5)
        
    return line_actual, line_pred, line_error

print("\nStarting live playback...")
# Run animation (interval=100 controls speed in milliseconds)
ani = animation.FuncAnimation(
    fig, 
    update_frame, 
    frames=len(y_test[:300]), 
    interval=100, 
    blit=False, 
    repeat=False
)

plt.tight_layout()
plt.show()
