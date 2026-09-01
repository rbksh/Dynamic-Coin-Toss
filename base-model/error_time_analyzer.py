import numpy as np
import matplotlib.pyplot as plt

# LINK TO YOUR MACHINE LEARNING ENGINE
# This imports the simulation function from your try_chaos_theory.py file
from try_chaos_theory import run_chaos_simulation

print("Initializing Chaos Simulation Engine to extract tracking data...")
# Execute the linked file and grab the actual physics data vs AI guesses
y_test, predictions = run_chaos_simulation()

# Flatten predictions into a simple 1D array to match y_test shape
predictions = predictions.flatten()
y_test = np.array(y_test)

# =====================================================================
# CALCULATION PHASE: DETECTING THE ERROR OVER TIME
# =====================================================================
# Error = Actual observed value minus what the ML model calculated
# A positive error means the AI guessed too low. Negative means it guessed too high.
raw_error = y_test - predictions

# Absolute Error = The pure size of the mistake, ignoring direction
absolute_error = np.abs(raw_error)

# Create a timeline array representing consecutive steps into the future
time_steps = np.arange(len(raw_error))

# Calculate the average baseline error to draw as a reference line on the graph
mean_absolute_error = np.mean(absolute_error)

# =====================================================================
# PLOT VISUALIZATION: ERROR TIMELINE
# =====================================================================
# We will create a two-panel chart to show both direction and size of errors over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Panel 1: Raw Error Tracker (Shows if the AI overshoots or undershoots)
ax1.plot(time_steps[:100], raw_error[:100], color='purple', label='Raw Error (Actual - Predicted)', alpha=0.85)
ax1.axhline(0, color='black', linestyle='--', alpha=0.5, label='Zero Error Baseline')
ax1.set_title('Tracking Prediction Errors Over Time (First 100 Chronological Steps)')
ax1.set_ylabel('Error Direction & Value')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

# Panel 2: Absolute Error Magnitude (Pure size of the mistake over time)
ax2.fill_between(time_steps[:100], absolute_error[:100], color='crimson', alpha=0.3, label='Error Size Area')
ax2.plot(time_steps[:100], absolute_error[:100], color='crimson', linewidth=1.5)
ax2.axhline(mean_absolute_error, color='darkred', linestyle='-.', linewidth=1.5, 
            label=f'Overall Average Mistake Size ({mean_absolute_error:.4f})')
ax2.set_title('Pure Magnitude of Mistakes Over Time')
ax2.set_xlabel('Timeline Steps (Consecutive Milliseconds of Flight)')
ax2.set_ylabel('Absolute Error Magnitude')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

