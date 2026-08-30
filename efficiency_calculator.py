import numpy as np
import matplotlib.pyplot as plt

# LINK TO PREVIOUS CODE: This imports the engine function from your try_chaos_theory.py file
from try_chaos_theory import run_chaos_simulation

print("Initializing Chaos Simulation Engine...")
# Execute the linked file and grab its outputs
y_test, predictions = run_chaos_simulation()

# Flatten predictions to a simple 1D array for easier math
predictions = predictions.flatten()

# =====================================================================
# CALCULATION PHASE
# =====================================================================
# Mean Absolute Error (MAE): Average size of the model's mistakes
mae = np.mean(np.abs(y_test - predictions))

# Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
mse = np.mean((y_test - predictions) ** 2)
rmse = np.sqrt(mse)

# Total Model Efficiency Percentage (R-Squared Score)
squared_errors_ai = np.sum((y_test - predictions) ** 2)
squared_errors_average = np.sum((y_test - np.mean(y_test)) ** 2)
r2_score = 1 - (squared_errors_ai / squared_errors_average)
efficiency_percentage = r2_score * 100

# =====================================================================
# DISPLAY SCORECARD TERMINAL OUTPUT
# =====================================================================
print("\n=========================================")
print("      ML CHAOS EFFICIENCY SCORECARD      ")
print("=========================================")
print(f"Average Mistake Size (MAE):     {mae:.5f}")
print(f"Big Mistake Penalty (RMSE):     {rmse:.5f}")
print("-----------------------------------------")
print(f"TOTAL MODEL EFFICIENCY SCORE:   {efficiency_percentage:.2f}%")
print("=========================================")

if efficiency_percentage > 90:
    print("Result: Excellent! The AI successfully decoded the hidden chaos rules.\n")
elif efficiency_percentage > 50:
    print("Result: Moderate. The AI sees the pattern but struggles with chaos sensitivity.\n")
else:
    print("Result: Poor. The AI failed to separate the chaos from pure randomness.\n")

# =====================================================================
# PLOT VISUALIZATION WITH ON-GRAPH EFFICIENCY TEXT
# =====================================================================
plt.figure(figsize=(10, 4.5))
plt.plot(y_test[:50], label='Actual Chaotic Value', color='black', marker='o')
plt.plot(predictions[:50], label='ML Prediction', color='red', linestyle='--', marker='x')

# Create a clean text box string to overlay onto the graph canvas
score_text = f"Model Efficiency: {efficiency_percentage:.2f}%"

# Place the text box on the graph dynamically
# transform=plt.gca().transAxes places the text relative to the graph box (0=bottom/left, 1=top/right)
plt.text(0.02, 0.05, score_text, transform=plt.gca().transAxes, fontsize=12,
         fontweight='bold', color='darkgreen',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

plt.title('Machine Learning Decoding Chaos (Isolated Calculator Model)')
plt.xlabel('Time Steps')
plt.ylabel('State Value')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

