import numpy as np
import matplotlib.pyplot as plt
from physics_engine import generate_5dof_kinematics
from model_predictor import train_and_predict
from metrics_error import calculate_errors

def run_covariance_analysis():
    print("Generating kinematics and training AI for Covariance Diagnostics...")
    
    # 1. Get raw data and AI predictions
    X_features, Y_target = generate_5dof_kinematics()
    y_true, y_pred = train_and_predict(X_features, Y_target)
    _, abs_error = calculate_errors(y_true, y_pred)
    
    # 2. Align the 5-DoF testing data with the AI's errors
    lookback = 10
    X_norm = (X_features - np.mean(X_features, axis=0)) / np.std(X_features, axis=0)
    
    # Reconstruct the exact X_test data the AI used
    X_aligned = []
    for i in range(len(X_norm) - lookback):
        X_aligned.append(X_norm[i : i + lookback])
    X_aligned = np.array(X_aligned)
    split = int(0.8 * len(X_aligned))
    X_test_raw = X_aligned[split:]
    
    # Compress the 10-step lookback window into a single average state per prediction
    X_test_compressed = np.mean(X_test_raw, axis=1)
    
    # 3. Calculate Pearson Correlation Coefficients
    # How strongly does the fluctuation in each specific variable correlate with the AI making a massive error?
    correlations = []
    for i in range(5):
        # np.corrcoef returns a 2x2 matrix; we want the correlation between feature 'i' and the error
        corr = np.corrcoef(X_test_compressed[:, i], abs_error)[0, 1]
        correlations.append(np.abs(corr))
        
    # Normalize to percentages for the research paper
    total_impact = np.sum(correlations)
    impact_percentages = [(c / total_impact) * 100 for c in correlations]
    
    # 4. Render Publication-Grade Bar Chart
    labels = ['Altitude (z)', 'Drop Speed (v_z)', 'X-Wobble (w1)', 'Y-Wobble (w2)', 'Main Spin (w3)']
    colors = ['#2ca02c', '#9467bd', '#ff7f0e', '#1f77b4', '#d62728']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, impact_percentages, color=colors, edgecolor='black')
    
    plt.title('Covariance Analysis: Which Variable Poisons the AI?', fontsize=14, fontweight='bold')
    plt.ylabel('Contribution to Total Predictive Error (%)', fontsize=12)
    plt.ylim(0, max(impact_percentages) + 10)
    
    # Add exact percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.text(0.02, 0.95, "Interpretation: Higher % means this variable's noise\ndrastically accelerated the Lyapunov divergence.", 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_covariance_analysis():!mv % new_filename.txt

