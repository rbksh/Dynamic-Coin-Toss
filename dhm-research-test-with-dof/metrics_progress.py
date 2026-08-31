def log_progress(step_name, status="COMPLETE"):
    print(f"[{status}] - {step_name}")

def compile_final_stats(efficiency, max_error):
    print("\n--- FINAL SIMULATION METRICS ---")
    print(f"Final Model Efficiency: {efficiency[-1]:.2f}%")
    print(f"Peak Chaotic Error Magnitude: {max_error:.4f}")
