import time
from animated_visualizer import run_animated_suite
from diagnostic_covariance import run_covariance_analysis

def execute_full_research_suite():
    print("==================================================")
    print("  COIN TOSS CHAOS THEORY & ML RESEARCH SUITE")
    print("==================================================")
    print("\n[PHASE 1] Initializing 5-Panel Live Simulation...")
    time.sleep(1) # Brief pause so you can read the terminal
    
    # This will open the 5 animated windows.
    # The script will pause here until you close the matplotlib window.
    run_animated_suite()
    
    print("\n[PHASE 2] Live Simulation Closed. Extracting Covariance Matrix...")
    time.sleep(1)
    
    # This automatically runs the moment you close the first window.
    run_covariance_analysis()
    
    print("\n==================================================")
    print("  RESEARCH SUITE COMPLETE.")
    print("  Save your graphs and efficiency scores for the paper!")
    print("==================================================")

if __name__ == "__main__":
    execute_full_research_suite()
