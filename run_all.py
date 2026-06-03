import os
import subprocess
import sys

def run_script(script_path: str) -> None:
    """
    Executes a python script and handles errors.

    Args:
        script_path: Path to the script to execute.
    """
    print(f"\n==================================================")
    print(f"Executing: {script_path}")
    print(f"==================================================")
    # Run the script using the current python executable
    res = subprocess.run([sys.executable, script_path], capture_output=False)
    if res.returncode != 0:
        print(f"Error: Script {script_path} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"Finished: {script_path} successfully.\n")

def main() -> None:
    """
    Orchestrates the entire Fuel Consumption & CO2 Emissions Prediction Pipeline.
    """
    print("==================================================")
    print("  Fuel Consumption & CO2 Emissions Prediction Pipeline  ")
    print("==================================================")
    
    # Define script paths
    cleaning_script = os.path.join('scripts', '01_data_cleaning.py')
    eda_script = os.path.join('scripts', '02_eda_visualization.py')
    model_script = os.path.join('scripts', '03_mlp_ggo_model.py')
    
    # 1. Run Data Cleaning
    run_script(cleaning_script)
    
    # 2. Run EDA & Visualizations
    run_script(eda_script)
    
    # 3. Run MLP + GGO Model
    run_script(model_script)
    
    print("==================================================")
    print("       ALL PIPELINE STEPS COMPLETED SUCCESSFULLY  ")
    print("==================================================")
    print("Results summary:")
    print("- Cleaned dataset created: data/processed/Fuel_Consumption_Cleaned.csv")
    print("- Visual figures saved in: results/figures/")
    print("- Reports written: ")
    print("  * results/reports/eda_report.md")
    print("  * results/reports/model_performance_report.md")
    print("==================================================")

if __name__ == '__main__':
    main()
