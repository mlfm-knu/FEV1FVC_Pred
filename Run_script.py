#
# Copyright (c) 2025 Machine Learning and Fluid Mechanics Lab at Kyungpook National University
#
# This file is part of [Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients].
#
# The full license text is available in the LICENSE file at the root of the project.
#
# License-Identifier: MLFM
#

import subprocess
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the list of scripts to run with their full paths
scripts_to_run = [
    os.path.join(script_dir, 'scripts', "3D-cPRM_extractor.py"),
    os.path.join(script_dir, 'scripts', "ANN_extractor.py"),
    os.path.join(script_dir, 'scripts', "EdgeGNN_extractor.py"),
    os.path.join(script_dir, 'scripts', "combine_features.py"),
    os.path.join(script_dir, 'scripts', "TabPFN_pred.py"),
    os.path.join(script_dir, 'scripts', "COPD_pred.py")
]

def run_script(script_name):
    python_executable = sys.executable
    print(f"\n--- Executing: {script_name} ---")

    try:
        result = subprocess.run([python_executable, script_name], check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"--- Finished: {script_name} successfully. ---")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Script '{script_name}' failed with return code {e.returncode}.")
        print("--- Details ---")
        print(f"Standard Output:\n{e.stdout}")
        print(f"Standard Error:\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nError: The script '{script_name}' was not found. "
              "Please ensure it is in the same directory as this script.")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting the full data processing and prediction pipeline...")
    
    # Run each script in the defined sequence
    for script in scripts_to_run:
        run_script(script)

    print("\nAll scripts in the pipeline completed successfully!")