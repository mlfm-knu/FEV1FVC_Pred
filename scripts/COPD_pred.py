#
# Copyright (c) 2025 Machine Learning and Fluid Mechanics Lab at Kyungpook National University
#
# This file is part of [Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients].
#
# The full license text is available in the LICENSE file at the root of the project.
#
# License-Identifier: MLFM
#

import pandas as pd
import os

# --- Configuration ---
input_prediction_file = 'output/pred.csv'
output_classification_file = 'output/copd_classification.csv'
# FEV1/FVC ratio threshold for COPD diagnosis
# The GOLD standard is FEV1/FVC < 0.70
COPD_THRESHOLD = 0.70
COPD_CATEGORY = "COPD"
NON_COPD_CATEGORY = "Non-COPD"

# --- Main Execution ---
def classify_copd():
    try:
        # Load the predicted data
        pred_df = pd.read_csv(input_prediction_file)
        print(f"Successfully loaded predictions from {input_prediction_file}")

    except FileNotFoundError:
        print(f"Error: The input file '{input_prediction_file}' was not found.")
        print("Please ensure the prediction script has been run and the file exists.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading the input file: {e}")
        return

    if 'Predicted_FEV1_FVC' not in pred_df.columns:
        print("Error: The column 'Predicted_FEV1_FVC' is missing from the input file.")
        print("Please ensure the prediction script is configured correctly.")
        return
    
    print("Classifying subjects based on predicted FEV1/FVC ratio...")
    
    pred_df['Diagnosis'] = pred_df['Predicted_FEV1_FVC'].apply(
        lambda x: COPD_CATEGORY if x < COPD_THRESHOLD else NON_COPD_CATEGORY
    )
    
    print("Classification complete.")
    
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_classification_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        pred_df.to_csv(output_classification_file, index=False)
        print(f"Classification results saved to {output_classification_file}")
    
    except Exception as e:
        print(f"An error occurred while saving the output file: {e}")

if __name__ == "__main__":
    classify_copd()