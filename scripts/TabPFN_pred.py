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
import joblib
import os

# --- Configuration ---
input_filename = 'features/combined_features.csv'

output_folder = 'output'
if output_folder and not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created directory: {output_folder}")
    
output_filename = f'{output_folder}/pred.csv'

model_filename = 'models/TabPFN_model.joblib'
scaler_filename = 'models/TabPFN_scaler.joblib'
final_features_list_path = 'models/TabPFN_features.joblib'

if __name__ == "__main__":
    try:
        model = joblib.load(model_filename)
        print(f"Successfully loaded model from {model_filename}")
        
        scaler = joblib.load(scaler_filename)
        print(f"Successfully loaded scaler from {scaler_filename}")

        feature_columns = joblib.load(final_features_list_path)
        print(f"Successfully loaded {len(feature_columns)} feature columns from {final_features_list_path}")

    except FileNotFoundError as e:
        print(f"Error: Required file not found. Please ensure the trained model, scaler, "
              f"and feature list files exist in the directory.")
        print(f"Missing file: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        exit()

    try:
        new_data_df = pd.read_csv(input_filename)
        print(f"Loaded new input data from {input_filename}")

    except FileNotFoundError as e:
        print(f"Error: New input data file not found at {input_filename}.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the input file: {e}")
        exit()

    print("Preparing data for prediction...")

    try:
        X_new = new_data_df[feature_columns]
        print("Successfully aligned new input data columns with trained model features.")
    except KeyError as e:
        print(f"Error: The new input data is missing a required feature column: {e}. "
              f"Please check your input data file.")
        exit()

    print("Applying RobustScaler to new input data...")
    X_new_scaled = scaler.transform(X_new)
    print("Normalization complete.")

    print("Making predictions on new input data...")
    predictions = model.predict(X_new_scaled)
    print("Predictions complete.")

    output_df = pd.DataFrame()
    if 'SUBJID' in new_data_df.columns:
        output_df['SUBJID'] = new_data_df['SUBJID']

    output_df['Predicted_FEV1_FVC'] = predictions

    output_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

    print("Script finished.")