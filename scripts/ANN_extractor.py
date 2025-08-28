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
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.preprocessing import StandardScaler
import os
import logging
import joblib 
from tensorflow.keras.losses import MeanSquaredError

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_ann_features(data_file_path, feature_names_file_path,
                         model_path, scaler_path,
                         output_features_csv_path,
                         feature_layer_index=7): 
    # 1. Validate input file paths
    if not os.path.exists(data_file_path):
        logger.error(f"Error: Data file not found at '{data_file_path}'")
        return
    if not os.path.exists(feature_names_file_path):
        logger.error(f"Error: Feature names file not found at '{feature_names_file_path}'")
        return
    if not os.path.exists(model_path):
        logger.error(f"Error: Trained model not found at '{model_path}'")
        return
    if not os.path.exists(scaler_path):
        logger.error(f"Error: StandardScaler not found at '{scaler_path}'. Please ensure it was saved during training.")
        return

    output_dir = os.path.dirname(output_features_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Ensured output directory exists at: '{output_dir}'")

    try:
        feature_df = pd.read_excel(feature_names_file_path)
        feature_columns = feature_df['FeatureName'].tolist()
        logger.info(f"Loaded {len(feature_columns)} feature names from '{feature_names_file_path}'")
    except Exception as e:
        logger.error(f"Error loading feature names from '{feature_names_file_path}': {e}")
        return

    try:
        data_df = pd.read_excel(data_file_path)
        logger.info(f"Successfully loaded data from '{data_file_path}'")
    except Exception as e:
        logger.error(f"Error reading data file '{data_file_path}': {e}")
        return

    required_cols = feature_columns
    metadata_cols = []
    if 'Filename' in data_df.columns:
        metadata_cols.append('Filename')
    else:
        logger.warning("No 'Filename' column found. Filenames will not be included.")
    
    if 'SUBJID' in data_df.columns:
        metadata_cols.append('SUBJID')
    else:
        logger.warning("No 'SUBJID' column found. SUBJIDs will not be included.")

    all_cols = required_cols + metadata_cols
    if not all(col in data_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data_df.columns]
        logger.error(f"Missing feature columns in data file: {missing}")
        return

    working_df = data_df[all_cols].copy()

    initial_rows = working_df.shape[0]
    working_df.dropna(subset=feature_columns, inplace=True)
    if working_df.shape[0] < initial_rows:
        logger.warning(f"Dropped {initial_rows - working_df.shape[0]} rows due to missing values in features.")
    
    if working_df.empty:
        logger.error("No valid data remaining after handling missing values. Cannot extract features.")
        return

    X_data = working_df[feature_columns]
    
    final_filenames = working_df['Filename'].values if 'Filename' in working_df.columns else [None] * len(working_df)
    final_subjid = working_df['SUBJID'].values if 'SUBJID' in working_df.columns else [None] * len(working_df)

    try:
        trained_model = models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
        logger.info(f"Successfully loaded trained model from '{model_path}'")
        scaler = joblib.load(scaler_path)
        logger.info(f"Successfully loaded StandardScaler from '{scaler_path}'")
    except Exception as e:
        logger.error(f"Error loading model or scaler: {e}")
        return

    if feature_layer_index >= len(trained_model.layers):
        logger.error(f"Error: feature_layer_index ({feature_layer_index}) is out of bounds.")
        logger.error(f"Available layers: {[layer.name for layer in trained_model.layers]}")
        return

    feature_extractor_model = models.Model(
        inputs=trained_model.inputs,
        outputs=trained_model.layers[feature_layer_index].output
    )
    
    extracted_feature_dim = feature_extractor_model.output_shape[1]
    logger.info(f"Feature extraction model created. It will extract {extracted_feature_dim}-dimensional features.")

    X_data_scaled = scaler.transform(X_data)
    logger.info("Input data scaled using the loaded StandardScaler.")

    extracted_features = feature_extractor_model.predict(X_data_scaled)
    logger.info(f"Extracted features for {len(extracted_features)} samples.")

    feature_col_names = [f'ANN_{i}' for i in range(extracted_feature_dim)]
    features_df = pd.DataFrame(extracted_features, columns=feature_col_names)

    output_df = pd.DataFrame({
        'Filename': final_filenames,
        'SUBJID': final_subjid
    })
    output_df = pd.concat([output_df, features_df], axis=1)

    try:
        output_df.to_csv(output_features_csv_path, index=False)
        logger.info(f"Extracted features saved to '{output_features_csv_path}'")
    except Exception as e:
        logger.error(f"Error saving extracted features to CSV: {e}")

if __name__ == "__main__":
    data_file = "data/data.xlsx"
    feature_names_file = "models/ANN_feature_cols.xlsx"
    model_file = "models/ANN_model.h5"
    scaler_file = "models/ANN_scaler.pkl"
    output_features_file = "features/ANN_features_19d.csv"

    logger.info("\n--- Starting ANN feature extraction process ---")
    extract_ann_features(data_file, feature_names_file,
                         model_file, scaler_file,
                         output_features_file, feature_layer_index=7)
    logger.info("\n--- ANN feature extraction process finished ---")