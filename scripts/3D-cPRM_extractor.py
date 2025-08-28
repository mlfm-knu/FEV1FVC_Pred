#
# Copyright (c) 2025 Machine Learning and Fluid Mechanics Lab at Kyungpook National University
#
# This file is part of [Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients].
#
# The full license text is available in the LICENSE file at the root of the project.
#
# License-Identifier: MLFM
#
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import nibabel as nib

# Configure GPU memory growth at the very top, before any TensorFlow operations
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs configured for memory growth.")
    except RuntimeError as e:
        print(e)

def extract_cnn_features(model_path, prm_nii_dir, output_dir, feature_layer_name='dense_1'):
    
    print(f"\n--- Starting Feature Extraction using model: {os.path.basename(model_path)} ---")

    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}.")
        return
    
    try:
        base_model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # Create a feature extraction model
    try:
        feature_extractor = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=base_model.get_layer(feature_layer_name).output
        )
        print(f"Feature extractor created using layer: '{feature_layer_name}'")
    except ValueError as e:
        print(f"Error creating feature extractor: {e}")
        print("Please check if 'feature_layer_name' is a valid layer in your model.")
        base_model.summary()
        return

    # Prepare list of image paths and subject IDs
    CNN_data_paths = []
    CNN_subids = []
    
    for filename in os.listdir(prm_nii_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            subject_id = os.path.splitext(os.path.splitext(filename)[0])[0]
            CNN_data_paths.append(os.path.join(prm_nii_dir, filename))
            CNN_subids.append(subject_id)
            
    if not CNN_data_paths:
        print("No valid PRM images (.nii or .nii.gz) found in the directory. Exiting feature extraction.")
        return

    # Load and preprocess images for feature extraction
    extracted_images_list = []
    extracted_subids = []
    expected_image_shape = (64, 64, 64) 

    print(f"Loading {len(CNN_data_paths)} images for feature extraction...")
    for i, img_path in enumerate(CNN_data_paths):
        try:
            img_data = nib.load(img_path).get_fdata().astype(np.float32)
            if img_data.shape == expected_image_shape:
                min_val, max_val = np.min(img_data), np.max(img_data)
                if (max_val - min_val) != 0:
                    img_data = (img_data - min_val) / (max_val - min_val)
                else:
                    img_data = np.zeros_like(img_data)

                extracted_images_list.append(img_data)
                extracted_subids.append(CNN_subids[i])
            else:
                print(f"Skipping {img_path}: unexpected shape {img_data.shape}. Expected {expected_image_shape}")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")

    if not extracted_images_list:
        print("No images successfully loaded for feature extraction. Exiting.")
        return

    images_np = np.array(extracted_images_list)
    images_reshaped = images_np.reshape(-1, *expected_image_shape, 1)

    print("Extracting features...")
    features = feature_extractor.predict(images_reshaped, batch_size=8) 
    print(f"Features shape: {features.shape}")

    if features.ndim > 2:
        features_flat = features.reshape(features.shape[0], -1)
    else:
        features_flat = features

    feature_column_names = [f'cPRM_{i}' for i in range(features_flat.shape[1])]
    
    features_df = pd.DataFrame(features_flat, columns=feature_column_names)
    features_df.insert(0, 'SUBJID', extracted_subids)
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'3D-cPRM_features_128d.csv')
    features_df.to_csv(output_filename, index=False)
    print(f"Extracted features saved to {output_filename}")
    print("--- Feature Extraction Complete ---")

if __name__ == '__main__':
    # Define your actual directories and file paths
    model_path = 'models/3D-cPRM_model.h5' 
    prm_images_folder = 'data/PRM_images' # Directory containing PRM .nii files
    feature_output_folder = 'features'

    print("\n--- Running Feature Extraction ---")
    extract_cnn_features(
        model_path, 
        prm_images_folder, 
        feature_output_folder,
        feature_layer_name='feature_layer_128' 
    )
    print("--- Finished Feature Extraction ---")