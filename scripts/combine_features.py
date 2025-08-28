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

def combine_specific_csv_files(file_list, output_file, on_column='SUBJID', columns_to_remove=None):
    if columns_to_remove is None:
        columns_to_remove = ['Filename', 'Filename_old']

    if not file_list:
        print("Error: The file list is empty. Please provide at least one CSV file.")
        return

    combined_df = pd.DataFrame()
    first_file_processed = False

    print(f"Starting to combine the following {len(file_list)} files:")
    for file_path in file_list:
        print(f"  - {os.path.basename(file_path)}")

    for file_path in file_list:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing '{file_name}'...")
        try:
            df = pd.read_csv(file_path)

            if on_column not in df.columns:
                print(f"Warning: '{file_name}' does not have a '{on_column}' column. Skipping this file.")
                continue

            cols_to_drop_found = [col for col in columns_to_remove if col in df.columns]
            if cols_to_drop_found:
                df.drop(columns=cols_to_drop_found, inplace=True)
                print(f"  Removed columns: {', '.join(cols_to_drop_found)}")

            if not first_file_processed:
                combined_df = df
                first_file_processed = True
                print(f"  Initialized combined data with '{file_name}'. Shape: {combined_df.shape}")
            else:
                combined_df = pd.merge(combined_df, df, on=on_column, how='outer', suffixes=('', f'_{file_name.replace(".csv", "")}'))
                print(f"  Merged '{file_name}'. New shape: {combined_df.shape}")

        except FileNotFoundError:
            print(f"Error: The file '{file_name}' was not found. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"Warning: '{file_name}' is empty. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{file_name}': {e}")

    if combined_df.empty:
        print("\nNo data was combined. Please check your input files and paths.")
        return

    try:
        combined_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully combined all specified files and saved to '{output_file}'.")
        print(f"Final combined data shape: {combined_df.shape}")
        print("First 5 rows of the combined data:")
        print(combined_df.head())
    except Exception as e:
        print(f"Error saving combined data to '{output_file}': {e}")

if __name__ == "__main__":
    input_folder = 'features'
    files_to_combine = ['ANN_features_19d.csv', '3D-cPRM_features_128d.csv', 'EdgeGNN_features_32d.csv']
    file_paths = [os.path.join(input_folder, f) for f in files_to_combine]
    output_file_path = f'{input_folder}/combined_features.csv'
    
    combine_specific_csv_files(file_paths, output_file_path)
    