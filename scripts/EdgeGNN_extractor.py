#
# Copyright (c) 2025 Machine Learning and Fluid Mechanics Lab at Kyungpook National University
#
# This file is part of [Integrated Multi-features with Tabular prior-data fitted network differentiate chronic obstructive pulmonary disease patients].
#
# The full license text is available in the LICENSE file at the root of the project.
#
# License-Identifier: MLFM
#
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, LeakyReLU, Tanh
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.nn import EdgeConv, global_mean_pool
import pandas as pd
import os
import logging
import numpy as np

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

best_lr = 0.001 
best_hidden_channels = 128
best_embedding_dim = 32 
best_num_conv_layers = 3
best_mlp1_layers_dims = [32, 32, 32] 
best_mlp2_layers_dims = [64, 64, 64] 
best_edgeconv_aggr = 'max'
best_use_batchnorm = True
best_activation_name = 'relu'

activation_map = {'relu': ReLU(), 'leaky_relu': LeakyReLU(), 'tanh': Tanh()}
best_mlp_activation = activation_map[best_activation_name]


# --- Define the Adjusted EdgeGNN Model (same as your training script) ---
class AirwayEdgeGNN(torch.nn.Module):
    def __init__(self, num_node_features, output_channels,
                 num_conv_layers=3, mlp1_layers=[64, 64], mlp2_layers=[64, 64],
                 mlp_activation=ReLU(), edgeconv_aggr='max', use_batchnorm=False):
        super(AirwayEdgeGNN, self).__init__()
        self.convs = torch.nn.ModuleList()

        mlp1 = []
        in_channels = 2 * num_node_features
        for h_dim in mlp1_layers:
            mlp1.append(Linear(in_channels, h_dim))
            if use_batchnorm:
                mlp1.append(BatchNorm1d(h_dim))
            mlp1.append(mlp_activation)
            in_channels = h_dim
        self.convs.append(EdgeConv(nn=Sequential(*mlp1), aggr=edgeconv_aggr))
        last_out_channels = mlp1_layers[-1] if mlp1_layers else 2 * num_node_features

        for _ in range(num_conv_layers - 1):
            mlp_intermediate = []
            in_channels = 2 * last_out_channels
            for h_dim in mlp2_layers:
                mlp_intermediate.append(Linear(in_channels, h_dim))
                if use_batchnorm:
                    mlp_intermediate.append(BatchNorm1d(h_dim))
                mlp_intermediate.append(mlp_activation)
                in_channels = h_dim
            self.convs.append(EdgeConv(nn=Sequential(*mlp_intermediate), aggr=edgeconv_aggr))
            last_out_channels = mlp2_layers[-1] if mlp2_layers else 2 * last_out_channels

        self.out = Linear(last_out_channels, output_channels)

    def forward(self, data):
        # This forward method is for prediction (regression)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return x

    def extract_features(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if not self.convs:
            raise ValueError("Model has no convolutional layers to extract features from.")

        x_after_first_conv = self.convs[0](x, edge_index)
        pooled_features = global_mean_pool(x_after_first_conv, batch)

        if pooled_features.shape[1] != best_embedding_dim:
            logger.warning(f"Extracted feature dimension ({pooled_features.shape[1]}) does not match best_embedding_dim ({best_embedding_dim}). "
                           "Ensure best_mlp1_layers_dims[-1] is set to best_embedding_dim for accurate extraction.")

        return pooled_features


def load_graph_from_excel(filepath):
    nodes_df = pd.read_excel(filepath, sheet_name='Nodes')
    edges_df = pd.read_excel(filepath, sheet_name='Edges')
    nodes = nodes_df['node_id'].tolist()
    edges = list(zip(edges_df['bp0'], edges_df['bp1']))
    node_features = torch.tensor(nodes_df[['x', 'y', 'z']].values, dtype=torch.float)
    edge_features = torch.tensor(edges_df[['generation', 'length', 'diameter', 'InArea', 'OutArea', 'InPeri', 'OutPeri', 'WT', 'WA', 'Din', 'Dout', 'Cr']].values, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    return data

def load_all_graphs_from_folder(folder_path):
    graph_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    airway_trees = [load_graph_from_excel(os.path.join(folder_path, f)) for f in graph_files]
    return airway_trees


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Paths ---
    OUTPUT_DIR = 'features'
    MODEL_SAVE_PATH = 'models/EdgeGNN_model.pth'
    GRAPH_FOLDER_FOR_FEATURES = 'data/airway_trees'
    COMBINED_DATA_FILE = 'data/data.xlsx'
    OUTPUT_FEATURES_DIR = 'features'
    os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)

    logger.info("Loading combined data for SUBJID mapping...")
    try:
        combined_df = pd.read_excel(COMBINED_DATA_FILE)
        # Filter out rows with 'NONE' in SUBJID, as done in your training script
        combined_df = combined_df[combined_df['SUBJID'] != 'NONE']
        filename_to_subj_id = {row['Filename']: row['SUBJID'] for _, row in combined_df.iterrows()}
    except FileNotFoundError:
        logger.error(f"Combined data file not found at {COMBINED_DATA_FILE}. Exiting.")
        exit()
    except Exception as e:
        logger.error(f"Error loading combined data file: {e}. Exiting.")
        exit()

    logger.info(f"Loading airway trees from {GRAPH_FOLDER_FOR_FEATURES}...")
    airway_trees_raw = load_all_graphs_from_folder(GRAPH_FOLDER_FOR_FEATURES)
    graph_filenames = [f for f in os.listdir(GRAPH_FOLDER_FOR_FEATURES) if f.endswith('.xlsx')]
    graph_filenames.sort() # Ensure filenames are sorted for consistent processing

    if not airway_trees_raw:
        logger.error(f"No airway trees found in {GRAPH_FOLDER_FOR_FEATURES}. Exiting.")
        exit()
    num_node_features = airway_trees_raw[0].x.shape[1]
    output_dim_placeholder = 1 # This value doesn't matter for feature extraction, but needed for model init

    # --- Initialize and Load the Trained GNN Model ---
    logger.info("Initializing EdgeGNN model for feature extraction...")
    model_feature_extractor = AirwayEdgeGNN(
        num_node_features=num_node_features,
        output_channels=output_dim_placeholder, # Placeholder, as we're not using the final output layer
        num_conv_layers=best_num_conv_layers,
        mlp1_layers=best_mlp1_layers_dims,
        mlp2_layers=best_mlp2_layers_dims,
        mlp_activation=best_mlp_activation,
        edgeconv_aggr=best_edgeconv_aggr,
        use_batchnorm=best_use_batchnorm
    ).to(device)

    logger.info(f"Loading trained model state dict from {MODEL_SAVE_PATH}...")
    try:
        model_feature_extractor.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device), strict=False)
        model_feature_extractor.eval() # Set model to evaluation mode
        logger.info("Trained model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Trained model not found at {MODEL_SAVE_PATH}. Please ensure the model is trained and saved. Exiting.")
        exit()
    except Exception as e:
        logger.error(f"Error loading trained model: {e}. Please check the model file and architecture. Exiting.")
        exit()

    # --- Feature Extraction ---
    logger.info("Starting feature extraction...")
    extracted_data_list = []

    DESIRED_FEATURE_DIMENSION = best_embedding_dim

    for i, file_name in enumerate(graph_filenames):
        if file_name in filename_to_subj_id:
            graph_data = airway_trees_raw[i]
            subj_id = filename_to_subj_id[file_name]

            # Prepare data for model
            data_to_process = Data(x=graph_data.x, edge_index=graph_data.edge_index, edge_attr=graph_data.edge_attr)
            data_to_process = data_to_process.to(device)

            # Extract features using the new method (no target_dimension argument needed)
            with torch.no_grad():
                features = model_feature_extractor.extract_features(data_to_process)

            # Convert features to numpy and flatten
            features_np = features.cpu().numpy().flatten()

            # Create a dictionary for the current row
            row_data = {'Filename': file_name, 'SUBJID': subj_id}
            for j, feature_val in enumerate(features_np):
                row_data[f'EdgeGNN_{j}'] = feature_val
            extracted_data_list.append(row_data)
        else:
            logger.warning(f"SUBJID for file {file_name} not found in {COMBINED_DATA_FILE}. Skipping feature extraction for this file.")

    # --- Save Extracted Features to CSV ---
    if extracted_data_list:
        extracted_features_df = pd.DataFrame(extracted_data_list)
        csv_filename = f'EdgeGNN_features_{DESIRED_FEATURE_DIMENSION}d.csv'
        extracted_features_csv_path = os.path.join(OUTPUT_FEATURES_DIR, csv_filename)
        extracted_features_df.to_csv(extracted_features_csv_path, index=False)
        logger.info(f"Extracted features saved to {extracted_features_csv_path}")
    else:
        logger.warning("No features were extracted. Check data paths and file mappings.")
