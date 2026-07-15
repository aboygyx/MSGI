"""
Time-Series Forecasting for Software Degradation Analysis
This script implements a deep learning framework (e.g., MSGI/Informer variant) 
for multivariate time-series forecasting. It includes rigorous data preprocessing, 
model training, evaluation, and visualization.
"""

import os
import random
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.model import InformerStack, MSGI

# ==========================================
# 1. Global Configurations & Reproducibility
# ==========================================
# Fix random seeds to ensure completely reproducible experimental results
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Ensure deterministic behavior in cuDNN for exact reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configure matplotlib for academic publication
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for sliding-window time-series forecasting.
    Converts raw sequences into input-target pairs for the model.
    """
    def __init__(self, data, seq_len, pred_len, target_col=-1):
        """
        Args:
            data (np.ndarray): Normalized multivariate time-series data.
            seq_len (int): Length of the historical input sequence (look-back window).
            pred_len (int): Length of the prediction horizon.
            target_col (int): Index of the target variable to predict (default is the last column).
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col

    def __len__(self):
        # Calculate the total number of valid sliding windows
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # Extract historical input features (X) and future target values (Y)
        x = self.data[index:index + self.seq_len]
        y = self.data[index + self.seq_len:index + self.seq_len + self.pred_len, self.target_col]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def prepare_data(csv_path, seq_len=96, pred_len=24):
    """
    Loads, splits, and scales the dataset, returning PyTorch DataLoaders.
    MinMaxScaler is fitted exclusively on the training set to prevent data leakage.
    """
    df = pd.read_csv(csv_path)
    data = df.values

    # Sequential split to preserve temporal dependencies
    # Note: 1867 for the Android dataset, 3044 for the OpenStack dataset
    train_size = 3044 
    train_data_raw = data[:train_size]
    test_data_raw = data[train_size:]

    # Apply Min-Max normalization to mitigate scale variance across different metrics
    scaler_train = MinMaxScaler()
    train_data = scaler_train.fit_transform(train_data_raw)

    scaler_test = MinMaxScaler()
    test_data = scaler_test.fit_transform(test_data_raw)

    # Store global min and max of the target variable for subsequent denormalization
    target_max_train = np.max(train_data_raw[:, -1])
    target_min_train = np.min(train_data_raw[:, -1])

    target_max_test = np.max(test_data_raw[:, -1])
    target_min_test = np.min(test_data_raw[:, -1])

    # Construct Datasets and DataLoaders
    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, target_max_test, target_min_test


def train_model(train_loader, test_loader, input_size, output_size, seq_len, pred_len, device, target_max, target_min):
    """
    Initializes and trains the deep learning model, evaluating performance on the test set.
    """
    # Initialize the forecasting architecture (MSGI)
    model = MSGI(
        enc_in=input_size,
        dec_in=input_size,
        c_out=output_size,
        seq_len=seq_len,
        label_len=seq_len // 2,
        out_len=pred_len,
        factor=5,
        d_model=32,
        n_heads=3,
        e_layers=1,
        d_layers=1,
        d_ff=128, 
        dropout=0.1,
        attn='prob',
        embed='fixed',
        freq='h',
        activation='gelu',
        output_attention=False,
        distil=False,
        mix=False,
        device=device
    ).to(device)

    print("=== Model Architecture ===")
    print(model)

    # Compute network complexity
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    epochs = 100

    last_true = None
    last_pred = None
    last_metrics = None

    # Epoch iteration loop
    for epoch in range(epochs):
        
        # ------------------- Training Phase -------------------
        model.train()
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)

        for x, y in train_loader_tqdm:
            x = x.to(device)
            y = y.to(device)
            
            # Construct decoder input: append zero-padding for the prediction horizon
            dec_inp = torch.zeros_like(x[:, -pred_len:, :]).to(device)
            dec_inp = torch.cat([x[:, -pred_len:, :], dec_inp], dim=1)

            optimizer.zero_grad()
            outputs = model(x, None, dec_inp, None)
            
            # Compute loss and perform backpropagation
            loss = criterion(outputs.squeeze(-1), y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            train_loader_tqdm.set_postfix(loss=loss.item())

        # ------------------- Testing Phase -------------------
        model.eval()
        test_loss = 0
        all_true = []
        all_pred = []

        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Test]", leave=False)
        with torch.no_grad():
            for x, y in test_loader_tqdm:
                x = x.to(device)
                y = y.to(device)
                
                # Construct decoder input for inference
                dec_inp = torch.zeros_like(x[:, -pred_len:, :]).to(device)
                dec_inp = torch.cat([x[:, -pred_len:, :], dec_inp], dim=1)

                outputs = model(x, None, dec_inp, None)
                loss = criterion(outputs.squeeze(-1), y)
                test_loss += loss.item()
                test_loader_tqdm.set_postfix(loss=loss.item())

                # Collect predictions and ground truth for comprehensive evaluation
                outputs_np = outputs.squeeze(-1).cpu().numpy()
                y_np = y.cpu().numpy()
                all_pred.append(outputs_np)
                all_true.append(y_np)

        # Average losses over the epoch
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        # Concatenate batch results
        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)

        # Denormalize predictions and ground truth to calculate physical metrics
        true_denorm = all_true * (target_max - target_min) + target_min
        pred_denorm = all_pred * (target_max - target_min) + target_min

        true_flat = true_denorm.ravel()
        pred_flat = pred_denorm.ravel()

        # Calculate standard academic evaluation metrics
        mae = mean_absolute_error(true_flat, pred_flat)
        mse = mean_squared_error(true_flat, pred_flat)
        rmse = sqrt(mse)
        r2 = r2_score(true_flat, pred_flat)
        eps = 1e-8
        mape = np.mean(np.abs((true_flat - pred_flat) / (true_flat + eps))) * 100

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, '
              f'MAE: {mae:.6f}, MAPE: {mape:.4f}%, RMSE: {rmse:.6f}, R2: {r2:.6f}')

        # Record the experimental results and arrays of the final epoch
        last_true = true_denorm.copy()
        last_pred = pred_denorm.copy()
        last_metrics = dict(MAE=mae, MAPE=mape, MSE=mse, RMSE=rmse, R2=r2,
                            TestLoss=test_loss, TrainLoss=train_loss)

    return model, last_metrics, last_true, last_pred


if __name__ == "__main__":
    
    # Configure hardware accelerator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define hyper-parameters and file paths
    csv_path = ''  # Specify the path to the target dataset (e.g., 'data.csv')
    seq_len = 20
    pred_len = 1

    # Load and preprocess data
    train_loader, test_loader, target_max, target_min = prepare_data(csv_path, seq_len, pred_len)

    # Dynamically extract input feature dimension from the dataset
    sample_x, sample_y = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    output_size = 1

    # Execute the training pipeline
    model, last_metrics, last_true, last_pred = train_model(
        train_loader, test_loader, input_size, output_size, seq_len, pred_len, device, target_max, target_min
    )

    # Display final evaluation metrics
    print("\n=== Final Epoch Evaluation Metrics ===")
    for k, v in last_metrics.items():
        if k == 'MAPE':
            print(f"{k}: {v:.4f}%")
        else:
            print(f"{k}: {v:.6f}")

    # ==========================================
    # 2. Prediction Visualization
    # ==========================================
    true_flat = last_true.ravel()
    pred_flat = last_pred.ravel()

    plt.figure(figsize=(10, 6))
    plt.plot(true_flat, label='Actual TTAF', linewidth=1.5)
    plt.plot(pred_flat, label='Predicted TTAF', linewidth=1.5, alpha=0.8)

    # Format axes and legends for publication quality
    plt.xlabel('Number of Samples', fontsize=20, labelpad=10)
    plt.ylabel('TTAF', fontsize=20, labelpad=10)
    plt.legend(fontsize=15, loc='best', frameon=True)

    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    
    plt.tight_layout() # Ensures labels are not cut off
    
    # Optional: Save high-resolution figure
    # plt.savefig('prediction_results.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()
