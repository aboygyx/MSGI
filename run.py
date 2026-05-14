import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from models.model import InformerStack, DGI22
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from tqdm import tqdm
import os

# 固定随机种子
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_col=-1):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + self.seq_len:index + self.seq_len + self.pred_len, self.target_col]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def prepare_data(csv_path, seq_len=96, pred_len=24):
    df = pd.read_csv(csv_path)
    data = df.values

    train_size = 3044 #android 1867 openstack 3044
    train_data_raw = data[:train_size]
    test_data_raw = data[train_size:]

    scaler_train = MinMaxScaler()
    train_data = scaler_train.fit_transform(train_data_raw)

    scaler_test = MinMaxScaler()
    test_data = scaler_test.fit_transform(test_data_raw)

    target_max_train = np.max(train_data_raw[:, -1])
    target_min_train = np.min(train_data_raw[:, -1])

    target_max_test = np.max(test_data_raw[:, -1])
    target_min_test = np.min(test_data_raw[:, -1])

    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, target_max_test, target_min_test


def train_model(train_loader, test_loader, input_size, output_size, seq_len, pred_len, device, target_max, target_min):

    model = DGI22(
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
        d_ff=128, #android 64 openstack 128
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

    print("===  ===")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    epochs = 100

    last_true = None
    last_pred = None
    last_metrics = None

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)

        for x, y in train_loader_tqdm:
            x = x.to(device)
            y = y.to(device)
            dec_inp = torch.zeros_like(x[:, -pred_len:, :]).to(device)
            dec_inp = torch.cat([x[:, -pred_len:, :], dec_inp], dim=1)

            optimizer.zero_grad()
            outputs = model(x, None, dec_inp, None)
            loss = criterion(outputs.squeeze(-1), y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_loader_tqdm.set_postfix(loss=loss.item())

        model.eval()
        test_loss = 0
        all_true = []
        all_pred = []

        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Test]", leave=False)
        with torch.no_grad():
            for x, y in test_loader_tqdm:
                x = x.to(device)
                y = y.to(device)
                dec_inp = torch.zeros_like(x[:, -pred_len:, :]).to(device)
                dec_inp = torch.cat([x[:, -pred_len:, :], dec_inp], dim=1)

                outputs = model(x, None, dec_inp, None)
                loss = criterion(outputs.squeeze(-1), y)
                test_loss += loss.item()
                test_loader_tqdm.set_postfix(loss=loss.item())

                outputs_np = outputs.squeeze(-1).cpu().numpy()
                y_np = y.cpu().numpy()
                all_pred.append(outputs_np)
                all_true.append(y_np)

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)

        all_pred = np.vstack(all_pred)
        all_true = np.vstack(all_true)

        true_denorm = all_true * (target_max - target_min) + target_min
        pred_denorm = all_pred * (target_max - target_min) + target_min

        true_flat = true_denorm.ravel()
        pred_flat = pred_denorm.ravel()

        mae = mean_absolute_error(true_flat, pred_flat)
        mse = mean_squared_error(true_flat, pred_flat)
        rmse = sqrt(mse)
        r2 = r2_score(true_flat, pred_flat)
        eps = 1e-8
        mape = np.mean(np.abs((true_flat - pred_flat) / (true_flat + eps))) * 100

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, '
              f'MAE: {mae:.6f}, MAPE: {mape:.4f}%, RMSE: {rmse:.6f}, R2: {r2:.6f}')

        # 记录最后一轮
        last_true = true_denorm.copy()
        last_pred = pred_denorm.copy()
        last_metrics = dict(MAE=mae, MAPE=mape, MSE=mse, RMSE=rmse, R2=r2,
                            TestLoss=test_loss, TrainLoss=train_loss)

    return model, last_metrics, last_true, last_pred


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csv_path = ''
    seq_len = 20
    pred_len = 1

    train_loader, test_loader, target_max, target_min = prepare_data(csv_path, seq_len, pred_len)

    sample_x, sample_y = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    output_size = 1

    model, last_metrics, last_true, last_pred = train_model(
        train_loader, test_loader, input_size, output_size, seq_len, pred_len, device, target_max, target_min
    )

    print("\n=== 最终一轮结果 ===")
    for k, v in last_metrics.items():
        if k == 'MAPE':
            print(f"{k}: {v:.4f}%")
        else:
            print(f"{k}: {v:.6f}")

    true_flat = last_true.ravel()
    pred_flat = last_pred.ravel()

    plt.figure(figsize=(10, 6))
    plt.plot(true_flat, label='Actual TTAF')
    plt.plot(pred_flat, label='Predicted TTAF')

    plt.xlabel('Number of Samples', fontsize=25)
    plt.ylabel('TTAF', fontsize=25)
    plt.legend(fontsize=15)

    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.show()
