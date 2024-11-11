#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import os
import statistics as stat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import time
# import kaggle_evaluation.jane_street_inference_server

# 设置路径和参数
jane_street_real_time_market_data_forecasting_path = '/root/js_competitions/data'
valid_from = 1455

# 高斯噪声层
class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

# LSTM模型定义
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.noise = GaussianNoise(std=.1)
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.noise(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Sequential(
            GaussianNoise(std=0.1),
            nn.Dropout(dropout)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x[:, -1, :])
        return x.squeeze()
    

means = {'feature_00': 0.640198826789856, 'feature_01': 0.03755598142743111, 'feature_02': 0.6368075609207153, 'feature_03': 0.6365063786506653, 'feature_04': 0.013741530478000641, 'feature_05': -0.02173694409430027, 'feature_06': -0.006415014620870352, 'feature_07': -0.010971736162900925, 'feature_08': -0.04653771221637726, 'feature_09': 32.596106194690265, 'feature_10': 4.95929203539823, 'feature_11': 167.6541592920354, 'feature_12': -0.13415881991386414, 'feature_13': -0.07573335617780685, 'feature_14': -0.12015637010335922, 'feature_15': -0.7470195889472961, 'feature_16': -0.6257441639900208, 'feature_17': -0.7294047474861145, 'feature_18': -0.042215555906295776, 'feature_19': -0.08798160403966904, 'feature_20': -0.15741558372974396, 'feature_21': 0.10528526455163956, 'feature_22': 0.018054703250527382, 'feature_23': 0.03165541961789131, 'feature_24': 2.733017921447754, 'feature_25': 0.39958420395851135, 'feature_26': -0.11045943945646286, 'feature_27': -0.5332594513893127, 'feature_28': -0.4522790312767029, 'feature_29': -0.5739678144454956, 'feature_30': -0.7905704975128174, 'feature_31': 0.10600688308477402, 'feature_32': 0.40044134855270386, 'feature_33': -0.021725023165345192, 'feature_34': 0.4226262867450714, 'feature_35': 0.42143046855926514, 'feature_36': -0.00023802756913937628, 'feature_37': 0.027961043640971184, 'feature_38': 0.010258913040161133, 'feature_39': 0.005768273025751114, 'feature_40': 0.017485467717051506, 'feature_41': 0.038347117602825165, 'feature_42': -0.06123563274741173, 'feature_43': -0.11644423753023148, 'feature_44': -0.12342483550310135, 'feature_45': -0.028769943863153458, 'feature_46': -0.015200662426650524, 'feature_47': 0.015717582777142525, 'feature_48': -0.0033910537604242563, 'feature_49': -0.0052393232472240925, 'feature_50': -0.2285808026790619, 'feature_51': -0.3548349440097809, 'feature_52': -0.358092725276947, 'feature_53': 0.2607136368751526, 'feature_54': 0.18796788156032562, 'feature_55': 0.3154229521751404, 'feature_56': -0.1471923440694809, 'feature_57': 0.15730056166648865, 'feature_58': -0.021774644032120705, 'feature_59': -0.0037768862675875425, 'feature_60': -0.010220836848020554, 'feature_61': -0.03178725391626358, 'feature_62': -0.3769100308418274, 'feature_63': -0.3229374587535858, 'feature_64': -0.3718394339084625, 'feature_65': -0.10233989357948303, 'feature_66': -0.13688170909881592, 'feature_67': -0.14402112364768982, 'feature_68': -0.06875362992286682, 'feature_69': -0.11862917989492416, 'feature_70': -0.11789549142122269, 'feature_71': -0.06013699993491173, 'feature_72': -0.10766122490167618, 'feature_73': -0.09921672940254211, 'feature_74': -0.10233042389154434, 'feature_75': -0.05991339311003685, 'feature_76': -0.06349952518939972, 'feature_77': -0.07424316555261612, 'feature_78': -0.07759837061166763}
stds = {'feature_00': 1.027751088142395, 'feature_01': 1.0967519283294678, 'feature_02': 1.0156300067901611, 'feature_03': 1.0170334577560425, 'feature_04': 1.0726385116577148, 'feature_05': 0.9639211297035217, 'feature_06': 1.0963259935379028, 'feature_07': 1.0789952278137207, 'feature_08': 0.7962697148323059, 'feature_09': 23.72976726545254, 'feature_10': 3.1867162933797224, 'feature_11': 163.44513161352285, 'feature_12': 0.6700984835624695, 'feature_13': 0.5805172920227051, 'feature_14': 0.664044201374054, 'feature_15': 0.37517768144607544, 'feature_16': 0.3393096327781677, 'feature_17': 0.3603287935256958, 'feature_18': 0.9911752939224243, 'feature_19': 1.0550744533538818, 'feature_20': 0.6643751263618469, 'feature_21': 0.38239365816116333, 'feature_22': 0.950261116027832, 'feature_23': 0.8119344711303711, 'feature_24': 1.4362775087356567, 'feature_25': 1.0947270393371582, 'feature_26': 1.077124834060669, 'feature_27': 1.0645726919174194, 'feature_28': 1.0676648616790771, 'feature_29': 0.2640742361545563, 'feature_30': 0.19689509272575378, 'feature_31': 0.3815343976020813, 'feature_32': 1.2996565103530884, 'feature_33': 0.9989405870437622, 'feature_34': 1.3409572839736938, 'feature_35': 1.3365675210952759, 'feature_36': 0.8695492148399353, 'feature_37': 0.7334080934524536, 'feature_38': 0.698810338973999, 'feature_39': 0.7965824604034424, 'feature_40': 0.518515944480896, 'feature_41': 0.6384949088096619, 'feature_42': 0.8168442249298096, 'feature_43': 0.5228385925292969, 'feature_44': 0.6521403193473816, 'feature_45': 0.8666537404060364, 'feature_46': 0.9039222002029419, 'feature_47': 3.2711963653564453, 'feature_48': 0.6570901274681091, 'feature_49': 0.7083076238632202, 'feature_50': 1.0132617950439453, 'feature_51': 0.6081287860870361, 'feature_52': 0.9250587224960327, 'feature_53': 1.0421689748764038, 'feature_54': 0.5859629511833191, 'feature_55': 0.9191848039627075, 'feature_56': 0.9549097418785095, 'feature_57': 1.0204777717590332, 'feature_58': 0.8327276110649109, 'feature_59': 0.8309783339500427, 'feature_60': 0.8389413356781006, 'feature_61': 1.192766547203064, 'feature_62': 1.388945460319519, 'feature_63': 0.09957146644592285, 'feature_64': 0.3396177291870117, 'feature_65': 1.01683509349823, 'feature_66': 1.0824761390686035, 'feature_67': 0.642227828502655, 'feature_68': 0.5312599539756775, 'feature_69': 0.6208390593528748, 'feature_70': 0.6724499464035034, 'feature_71': 0.5356909036636353, 'feature_72': 0.6534596681594849, 'feature_73': 1.0855497121810913, 'feature_74': 1.0880277156829834, 'feature_75': 1.2321789264678955, 'feature_76': 1.2345560789108276, 'feature_77': 1.0921478271484375, 'feature_78': 1.0924347639083862}

def normalize_dataframe(df: pl.DataFrame, means: dict, stds: dict) -> pl.DataFrame:
    normalize_exprs = []
    for col in df.columns:
        if col in means and col in stds:
            if stds[col] != 0:
                normalize_exprs.append(
                    ((pl.col(col) - means[col]) / stds[col]).alias(col)
                )
            else:
                normalize_exprs.append(
                    (pl.col(col) - means[col]).alias(col)
                )
        else:
            normalize_exprs.append(pl.col(col))
    return df.select(normalize_exprs)

def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f} MB'.format(size_all_mb))

def r2_score(y_true, y_pred, weights):
    numerator = np.sum(weights * (y_true - y_pred)**2)
    denominator = np.sum(weights * y_true**2)
    return 1 - numerator / denominator


def train_model(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    all_probs = []
    all_targets = []
    all_weights = []

    pbar = tqdm(loader, desc='Training')
    start_time = time.time()

    for batch_idx, (X_batch, y_batch, weights_batch) in enumerate(pbar):        
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        weights_batch = weights_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss_per_sample = loss_function(outputs, y_batch)
        weighted_loss = loss_per_sample * weights_batch
        loss = weighted_loss.mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_probs.append(outputs.detach().cpu())
        all_targets.append(y_batch.cpu())
        all_weights.append(weights_batch.cpu())

        if batch_idx > 0:
            time_per_batch = (time.time() - start_time) / batch_idx
            remaining_batches = len(loader) - batch_idx
            eta = remaining_batches * time_per_batch
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'ETA': f'{eta/60:.1f}min'})

    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_weights = torch.cat(all_weights).numpy()
    
    mse = mean_squared_error(all_targets, all_probs, sample_weight=all_weights)
    r2 = r2_score(all_targets, all_probs, all_weights)

    return total_loss / len(loader), mse, r2

def evaluate_model(model, loader):
    model.eval()
    all_probs = []
    all_targets = []
    all_weights = []
    
    with torch.no_grad():
        for X_batch, y_batch, weights_batch in loader:
            outputs = model(X_batch)
            all_probs.append(outputs.cpu())
            all_targets.append(y_batch.cpu())
            all_weights.append(weights_batch.cpu())
            
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_weights = torch.cat(all_weights).numpy()
    
    mse = mean_squared_error(all_targets, all_probs, sample_weight=all_weights)
    r2 = r2_score(all_targets, all_probs, all_weights)
    
    return mse, r2


# 加载数据
alltraindata = pl.scan_parquet(f"{jane_street_real_time_market_data_forecasting_path}/train.parquet")
train = alltraindata.filter(pl.col("date_id")>=valid_from).collect()

# 准备特征
feature_names = [f"feature_{i:02d}" for i in range(79)]
train_features = train.select(feature_names)
train_features = train_features.fill_null(strategy='forward').fill_null(0)
train_features = normalize_dataframe(train_features, means, stds)

# 准备数据
X = train_features.to_numpy()
y = train.select('responder_6').to_numpy().reshape(-1)
weights = train.select('weight').to_numpy().reshape(-1)

# 数据集分割
n_test = int(len(X) * .2)
train_X, t_X = X[:-n_test], X[-n_test:]
train_y, t_y = y[:-n_test], y[-n_test:]
train_weights, t_weights = weights[:-n_test], weights[-n_test:]
val_n = int(len(t_y) * .5)
val_X, test_X = t_X[:-val_n], t_X[-val_n:]
val_y, test_y = t_y[:-val_n], t_y[-val_n:]
val_weights, test_weights = t_weights[:-val_n], t_weights[-val_n:]

# 转换为tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
val_X = torch.tensor(val_X, dtype=torch.float32).to(device)
val_y = torch.tensor(val_y, dtype=torch.float32).to(device)
test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
test_y = torch.tensor(test_y, dtype=torch.float32).to(device)
train_weights = torch.tensor(train_weights, dtype=torch.float32).to(device)
val_weights = torch.tensor(val_weights, dtype=torch.float32).to(device)
test_weights = torch.tensor(test_weights, dtype=torch.float32).to(device)

# 创建数据加载器
batch_size = 4096 * 2
train_dataset = TensorDataset(train_X, train_y, train_weights)
val_dataset = TensorDataset(val_X, val_y, val_weights)
test_dataset = TensorDataset(test_X, test_y, test_weights)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型训练
run_type = 'Transformer'

if run_type == 'Transformer':
    # 模型参数
    input_size = 79
    d_model = 512
    nhead = 8
    num_layers = 3
    dropout = 0.1

    # 初始化模型
    model = TransformerModel(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    print_model_size(model)

    # 优化器设置
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    loss_function = nn.MSELoss(reduction='none')

    # 训练循环
    epochs = 20
    best = float('-inf')
    degraded = 0
    best_model = model

    for epoch in range(epochs):
        train_loss, train_mse, train_r2 = train_model(
            model, train_loader, optimizer, loss_function, device
        )
        
        scheduler.step()
        val_mse, val_r2 = evaluate_model(model, val_loader)
        
        print(f'epoch {epoch}:')
        print(f'train loss {train_loss:.4f}, train_r2 {train_r2:.4f}, '
              f'train_mse {train_mse:.4f}')
        print(f'val_mse {val_mse:.4f}, val_r2 {val_r2:.4f}')
        print(f'lr: {scheduler.get_last_lr()[0]:.6f}')
        
        if val_r2 > best:
            best = val_r2
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), f'./model/js_{run_type}_unnorm.pth')
            degraded = 0
        else:
            degraded += 1
            
        if degraded > 10:
            print("Early stopping triggered")
            break
            
        torch.cuda.empty_cache()
    
    model = best_model
    test_mse, test_r2 = evaluate_model(model, test_loader)
    print(f'test R2 score is {test_r2}')

if run_type == 'LSTM':
    model = LSTM(input_size=79, hidden_dim=512, output_size = 1, num_layers = 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.MSELoss(reduction='none')
    epochs = 20
    best = float('-inf')
    degraded = 0
    best_model = model
    for epoch in range(epochs):
        train_loss, train_mse, train_r2 = train_model(model, train_loader, optimizer, loss_function, device)
        val_mse,val_r2 = evaluate_model(model, val_loader)
        print(f'epoch {epoch } train loss {train_loss:.4f}, train_r2 {train_r2:.4f}, train_mse {train_mse:.4f},val_mse {val_mse:.4f}, val_r2 { val_r2:.4f}')
        if val_r2 > best:
            best = val_r2
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), f'./model/js_{run_type}_unnorm.pth')
            degraded = 0
        else:
            degraded += 1
        if degraded > 10:
            break
    model = best_model

    test_mse, test_r2 = evaluate_model(model, test_loader)
    print(f'test R2 score is {test_r2}')