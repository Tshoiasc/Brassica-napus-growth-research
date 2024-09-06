from data_process.data_loader_ import Radar_Dataset
from torch.utils.data import DataLoader
import torch
from models.SmaAt_UNet import SmaAt_UNet
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 参数设置
path_radars = 'total_subfolders/'
path_Model_Save_model_paras_ori = 'model_params_NOGAN.pth'
n_channels_ori = 6
n_classes = 5
lr = 0.001
batch_size = 1

# 初始化模型
model = SmaAt_UNet(n_channels=n_channels_ori, n_classes=n_classes).to(device)
# model.load_state_dict(torch.load(path_Model_Save_model_paras_ori, map_location=lambda storage, loc: storage).module.state_dict())

# 定义损失函数和优化器
mse_loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 加载数据
radar_dataset_train = Radar_Dataset('train', path_data=path_radars)
dataloader_train = DataLoader(radar_dataset_train, batch_size=batch_size, shuffle=False, num_workers=0)
radar_dataset_valid = Radar_Dataset('valid', path_data=path_radars)
dataloader_valid = DataLoader(radar_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)

# 定义训练过程
num_epochs = 100
best_val_loss = float('inf')
epochs_no_improve = 0
patience = 20

# 训练
print("Starting Training Loop...")
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in dataloader_train:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

    # 验证阶段
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader_valid:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            val_loss += mse_loss(outputs, targets).item()

    val_loss /= len(dataloader_valid)
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
    # 早停条件
    if epochs_no_improve >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

# 保存最终模型
torch.save(model.state_dict(), path_Model_Save_model_paras_ori)
