from data_process.data_loader import Radar_Dataset
from torch.utils.data import DataLoader
import torch
from models.SmaAt_UNet import SmaAt_UNet
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from networks import Discriminator

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 参数设置
path_radars = 'plant_generate//'
path_Model_Save_model_paras_ori = 'model_params_ori.pth'
n_channels_ori = 6
n_classes = 5
p_valid = 1
num_workers = 0
lr = 0.0002
batch_size = 2

# 初始化生成器和判别器
generator = SmaAt_UNet(n_channels=n_channels_ori, n_classes=n_classes).to(device)
# generator.load_state_dict(torch.load(path_Model_Save_model_paras_ori, map_location=lambda storage, loc: storage).module.state_dict())

discriminator = Discriminator().to(device)

# 定义损失函数和优化器
criterion = torch.nn.BCELoss().to(device)
mse_loss = torch.nn.MSELoss().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

# 加载数据
radar_dataset_train = Radar_Dataset('train', path_data=path_radars)
dataloader_train = DataLoader(radar_dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
radar_dataset_valid = Radar_Dataset('valid', path_data=path_radars)
dataloader_valid = DataLoader(radar_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 定义训练过程
num_epochs = 100
real_label = 1
fake_label = 0
patience = 20

# 创建保存结果的txt文件
with open('loss_results.txt', 'w') as f:
    f.write('Weight\tBCE_Loss\tMSE_Loss\n')

# 记录最优BCE Loss和对应的模型
best_global_bce_loss = float('inf')
best_global_generator_state = None
best_global_discriminator_state = None

# 训练
print("Starting Training Loop...")

for weight in np.arange(0.01, 0.51, 0.01):
    best_val_loss = float('inf')
    best_val_bce_loss = float('inf')
    best_generator_state = None
    best_discriminator_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        i = 0
        for inputs, targets in dataloader_train:
            # 更新判别器
            inputs = inputs.to(device)
            targets = targets.to(device)
            discriminator.zero_grad()
            b_size = targets.size(0)
            label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            
            output_real = discriminator(targets).view(-1)
            errD_real = criterion(output_real, label_real)

            fake = generator(inputs)
            output_fake = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output_fake, label_fake)

            errD = errD_real + errD_fake
            errD.backward()
            optimizer_D.step()

            D_x = output_real.mean().item()
            D_G_z1 = output_fake.mean().item()

            # 更新生成器
            generator.zero_grad()
            output = discriminator(fake).view(-1)
            errG = (weight) * criterion(output, label_real) + (1 - weight) * mse_loss(fake, targets)
            errG.backward()
            optimizer_G.step()

            D_G_z2 = output.mean().item()

            i += 1

        # 验证阶段
        generator.eval()
        val_bce_loss = 0.0
        val_mse_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader_valid:
                b_size = targets.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                inputs = inputs.to(device)
                targets = targets.to(device)
                fake = generator(inputs)
                label.fill_(real_label)
                output = discriminator(fake).view(-1)
                val_bce_loss += criterion(output, label).item()
                val_mse_loss += mse_loss(fake, targets).item()

        val_bce_loss /= len(dataloader_valid)
        val_mse_loss /= len(dataloader_valid)
        val_loss = (1 - weight) * val_bce_loss + (weight) * val_mse_loss
        

        # 保存最优模型和损失值
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_bce_loss = val_bce_loss
            best_generator_state = generator.state_dict()
            best_discriminator_state = discriminator.state_dict()
            # 创建文件夹
            weight_folder = f'weight_{weight:.2f}'
            os.makedirs(weight_folder, exist_ok=True)
            torch.save(best_generator_state, weight_folder + '/' + 'best_generator.pth')
            torch.save(best_discriminator_state, weight_folder + '/' + 'best_discriminator.pth')
            epochs_no_improve = 0
            print(f'Current weight: {weight:.2f} - Validation BCE Loss: {val_bce_loss:.4f} MSE Loss: {val_mse_loss:.4f} Weighted Loss: {val_loss:.4f}')
            with open('loss_results.txt', 'a') as f:
                f.write(f'{weight:.2f}\t{val_bce_loss:.4f}\t{val_mse_loss:.4f}\n')
        else:
            epochs_no_improve += 1

        # 早停机制
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch + 1} for weight {weight:.2f}')
            break

    # # 保存最优模型图像
    # if best_generator_state and best_discriminator_state:
    #     generator.load_state_dict(best_generator_state)
    #     discriminator.load_state_dict(best_discriminator_state)
        # generator.eval()
        # with torch.no_grad():
        #     for inputs, targets in dataloader_valid:
        #         inputs = inputs.to(device)
        #         targets = targets.to(device)
        #         fake = generator(inputs).detach().cpu()
        #         targets = targets.detach().cpu()

        #         # 创建文件夹
        #         weight_folder = f'weight_{weight:.2f}'
        #         os.makedirs(weight_folder, exist_ok=True)

        #         for t in range(20):
        #             plt.figure(figsize=(12, 6))
        #             plt.subplot(1, 2, 1)
        #             plt.imshow(fake[0, t], cmap='jet', vmin=0, vmax=1)
        #             plt.title(f"Generated Image - Time {t}")
        #             plt.axis("off")

        #             plt.subplot(1, 2, 2)
        #             plt.imshow(targets[0, t], cmap='jet', vmin=0, vmax=1)
        #             plt.title(f"Target Image - Time {t}")
        #             plt.axis("off")

        #             plt.savefig(os.path.join(weight_folder, f'image_{t}.png'))
        #             plt.close()

        # 更新全局最优模型
        if best_val_bce_loss < best_global_bce_loss:
            best_global_bce_loss = best_val_bce_loss
            best_global_generator_state = best_generator_state
            best_global_discriminator_state = best_discriminator_state

# 保存最终最优模型
if best_global_generator_state and best_global_discriminator_state:
    torch.save(best_global_generator_state, 'best_generator_global.pth')
    torch.save(best_global_discriminator_state, 'best_discriminator_global.pth')

# 保存最终模型
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

print(f'The weight with the lowest validation BCE loss is: {best_global_bce_loss}')
