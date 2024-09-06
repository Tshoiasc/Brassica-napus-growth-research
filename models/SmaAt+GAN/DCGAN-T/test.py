from data_process.data_loader import Radar_Dataset
from torch.utils.data import DataLoader
import torch
from models.SmaAt_UNet import SmaAt_UNet
import os
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 参数设置
path_error_file_dict = 'Process/error_file_dict.txt'
path_radars = 'hn/'
path_Model_Save_model_paras_ori = 'weight_0.026000/best_generator.pth'
n_channels_ori = 5
n_classes = 20
batch_size = 1

# 初始化生成器
net = SmaAt_UNet(n_channels=n_channels_ori, n_classes=n_classes).to(device)
net.load_state_dict(torch.load(path_Model_Save_model_paras_ori))

# 加载数据
radar_dataset_valid = Radar_Dataset('valid', path_data=path_radars, path_error_file_dict=path_error_file_dict)
dataloader_valid = DataLoader(radar_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)

# 定义损失函数
criterion = torch.nn.BCELoss().to(device)
mse_loss = torch.nn.MSELoss().to(device)

# 创建文件夹
result_folder = 'validation_results'
os.makedirs(result_folder, exist_ok=True)

# 测试模型并保存结果
net.eval()
batch_num = 0

with torch.no_grad():
    for inputs, targets in dataloader_valid:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        outputs = torch.clamp(outputs, 0, 65 / 70)
        outputs[outputs <= 10 / 70] = float('nan')
        targets[targets <= 10 / 70] = float('nan')
        batch_folder = os.path.join(result_folder, f'batch_{batch_num}')
        os.makedirs(batch_folder, exist_ok=True)

        # 保存每个样本的预测结果和实际结果
        for i in range(targets.size(1)):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            time_minute = (i + 1) * 6
            
            # 预测结果
            ax1 = axes[0]
            img1 = ax1.imshow(outputs[0, i].detach().cpu().numpy() * 70, cmap='jet', vmin=0, vmax=70)
            ax1.set_title('GAN')
            ax1.axis('off')
            ax1.grid(False)
            for spine in ax1.spines.values():
                spine.set_color('black')
            cbar1 = fig.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('dBz')

            # 实际结果
            ax2 = axes[1]
            img2 = ax2.imshow(targets[0, i].detach().cpu().numpy() * 70, cmap='jet', vmin=0, vmax=70)
            ax2.set_title('OBS')
            ax2.axis('off')
            ax2.grid(False)
            for spine in ax2.spines.values():
                spine.set_color('black')
            cbar2 = fig.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('dBz')

            # 在顶部添加时间信息
            fig.suptitle(f'Time: {time_minute} minutes')

            plt.savefig(os.path.join(batch_folder, f'image_{i}.png'))
            plt.close()

        batch_num += 1
