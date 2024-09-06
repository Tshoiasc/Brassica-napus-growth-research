import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.SmaAt_UNet import SmaAt_UNet
from data_process.data_loader import Radar_Dataset

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 参数设置
path_radars = 'F:/mn/5000/plant_generate/total_subfolders/'
path_Model_Save_model_paras_ori = 'model_params_NOGAN.pth'
n_channels_ori = 5
n_classes = 5
batch_size = 1
output_folder = 'output_images'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 初始化模型并加载已保存的参数
model = SmaAt_UNet(n_channels=n_channels_ori, n_classes=n_classes).to(device)
model.load_state_dict(torch.load(path_Model_Save_model_paras_ori, map_location=device))

# 加载数据
radar_dataset_valid = Radar_Dataset('valid', path_data=path_radars)
dataloader_valid = DataLoader(radar_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)

# 测试过程
model.eval()

print("Starting Testing Loop...")
batch_index = 0
with torch.no_grad():
    for inputs, targets in dataloader_valid:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        targets = targets.cpu().numpy() * 255
        outputs = outputs.cpu().numpy() * 255
        
        # 为每个batch创建一个单独的文件夹
        batch_folder = os.path.join(output_folder, f'batch_{batch_index}')
        os.makedirs(batch_folder, exist_ok=True)
        
        # 绘制并保存结果
        for t in range(outputs.shape[1]):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            im1 = axes[0].imshow(targets[0, t, :, :], cmap='jet', vmin=0, vmax=255)
            axes[0].set_title(f'Target - Time {t}')
            fig.colorbar(im1, ax=axes[0])
            im2 = axes[1].imshow(outputs[0, t, :, :], cmap='jet', vmin=0, vmax=255)
            axes[1].set_title(f'Prediction - Time {t}')
            fig.colorbar(im2, ax=axes[1])
            plt.tight_layout()
            plt.savefig(os.path.join(batch_folder, f'time_{t}.png'))
            plt.close()
        
        batch_index += 1
