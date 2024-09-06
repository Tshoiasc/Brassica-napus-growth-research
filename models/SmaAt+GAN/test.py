import logging
import numpy as np
from torch.utils.data import DataLoader
import torch
from models.SmaAt_UNet import SmaAt_UNet
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.functional import mean_squared_error, mean_absolute_error

class Radar_Dataset:
    def __init__(self, mode, path_data):
        self.mode = mode
        self.data_path = path_data
        self.radar_seq_list = self._get_radar_seq_list()

    def _get_radar_seq_list(self):
        return [f for f in os.listdir(self.data_path) if
                not f.startswith('.') and os.path.isdir(os.path.join(self.data_path, f))]

    def __len__(self):
        return len(self.radar_seq_list)

    def __getitem__(self, idx):
        radar_seq = self.radar_seq_list[idx]
        radar_seq_path = os.path.join(self.data_path, radar_seq)

        image_files = [f for f in os.listdir(radar_seq_path) if not f.startswith('.')]
        image_files.sort()

        inputs = []
        targets = []
        for i, file in enumerate(image_files):
            try:
                img_path = os.path.join(radar_seq_path, file)
                with Image.open(img_path) as img:
                    img_array = np.array(img.convert('L'))
                    img_array = img_array / 255.0
                    if i < 5:
                        inputs.append(img_array)
                    else:
                        targets.append(img_array)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue

        if len(inputs) != 5 or not targets:
            raise ValueError(f"Invalid data in sequence {radar_seq}: inputs={len(inputs)}, targets={len(targets)}")

        inputs = np.array(inputs)
        targets = np.array(targets)

        return inputs, targets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

path_radars = 'total_subfolders'
path_Model_Save_model_paras_ori = 'weight_0.02/best_generator.pth'
n_channels_ori = 5
n_classes = 5
batch_size = 1

net = SmaAt_UNet(n_channels=n_channels_ori, n_classes=n_classes).to(device)
net.load_state_dict(torch.load(path_Model_Save_model_paras_ori, map_location=device))
logging.info("Model loaded successfully")

try:
    radar_dataset_valid = Radar_Dataset('valid', path_data=path_radars)
    dataloader_valid = DataLoader(radar_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=0)
    logging.info(f"Dataset loaded successfully. Total samples: {len(radar_dataset_valid)}")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    raise

criterion = torch.nn.BCELoss().to(device)
mse_loss = torch.nn.MSELoss().to(device)

result_folder = 'validation_results_2'
os.makedirs(result_folder, exist_ok=True)

net.eval()
batch_num = 0

# 初始化性能指标
ssim = StructuralSimilarityIndexMeasure().to(device)
total_mse = 0
total_mae = 0
total_ssim = 0
total_samples = 0

with torch.no_grad():
    for inputs, targets in dataloader_valid:
        try:
            if inputs.shape[0] == 0 or targets.shape[0] == 0:
                logging.warning(f"Empty batch encountered. Skipping batch {batch_num}")
                continue

            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                logging.warning(f"NaN values detected in batch {batch_num}. Skipping.")
                continue

            outputs = net(inputs)
            outputs = torch.clamp(outputs, 0, 1)

            # 计算性能指标
            mse = mean_squared_error(outputs, targets)
            mae = mean_absolute_error(outputs, targets)
            ssim_value = ssim(outputs, targets)

            total_mse += mse.item() * inputs.size(0)
            total_mae += mae.item() * inputs.size(0)
            total_ssim += ssim_value.item() * inputs.size(0)
            total_samples += inputs.size(0)

            batch_folder = os.path.join(result_folder, f'batch_{batch_num}')
            os.makedirs(batch_folder, exist_ok=True)

            for i in range(targets.size(1)):
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))

                # 显示输入的5帧图像
                for j in range(5):
                    ax = axes[0, j] if j < 4 else axes[1, 0]
                    img = ax.imshow(inputs[0, j].detach().cpu().numpy() * 255, cmap='jet', vmin=0, vmax=255)
                    ax.set_title(f'Input Frame {j + 1}')
                    ax.axis('off')
                    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

                # 显示预测结果和实际图像
                ax1 = axes[1, 1]
                img1 = ax1.imshow(outputs[0, i].detach().cpu().numpy() * 255, cmap='jet', vmin=0, vmax=255)
                ax1.set_title('GAN Prediction')
                ax1.axis('off')
                fig.colorbar(img1, ax=ax1, fraction=0.046, pad=0.04)

                ax2 = axes[1, 2]
                img2 = ax2.imshow(targets[0, i].detach().cpu().numpy() * 255, cmap='jet', vmin=0, vmax=255)
                ax2.set_title('Ground Truth')
                ax2.axis('off')
                fig.colorbar(img2, ax=ax2, fraction=0.046, pad=0.04)

                # 移除多余的子图
                fig.delaxes(axes[1, 3])

                fig.suptitle(f'Prediction for Frame {i + 6}\nMSE: {mse:.4f}, MAE: {mae:.4f}, SSIM: {ssim_value:.4f}')
                plt.tight_layout()
                plt.savefig(os.path.join(batch_folder, f'prediction_{i + 6}.png'))
                plt.close()

            batch_num += 1
            logging.info(f"Processed batch {batch_num}")
        except Exception as e:
            logging.error(f"Error processing batch {batch_num}: {e}")
            continue

# 计算并输出平均性能指标
if total_samples > 0:
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    avg_ssim = total_ssim / total_samples

    logging.info(f"Average MSE: {avg_mse:.4f}")
    logging.info(f"Average MAE: {avg_mae:.4f}")
    logging.info(f"Average SSIM: {avg_ssim:.4f}")
else:
    logging.warning("No valid samples processed.")

logging.info("Processing completed")