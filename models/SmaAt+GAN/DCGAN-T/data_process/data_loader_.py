# coding=utf-8
#%%
from torch.utils.data import DataLoader
import os
import torch.utils.data as data
import numpy as np
import torch
from PIL import Image

#%%
class Radar_Dataset(data.Dataset):
    def __init__(self, flag, path_data):
        self.data_path = path_data
        self.data_use = []
        self.flag = flag
        train_files = os.listdir(self.data_path)
        train_files.sort()
        for train_file in train_files:
            # if error_file_dict[train_file] == []:
            self.data_use.append(train_file)
        # self.data_use = self.data_use[:1]
        if self.flag == 'train':
            self.data_use = self.data_use
        else:
            self.data_use = self.data_use
    def __getitem__(self, index):
        radar_seq = self.data_use[index]
        t = np.int(radar_seq.split('-')[1][3])
        image_path = os.listdir(self.data_path + radar_seq + '/')
        image_path.sort()
        image_x, image_y = [], []
        for i in range(0, 5):  # 加载40-35作为输入
            image = Image.open(self.data_path + radar_seq + '/' + image_path[i]).convert('L')
            image = image.resize((256, 256), Image.BILINEAR)  # 插值为256x256
            image = np.array(image) / 255
            image = torch.tensor(image.astype(np.float32))
            image_x.append(image[:, :])

        for i in range(5, 10):  # 加载35-15作为输出
            image = Image.open(self.data_path + radar_seq + '/' + image_path[i]).convert('L')
            image = image.resize((256, 256), Image.BILINEAR)  # 插值为256x256
            image = np.array(image) / 255
            image = torch.tensor(image.astype(np.float32))
            image_y.append(image[:, :])

        image_x = torch.stack(image_x, dim=0)
        image_y = torch.stack(image_y, dim=0)
        input_radar = image_x
        target_radar = image_y
        t = np.ones(image_x[:1].size()) * t
        t = torch.tensor(t.astype(np.float32))
        input_radar = torch.concat([input_radar, t],dim = 0)
        return input_radar, target_radar

    def __len__(self):
        return len(self.data_use)


#path_data = 'F:/mn/5000/plant_generate/total_subfolders/'
#dataset = Radar_Dataset(flag='train', path_data=path_data)
#dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

#for input_radar, target_radar in dataloader:
#    print(input_radar.shape, target_radar.shape)
