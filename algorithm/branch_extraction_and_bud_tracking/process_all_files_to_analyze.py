img_all_path = "/Users/tshoiasc/Documents/文稿 - 陈跃千纪的MacBook Pro (325)/durham/毕业论文"

import os
import pandas as pd
import json
import glob
from datetime import datetime
import re

# 读取 data_available.json 文件
with open('../data_available.json', 'r') as f:
    plant_info = json.load(f)

# 将 plant_info 转换为更容易查询的字典格式
plant_info_dict = {item['plant_id']: item for item in plant_info}

# 创建一个空的 DataFrame 来存储所有数据
all_data = pd.DataFrame()

# 遍历所有 CSV 文件
for csv_file in glob.glob(img_all_path+'/skeleton/result/BR017-*-sv-*/branch_and_trajectory_data.csv'):
    # 从文件路径中提取信息
    folder_name = os.path.basename(os.path.dirname(csv_file))
    parts = folder_name.split('-')
    plant_id = f"{parts[0]}-{parts[1]}"
    view = f"{parts[3]}"
    print(plant_id, view)
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 从图像名称中提取日期
    if 'Image Name' in df.columns:
        df['Date'] = df['Image Name'].apply(lambda x: datetime.strptime('-'.join(re.split(r"[-_]", x)[2:5]), '%Y-%m-%d'))
    else:
        print(f"Warning: 'Image Name' column not found in {csv_file}")
        continue

    # 添加植株ID和视角信息
    df['plant_id'] = plant_id
    df['view'] = view

    # 从 plant_id 中提取春化温度信息
    vernalization_temp = '5°C' if parts[1][-2] == '1' else '10°C'
    df['vernalization_temp'] = vernalization_temp

    # 添加基因型和作物类型信息
    if plant_id in plant_info_dict:
        df['crop_type'] = plant_info_dict[plant_id]['crop_type']
        df['geno_type'] = plant_info_dict[plant_id]['geno_type']
    else:
        df['crop_type'] = 'Unknown'
        df['geno_type'] = 'Unknown'

    # 将数据添加到主 DataFrame
    all_data = pd.concat([all_data, df], ignore_index=True)

all_data.to_csv('processed_rapeseed_data.csv', index=False)

print("数据预处理完成，结果已保存到 processed_rapeseed_data.csv")