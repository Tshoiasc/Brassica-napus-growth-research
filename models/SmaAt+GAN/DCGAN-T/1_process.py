import os
import shutil

def create_subfolders(src_folder, dst_folder):
    # 创建总文件夹
    os.makedirs(dst_folder, exist_ok=True)
    
    # 获取主文件夹下的所有子文件夹
    subfolders = [os.path.join(src_folder, name) for name in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, name))]
    
    for subfolder in subfolders:
        # 获取子文件夹中的所有图片文件，并按名称排序
        images = sorted([f for f in os.listdir(subfolder) if f.endswith(('jpg', 'jpeg', 'png'))])
        
        # 划分图片并创建新的子文件夹
        for i in range(0, len(images) - 10 + 1, 5):
            # 创建新的子文件夹路径
            new_subfolder_name = f"{os.path.basename(subfolder)}_subfolder_{i}_{i+10}"
            new_subfolder_path = os.path.join(dst_folder, new_subfolder_name)
            os.makedirs(new_subfolder_path, exist_ok=True)
            
            # 复制图片到新的子文件夹
            for j in range(i, i + 10):
                src_image_path = os.path.join(subfolder, images[j])
                dst_image_path = os.path.join(new_subfolder_path, images[j])
                shutil.copy(src_image_path, dst_image_path)
            print(f"Created {new_subfolder_path} with images {i} to {i+9}")

# 设置主文件夹路径
src_folder = '/home/dl/mn/plant_generate/'
# 设置总文件夹路径
dst_folder = '/home/dl/mn/total_subfolders/'

# 创建子文件夹并划分图片
create_subfolders(src_folder, dst_folder)
