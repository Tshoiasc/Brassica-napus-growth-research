import sys
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 加载本地训练好的 YOLOv8 模型
model = YOLO('bets2.pt')  # 替换为你训练好的模型权重文件的路径

# 加载预训练的ResNet模型
resnet = resnet50(pretrained=True)
resnet = resnet.eval()

# 设置图像路径
image_dir = r'images'  # 替换为你的图像文件夹路径

# 读取图像文件列表
image_files = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.png'):
            image_files.append(os.path.join(root, file))

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 特征提取函数（使用ResNet提取特征）
def extract_features(image, bbox):
    x1, y1, x2, y2 = bbox
    flower = image[int(y1):int(y2), int(x1):int(x2)]
    flower = cv2.cvtColor(flower, cv2.COLOR_BGR2RGB)
    flower_tensor = preprocess(flower).unsqueeze(0)
    with torch.no_grad():
        features = resnet(flower_tensor).numpy().flatten()
    return features


# 追踪花蕾并赋予ID
flower_tracks = {}
current_id = 0
confidence_threshold = 0.01
batch_size = 8  # 设置批量大小

# 存储三维点云
flower_positions = {}


def process_batch(batch_images, batch_files):
    global current_id
    results = model(batch_images)
    for i, image_file in enumerate(batch_files):
        detected_flowers = results[i].boxes

        for det in detected_flowers:
            bbox = det.xyxy[0].numpy()
            conf = det.conf[0].numpy()
            if conf < confidence_threshold:
                continue

            features = extract_features(batch_images[i], bbox)

            # 花蕾匹配
            matched_id = None
            max_similarity = 0.95  # 设置一个相似度阈值
            for flower_id, track in flower_tracks.items():
                similarity = cosine_similarity([track['features']], [features])[0][0]
                if similarity > max_similarity:
                    matched_id = flower_id
                    max_similarity = similarity

            if matched_id is None:
                current_id += 1
                matched_id = current_id
                flower_tracks[matched_id] = {'features': features, 'bboxes': [bbox], 'images': [image_file]}
                flower_positions[matched_id] = []
            else:
                flower_tracks[matched_id]['features'] = features
                flower_tracks[matched_id]['bboxes'].append(bbox)
                flower_tracks[matched_id]['images'].append(image_file)

            # 添加三维点云数据（假设每张图像的z坐标不同）
            z_coord = len(flower_tracks[matched_id]['images'])
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            flower_positions[matched_id].append([x_center, y_center, z_coord])


batch_images = []
batch_files = []

for image_file in image_files:
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Image {image_file} not found or unable to load.")
        continue

    batch_images.append(image)
    batch_files.append(image_file)

    if len(batch_images) == batch_size:
        process_batch(batch_images, batch_files)
        batch_images = []
        batch_files = []

# 处理剩余的图像
if batch_images:
    process_batch(batch_images, batch_files)

# 输出结果
for flower_id, track in flower_tracks.items():
    print(f"Flower ID: {flower_id}")
    for bbox, img in zip(track['bboxes'], track['images']):
        print(f"Image: {img}, BBox: {bbox}")

# 可视化
for flower_id, track in flower_tracks.items():
    for bbox, img in zip(track['bboxes'], track['images']):
        image = cv2.imread(img)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"ID {flower_id}"
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow(f'Flower {flower_id}', image)
        cv2.waitKey(1000)  # 显示1秒钟
        cv2.destroyAllWindows()

# 生成三维追踪线路图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for flower_id, positions in flower_positions.items():
    positions = np.array(positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=f'Flower {flower_id}')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.legend()
plt.show()


# 多视图融合生成更全面的花序生长模型

def load_and_preprocess_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
    return images


# 替换为你的多视角图像路径列表
top_view_image_paths = [...]  # 顶视图图像路径列表
side_view_image_paths = [...]  # 侧视图图像路径列表

top_view_images = load_and_preprocess_images(top_view_image_paths)
side_view_images = load_and_preprocess_images(side_view_image_paths)


def extract_and_match_features(images1, images2):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    matches = []
    for img1 in images1:
        kp1, des1 = sift.detectAndCompute(img1, None)
        for img2 in images2:
            kp2, des2 = sift.detectAndCompute(img2, None)
            match = bf.knnMatch(des1, des2, k=2)

            # 应用Lowe's ratio test
            good_matches = []
            for m, n in match:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 0:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
                matches.append((src_pts, dst_pts))

    return matches


# 提取并匹配特征
matched_features = extract_and_match_features(top_view_images, side_view_images)


def generate_3d_point_cloud(matched_features, top_view_images, side_view_images):
    point_cloud = []

    for (src_pts, dst_pts) in matched_features:
        for (src, dst) in zip(src_pts, dst_pts):
            x, y = src
            z = dst[1]  # 假设y轴方向的偏移代表z坐标

            point_cloud.append([x, y, z])

    return np.array(point_cloud)


# 生成三维点云
point_cloud = generate_3d_point_cloud(matched_features, top_view_images, side_view_images)

# 可视化三维点云
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
