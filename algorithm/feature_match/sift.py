import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)

# 创建SIFT对象
sift = cv2.SIFT_create()

# 查找关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN参数
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

# 使用FLANN匹配器
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# 应用比率测试
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果
plt.figure(figsize=(20,10))
plt.imshow(img_matches)
plt.show()

# 打印匹配点数量
print(f"找到 {len(good_matches)} 个良好匹配点")