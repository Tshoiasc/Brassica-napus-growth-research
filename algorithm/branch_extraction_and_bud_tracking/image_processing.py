import cv2
import numpy as np
from skimage.morphology import skeletonize

def process_green_with_detection(image_path, result, lowest_box):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = 8
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 0
    skeleton = skeletonize(binary // 255)

    skeleton = (skeleton * 255).astype(np.uint8)

    skeleton[int(lowest_box[3]):, :] = 0
    color_mask = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    boxes = result[0].boxes.xyxy.cpu().numpy()
    red_centers = []
    green_centers = []
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        if center[1] < int(lowest_box[3]):
            cv2.rectangle(color_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)
            green_centers.append(center)
    x1, y1, x2, y2 = lowest_box.astype(int)
    red_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    cv2.rectangle(color_mask, (x1, y1), (x2, y2), (255, 0, 0), 2)
    red_centers.append(red_center)
    return color_mask, red_centers, green_centers, skeleton