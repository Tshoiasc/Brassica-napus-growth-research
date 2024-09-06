import json
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
from ultralytics import YOLO


def get_image_paths(img_path_sub):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_paths = []

    for root, dirs, files in os.walk(img_path_sub):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def save_preds(preds, plant_id):
    filename = f"results/{plant_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(preds, f, ensure_ascii=False, indent=4)


def serialize_pred(pred, image_path):
    serialized = {
        'image_name': os.path.basename(image_path),
        'image_path': image_path,
        'predictions': []
    }
    for p in pred:
        serialized['predictions'].append({
            'boxes': p.boxes.xyxy.tolist(),
            'conf': p.boxes.conf.tolist(),
            'cls': p.boxes.cls.tolist()
        })
    return serialized


if __name__ == '__main__':
    model = YOLO('bets2.pt')  # select your bets2.pt path
    img_path = "/Volumes/Elements SE/植物数据集处理"

    # 读取json
    with open("data_available.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历data with progress bar
    for item in tqdm(data, desc="Processing plants"):
        preds = {}
        for j in ["sv-000", "sv-045", "sv-090"]:
            img_path_sub = os.path.join(img_path, item["plant_id"], j)
            print(f"Processing: {img_path_sub}")

            # 获取指定目录下所有图片
            image_paths = get_image_paths(img_path_sub)
            # 按照名称排序
            image_paths.sort()

            # 预测图片
            results = model.predict(source=image_paths)

            # 序列化预测结果
            preds[j] = [serialize_pred(pred, img_path) for pred, img_path in zip(results, image_paths)]

        # 保存当前植物的预测结果
        save_preds(preds, item["plant_id"])

    print("All predictions have been saved.")