import os
from typing import List, Dict
import sys
from PIL import Image
from ultralytics import YOLO
from urllib.parse import urlparse
import logging
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys

# 设置日志级别
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

# 验证 API Key 设置
api_key = os.environ.get('LABEL_STUDIO_API_KEY')
if not api_key:
    raise EnvironmentError("LABEL_STUDIO_API_KEY is not set. Please set this environment variable to access uploaded files.")
logger.info(f"API Key is set: {api_key[:5]}...{api_key[-5:]}")

class NewModel(LabelStudioMLBase):
    _model_version = "v8s"  # 直接在类定义中初始化

    def __init__(self, **kwargs):
        # 先初始化父类
        super(NewModel, self).__init__(**kwargs)
        
        self._conf_threshold = 0.5
        self._model = YOLO("best.pt")  # 确保这个路径是正确的

        # 初始化用于边界框检测的变量
        self.from_name, self.to_name, self.value, self.labels = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        
        logger.info("NewModel initialized")

    @property
    def model_version(self):
        return self._model_version

    @property
    def conf_threshold(self):
        return self._conf_threshold

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        logger.info("Entering predict method")
        
        if not tasks:
            logger.warning("No tasks provided")
            return []
    
        predictions = []
        for task in tasks:
            logger.info(f"Processing task: {task}")
    
            # Getting path of the image
            original_path = task['data'][self.value]
            logger.info(f"Original image path: {original_path}")
    
            # 处理三种不同的路径情况
            if urlparse(original_path).scheme:
                logger.info("Case 1: Direct image URL")
                image_path = original_path
            elif original_path.startswith('/data/local-files/'):
                logger.info("Case 2: Local dataset path")
                image_path = original_path.replace('/data/local-files/?d=', '/')
            elif original_path.startswith('/data/upload/'):
                logger.info("Case 3: Uploaded local image")
                image_path = original_path.replace('/data/upload/', '/workspace/label-studio-dataset/media/upload/')
            else:
                logger.warning(f"Unrecognized path format: {original_path}")
                continue
    
            logger.info(f"Processed image path: {image_path}")
    
            # 检查文件是否存在
            if not os.path.exists(image_path) and not urlparse(image_path).scheme:
                logger.error(f"File not found: {image_path}")
                continue
    
            # Load the image
            try:
                image = Image.open(image_path)
                logger.info(f"Image loaded successfully. Size: {image.size}")
            except Exception as e:
                logger.error(f"Error loading image: {str(e)}")
                continue
    
            # Height and width of image
            original_width, original_height = image.size
    
            # Getting prediction using model
            logger.info("Starting YOLO prediction")
            results = self._model.predict(source=image_path, conf=self._conf_threshold)
            logger.info(f"YOLO prediction completed. Number of results: {len(results)}")
    
            task_predictions = []
            total_score = 0
            num_predictions = 0
    
            # Getting boxes from model prediction
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    class_id = int(box.cls.item())
                    label = self._model.names[class_id]
    
                    # 将坐标转换为相对值
                    x = x1 / original_width * 100
                    y = y1 / original_height * 100
                    width = (x2 - x1) / original_width * 100
                    height = (y2 - y1) / original_height * 100
    
                    task_predictions.append({
                        "from_name": self.from_name,
                        "to_name": self.to_name,
                        "type": "rectanglelabels",
                        "score": conf,
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "rectanglelabels": [label]
                        }
                    })
    
                    total_score += conf
                    num_predictions += 1
                    logger.info(f"Prediction: label={label}, score={conf:.3f}")
    
            logger.info(f"Total predictions for this task: {num_predictions}")
            average_score = total_score / num_predictions if num_predictions > 0 else 0
            logger.info(f"Average prediction score: {average_score:.3f}")
    
            predictions.append({
                "result": task_predictions,
                "score": average_score,
                "model_version": self._model_version
            })
    
        logger.info("Predict method completed")
        return predictions
    
    def fit(self, event, data, **kwargs):
        """
        目前，由于我们使用的是预训练的YOLO模型，这个方法可以保持为空
        或者您可以在这里实现模型的微调逻辑
        """
        pass