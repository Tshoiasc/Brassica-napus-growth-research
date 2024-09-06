import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/data/dataset/ultralytics-main/ultralytics/cfg/models/v8/yolov8n-C2f-DySnakeConv.yaml')
    model.load('yolov8m.pt') # loading pretrain weights
    model.train(data='/root/data/dataset/ultralytics-main/shishi/data.yaml',
                cache=True,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='AdamW', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs2/train',
                name='exp',
                )