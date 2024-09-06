import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
import json
import numpy as np
import random

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dic = [
    # "yolov8-C2f-DySnakeConv-CBAM-adown-detectAux",
    # "yolov8-C2f-DySnakeConv-CBAM-adown-dyhead",
    # "yolov8-C2f-DySnakeConv-CBAM-adown-ladh",
    # "yolov8-C2f-DySnakeConv-CBAM-adown-lscd",
    # "yolov8-C2f-DySnakeConv-CBAM-adown-lscsbd",
    # "yolov8-C2f-DySnakeConv-CBAM-adown-lsdecd", 
    # "yolov8-C2f-DySnakeConv-CBAM-adown-rtdetrdecoder",  走不通
    # "yolov8-C2f-DySnakeConv-CBAM-adown-seam"
    # "yolov8n-C2f-DySnakeConv-Triplet-adown-small",
    # "yolov8-C2f-DySnakeConv-CBAM-ASFP1",
    # "yolov8-C2f-DySnakeConv-CBAM-DBB",
    # "yolov8-C2f-DySnakeConv-CBAM-DySample",
    # "yolov8-C2f-DySnakeConv-CBAM-head",
    # "yolov8-C2f-DySnakeConv-CBAM-p6",
    # "yolov8-C2f-DySnakeConv-CBAM-ScalSeq",
    # "yolov8-C2f-DySnakeConv-CBAM-V7Down",
    # "yolov8-C2f-DySnakeConv-CBAM-VoVGSCSP"
    # "yolov8n",
    # "yolov8s",
    # "yolov8m",
    # "yolov8l",
    # # "yolov8x"
    # "yolov8s-C2f-AKConv", "yolov8s-C2f-ContextGuided", "yolov8s-C2f-DBB", 
    # "yolov8s-C2f-DCNV2", 
    # "yolov8s-C2f-DRB",
    # "yolov8s-C2f-DWR",
    # "yolov8-C2f-DySnakeConv-CBAM-DySample",
    # "yolov8-C2f-DySnakeConv-CBAM-DySample-Adown"
    # ,

    # "yolov8n-C2f-DySnakeConv-CBAM",
    # "yolov8n-C2f-DySnakeConv-CBAM",
    # "yolov8n-C2f-DySnakeConv-CBAM",
    # "yolov8n-C2f-DySnakeConv-CBAM",
    # "yolov8n-C2f-DySnakeConv-CBAM",
    # "yolov8n-C2f-DySnakeConv-CBAM",
    # "yolov8n-C2f-DySnakeConv-CBAM",
    # "yolov8s-C2f-DySnakeConv",
    # "yolov8s-C2f-DWR-SimAM",
    # "yolov8s-C2f-DWR-CoordAtt",
    # "yolov8s-C2f-DWR-SE",
    "yolov8s-C2f-DWR-Triplet",
    "yolov8s-C2f-DWR-BAMBlock",
    "yolov8s-C2f-DWR-BiLevel",
    "yolov8s-C2f-DWR-BiLevelNCHW",
    "yolov8s-C2f-DWR-CoordAtt",
    "yolov8s-C2f-DWR-EffectiveSE",
    "yolov8s-C2f-DWR-EMA"
    
    # "yolov8n-C2f-DySnakeConv",
    # "yolov8n-C2f-DySnakeConv",
    # "yolov8n-C2f-DySnakeConv",
    # "yolov8n-C2f-DySnakeConv",
    # "yolov8n-C2f-DySnakeConv-SimAM",
    # "yolov8n-C2f-DySnakeConv-SimAM",
    # "yolov8n-C2f-DySnakeConv-SimAM",
    # "yolov8n-C2f-DySnakeConv-SimAM",


    # "yolov8s-C2f-DySnakeConv-SimAM",
    # "yolov8s-C2f-DySnakeConv-SimAM-Adown",
    # "yolov8s-C2f-DySnakeConv-CoordAtt-Adown",
    # "yolov8n-C2f-DySnakeConv-CoordAtt-DySample",
    # "yolov8s-C2f-DySnakeConv-CoordAtt-Seam",
    # "yolov8s-C2f-DySnakeConv-h-CoordAtt",
    # "yolov8s-C2f-DySnakeConv-h-EffectiveSE",
    # "yolov8s-C2f-DySnakeConv-h-EMA",
    # "yolov8s-C2f-DySnakeConv-h-SE",
    # "yolov8s-C2f-DySnakeConv-h-SimAM",
    # "yolov8s-C2f-DySnakeConv-h-Triplet",
    # "yolov8s-C2f-DySnakeConv-SimAM",
    # "yolov8n-C2f-DySnakeConv-test"
]

base_folder = '/root/data/dataset/ultralytics-main/finalRun'

seed_everything(11)

for imp in dic:
    # Create a timestamp for unique experiment folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model-specific folder
    model_folder = os.path.join(base_folder, imp)
    os.makedirs(model_folder, exist_ok=True)
    
    # Create experiment folder within the model folder
    exp_folder = os.path.join(model_folder, f'exp_{timestamp}')
    os.makedirs(exp_folder, exist_ok=True)
    
    # Create train and val folders
    train_folder = os.path.join(exp_folder, 'train')
    val_folder = os.path.join(exp_folder, 'val')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # Create a log file to record validation results
    log_file = os.path.join(exp_folder, 'validation_results.log')
    
    yaml_file = f'/root/data/dataset/ultralytics-main/ultralytics/cfg/models/v8/{imp}.yaml'
    model = YOLO(yaml_file)
    
    # Set batch size
    batch_size = 16 if imp in ["yolov8s-C2f-AKConv", "yolov8s-C2f-DySnakeConv-h-BiLevel", "yolov8s-C2f-DySnakeConv-h-BiLevelNCHW"] else 32
    print(f"Batch size: {batch_size}")
    
    # Train the model
    model.train(data='/root/data/dataset/ultralytics-main/data/project_yolo/data.yaml',
                cache=True,
                imgsz=640,
                epochs=300,
                batch=batch_size,
                close_mosaic=20,
                workers=8,
                device='0',
                single_cls=True,
                optimizer='Adam',
                project=train_folder,
                name=imp
                )
    
    print(f"Training completed for {imp}")

    # Validate the model
    best_model_path = os.path.join(train_folder, imp, 'weights', 'best.pt')
    val_model = YOLO(best_model_path)
    results = val_model.val(data='/root/data/dataset/ultralytics-main/data/project_yolo/data.yaml',
                  split='test',
                  imgsz=640,
                  batch=1,
                  save_json=True,
                  project=val_folder,
                  name=imp,
                  plots=True
                  )

    print(f"Validation completed for {imp}")

    # Record validation results
    with open(log_file, 'a') as f:
        f.write(f"Validation results for {imp}:\n")
        json.dump(results.results_dict, f, indent=2)
        f.write('\n\n')

    # Clean up memory
    del model
    del val_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

print(f"All trainings and validations completed. Results saved in {base_folder}")