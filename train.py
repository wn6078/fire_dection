from ultralytics import YOLO
import os
import yaml
from datetime import datetime
# from config import TrainingConfig

def parse_args():
    # 不再使用 argparse 解析命令行参数
    class Args:
        def __init__(self):
            self.data_yaml = 'data.yaml'
            self.model_type = 'yolov8s.pt'
            self.pretrained = True
            self.resume = False
            self.epochs = 100
            self.batch_size = 32
            self.imgsz = 640
            self.optimizer = 'SGD'
            self.lr0 = 0.01
            self.lrf = 0.001
            self.momentum = 0.937
            self.weight_decay = 0.0005
            self.warmup_epochs = 3.0
            self.mosaic = 1.0
            self.mixup = 0.1
            self.copy_paste = 0.1
            self.project = 'runs/train'
            self.name = 'improved_exp'
            self.device = ''
            self.resume = False

    return Args()

def train_yolo(args):
    # 使用从 parse_args 函数获取的参数
    model_type = args.model_type
    epochs = args.epochs
    batch_size = args.batch_size

    # 初始化模型
    if args.pretrained:
        model = YOLO(model_type)
    else:
        model = YOLO(model_type.replace('.pt', '.yaml'))

    # 训练模型
    results = model.train(
        data=args.data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        project=args.project,
        name=args.name,
        device=args.device,
        resume=args.resume,
        patience=50,  # 早停策略
        verbose=True,
        amp=True,  # 混合精度训练
        cos_lr=True,  # 余弦退火学习率
    )

    return results

def validate_yolo(model_path, data_yaml='data.yaml'):
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate the model
    results = model.val(data=data_yaml)
    
    return results

def test_yolo(model_path, data_yaml='data.yaml'):
    # Load the trained model
    model = YOLO(model_path)
    
    # Test the model
    results = model.val(data=data_yaml, split='test')
    
    return results

if __name__ == '__main__':
    args = parse_args()
    results = train_yolo(args)
    print("Training completed. Results:", results)
    
    # Validate the best model
    best_model_path = os.path.join(args.project, args.name, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        val_results = validate_yolo(best_model_path, args.data_yaml)
        print(f"Validation results for best model: {val_results}")
        
        # Test the best model
        test_results = test_yolo(best_model_path, args.data_yaml)
        print(f"Test results for best model: {test_results}") 