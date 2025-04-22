import cv2
import numpy as np
import os
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import Qt

# 为每个类别预定义鲜艳的颜色
# 烟雾用蓝色，火灾用红色
CLASS_COLORS = {
    "smoke": (0, 0, 255),    # 红色(BGR)
    "fire": (0, 165, 255),   # 橙色(BGR)
    0: (0, 0, 255),          # 红色(BGR) - 数字索引
    1: (0, 165, 255),        # 橙色(BGR) - 数字索引
    # 可以根据需要添加更多类别颜色
}

# 默认颜色集合（如果没有为某个类别指定颜色）
DEFAULT_COLORS = [
    (255, 0, 0),      # 蓝色(BGR)
    (0, 255, 0),      # 绿色(BGR)
    (0, 0, 255),      # 红色(BGR)
    (0, 255, 255),    # 黄色(BGR)
    (255, 0, 255),    # 紫色(BGR)
    (255, 255, 0),    # 青色(BGR)
    (128, 0, 255),    # 粉色(BGR)
    (0, 128, 255),    # 橙色(BGR)
]

def get_color_for_class(class_id, class_name=None):
    """获取类别对应的颜色"""
    # 优先使用类名查找
    if class_name and class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    
    # 其次使用类ID查找
    if class_id in CLASS_COLORS:
        return CLASS_COLORS[class_id]
    
    # 最后使用默认颜色循环
    return DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]

def cv2_to_qpixmap(cv_img):
    """将OpenCV图像转换为QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_image)

def count_detections(results):
    """统计每个类别的检测数量"""
    class_counts = {}
    
    for r in results:
        # 获取检测到的类别和数量
        if hasattr(r, 'boxes') and hasattr(r.boxes, 'cls'):
            boxes = r.boxes
            for cls in boxes.cls.cpu().numpy():
                cls_id = int(cls)
                cls_name = r.names[cls_id] if hasattr(r, 'names') else f"类别{cls_id}"
                
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                else:
                    class_counts[cls_name] = 1
    
    return class_counts

def scale_pixmap_to_label(pixmap, label):
    """按比例缩放图像以适应标签大小"""
    label_size = label.size()
    return pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def plot_with_custom_colors(result, line_width=2):
    """使用自定义颜色绘制检测结果"""
    # 复制原始图像以避免修改源图像
    annotated_frame = result.orig_img.copy()
    
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            
            # 获取类别ID和名称
            cls_id = int(box.cls.cpu().numpy()[0])
            cls_name = result.names[cls_id] if hasattr(result, 'names') else f"类别{cls_id}"
            
            # 获取置信度
            conf = float(box.conf.cpu().numpy()[0])
            
            # 获取该类别的颜色
            color = get_color_for_class(cls_id, cls_name)
            
            # 绘制边界框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, line_width)
            
            # 绘制类别标签和置信度
            label = f"{cls_name} {conf:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return annotated_frame

def check_model_path(model_path):
    """检查模型路径是否有效"""
    # 首先检查是否为绝对路径
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return True, model_path
    
    # 检查相对路径
    if os.path.exists(model_path):
        return True, os.path.abspath(model_path)
    
    # 检查预训练模型名称 (yolov8n.pt, yolov8s.pt 等)
    if model_path.startswith("yolov8") and model_path.endswith(".pt"):
        # 使用ultralytics默认下载位置
        home_dir = os.path.expanduser("~")
        model_dir = os.path.join(home_dir, ".ultralytics", "models")
        full_path = os.path.join(model_dir, model_path)
        
        if os.path.exists(full_path):
            return True, full_path
    
    return False, model_path 