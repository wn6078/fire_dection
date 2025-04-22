"""
YOLOv8训练高级配置文件
"""

class TrainingConfig:
    # 数据集配置
    data_yaml = 'data.yaml'  # 数据集配置文件路径
    
    # 模型配置
    model_type = 'yolov8n.pt'  # 模型类型：yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    pretrained = True          # 是否使用预训练权重
    resume = False             # 是否从上次中断处继续训练
    
    # 训练超参数
    epochs = 100               # 训练轮数
    batch_size = 16            # 批次大小
    imgsz = 640                # 图像大小
    
    # 优化器配置
    optimizer = 'Adam'         # 可选: SGD, Adam, AdamW
    lr0 = 0.01                 # 初始学习率
    lrf = 0.01                 # 最终学习率 = lr0 * lrf
    momentum = 0.937           # SGD动量
    weight_decay = 0.0005      # 权重衰减
    
    # 学习率调度
    warmup_epochs = 3.0        # 预热轮数
    warmup_momentum = 0.8      # 预热动量
    warmup_bias_lr = 0.1       # 预热偏置学习率
    
    # 损失函数权重
    box = 7.5                  # 框损失增益
    cls = 0.5                  # 类别损失增益
    dfl = 1.5                  # DFL损失增益
    
    # 训练技巧
    amp = True                 # 自动混合精度训练
    cache = True               # 缓存图像以加速训练
    workers = 8                # 数据加载线程数
    device = '0'               # 使用的设备，'0'表示第一个GPU，'cpu'表示CPU
    
    # 增强配置
    augment = True             # 是否使用数据增强
    mosaic = 1.0               # 马赛克增强系数，0.0表示禁用
    mixup = 0.1                # 混合增强系数，0.0表示禁用
    
    # 输出配置
    project = 'runs/train'     # 保存结果的项目文件夹
    name = None                # 实验名称，None会自动生成时间戳名称
    exist_ok = True            # 是否允许覆盖现有实验文件夹
    
    # 验证配置
    val = True                 # 是否在训练期间进行验证
    save_period = -1           # 每隔多少epoch保存一次模型，-1表示只保存最后一个
    
    # 回调函数配置
    plots = True               # 是否绘制训练图表
    save = True                # 是否保存训练结果
    save_json = False          # 是否保存json格式的预测结果

class PredictionConfig:
    # 预测配置
    conf_threshold = 0.25      # 置信度阈值
    iou_threshold = 0.45       # IOU阈值
    max_det = 300              # 每张图像最大检测框数量
    classes = None             # 过滤特定类别，None表示所有类别
    
    # 可视化配置
    line_width = None          # 边界框线宽，None表示自动
    hide_labels = False        # 是否隐藏标签
    hide_conf = False          # 是否隐藏置信度
    half = False               # 是否使用FP16推理
    
    # 输出配置
    save_crop = False          # 是否保存裁剪的预测框
    save_txt = False           # 是否将预测结果保存为txt文件
    save_conf = False          # 是否在txt结果中包含置信度
    save_json = False          # 是否保存json格式的预测结果
    project = 'runs/predict'   # 保存结果的项目文件夹
    name = 'exp'               # 实验名称
    exist_ok = True            # 是否允许覆盖现有实验文件夹 