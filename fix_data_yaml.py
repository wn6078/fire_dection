import os
import yaml

def fix_data_yaml(yaml_path='data.yaml'):
    """
    修复data.yaml文件，使用本地路径替代Kaggle路径
    
    参数:
        yaml_path: 数据集配置文件路径
    """
    # 读取当前data.yaml文件
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # 获取当前工作目录
    cwd = os.getcwd()
    
    # 更新路径
    data['train'] = os.path.join(cwd, 'data/train/images')
    data['val'] = os.path.join(cwd, 'data/val/images')
    data['test'] = os.path.join(cwd, 'data/test/images')
    
    # 保存更新后的data.yaml文件
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"已更新 {yaml_path} 为本地路径")

if __name__ == '__main__':
    fix_data_yaml() 