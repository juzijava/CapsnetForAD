# config.py
import torch

# 数据配置
DATA_CONFIG = {
    # 音频数据路径
    'audio_dir': r"D:\tools\Pycode\CapsnetForAD\data\ToyCar\train",
    'output_file': r"D:\tools\Pycode\CapsnetForAD\workresult\audio_features.h5",

    # 特征提取参数
    'feature_extraction': {
        'duration': 2.0,
        'sample_rate': 16000,
        'n_window': 1024,
        'hop_length': 512,
        'n_mels': 64,
        'recompute': False
    },

    # 模型输入参数
    'input_shape': (61, 64),
    'num_classes': 2
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'step_size': 20,
    'gamma': 0.1
}

# 模型配置
MODEL_CONFIG = {
    'input_shape': (61, 64),
    'n_classes': 2
}

# 文件路径配置
PATH_CONFIG = {
    'model_save_path': 'gccaps_model.pth'
}

# 设备配置
DEVICE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}