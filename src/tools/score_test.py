import torch
import os
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset
from src.network.resAcapsnet import create_resnet_capsnet  # 确保模型导入正确
from src.tools.newmain import robust_data_preprocessing  # 假设数据预处理代码在此文件中


class H5Dataset(Dataset):
    """自定义数据集类，用于加载 .h5 数据文件"""
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform

        # 加载 .h5 文件
        with h5py.File(self.file_path, 'r') as f:
            # 使用 'F' 数据集作为输入数据
            if 'F' in f:
                self.data = np.array(f['F'])  # 加载 'F' 数据集
            else:
                raise KeyError("没有找到数据集 'F'，请检查文件结构")

            # 如果文件中有标签，按照实际情况加载
            # self.labels = np.array(f['labels'])  # 如果有标签，加载标签（如果需要）

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取单个数据
        data = torch.tensor(self.data[idx], dtype=torch.float32)

        # 如果有transform（如数据预处理），可以在这里应用
        if self.transform:
            data = self.transform(data)

        return data, idx  # 返回数据和索引


# 加载训练好的模型
def load_trained_model(checkpoint_path, device='cpu'):
    """加载训练好的模型"""
    print("加载模型...")

    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    input_shape = checkpoint.get('input_shape', (311, 64))  # 假设输入形状为 (311, 64)，根据实际情况调整
    print(f"模型输入形状: {input_shape}")

    model = create_resnet_capsnet(input_shape=input_shape)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


# 计算异常分数
def generate_anomaly_scores(model, data_loader, device):
    """生成异常分数"""
    anomaly_scores = []
    with torch.no_grad():
        for batch_idx, (data, idx) in enumerate(data_loader):
            data = data.to(device)
            scores = model.compute_anomaly_score(data)
            anomaly_scores.extend(scores)

            if batch_idx % 10 == 0:
                print(f"生成异常分数进度: {batch_idx * data_loader.batch_size}/{len(data_loader.dataset)}")

    return np.array(anomaly_scores)


# 生成异常分数
def generate_scores(checkpoint_path, file_path, device='cpu', save_path=None):
    """加载模型并生成异常分数"""
    model = load_trained_model(checkpoint_path, device)

    # 创建数据集和DataLoader
    dataset = H5Dataset(file_path=file_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 计算测试集的异常分数
    anomaly_scores = generate_anomaly_scores(model, data_loader, device)

    # 保存异常分数到文件
    if save_path:
        np.save(os.path.join(save_path, 'test_anomaly_scores.npy'), anomaly_scores)
        print(f"异常分数已保存至: {os.path.join(save_path, 'test_anomaly_scores.npy')}")
    else:
        print(f"异常分数：{anomaly_scores}")

    return anomaly_scores


# 示例代码：如何使用
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 你的 .h5 文件路径
    file_path = 'D:/tools/Pycode/CapsnetForAD/workresult/testfeatures.h5'  # 你提供的 .h5 文件路径

    # 模型检查点路径
    checkpoint_path = 'D:/tools/Pycode/CapsnetForAD/workresult/superResnet_capsnet_results/resnet_capsnet_model.pth'

    # 生成异常分数
    anomaly_scores = generate_scores(checkpoint_path, file_path, device=device, save_path='D:/tools/Pycode/CapsnetForAD/workresult')

