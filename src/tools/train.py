# tools/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.capsnet import GCCaps, create_gccaps


class H5AudioDataset(Dataset):
    """从HDF5文件加载音频特征的Dataset"""

    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]  # shape: (61, 64)
        label = self.labels[idx]

        # 转换为PyTorch张量
        feature = torch.FloatTensor(feature)

        # 处理标签格式
        if isinstance(label, (int, np.integer)):
            label = torch.LongTensor([label])
        else:
            label = torch.FloatTensor(label)

        # 数据增强
        if self.transform:
            feature = self.transform(feature)

        return feature, label.squeeze()


def load_labels_from_csv(csv_path, filename_col='file_name', label_col='label'):
    """从CSV文件加载标签"""
    df = pd.read_csv(csv_path)
    print(f"从CSV加载数据: {len(df)} 行")
    print(f"列名: {df.columns.tolist()}")

    # 检查必需的列是否存在
    if filename_col not in df.columns:
        print(f"错误: CSV文件中没有 '{filename_col}' 列")
        print(f"可用的列: {df.columns.tolist()}")
        return {}

    # 检查标签列 - 如果没有label列，使用第一个数值列
    if label_col not in df.columns:
        # 查找第一个数值列作为标签
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            label_col = numeric_cols[0]
            print(f"使用 '{label_col}' 作为标签列")
        else:
            print("错误: 没有找到数值列作为标签")
            return {}

    # 创建文件名到标签的映射
    labels_dict = {}
    for _, row in df.iterrows():
        filename = row[filename_col]
        label = row[label_col]
        labels_dict[filename] = label

    print(f"成功加载 {len(labels_dict)} 个文件的标签")
    print(f"标签范围: {min(labels_dict.values())} 到 {max(labels_dict.values())}")

    return labels_dict


def match_features_with_labels(h5_path, labels_dict):
    """将HDF5特征与CSV标签匹配"""

    with h5py.File(h5_path, 'r') as f:
        features = f['F'][:]
        print(f"HDF5文件中的特征形状: {features.shape}")

        # 尝试获取文件名信息
        filenames = []
        if 'filenames' in f:
            filenames = [name.decode() if isinstance(name, bytes) else name for name in f['filenames'][:]]
            print("使用 'filenames' 数据集")
        elif 'timestamps' in f:
            # 时间戳不能用作文件名，需要其他方式
            timestamps = f['timestamps'][:]
            print(f"找到时间戳，但无法用于文件名匹配")
            # 使用索引创建临时文件名
            filenames = [f"section_00_source_train_normal_{i:04d}_car_D1_spd_31V_mic_1.wav" for i in
                         range(len(features))]
        else:
            # 创建基于索引的文件名
            print("创建基于索引的文件名")
            filenames = [f"section_00_source_train_normal_{i:04d}_car_D1_spd_31V_mic_1.wav" for i in
                         range(len(features))]

    print(f"HDF5中的文件数量: {len(filenames)}")
    print(f"CSV标签中的文件数量: {len(labels_dict)}")

    # 显示一些示例文件名进行调试
    print("\nHDF5中的前5个文件名:")
    for i in range(min(5, len(filenames))):
        print(f"  {filenames[i]}")

    print("\nCSV中的前5个文件名:")
    csv_filenames = list(labels_dict.keys())[:5]
    for filename in csv_filenames:
        print(f"  {filename}")

    # 创建匹配的特征和标签数组
    matched_features = []
    matched_labels = []
    unmatched_count = 0

    for i, h5_filename in enumerate(filenames):
        # 提取基本文件名（去掉路径）
        h5_basename = os.path.basename(h5_filename)

        # 尝试不同的匹配策略
        matched = False

        # 策略1: 精确匹配
        if h5_basename in labels_dict:
            matched_features.append(features[i])
            matched_labels.append(labels_dict[h5_basename])
            matched = True

        # 策略2: 去掉扩展名匹配
        elif not matched:
            h5_no_ext = os.path.splitext(h5_basename)[0]
            for csv_filename, label in labels_dict.items():
                csv_basename = os.path.basename(csv_filename)
                csv_no_ext = os.path.splitext(csv_basename)[0]
                if h5_no_ext == csv_no_ext:
                    matched_features.append(features[i])
                    matched_labels.append(label)
                    matched = True
                    break

        # 策略3: 部分匹配（包含关系）
        elif not matched:
            for csv_filename, label in labels_dict.items():
                csv_basename = os.path.basename(csv_filename)
                if h5_basename in csv_basename or csv_basename in h5_basename:
                    matched_features.append(features[i])
                    matched_labels.append(label)
                    matched = True
                    break

        # 策略4: 数字索引匹配
        elif not matched:
            # 提取文件名中的数字
            import re
            h5_numbers = re.findall(r'\d+', h5_basename)
            if h5_numbers:
                for csv_filename, label in labels_dict.items():
                    csv_basename = os.path.basename(csv_filename)
                    csv_numbers = re.findall(r'\d+', csv_basename)
                    if h5_numbers and csv_numbers and h5_numbers[-1] == csv_numbers[-1]:
                        matched_features.append(features[i])
                        matched_labels.append(label)
                        matched = True
                        break

        if not matched:
            unmatched_count += 1
            if unmatched_count <= 5:  # 只显示前5个未匹配的文件
                print(f"未匹配: {h5_basename}")

    print(f"\n匹配结果:")
    print(f"成功匹配 {len(matched_features)} 个文件")
    print(f"未匹配 {unmatched_count} 个文件")

    if len(matched_features) > 0:
        print(f"匹配的标签分布:")
        unique_labels, counts = np.unique(matched_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  类别 {label}: {count} 个样本")

    return np.array(matched_features), np.array(matched_labels)


def match_features_with_labels_simple(h5_path, labels_dict):
    """简单匹配：假设HDF5和CSV文件的顺序一致"""

    with h5py.File(h5_path, 'r') as f:
        features = f['F'][:]

    # 获取CSV文件名的有序列表
    csv_filenames = list(labels_dict.keys())

    # 检查数量是否匹配
    if len(features) != len(csv_filenames):
        print(f"警告: HDF5文件数 ({len(features)}) 与CSV文件数 ({len(csv_filenames)}) 不匹配")
        # 使用较小的数量
        min_len = min(len(features), len(csv_filenames))
        features = features[:min_len]
        csv_filenames = csv_filenames[:min_len]

    # 直接按顺序匹配
    matched_features = features
    matched_labels = [labels_dict[name] for name in csv_filenames]

    print(f"简单匹配结果: {len(matched_features)} 个文件")
    return np.array(matched_features), np.array(matched_labels)


def prepare_data(h5_path, csv_path, test_size=0.2, random_state=42, use_simple_match=True):
    """准备训练和测试数据"""

    print("加载CSV标签...")
    labels_dict = load_labels_from_csv(csv_path)

    print("匹配特征和标签...")
    if use_simple_match:
        features, labels = match_features_with_labels_simple(h5_path, labels_dict)
    else:
        features, labels = match_features_with_labels(h5_path, labels_dict)

    if len(features) == 0:
        raise ValueError("没有成功匹配任何特征和标签！")

    # 分析标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  类别 {label}: {count} 个样本")

    # 分割训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """训练模型"""

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            optimizer.zero_grad()
            output = model(data)

            # 计算损失
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs} | '
                      f'Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')

        # 计算训练准确率
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()

        print(f'Epoch: {epoch + 1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%')

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }


def main():
    # 配置参数
    base_dir = r"D:\tools\Pycode\CapsnetForAD"
    h5_path = os.path.join(base_dir, "workresult", "audio_features.h5")
    csv_path = os.path.join(base_dir, "data", "ToyCar", "training.csv")  # 根据您的实际CSV路径调整

    # 检查文件是否存在
    if not os.path.exists(h5_path):
        print(f"错误: HDF5文件不存在: {h5_path}")
        # 列出data目录下的文件
        data_dir = os.path.join(base_dir, "data")
        if os.path.exists(data_dir):
            print("data目录下的文件:")
            for file in os.listdir(data_dir):
                print(f"  {file}")
        return

    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return

    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    input_shape = (61, 64)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("准备数据...")
    X_train, X_test, y_train, y_test = prepare_data(h5_path, csv_path, use_simple_match=True)

    # 确保标签从0开始
    y_train = y_train - 1  # 将1,2转换为0,1
    y_test = y_test - 1  # 将1,2转换为0,1

    print("修正后的标签分布:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"训练集 - 类别 {unique_train}: {counts_train}")
    print(f"测试集 - 类别 {unique_test}: {counts_test}")

    # 创建数据集
    train_dataset = H5AudioDataset(X_train, y_train)
    test_dataset = H5AudioDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    print("创建模型...")
    model = create_gccaps(input_shape=input_shape, n_classes=2)
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 训练模型
    print("开始训练...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

    # 保存模型和训练历史
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'input_shape': input_shape,
        'num_classes': 2,
        'best_val_acc': history['best_val_acc']
    }

    torch.save(checkpoint, '../../workresult/gccaps_model.pth')

    # 保存训练历史为JSON
    with open('../../workresult/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"训练完成！最佳验证准确率: {history['best_val_acc']:.2f}%")
    print("模型已保存为 'gccaps_model.pth'")
    print("训练历史已保存为 'training_history.json'")

    return model, history


if __name__ == "__main__":
    model, history = main()