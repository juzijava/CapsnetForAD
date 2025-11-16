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


class AnomalyDetectionModel(nn.Module):
    """修改为异常检测的模型"""

    def __init__(self, input_shape, n_classes=2):
        super().__init__()
        # 使用你现有的CapsNet架构
        self.capsnet = create_gccaps(input_shape=input_shape, n_classes=n_classes)

        # 添加异常分数输出层
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(n_classes, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1的异常分数
        )

    def forward(self, x):
        # 获取CapsNet的特征
        features = self.capsnet(x)

        # 计算异常分数
        anomaly_score = self.anomaly_scorer(features)

        return anomaly_score.squeeze()

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

        # 修复标签处理
        if isinstance(label, (int, np.integer, np.float32, np.float64)):
            # 单个数值，直接转换为标量张量
            label = torch.tensor(label, dtype=torch.float32)
        elif isinstance(label, (list, np.ndarray)):
            # 数组类型
            label = torch.FloatTensor(label)
        else:
            # 其他情况
            label = torch.tensor(label, dtype=torch.float32)

        # 数据增强
        if self.transform:
            feature = self.transform(feature)

        return feature, label


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


def train_anomaly_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """训练异常检测模型"""

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 将标签转换为异常分数格式：正常=0, 异常=1
            anomaly_target = target.float()

            # 前向传播
            optimizer.zero_grad()
            anomaly_scores = model(data)

            # 计算损失
            loss = criterion(anomaly_scores, anomaly_target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs} | '
                      f'Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')

        # 计算训练损失
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                anomaly_target = target.float()

                anomaly_scores = model(data)
                loss = criterion(anomaly_scores, anomaly_target)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        # 保存最佳模型（基于验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        print(f'Epoch: {epoch + 1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f}')

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


def generate_anomaly_scores(model, data_loader, device):
    """生成异常分数"""
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            scores = model(data)
            anomaly_scores.extend(scores.cpu().numpy())

    return anomaly_scores


def save_anomaly_scores(anomaly_scores, output_dir):
    """保存异常分数到文件"""
    # 创建示例文件名（根据实际数据调整）
    filenames = [f"test_{i:03d}.wav" for i in range(len(anomaly_scores))]

    # 保存异常分数CSV
    df_anomaly = pd.DataFrame({
        'filename': filenames,
        'anomaly_score': anomaly_scores
    })
    df_anomaly.to_csv(os.path.join(output_dir, 'anomaly_scores.csv'), index=False)

    # 保存决策结果CSV（基于阈值0.5）
    df_decision = pd.DataFrame({
        'filename': filenames,
        'prediction': (np.array(anomaly_scores) > 0.5).astype(int)
    })
    df_decision.to_csv(os.path.join(output_dir, 'decision_results.csv'), index=False)

    print(f"异常分数已保存到 {output_dir}")
    print(f"异常分数范围: {min(anomaly_scores):.4f} - {max(anomaly_scores):.4f}")

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

    # 修改标签：将分类标签转换为异常检测标签
    # 假设正常样本标签为1，异常样本标签为2
    # 转换为：正常=0，异常=1
    print("转换标签格式...")
    y_train = (y_train - 1).astype(float)  # 1->0.0(正常), 2->1.0(异常)
    y_test = (y_test - 1).astype(float)  # 1->0.0(正常), 2->1.0(异常)

    print("转换后的标签分布:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"训练集 - 正常: {counts_train[0]}, 异常: {counts_train[1]}")
    print(f"测试集 - 正常: {counts_test[0]}, 异常: {counts_test[1]}")
    # 创建数据集
    train_dataset = H5AudioDataset(X_train, y_train)
    test_dataset = H5AudioDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    print("创建异常检测模型...")
    model = AnomalyDetectionModel(input_shape=input_shape, n_classes=2)
    model = model.to(device)

    # 使用加权MSELoss
    class WeightedMSELoss(nn.Module):
        def __init__(self, weight_normal=1.0, weight_anomaly=100.0):
            super().__init__()
            self.weight_normal = weight_normal
            self.weight_anomaly = weight_anomaly

        def forward(self, pred, target):
            # 为异常样本分配更高权重
            weights = torch.where(target == 1,
                                  torch.tensor(self.weight_anomaly).to(pred.device),
                                  torch.tensor(self.weight_normal).to(pred.device))
            loss = (pred - target) ** 2
            weighted_loss = loss * weights
            return weighted_loss.mean()

    # 使用高权重处理类别不平衡
    criterion = WeightedMSELoss(weight_normal=1.0, weight_anomaly=500.0)  # 异常样本权重500倍
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 修改训练函数调用
    print("开始训练异常检测模型...")
    history = train_anomaly_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

    # 保存模型
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'input_shape': input_shape,
        'best_val_loss': history['best_val_loss']
    }

    torch.save(checkpoint, '../../workresult/gccaps_anomaly_model.pth')

    print(f"训练完成！最佳验证损失: {history['best_val_loss']:.4f}")
    print("异常检测模型已保存为 'gccaps_anomaly_model.pth'")

    # 新增：生成测试集的异常分数
    # 生成测试集的异常分数
    print("生成测试集异常分数...")
    test_anomaly_scores = generate_anomaly_scores(model, train_loader, device)

    # 保存异常分数
    save_anomaly_scores(test_anomaly_scores, '../../workresult/')

    # 在main函数内部分析模型性能
    print("\n分析模型性能...")
    analyze_model_performance(model, train_loader, device)

    return model, history, test_anomaly_scores


def analyze_model_performance(model, data_loader, device):
    """分析模型性能"""
    model.eval()
    normal_scores = []
    anomaly_scores = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            scores = model(data)

            # 分离正常和异常样本的分数
            normal_mask = (target == 0)
            anomaly_mask = (target == 1)

            if normal_mask.any():
                normal_scores.extend(scores[normal_mask].cpu().numpy())
            if anomaly_mask.any():
                anomaly_scores.extend(scores[anomaly_mask].cpu().numpy())

    print("\n" + "=" * 50)
    print("模型性能详细分析:")
    print("=" * 50)

    if normal_scores:
        print(f"正常样本 ({len(normal_scores)}个):")
        print(f"  分数范围: {np.min(normal_scores):.4f} - {np.max(normal_scores):.4f}")
        print(f"  平均值: {np.mean(normal_scores):.4f}")
        print(f"  标准差: {np.std(normal_scores):.4f}")

    if anomaly_scores:
        print(f"异常样本 ({len(anomaly_scores)}个):")
        print(f"  分数范围: {np.min(anomaly_scores):.4f} - {np.max(anomaly_scores):.4f}")
        print(f"  平均值: {np.mean(anomaly_scores):.4f}")
        print(f"  标准差: {np.std(anomaly_scores):.4f}")

    if normal_scores and anomaly_scores:
        separation = np.mean(anomaly_scores) - np.mean(normal_scores)
        print(f"\n分离度分析:")
        print(f"  均值差异: {separation:.4f}")
        print(f"  分离效果: {'好' if separation > 0.1 else '一般' if separation > 0.05 else '差'}")

        # 计算AUC（如果可能）
        from sklearn.metrics import roc_auc_score
        try:
            all_scores = normal_scores + anomaly_scores
            all_labels = [0] * len(normal_scores) + [1] * len(anomaly_scores)
            auc = roc_auc_score(all_labels, all_scores)
            print(f"  AUC: {auc:.4f}")
        except:
            print("  AUC: 无法计算（可能需要更多样本）")


if __name__ == "__main__":
    model, history, test_anomaly_scores = main()