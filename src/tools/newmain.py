import os
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import json
from datetime import datetime

# 导入修复后的ResNet胶囊网络
from src.network.resAcapsnet import create_resnet_capsnet, train_resnet_capsnet


class H5AudioDataset(torch.utils.data.Dataset):
    """音频数据集类"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def robust_data_preprocessing(features):
    """鲁棒的数据预处理"""
    print("进行数据预处理...")

    # 处理NaN和Inf
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    # 检查数据统计
    print(f"原始数据 - 范围: [{np.min(features):.6f}, {np.max(features):.6f}]")
    print(f"原始数据 - 均值: {np.mean(features):.6f}, 标准差: {np.std(features):.6f}")

    # 稳健的标准化
    median = np.median(features)
    mad = np.median(np.abs(features - median))
    if mad > 0:
        features = (features - median) / (mad + 1e-8)
        print("使用中位数绝对偏差标准化")
    else:
        std = np.std(features)
        if std > 0:
            features = (features - np.mean(features)) / (std + 1e-8)
            print("使用标准差标准化")
        else:
            print("警告: 数据方差为0，跳过标准化")

    # 最终裁剪
    features = np.clip(features, -10, 10)

    print(f"预处理后 - 范围: [{np.min(features):.6f}, {np.max(features):.6f}]")
    print(f"预处理后 - 均值: {np.mean(features):.6f}, 标准差: {np.std(features):.6f}")

    return features


def generate_anomaly_scores(model, data_loader, device):
    """生成异常分数"""
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            scores = model.compute_anomaly_score(data)
            anomaly_scores.extend(scores)

            if batch_idx % 10 == 0:
                print(f"生成异常分数进度: {batch_idx * data_loader.batch_size}/{len(data_loader.dataset)}")

    return np.array(anomaly_scores)


def analyze_anomaly_scores_detailed(anomaly_scores, true_labels=None, save_path=None):
    """详细分析异常分数"""
    print("\n" + "=" * 60)
    print("异常分数详细分析")
    print("=" * 60)

    # 基本统计
    print(f"样本数量: {len(anomaly_scores)}")
    print(f"分数范围: {np.min(anomaly_scores):.6f} - {np.max(anomaly_scores):.6f}")
    print(f"平均值: {np.mean(anomaly_scores):.6f}")
    print(f"中位数: {np.median(anomaly_scores):.6f}")
    print(f"标准差: {np.std(anomaly_scores):.6f}")

    # 分数分布
    percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    print("\n分数分位数:")
    percentile_data = {}
    for p in percentiles:
        value = np.percentile(anomaly_scores, p)
        percentile_data[p] = float(value)
        print(f"  {p}%: {value:.6f}")

    # 推荐阈值
    threshold_90 = np.percentile(anomaly_scores, 90)
    threshold_95 = np.percentile(anomaly_scores, 95)
    threshold_99 = np.percentile(anomaly_scores, 99)

    print(f"\n推荐异常阈值:")
    print(f"  90%分位数: {threshold_90:.6f}")
    print(f"  95%分位数: {threshold_95:.6f}")
    print(f"  99%分位数: {threshold_99:.6f}")

    # 如果有真实标签，计算性能
    performance_data = {}
    if true_labels is not None:
        try:
            # 计算AUC
            auc = roc_auc_score(true_labels, anomaly_scores)
            print(f"\n性能指标:")
            print(f"  AUC: {auc:.4f}")

            # 找到最佳F1分数的阈值
            precision, recall, thresholds = precision_recall_curve(true_labels, anomaly_scores)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
            best_f1 = f1_scores[best_idx]

            print(f"  最佳F1分数: {best_f1:.4f}")
            print(f"  对应阈值: {best_threshold:.6f}")

            performance_data = {
                'auc': float(auc),
                'best_f1': float(best_f1),
                'best_threshold': float(best_threshold)
            }

        except Exception as e:
            print(f"性能计算错误: {e}")

    # 绘制分布图
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(anomaly_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(threshold_95, color='red', linestyle='--', label=f'95%Threshold: {threshold_95:.4f}')
    plt.axvline(threshold_99, color='orange', linestyle='--', label=f'99%Threshold: {threshold_99:.4f}')
    plt.xlabel('Abnormal_score')
    plt.ylabel('Frequency')
    plt.title('Abnormal_score_distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.boxplot(anomaly_scores)
    plt.ylabel('Abnormal_score')
    plt.title('Abnormal score box plot')

    plt.subplot(2, 2, 3)
    sorted_scores = np.sort(anomaly_scores)
    plt.plot(sorted_scores, np.arange(len(sorted_scores)) / len(sorted_scores))
    plt.xlabel('Abnormal_score')
    plt.ylabel('Cumulative probability')
    plt.title('Cumulative distribution function of abnormal scores')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    percentiles_plot = list(percentile_data.keys())
    values_plot = list(percentile_data.values())
    plt.plot(percentiles_plot, values_plot, 'o-')
    plt.xlabel('Quantile (%)')
    plt.ylabel('Abnormal_score')
    plt.title('Quantile_plot')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'anomaly_scores_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"分析图已保存至: {os.path.join(save_path, 'anomaly_scores_analysis.png')}")
    else:
        plt.savefig('anomaly_scores_analysis.png', dpi=300, bbox_inches='tight')

    plt.close()

    # 保存分析结果
    analysis_results = {
        'basic_stats': {
            'n_samples': len(anomaly_scores),
            'min_score': float(np.min(anomaly_scores)),
            'max_score': float(np.max(anomaly_scores)),
            'mean_score': float(np.mean(anomaly_scores)),
            'median_score': float(np.median(anomaly_scores)),
            'std_score': float(np.std(anomaly_scores))
        },
        'percentiles': percentile_data,
        'recommended_thresholds': {
            'threshold_90': float(threshold_90),
            'threshold_95': float(threshold_95),
            'threshold_99': float(threshold_99)
        },
        'performance': performance_data,
        'timestamp': datetime.now().isoformat()
    }

    if save_path:
        with open(os.path.join(save_path, 'anomaly_analysis_results.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"分析结果已保存至: {os.path.join(save_path, 'anomaly_analysis_results.json')}")
    else:
        with open('anomaly_analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)

    return analysis_results


def visualize_reconstruction_results(model, test_loader, device, num_samples=5, save_path=None):
    """可视化重建结果"""
    model.eval()

    # 获取测试样本
    data_iter = iter(test_loader)
    data, _ = next(data_iter)
    data = data[:num_samples].to(device)

    with torch.no_grad():
        reconstructed, _ = model(data)

    # 转换为numpy
    original_np = data.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()

    # 绘制对比图
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # 原始频谱
        im0 = axes[i, 0].imshow(original_np[i].T, aspect='auto', origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'Sample {i + 1} - Original spectrum')
        axes[i, 0].set_xlabel('Time frame')
        axes[i, 0].set_ylabel('Frequency bin')
        plt.colorbar(im0, ax=axes[i, 0])

        # 重建频谱
        im1 = axes[i, 1].imshow(reconstructed_np[i].T, aspect='auto', origin='lower', cmap='viridis')
        axes[i, 1].set_title(f'Sample {i + 1} - Reconstruction of the spectrum')
        axes[i, 1].set_xlabel('Time frame')
        axes[i, 1].set_ylabel('Frequency bin')
        plt.colorbar(im1, ax=axes[i, 1])

        # 重建误差
        error = np.abs(original_np[i] - reconstructed_np[i])
        im2 = axes[i, 2].imshow(error.T, aspect='auto', origin='lower', cmap='hot')
        axes[i, 2].set_title(f'Sample {i + 1} - Reconstruction of the spectrum')
        axes[i, 2].set_xlabel('Time frame')
        axes[i, 2].set_ylabel('Frequency bin')
        plt.colorbar(im2, ax=axes[i, 2])

    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'reconstruction_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"重建结果可视化已保存至: {os.path.join(save_path, 'reconstruction_comparison.png')}")
    else:
        plt.savefig('reconstruction_comparison.png', dpi=300, bbox_inches='tight')

    plt.close()


def evaluate_model_performance(model, train_loader, test_loader, threshold_95, device, save_path=None):
    """评估模型性能，使用训练时计算的95%阈值"""
    print("\n评估模型性能...")

    # 计算训练集和测试集的异常分数
    print("计算训练集异常分数...")
    train_scores = generate_anomaly_scores(model, train_loader, device)
    print("计算测试集异常分数...")
    test_scores = generate_anomaly_scores(model, test_loader, device)

    # 计算统计量
    train_mean = np.mean(train_scores)
    train_std = np.std(train_scores)
    test_mean = np.mean(test_scores)
    test_std = np.std(test_scores)

    print(f"训练集异常分数 - 均值: {train_mean:.6f}, 标准差: {train_std:.6f}")
    print(f"测试集异常分数 - 均值: {test_mean:.6f}, 标准差: {test_std:.6f}")

    # 使用训练时计算的95%阈值
    anomalies_95percent = np.sum(test_scores > threshold_95)

    # 同时计算Z-score阈值作为参考
    z_score_threshold_3 = train_mean + 3 * train_std
    z_score_threshold_5 = train_mean + 5 * train_std
    anomalies_3sigma = np.sum(test_scores > z_score_threshold_3)
    anomalies_5sigma = np.sum(test_scores > z_score_threshold_5)

    print(f"\n基于训练集统计的异常检测:")
    print(
        f"  [主要] 95%分位数阈值: {threshold_95:.6f}, 异常样本: {anomalies_95percent}/{len(test_scores)} ({anomalies_95percent / len(test_scores):.2%})")
    print(
        f"  [参考] 3σ阈值: {z_score_threshold_3:.6f}, 异常样本: {anomalies_3sigma}/{len(test_scores)} ({anomalies_3sigma / len(test_scores):.2%})")
    print(
        f"  [参考] 5σ阈值: {z_score_threshold_5:.6f}, 异常样本: {anomalies_5sigma}/{len(test_scores)} ({anomalies_5sigma / len(test_scores):.2%})")

    # 绘制对比图（在图中标注95%阈值）
    plt.figure(figsize=(10, 6))

    plt.hist(train_scores, bins=50, alpha=0.7, label='train_set', color='blue', edgecolor='black')
    plt.hist(test_scores, bins=50, alpha=0.7, label='test_set', color='red', edgecolor='black')
    plt.axvline(threshold_95, color='green', linestyle='-', linewidth=2, label=f'95%threshold: {threshold_95:.4f}')
    plt.axvline(z_score_threshold_3, color='orange', linestyle='--', label=f'3σThreshold: {z_score_threshold_3:.4f}')
    plt.axvline(z_score_threshold_5, color='red', linestyle='--', label=f'5σThreshold: {z_score_threshold_5:.4f}')

    plt.xlabel('Abnormal score')
    plt.ylabel('frequency')
    plt.title('distribution of abnormal scores  train_set vs test_set')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(os.path.join(save_path, 'train_test_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"训练测试对比图已保存至: {os.path.join(save_path, 'train_test_comparison.png')}")
    else:
        plt.savefig('train_test_comparison.png', dpi=300, bbox_inches='tight')

    plt.close()

    return {
        'threshold_95': float(threshold_95),
        'train_scores': train_scores.tolist(),
        'test_scores': test_scores.tolist(),
        'z_score_threshold_3': float(z_score_threshold_3),
        'z_score_threshold_5': float(z_score_threshold_5),
        'anomalies_95percent': int(anomalies_95percent),
        'anomalies_3sigma': int(anomalies_3sigma),
        'anomalies_5sigma': int(anomalies_5sigma)
    }


def save_anomaly_scores(anomaly_scores, save_path, filename='anomaly_scores.npy'):
    """保存异常分数"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, filename), anomaly_scores)
    print(f"异常分数已保存至: {os.path.join(save_path, filename)}")


def main_unsupervised():
    """主函数 - 无监督异常检测"""
    # 配置参数
    base_dir = r"D:\tools\Pycode\CapsnetForAD"
    h5_path = os.path.join(base_dir, "workresult", "testfeatures.h5")
    output_dir = os.path.join(base_dir, "workresult", "superResnet_capsnet_results")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 16
    num_epochs = 150
    input_shape = (311, 64)
    learning_rate = 0.0005
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 准备数据
    print("准备数据...")
    with h5py.File(h5_path, 'r') as f:
        all_features = f['F'][:]
        print(f"加载所有特征形状: {all_features.shape}")

        # 验证形状
        if all_features.shape[1:] != input_shape:
            print(f"警告: 数据形状 {all_features.shape[1:]} 与预期形状 {input_shape} 不匹配")
            # 可以选择自动调整或报错
            # input_shape = all_features.shape[1:]  # 自动调整

    # 数据预处理
    # all_features = robust_data_preprocessing(all_features)

    # 划分数据集
    total_size = len(all_features)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    X_train = all_features[:train_size]
    X_val = all_features[train_size:train_size + val_size]
    X_test = all_features[train_size + val_size:]

    print(f"\n数据集划分:")
    print(f"训练集: {X_train.shape} ({len(X_train) / total_size:.1%})")
    print(f"验证集: {X_val.shape} ({len(X_val) / total_size:.1%})")
    print(f"测试集: {X_test.shape} ({len(X_test) / total_size:.1%})")

    # 创建数据加载器
    train_dataset = H5AudioDataset(X_train, np.zeros(len(X_train)))
    val_dataset = H5AudioDataset(X_val, np.zeros(len(X_val)))
    test_dataset = H5AudioDataset(X_test, np.zeros(len(X_test)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建ResNet胶囊网络模型
    print("\n创建ResNet胶囊网络...")
    model = create_resnet_capsnet(input_shape=input_shape)
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 训练模型
    print("\n开始训练ResNet胶囊网络...")
    history = train_resnet_capsnet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        verbose=True
    )

    # === 新增：在训练集上计算95%阈值 ===
    print("\n计算训练集95%阈值...")
    model.eval()
    train_anomaly_scores = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            scores = model.compute_anomaly_score(data)
            train_anomaly_scores.extend(scores)

            if batch_idx % 10 == 0:
                print(f"计算训练集异常分数进度: {batch_idx * train_loader.batch_size}/{len(train_loader.dataset)}")

    train_anomaly_scores = np.array(train_anomaly_scores)
    threshold_95 = np.percentile(train_anomaly_scores, 95)

    print(f"训练集异常分数统计:")
    print(f"  范围: [{np.min(train_anomaly_scores):.6f}, {np.max(train_anomaly_scores):.6f}]")
    print(f"  均值: {np.mean(train_anomaly_scores):.6f}, 标准差: {np.std(train_anomaly_scores):.6f}")
    print(f"  95%分位数阈值: {threshold_95:.6f}")

    # 保存模型（包含阈值）
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'history': history,
        'input_shape': input_shape,
        'threshold_95': threshold_95,  # 新增：保存95%阈值
        'train_anomaly_stats': {  # 新增：保存训练集统计信息
            'min': float(np.min(train_anomaly_scores)),
            'max': float(np.max(train_anomaly_scores)),
            'mean': float(np.mean(train_anomaly_scores)),
            'std': float(np.std(train_anomaly_scores)),
            'percentile_95': float(threshold_95),
            'n_samples': len(train_anomaly_scores)
        },
        'training_time': datetime.now().isoformat()
    }
    model_path = os.path.join(output_dir, 'resnet_capsnet_model.pth')
    torch.save(checkpoint, model_path)
    print(f"\n模型和阈值已保存至: {model_path}")
    print(f"训练完成！最佳验证损失: {history['best_val_loss']:.6f}")
    print(f"95%阈值: {threshold_95:.6f}")

    # 保存阈值到单独的JSON文件以便查看
    threshold_info = {
        'threshold_95': float(threshold_95),
        'train_stats': checkpoint['train_anomaly_stats'],
        'best_val_loss': float(history['best_val_loss']),
        'calculation_time': datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'threshold_info.json'), 'w') as f:
        json.dump(threshold_info, f, indent=2)
    print(f"阈值信息已保存至: {os.path.join(output_dir, 'threshold_info.json')}")

    # 生成测试集异常分数
    print("\n生成测试集异常分数...")
    test_anomaly_scores = generate_anomaly_scores(model, test_loader, device)

    # 保存异常分数
    save_anomaly_scores(test_anomaly_scores, output_dir, 'test_anomaly_scores.npy')
    print(f"异常分数范围: {np.min(test_anomaly_scores):.6f} - {np.max(test_anomaly_scores):.6f}")

    # 分析异常分数
    analysis_results = analyze_anomaly_scores_detailed(
        test_anomaly_scores,
        save_path=output_dir
    )

    # 可视化重建结果
    print("\n生成重建结果可视化...")
    visualize_reconstruction_results(
        model, test_loader, device,
        num_samples=3,
        save_path=output_dir
    )

    # 评估模型性能
    print("\n进行模型性能评估...")
    performance_results = evaluate_model_performance(
        model, train_loader, test_loader, threshold_95, device,  # 传入threshold_95
        save_path=output_dir
    )

    # 保存训练历史
    training_history = {
        'train_losses': [float(x) for x in history['train_losses']],
        'val_losses': [float(x) for x in history['val_losses']],
        'best_val_loss': float(history['best_val_loss']),
        'performance': performance_results,
        'analysis': analysis_results
    }

    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n训练历史已保存至: {os.path.join(output_dir, 'training_history.json')}")

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training loss')
    plt.plot(history['val_losses'], label='Verification loss')
    plt.xlabel('Round')
    plt.ylabel('loss')
    plt.title('training process')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练损失曲线已保存至: {os.path.join(output_dir, 'training_loss_curve.png')}")

    print(f"\n{'=' * 60}")
    print("所有任务完成！")
    print(f"结果保存在: {output_dir}")
    print(f"{'=' * 60}")

    return model, history, test_anomaly_scores


if __name__ == "__main__":
    print("开始ResNet胶囊网络无监督异常检测...")
    model, history, test_anomaly_scores = main_unsupervised()