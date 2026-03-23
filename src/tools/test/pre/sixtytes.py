import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neighbors import NearestNeighbors
import os

# 导入你的模型架构
from src.network.hundernet import FanAutoEncoder


def evaluate_060_model():
    # --- 1. 环境配置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "/workresult/model_toycar_nomon_best.pth"
    test_data_path = "/workresult/test_car_nomon.h5"

    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    # --- 2. 加载模型 ---
    # 确保参数与训练时一致：input_dim=384, hidden_dim=32
    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f">>> 成功加载最佳模型: {model_path}")

    # --- 3. 加载测试数据 ---
    # 注意：这里的 h5 必须是【没有经过归一化】提取的版本
    with h5py.File(test_data_path, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]
    print(f">>> 测试样本数量: {len(X_test)}")

    # --- 4. 执行推理 ---
    raw_scores = []
    all_features = []

    # 定义评估权重（与训练后期逻辑对齐：侧重 Delta 区域）
    eval_weights = torch.ones(384).to(device)
    eval_weights[:128] = 0.1  # 压低 Log-Mel
    eval_weights[128:] = 5.0  # 增强 Delta 结构感

    print(">>> 正在计算重构误差与特征分布...")
    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i:i + 1].to(device)
            # 评估时 mask_ratio 必须为 0
            recon, feat = model(sample, mask_ratio=0.0)

            # 计算加权 MSE
            diff = (recon - sample) ** 2
            weighted_diff = diff * eval_weights.view(1, 1, -1)

            # 采用 Mean 策略
            score = torch.mean(weighted_diff).item()
            raw_scores.append(score)

            # 提取 Bottleneck 特征用于 LDN (时间维度取平均)
            all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

    # --- 5. 核心：LDN (Local Density Normalization) ---
    print(">>> 正在执行 LDN 局部密度校准...")
    all_features = np.array(all_features)
    raw_scores = np.array(raw_scores)

    # 使用 Cosine 距离寻找最近邻
    knn = NearestNeighbors(n_neighbors=10, metric='cosine')
    knn.fit(all_features)
    _, indices = knn.kneighbors(all_features)

    ldn_scores = []
    for i in range(len(raw_scores)):
        # 排除自身后的 9 个邻居
        neighbor_indices = indices[i][1:]
        neighbor_avg = np.mean(raw_scores[neighbor_indices])
        # 计算异常率：自身得分 / 邻居平均得分
        ldn_scores.append(raw_scores[i] / (neighbor_avg + 1e-8))

    # --- 6. 计算指标 ---
    final_auc = roc_auc_score(Y_test, ldn_scores)

    # 额外计算 pAUC (DCASE 常用指标，聚焦低误报率区域)
    # 虽然这里只算标准 AUC，但可以作为参考
    print("\n" + "=" * 30)
    print(f"🏆 最终评测结果 (ToyCar)")
    print(f"LDN AUC: {final_auc:.4f}")
    print("=" * 30)

    # 如果你想看 PR 曲线（可选）
    precision, recall, _ = precision_recall_curve(Y_test, ldn_scores)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    evaluate_060_model()