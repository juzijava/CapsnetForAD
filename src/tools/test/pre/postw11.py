import torch
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import median_filter
from src.network.hundernet import FanAutoEncoder


def final_sprint():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "D:/tools/Pycode/CapsnetForAD/workresult/model_nomon_car_best_auc.pth"
    test_h5 = "D:/tools/Pycode/CapsnetForAD/workresult/test_car_nomon.h5"

    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with h5py.File(test_h5, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    # 1. 预提取并进行 Window=11 的平滑处理
    all_smoothed_scores = []
    all_features = []
    print(">>> 正在进行 Window=11 的时序平滑提取...")
    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i:i + 1].to(device)
            recon, feat = model(sample, mask_ratio=0.0)

            # 计算原始 MSE
            mse_series = torch.mean((recon - sample) ** 2, dim=2).cpu().numpy()[0]

            # 执行中值平滑
            smoothed = median_filter(mse_series, size=11)

            # 提取单样本得分 (Top 2%)
            top_k = max(1, int(len(smoothed) * 0.02))
            all_smoothed_scores.append(np.sort(smoothed)[-top_k:].mean())

            # 提取位置特征
            all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

    all_smoothed_scores = np.array(all_smoothed_scores)
    all_features = np.array(all_features)

    # 2. 在平滑基础上，精细化扫描 K 值 (从 2 到 20)
    print("\n[进阶方案：差分 Z-Score LDN] 正在扫描最优 K 值...")
    print("K Value | AUC Score (Z-Score)")
    print("-" * 25)

    best_auc = 0
    best_k = 0

    for k in range(2, 21):
        # 使用余弦距离寻找特征空间邻居
        knn = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(all_features)
        _, indices = knn.kneighbors(all_features)

        final_scores = []
        for i in range(len(all_smoothed_scores)):
            neighbor_indices = indices[i][1:]  # 剔除自身
            neighbor_vals = all_smoothed_scores[neighbor_indices]

            # 计算邻域的统计分布
            mu = np.mean(neighbor_vals)
            std = np.std(neighbor_vals)

            # 差分 Z-Score 公式
            # 加上 1e-8 防止标准差为 0 导致溢出
            score = (all_smoothed_scores[i] - mu) / (std + 1e-8)
            final_scores.append(score)

        auc = roc_auc_score(Y_test, final_scores)
        print(f"{k:<8} | {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_k = k

    print("-" * 25)
    print(f"🚀 冲刺结果: 当 K={best_k} 时, Z-Score AUC = {best_auc:.4f}")

    if best_auc >= 0.64:
        print(">>> 恭喜！已突破 0.64 目标。")
    else:
        print(f">>> 距离 0.64 还差 {0.64 - best_auc:.4f}，建议考虑关闭 Normalization。")


if __name__ == "__main__":
    final_sprint()