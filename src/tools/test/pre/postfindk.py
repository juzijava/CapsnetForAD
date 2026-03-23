import torch
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import median_filter
from src.network.hundernet import FanAutoEncoder


def run_window_scan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "D:/tools/Pycode/CapsnetForAD/workresult/model_nomon_car_best_auc_second.pth"
    test_h5 = "D:/tools/Pycode/CapsnetForAD/workresult/test_car_nomon.h5"

    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with h5py.File(test_h5, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    # 1. 预先提取原始 MSE 序列和特征
    all_mse_series = []
    all_features = []
    print(">>> 正在预提取原始数据...")
    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i:i + 1].to(device)
            recon, feat = model(sample, mask_ratio=0.0)
            mse_series = torch.mean((recon - sample) ** 2, dim=2).cpu().numpy()[0]
            all_mse_series.append(mse_series)
            all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

    all_features = np.array(all_features)

    # 2. 遍历不同的中值滤波窗口大小
    results = []
    windows = [1, 3, 5, 7, 9, 11, 13, 15]  # 1 表示不平滑

    print("\nWindow Size | AUC Score")
    print("-" * 25)

    for size in windows:
        raw_scores = []
        for mse_series in all_mse_series:
            # 应用不同尺寸的平滑
            if size > 1:
                smoothed = median_filter(mse_series, size=size)
            else:
                smoothed = mse_series

            top_k = max(1, int(len(smoothed) * 0.02))
            raw_scores.append(np.sort(smoothed)[-top_k:].mean())

        raw_scores = np.array(raw_scores)

        # 计算 LDN (固定 K=5)
        knn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(all_features)
        _, indices = knn.kneighbors(all_features)

        final_scores = []
        for i in range(len(raw_scores)):
            neighbor_avg = np.mean(raw_scores[indices[i][1:]])
            final_scores.append(raw_scores[i] / (neighbor_avg + 1e-8))

        auc = roc_auc_score(Y_test, final_scores)
        print(f"{size:<11} | {auc:.4f}")
        results.append(auc)

    best_idx = np.argmax(results)
    print("-" * 25)
    print(f"最优 Window Size: {windows[best_idx]}, 最高 AUC: {results[best_idx]:.4f}")


if __name__ == "__main__":
    run_window_scan()