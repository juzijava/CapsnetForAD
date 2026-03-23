import torch
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import median_filter
from src.network.hundernet import FanAutoEncoder


def run_smooth_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "D:/tools/Pycode/CapsnetForAD/workresult/model_car_best_auc.pth"
    test_h5 = "D:/tools/Pycode/CapsnetForAD/workresult/test_car_pro.h5"

    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with h5py.File(test_h5, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    raw_scores, all_features = [], []
    print(">>> 提取特征中...")
    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i:i + 1].to(device)
            recon, feat = model(sample, mask_ratio=0.0)

            # 计算原始 MSE 序列
            diff = (recon - sample) ** 2
            mse_series = torch.mean(diff, dim=2).cpu().numpy()[0]

            # --- 关键：时序平滑处理 ---
            # 使用窗口大小为 5 的中值滤波，消除瞬间噪点
            smoothed_mse = median_filter(mse_series, size=5)

            # 提取平滑后的 Top 2% 得分作为该样本的原始分
            top_k = max(1, int(len(smoothed_mse) * 0.02))
            raw_scores.append(np.sort(smoothed_mse)[-top_k:].mean())

            all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

    raw_scores = np.array(raw_scores)
    all_features = np.array(all_features)

    # 计算 K=5 的 LDN
    knn = NearestNeighbors(n_neighbors=6, metric='cosine').fit(all_features)
    _, indices = knn.kneighbors(all_features)

    final_scores = []
    for i in range(len(raw_scores)):
        neighbor_avg = np.mean(raw_scores[indices[i][1:]])
        final_scores.append(raw_scores[i] / (neighbor_avg + 1e-8))

    auc = roc_auc_score(Y_test, final_scores)
    print("\n" + "=" * 30)
    print(f"应用时序中值平滑 (Window=5)")
    print(f"当前 AUC: {auc:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    run_smooth_eval()