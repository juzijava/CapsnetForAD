import torch
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
import os
from src.network.hundernet import FanAutoEncoder
# =======================================================
# Fan Domain      | AUC (Top2%)     | pAUC (0.1)
# -------------------------------------------------------
# Overall         | 0.6862         | 0.5237
# Source          | 0.7568         | 0.5726
# Target          | 0.6088         | 0.4905
# =======================================================

def evaluate_fan_with_topk_and_domains():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 路径配置 (确保指向 Fan 的模型和 H5)
    model_path = "D:/tools/Pycode/CapsnetForAD/workresult/model_nmon_fan_best_auc.pth"
    test_h5 = "D:/tools/Pycode/CapsnetForAD/workresult/test_fan_nomon.h5"
    test_dir = "D:/tools/Pycode/CapsnetForAD/data/Fan/test"  # 用于解析 Domain

    # 2. 加载模型
    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f">>> 成功加载 Fan 最佳模型，采用 Top 2% 评估策略...")

    # 3. 加载数据与 Domain 标签
    with h5py.File(test_h5, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    # 解析 Domain (0=Source, 1=Target)
    audio_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.wav')])
    domain_labels = np.array([1 if 'target' in f.lower() else 0 for f in audio_files])

    # 4. 推理 (完全对齐 newstrain.py 逻辑)
    raw_scores = []
    all_features = []

    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i:i + 1].to(device)
            recon, feat = model(sample, mask_ratio=0.0)

            # 计算 Top 2% MSE
            diff = (recon - sample) ** 2
            mse_per_frame = torch.mean(diff, dim=2).cpu().numpy()[0]  # [T]
            top_k = max(1, int(len(mse_per_frame) * 0.02))
            score = np.sort(mse_per_frame)[-top_k:].mean()

            raw_scores.append(score)
            all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

    # 5. LDN 校准 (对齐 newstrain.py 的 K=6)
    all_features = np.array(all_features)
    raw_scores = np.array(raw_scores)

    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(all_features)
    _, indices = knn.kneighbors(all_features)

    ldn_scores = []
    for i in range(len(raw_scores)):
        neighbor_indices = indices[i][1:]
        neighbor_avg = np.mean(raw_scores[neighbor_indices])
        ldn_scores.append(raw_scores[i] / (neighbor_avg + 1e-8))
    ldn_scores = np.array(ldn_scores)

    # 6. 输出结果
    print("\n" + "=" * 55)
    print(f"{'Fan Domain':<15} | {'AUC (Top2%)':<15} | {'pAUC (0.1)':<15}")
    print("-" * 55)

    def print_metrics(mask, name):
        y, s = Y_test[mask], ldn_scores[mask]
        auc_val = roc_auc_score(y, s)
        pauc_val = roc_auc_score(y, s, max_fpr=0.1)
        print(f"{name:<15} | {auc_val:.4f}         | {pauc_val:.4f}")

    print_metrics(np.ones_like(Y_test, dtype=bool), "Overall")
    print_metrics(domain_labels == 0, "Source")
    print_metrics(domain_labels == 1, "Target")
    print("=" * 55)


if __name__ == "__main__":
    evaluate_fan_with_topk_and_domains()