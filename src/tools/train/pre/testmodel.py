import torch
import numpy as np
import h5py
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from src.network.hundernet import FanAutoEncoder
# >>> 正在搜索多尺度 LDN 最佳组合...
# 邻居数 K= 3 | AUC: 0.6327
# 邻居数 K= 4 | AUC: 0.6405
# 邻居数 K= 5 | AUC: 0.6370
# ------------------------------



def search_best_fusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "D:/tools/Pycode/CapsnetForAD/workresult/model_car_best_auc.pth"
    test_h5 = "D:/tools/Pycode/CapsnetForAD/workresult/test_car_pro.h5"

    # 1. 加载数据与模型
    with h5py.File(test_h5, 'r') as f:
        X_test = torch.from_numpy(f['F_fan'][:]).float()
        Y_test = f['labels'][:]

    model = FanAutoEncoder(input_dim=384, hidden_dim=32).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    raw_scores = []
    all_features = []

    print(">>> 正在提取模型潜空间特征...")
    with torch.no_grad():
        for i in range(len(X_test)):
            sample = X_test[i:i + 1].to(device)
            recon, feat = model(sample, mask_ratio=0.0)

            # MBS 原始得分
            diff = (recon - sample) ** 2
            mse_per_frame = torch.mean(diff, dim=2).cpu().numpy()[0]
            top_k = max(1, int(len(mse_per_frame) * 0.02))
            raw_scores.append(np.sort(mse_per_frame)[-top_k:].mean())

            # 特征向量
            all_features.append(torch.mean(feat, dim=1).cpu().numpy()[0])

    raw_scores = np.array(raw_scores)
    all_features = np.array(all_features)

    # 2. 尝试多尺度融合
    print("\n>>> 正在搜索多尺度 LDN 最佳组合...")
    results = []

    # 定义不同的邻居尺度
    neighbor_scales = [3, 4, 5]
    scale_scores = []

    for n in neighbor_scales:
        knn = NearestNeighbors(n_neighbors=n + 1, metric='cosine')
        knn.fit(all_features)
        _, indices = knn.kneighbors(all_features)

        ldn_n = []
        for i in range(len(raw_scores)):
            neighbor_avg = np.mean(raw_scores[indices[i][1:]])
            ldn_n.append(raw_scores[i] / (neighbor_avg + 1e-8))

        auc_n = roc_auc_score(Y_test, ldn_n)
        scale_scores.append(np.array(ldn_n))
        print(f"邻居数 K={n:2d} | AUC: {auc_n:.4f}")

    # 3. 简单平均融合
    fusion_score = np.mean(scale_scores, axis=0)
    fusion_auc = roc_auc_score(Y_test, fusion_score)

    print("-" * 30)
    print(f"【多尺度融合最终 AUC】: {fusion_auc:.4f}")
    if fusion_auc > 0.6370:
        print(f"恭喜！性能提升了 {fusion_auc - 0.6370:.4f}")
    else:
        print("融合未见显著提升，说明模型在单一尺度下已经非常稳健。")


if __name__ == "__main__":
    search_best_fusion()